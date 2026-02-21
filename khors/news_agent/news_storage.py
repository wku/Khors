"""
Хранилище данных новостного агента с дедупликацией.
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Set, Dict, Any
from contextlib import contextmanager

from .models import RawNews, AnalyzedNews, NewsDigest, NewsImportance, NewsCategory, AgentConfig


logger = logging.getLogger(__name__)


class NewsStorage:
    """Хранилище данных новостного агента."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.storage_path = Path(config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_path / "news.db"
        self._init_database()
    
    def _init_database(self):
        """Инициализация базы данных."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    url TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    published_at TIMESTAMP NOT NULL,
                    author TEXT,
                    tags TEXT,  -- JSON array
                    raw_data TEXT,  -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analyzed_news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    raw_news_id INTEGER NOT NULL,
                    importance TEXT NOT NULL,
                    category TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    key_points TEXT NOT NULL,  -- JSON array
                    social_impact_score REAL NOT NULL,
                    analysis_reasoning TEXT NOT NULL,
                    analyzed_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (raw_news_id) REFERENCES raw_news (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_digests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    digest_id TEXT UNIQUE NOT NULL,
                    period_start TIMESTAMP NOT NULL,
                    period_end TIMESTAMP NOT NULL,
                    summary_text TEXT NOT NULL,
                    total_news_processed INTEGER NOT NULL,
                    news_ids TEXT NOT NULL,  -- JSON array of analyzed_news IDs
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Индексы для производительности
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_news_hash ON raw_news (content_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_news_published ON raw_news (published_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analyzed_importance ON analyzed_news (importance)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analyzed_category ON analyzed_news (category)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Контекстный менеджер для подключения к БД."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Для доступа по именам колонок
        try:
            yield conn
        finally:
            conn.close()
    
    async def store_raw_news(self, news_list: List[RawNews]) -> List[RawNews]:
        """Сохранить сырые новости, исключив дубликаты."""
        if not news_list:
            return []
        
        # Получаем существующие хеши
        existing_hashes = self._get_existing_hashes([news.content_hash for news in news_list])
        
        # Фильтруем новые новости
        new_news = [news for news in news_list if news.content_hash not in existing_hashes]
        
        if not new_news:
            logger.info("No new news to store (all duplicates)")
            return []
        
        # Сохраняем новые новости
        with self._get_connection() as conn:
            for news in new_news:
                try:
                    conn.execute("""
                        INSERT INTO raw_news 
                        (content_hash, title, content, url, source_name, published_at, author, tags, raw_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        news.content_hash,
                        news.title,
                        news.content,
                        news.url,
                        news.source_name,
                        news.published_at,
                        news.author,
                        json.dumps(news.tags, ensure_ascii=False),
                        json.dumps(news.raw_data, ensure_ascii=False, default=str)
                    ))
                except sqlite3.IntegrityError:
                    # Дубликат по хешу (race condition)
                    logger.debug(f"Duplicate hash detected for: {news.title}")
                    continue
            
            conn.commit()
        
        logger.info(f"Stored {len(new_news)} new raw news items")
        return new_news
    
    async def store_analyzed_news(self, analyzed_news_list: List[AnalyzedNews]) -> None:
        """Сохранить проанализированные новости."""
        if not analyzed_news_list:
            return
        
        with self._get_connection() as conn:
            for analyzed_news in analyzed_news_list:
                # Находим ID сырой новости
                raw_news_id = self._get_raw_news_id(conn, analyzed_news.raw_news.content_hash)
                if raw_news_id is None:
                    logger.warning(f"Raw news not found for hash: {analyzed_news.raw_news.content_hash}")
                    continue
                
                conn.execute("""
                    INSERT INTO analyzed_news 
                    (raw_news_id, importance, category, summary, key_points, 
                     social_impact_score, analysis_reasoning, analyzed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    raw_news_id,
                    analyzed_news.importance.value,
                    analyzed_news.category.value,
                    analyzed_news.summary,
                    json.dumps(analyzed_news.key_points, ensure_ascii=False),
                    analyzed_news.social_impact_score,
                    analyzed_news.analysis_reasoning,
                    analyzed_news.analyzed_at
                ))
            
            conn.commit()
        
        logger.info(f"Stored {len(analyzed_news_list)} analyzed news items")
    
    async def store_digest(self, digest: NewsDigest) -> None:
        """Сохранить дайджест."""
        # Собираем ID новостей из дайджеста
        news_ids = []
        all_news = digest.critical_news + digest.important_news
        
        with self._get_connection() as conn:
            for news in all_news:
                raw_news_id = self._get_raw_news_id(conn, news.raw_news.content_hash)
                if raw_news_id:
                    # Находим ID проанализированной новости
                    cursor = conn.execute("""
                        SELECT id FROM analyzed_news 
                        WHERE raw_news_id = ? 
                        ORDER BY created_at DESC LIMIT 1
                    """, (raw_news_id,))
                    row = cursor.fetchone()
                    if row:
                        news_ids.append(row['id'])
            
            conn.execute("""
                INSERT INTO news_digests 
                (digest_id, period_start, period_end, summary_text, total_news_processed, news_ids)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                digest.digest_id,
                digest.period_start,
                digest.period_end,
                digest.summary_text,
                digest.total_news_processed,
                json.dumps(news_ids)
            ))
            
            conn.commit()
        
        logger.info(f"Stored digest {digest.digest_id}")
    
    def _get_existing_hashes(self, hashes: List[str]) -> Set[str]:
        """Получить существующие хеши из БД."""
        if not hashes:
            return set()
        
        with self._get_connection() as conn:
            placeholders = ','.join(['?' for _ in hashes])
            cursor = conn.execute(f"""
                SELECT content_hash FROM raw_news 
                WHERE content_hash IN ({placeholders})
            """, hashes)
            
            return {row['content_hash'] for row in cursor.fetchall()}
    
    def _get_raw_news_id(self, conn: sqlite3.Connection, content_hash: str) -> Optional[int]:
        """Получить ID сырой новости по хешу."""
        cursor = conn.execute("""
            SELECT id FROM raw_news WHERE content_hash = ?
        """, (content_hash,))
        
        row = cursor.fetchone()
        return row['id'] if row else None
    
    async def get_recent_analyzed_news(
        self, 
        hours: int = 24,
        min_importance: Optional[NewsImportance] = None,
        categories: Optional[List[NewsCategory]] = None,
        limit: Optional[int] = None
    ) -> List[AnalyzedNews]:
        """Получить недавние проанализированные новости."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        query = """
            SELECT an.*, rn.* FROM analyzed_news an
            JOIN raw_news rn ON an.raw_news_id = rn.id
            WHERE rn.published_at >= ?
        """
        params = [cutoff_time]
        
        if min_importance:
            importance_order = {
                NewsImportance.SKIP: 0,
                NewsImportance.LOW: 1,
                NewsImportance.MEDIUM: 2,
                NewsImportance.HIGH: 3,
                NewsImportance.CRITICAL: 4
            }
            min_level = importance_order[min_importance]
            
            query += " AND CASE an.importance"
            query += " WHEN 'skip' THEN 0"
            query += " WHEN 'low' THEN 1"
            query += " WHEN 'medium' THEN 2"
            query += " WHEN 'high' THEN 3"
            query += " WHEN 'critical' THEN 4"
            query += " END >= ?"
            params.append(min_level)
        
        if categories:
            placeholders = ','.join(['?' for _ in categories])
            query += f" AND an.category IN ({placeholders})"
            params.extend([cat.value for cat in categories])
        
        query += " ORDER BY rn.published_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_analyzed_news(row) for row in rows]
    
    def _row_to_analyzed_news(self, row: sqlite3.Row) -> AnalyzedNews:
        """Преобразовать строку БД в AnalyzedNews."""
        raw_news = RawNews(
            title=row['title'],
            content=row['content'],
            url=row['url'],
            source_name=row['source_name'],
            published_at=datetime.fromisoformat(row['published_at']),
            author=row['author'],
            tags=json.loads(row['tags']) if row['tags'] else [],
            raw_data=json.loads(row['raw_data']) if row['raw_data'] else {}
        )
        
        return AnalyzedNews(
            raw_news=raw_news,
            importance=NewsImportance(row['importance']),
            category=NewsCategory(row['category']),
            summary=row['summary'],
            key_points=json.loads(row['key_points']),
            social_impact_score=row['social_impact_score'],
            analysis_reasoning=row['analysis_reasoning'],
            analyzed_at=datetime.fromisoformat(row['analyzed_at'])
        )
    
    async def cleanup_old_data(self) -> None:
        """Очистка старых данных."""
        cutoff_date = datetime.now() - timedelta(days=self.config.max_storage_days)
        
        with self._get_connection() as conn:
            # Удаляем старые дайджесты
            cursor = conn.execute("""
                DELETE FROM news_digests WHERE created_at < ?
            """, (cutoff_date,))
            digests_deleted = cursor.rowcount
            
            # Удаляем старые анализы
            cursor = conn.execute("""
                DELETE FROM analyzed_news WHERE created_at < ?
            """, (cutoff_date,))
            analyzed_deleted = cursor.rowcount
            
            # Удаляем сырые новости без анализов
            cursor = conn.execute("""
                DELETE FROM raw_news 
                WHERE created_at < ? 
                AND id NOT IN (SELECT raw_news_id FROM analyzed_news)
            """, (cutoff_date,))
            raw_deleted = cursor.rowcount
            
            conn.commit()
        
        logger.info(f"Cleanup complete: {digests_deleted} digests, {analyzed_deleted} analyses, {raw_deleted} raw news deleted")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Получить статистику хранилища."""
        with self._get_connection() as conn:
            stats = {}
            
            # Общая статистика
            cursor = conn.execute("SELECT COUNT(*) as count FROM raw_news")
            stats['total_raw_news'] = cursor.fetchone()['count']
            
            cursor = conn.execute("SELECT COUNT(*) as count FROM analyzed_news")
            stats['total_analyzed_news'] = cursor.fetchone()['count']
            
            cursor = conn.execute("SELECT COUNT(*) as count FROM news_digests")
            stats['total_digests'] = cursor.fetchone()['count']
            
            # Статистика за последние 24 часа
            cutoff_24h = datetime.now() - timedelta(hours=24)
            cursor = conn.execute("""
                SELECT COUNT(*) as count FROM raw_news WHERE created_at >= ?
            """, (cutoff_24h,))
            stats['raw_news_24h'] = cursor.fetchone()['count']
            
            cursor = conn.execute("""
                SELECT COUNT(*) as count FROM analyzed_news WHERE created_at >= ?
            """, (cutoff_24h,))
            stats['analyzed_news_24h'] = cursor.fetchone()['count']
            
            # Статистика по важности
            cursor = conn.execute("""
                SELECT importance, COUNT(*) as count 
                FROM analyzed_news 
                WHERE created_at >= ?
                GROUP BY importance
            """, (cutoff_24h,))
            stats['importance_distribution_24h'] = {row['importance']: row['count'] for row in cursor.fetchall()}
            
            # Статистика по категориям
            cursor = conn.execute("""
                SELECT category, COUNT(*) as count 
                FROM analyzed_news 
                WHERE created_at >= ?
                GROUP BY category
            """, (cutoff_24h,))
            stats['category_distribution_24h'] = {row['category']: row['count'] for row in cursor.fetchall()}
            
            return stats