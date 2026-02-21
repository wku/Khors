"""
Планировщик новостного агента для автоматической отправки дайджестов.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Callable, Dict, Any

from .models import AnalyzedNews, NewsDigest, NewsImportance, AgentConfig
from .news_storage import NewsStorage


logger = logging.getLogger(__name__)


class NewsScheduler:
    """Планировщик для автоматической отправки дайджестов."""
    
    def __init__(self, config: AgentConfig, storage: NewsStorage):
        self.config = config
        self.storage = storage
        self.is_running = False
        self.digest_callback: Optional[Callable[[NewsDigest], None]] = None
        
        # Время последней отправки дайджестов
        self.last_digest_times: Dict[int, datetime] = {}
    
    def set_digest_callback(self, callback: Callable[[NewsDigest], None]):
        """Установить callback для отправки дайджестов."""
        self.digest_callback = callback
    
    async def start(self):
        """Запустить планировщик."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        logger.info("News scheduler started")
        
        try:
            while self.is_running:
                await self._check_and_send_digests()
                await asyncio.sleep(60)  # Проверяем каждую минуту
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
        finally:
            self.is_running = False
            logger.info("News scheduler stopped")
    
    async def stop(self):
        """Остановить планировщик."""
        self.is_running = False
    
    async def _check_and_send_digests(self):
        """Проверить и отправить дайджесты по расписанию."""
        now = datetime.now()
        current_hour = now.hour
        
        # Проверяем, нужно ли отправлять дайджест в этот час
        if current_hour not in self.config.digest_schedule:
            return
        
        # Проверяем, не отправляли ли уже дайджест в этот час
        last_digest_time = self.last_digest_times.get(current_hour)
        if last_digest_time and last_digest_time.date() == now.date():
            return  # Уже отправляли сегодня в этот час
        
        logger.info(f"Creating digest for hour {current_hour}")
        
        try:
            # Создаем дайджест
            digest = await self.create_digest()
            
            if digest and self.digest_callback:
                # Отправляем дайджест
                await self.digest_callback(digest)
                
                # Сохраняем дайджест в БД
                await self.storage.store_digest(digest)
                
                # Запоминаем время отправки
                self.last_digest_times[current_hour] = now
                
                logger.info(f"Digest sent successfully for hour {current_hour}")
            else:
                logger.info(f"No digest created for hour {current_hour} (no relevant news)")
                
        except Exception as e:
            logger.error(f"Error creating/sending digest for hour {current_hour}: {e}")
    
    async def create_digest(self, hours_back: int = 6) -> Optional[NewsDigest]:
        """Создать дайджест новостей за указанный период."""
        now = datetime.now()
        period_start = now - timedelta(hours=hours_back)
        
        # Получаем новости за период
        all_news = await self.storage.get_recent_analyzed_news(
            hours=hours_back,
            min_importance=self.config.min_importance_for_digest,
            limit=self.config.max_news_per_digest * 2  # Берем с запасом для фильтрации
        )
        
        if not all_news:
            logger.info("No news found for digest")
            return None
        
        # Разделяем по важности
        critical_news = [
            news for news in all_news 
            if news.importance == NewsImportance.CRITICAL
        ][:self.config.max_critical_news_per_digest]
        
        important_news = [
            news for news in all_news 
            if news.importance == NewsImportance.HIGH and news not in critical_news
        ]
        
        # Добавляем средние новости, если есть место
        remaining_slots = self.config.max_news_per_digest - len(critical_news) - len(important_news)
        if remaining_slots > 0:
            medium_news = [
                news for news in all_news 
                if news.importance == NewsImportance.MEDIUM and news not in critical_news and news not in important_news
            ][:remaining_slots]
            important_news.extend(medium_news)
        
        # Если нет важных новостей, не создаем дайджест
        if not critical_news and not important_news:
            logger.info("No important news found for digest")
            return None
        
        # Создаем саммарі дайджеста
        summary_text = await self._create_digest_summary(critical_news, important_news)
        
        digest = NewsDigest(
            created_at=now,
            period_start=period_start,
            period_end=now,
            critical_news=critical_news,
            important_news=important_news,
            summary_text=summary_text,
            total_news_processed=len(all_news)
        )
        
        logger.info(f"Created digest with {len(critical_news)} critical and {len(important_news)} important news")
        return digest
    
    async def _create_digest_summary(self, critical_news: List[AnalyzedNews], important_news: List[AnalyzedNews]) -> str:
        """Создать текстовое саммарі дайджеста."""
        lines = []
        
        # Заголовок
        now = datetime.now()
        lines.append(f"📰 Дайджест новин • {now.strftime('%d.%m.%Y %H:%M')}")
        lines.append("")
        
        # Критичні новини
        if critical_news:
            lines.append("🔴 КРИТИЧНО ВАЖЛИВО:")
            for i, news in enumerate(critical_news, 1):
                lines.append(f"{i}. **{news.raw_news.title}**")
                lines.append(f"   {news.summary}")
                if news.raw_news.url:
                    lines.append(f"   🔗 [Детальніше]({news.raw_news.url})")
                lines.append("")
        
        # Важливі новини
        if important_news:
            lines.append("🔶 ВАЖЛИВО:")
            for i, news in enumerate(important_news, 1):
                lines.append(f"{i}. **{news.raw_news.title}**")
                lines.append(f"   {news.summary}")
                if news.raw_news.url:
                    lines.append(f"   🔗 [Детальніше]({news.raw_news.url})")
                lines.append("")
        
        # Статистика
        total_news = len(critical_news) + len(important_news)
        lines.append(f"📊 Всього новин у дайджесті: {total_news}")
        
        # Категории
        categories = {}
        for news in critical_news + important_news:
            cat = news.category.value
            categories[cat] = categories.get(cat, 0) + 1
        
        if categories:
            cat_text = ", ".join([f"{cat}: {count}" for cat, count in categories.items()])
            lines.append(f"📂 Категорії: {cat_text}")
        
        return "\n".join(lines)
    
    async def create_manual_digest(self, hours_back: int = 6) -> Optional[NewsDigest]:
        """Создать дайджест вручную (для тестирования)."""
        return await self.create_digest(hours_back)
    
    def get_next_digest_time(self) -> Optional[datetime]:
        now = datetime.now()
        current_hour = now.hour
        next_hours = [h for h in self.config.digest_schedule if h > current_hour]
        if next_hours:
            next_hour = min(next_hours)
            return now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
        next_hour = min(self.config.digest_schedule)
        return (now + timedelta(days=1)).replace(hour=next_hour, minute=0, second=0, microsecond=0)

    def get_schedule_info(self) -> Dict[str, Any]:
        next_digest = self.get_next_digest_time()
        return {
            'digest_hours': self.config.digest_schedule,
            'is_running': self.is_running,
            'last_digest_times': {
                hour: t.isoformat()
                for hour, t in self.last_digest_times.items()
            },
            'next_digest_time': next_digest.isoformat() if next_digest else None,
        }