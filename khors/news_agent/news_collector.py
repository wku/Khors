"""
Сборщик новостей из RSS и веб-источников.
"""

import asyncio
import aiohttp
import feedparser
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from .models import NewsSource, RawNews, AgentConfig


logger = logging.getLogger(__name__)


class NewsCollector:
    """Сборщик новостей из различных источников."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Украинские новостные источники
        self.default_sources = [
            NewsSource(
                name="Українська Правда",
                url="https://www.pravda.com.ua/rss/",
                source_type="rss"
            ),
            NewsSource(
                name="BBC Україна",
                url="https://feeds.bbci.co.uk/ukrainian/rss.xml",
                source_type="rss"
            ),
            NewsSource(
                name="Суспільне",
                url="https://suspilne.media/rss/",
                source_type="rss"
            ),
            NewsSource(
                name="УНІАН",
                url="https://rss.unian.net/site/news_ukr.rss",
                source_type="rss"
            ),
            NewsSource(
                name="Цензор.НЕТ",
                url="https://censor.net/includes/news_rss.php",
                source_type="rss"
            ),
            NewsSource(
                name="Детектор медіа",
                url="https://detector.media/rss",
                source_type="rss"
            ),
            NewsSource(
                name="Економічна правда",
                url="https://www.epravda.com.ua/rss/",
                source_type="rss"
            ),
            NewsSource(
                name="Дзеркало тижня",
                url="https://zn.ua/rss/full.rss",
                source_type="rss"
            )
        ]
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (compatible; NewsAgent/1.0)'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def collect_from_all_sources(self, sources: Optional[List[NewsSource]] = None) -> List[RawNews]:
        """Собрать новости из всех источников."""
        if sources is None:
            sources = self.default_sources
        
        all_news = []
        tasks = []
        
        for source in sources:
            if not source.enabled:
                continue
                
            if self._should_check_source(source):
                if source.source_type == "rss":
                    tasks.append(self._collect_from_rss(source))
                elif source.source_type == "web":
                    tasks.append(self._collect_from_web(source))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error collecting from {sources[i].name}: {result}")
                else:
                    all_news.extend(result)
        
        # Ограничиваем количество новостей
        if len(all_news) > self.config.max_news_per_collection:
            # Сортируем по времени публикации (новые первыми)
            all_news.sort(key=lambda x: x.published_at, reverse=True)
            all_news = all_news[:self.config.max_news_per_collection]
        
        logger.info(f"Collected {len(all_news)} news items from {len([s for s in sources if s.enabled])} sources")
        return all_news
    
    def _should_check_source(self, source: NewsSource) -> bool:
        """Проверить, нужно ли обновлять источник."""
        if source.last_check is None:
            return True
        
        time_since_check = datetime.now() - source.last_check
        return time_since_check.total_seconds() >= source.check_interval_minutes * 60
    
    async def _collect_from_rss(self, source: NewsSource) -> List[RawNews]:
        """Собрать новости из RSS."""
        try:
            async with self.session.get(source.url) as response:
                if response.status != 200:
                    logger.warning(f"RSS {source.name}: HTTP {response.status}")
                    return []
                
                content = await response.text()
                feed = feedparser.parse(content)
                
                news_items = []
                cutoff_time = datetime.now() - timedelta(hours=24)  # Только за последние 24 часа
                
                for entry in feed.entries[:50]:  # Ограничиваем количество
                    try:
                        # Парсим дату публикации
                        published_at = self._parse_date(entry.get('published', ''))
                        if published_at and published_at < cutoff_time:
                            continue
                        
                        # Извлекаем контент
                        content = self._extract_content_from_entry(entry)
                        if not content or len(content.strip()) < 50:
                            continue
                        
                        news_item = RawNews(
                            title=entry.get('title', '').strip(),
                            content=content,
                            url=entry.get('link', ''),
                            source_name=source.name,
                            published_at=published_at or datetime.now(),
                            author=entry.get('author', ''),
                            tags=self._extract_tags(entry),
                            raw_data={'entry': entry}
                        )
                        
                        if self._is_valid_news(news_item):
                            news_items.append(news_item)
                            
                    except Exception as e:
                        logger.warning(f"Error parsing RSS entry from {source.name}: {e}")
                        continue
                
                source.last_check = datetime.now()
                logger.info(f"Collected {len(news_items)} items from RSS {source.name}")
                return news_items
                
        except Exception as e:
            logger.error(f"Error collecting RSS from {source.name}: {e}")
            return []
    
    async def _collect_from_web(self, source: NewsSource) -> List[RawNews]:
        """Собрать новости с веб-страницы (для будущего расширения)."""
        # Заглушка для веб-скрапинга
        # Можно реализовать позже для источников без RSS
        logger.info(f"Web scraping not implemented yet for {source.name}")
        return []
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        if not date_str:
            return None
        import email.utils
        try:
            return datetime(*email.utils.parsedate(date_str)[:6])
        except Exception:
            pass
        try:
            from dateutil import parser as _dp
            return _dp.parse(date_str)
        except Exception:
            pass
        return datetime.now()
    
    def _extract_content_from_entry(self, entry: Dict[str, Any]) -> str:
        """Извлечь текстовый контент из RSS entry."""
        # Пробуем разные поля
        content_fields = ['content', 'summary', 'description']
        
        for field in content_fields:
            if field in entry:
                content = entry[field]
                if isinstance(content, list) and content:
                    content = content[0].get('value', '')
                elif isinstance(content, dict):
                    content = content.get('value', '')
                
                if content:
                    # Очищаем HTML
                    soup = BeautifulSoup(content, 'html.parser')
                    text = soup.get_text(separator=' ', strip=True)
                    return text[:2000]  # Ограничиваем длину
        
        return ""
    
    def _extract_tags(self, entry: Dict[str, Any]) -> List[str]:
        """Извлечь теги из RSS entry."""
        tags = []
        
        if 'tags' in entry:
            for tag in entry['tags']:
                if isinstance(tag, dict) and 'term' in tag:
                    tags.append(tag['term'])
                elif isinstance(tag, str):
                    tags.append(tag)
        
        return tags[:10]  # Ограничиваем количество тегов
    
    def _is_valid_news(self, news: RawNews) -> bool:
        """Проверить валидность новости."""
        if not news.title or len(news.title.strip()) < 10:
            return False
        
        if not news.content or len(news.content.strip()) < 50:
            return False
        
        if not news.url:
            return False
        
        # Фильтруем спам и рекламу
        spam_keywords = ['реклама', 'акція', 'знижка', 'купити', 'продати']
        title_lower = news.title.lower()
        
        for keyword in spam_keywords:
            if keyword in title_lower:
                return False
        
        return True