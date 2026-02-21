"""
Тесты для новостного агента.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from khors.news_agent.models import (
    AgentConfig, RawNews, AnalyzedNews, NewsImportance, 
    NewsCategory, NewsDigest
)
from khors.news_agent.news_storage import NewsStorage
from khors.news_agent.news_scheduler import NewsScheduler


@pytest.fixture
def test_config():
    """Тестовая конфигурация."""
    return AgentConfig(
        telegram_bot_token="test_token",
        telegram_chat_id="test_chat",
        llm_api_key="test_key",
        storage_path="test_data",
        digest_schedule=[6, 12, 18, 22],
        max_news_per_digest=10
    )


@pytest.fixture
def sample_raw_news():
    """Примеры сырых новостей."""
    return [
        RawNews(
            title="Важная новость 1",
            content="Содержание важной новости",
            url="https://example.com/news1",
            source="test_source",
            published_at=datetime.now()
        ),
        RawNews(
            title="Критическая новость",
            content="Содержание критической новости",
            url="https://example.com/news2",
            source="test_source",
            published_at=datetime.now()
        )
    ]


@pytest.fixture
def sample_analyzed_news():
    """Примеры проанализированных новостей."""
    raw_news = RawNews(
        title="Тестовая новость",
        content="Содержание тестовой новости",
        url="https://example.com/test",
        source="test_source",
        published_at=datetime.now()
    )
    
    return [
        AnalyzedNews(
            raw_news=raw_news,
            importance=NewsImportance.CRITICAL,
            category=NewsCategory.POLITICS,
            summary="Краткое содержание",
            keywords=["тест", "новость"],
            analyzed_at=datetime.now()
        )
    ]


class TestNewsStorage:
    """Тесты для NewsStorage."""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_raw_news(self, test_config, sample_raw_news):
        """Тест сохранения и получения сырых новостей."""
        storage = NewsStorage(test_config)
        
        # Сохраняем новости
        stored = await storage.store_raw_news(sample_raw_news)
        assert len(stored) == len(sample_raw_news)
        
        # Получаем новости
        retrieved = await storage.get_recent_raw_news(hours=1)
        assert len(retrieved) >= len(sample_raw_news)
    
    @pytest.mark.asyncio
    async def test_deduplication(self, test_config, sample_raw_news):
        """Тест дедупликации новостей."""
        storage = NewsStorage(test_config)
        
        # Сохраняем новости первый раз
        stored1 = await storage.store_raw_news(sample_raw_news)
        assert len(stored1) == len(sample_raw_news)
        
        # Сохраняем те же новости второй раз
        stored2 = await storage.store_raw_news(sample_raw_news)
        assert len(stored2) == 0  # Дубликаты не должны сохраниться
    
    @pytest.mark.asyncio
    async def test_store_analyzed_news(self, test_config, sample_analyzed_news):
        """Тест сохранения проанализированных новостей."""
        storage = NewsStorage(test_config)
        
        # Сохраняем проанализированные новости
        await storage.store_analyzed_news(sample_analyzed_news)
        
        # Получаем новости
        retrieved = await storage.get_recent_analyzed_news(hours=1)
        assert len(retrieved) >= len(sample_analyzed_news)


class TestNewsScheduler:
    """Тесты для NewsScheduler."""
    
    @pytest.mark.asyncio
    async def test_create_digest(self, test_config, sample_analyzed_news):
        """Тест создания дайджеста."""
        # Мокаем storage
        storage_mock = Mock()
        storage_mock.get_recent_analyzed_news = AsyncMock(return_value=sample_analyzed_news)
        
        scheduler = NewsScheduler(test_config, storage_mock)
        
        # Создаем дайджест
        digest = await scheduler.create_digest(hours_back=6)
        
        assert digest is not None
        assert isinstance(digest, NewsDigest)
        assert len(digest.critical_news) > 0 or len(digest.important_news) > 0
        assert digest.summary_text is not None
    
    @pytest.mark.asyncio
    async def test_empty_digest(self, test_config):
        """Тест создания дайджеста при отсутствии новостей."""
        # Мокаем storage с пустым результатом
        storage_mock = Mock()
        storage_mock.get_recent_analyzed_news = AsyncMock(return_value=[])
        
        scheduler = NewsScheduler(test_config, storage_mock)
        
        # Создаем дайджест
        digest = await scheduler.create_digest(hours_back=6)
        
        assert digest is None
    
    def test_schedule_info(self, test_config):
        """Тест получения информации о расписании."""
        storage_mock = Mock()
        scheduler = NewsScheduler(test_config, storage_mock)
        
        info = scheduler.get_schedule_info()
        
        assert 'digest_hours' in info
        assert 'is_running' in info
        assert info['digest_hours'] == test_config.digest_schedule


@pytest.mark.asyncio
async def test_integration_flow(test_config, sample_raw_news):
    """Интеграционный тест основного потока."""
    storage = NewsStorage(test_config)
    
    # 1. Сохраняем сырые новости
    stored_raw = await storage.store_raw_news(sample_raw_news)
    assert len(stored_raw) > 0
    
    # 2. Создаем проанализированные новости (мокаем анализ)
    analyzed_news = []
    for raw in stored_raw:
        analyzed = AnalyzedNews(
            raw_news=raw,
            importance=NewsImportance.HIGH,
            category=NewsCategory.GENERAL,
            summary=f"Краткое содержание: {raw.title}",
            keywords=["тест"],
            analyzed_at=datetime.now()
        )
        analyzed_news.append(analyzed)
    
    # 3. Сохраняем проанализированные новости
    await storage.store_analyzed_news(analyzed_news)
    
    # 4. Создаем дайджест
    scheduler = NewsScheduler(test_config, storage)
    digest = await scheduler.create_digest(hours_back=1)
    
    assert digest is not None
    assert len(digest.important_news) > 0
    
    # 5. Сохраняем дайджест
    await storage.store_digest(digest)
    
    # 6. Получаем статистику
    stats = await storage.get_stats()
    assert stats['total_raw_news'] > 0
    assert stats['total_analyzed_news'] > 0
    assert stats['total_digests'] > 0


if __name__ == "__main__":
    pytest.main([__file__])