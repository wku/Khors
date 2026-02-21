"""
Главный модуль автономного новостного агента.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from .models import AgentConfig
from .news_collector import NewsCollector
from .news_analyzer import NewsAnalyzer
from .news_storage import NewsStorage
from .news_scheduler import NewsScheduler
from .news_sender import NewsSender


logger = logging.getLogger(__name__)


class NewsAgent:
    """Автономный новостной агент."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        
        # Инициализация компонентов
        self.storage = NewsStorage(config)
        self.collector = NewsCollector(config)
        self.analyzer = NewsAnalyzer(config)
        self.scheduler = NewsScheduler(config, self.storage)
        self.sender = NewsSender(config)
        
        # Состояние агента
        self.is_running = False
        self.collection_task: Optional[asyncio.Task] = None
        self.scheduler_task: Optional[asyncio.Task] = None
        
        # Настройка callback для отправки дайджестов
        self.scheduler.set_digest_callback(self._send_digest_callback)
    
    async def start(self):
        """Запустить агента."""
        if self.is_running:
            logger.warning("News agent is already running")
            return
        
        logger.info("Starting news agent...")
        
        try:
            # Тестируем подключения
            await self._test_connections()
            
            # Запускаем компоненты
            self.is_running = True
            
            # Запускаем планировщик дайджестов
            self.scheduler_task = asyncio.create_task(self.scheduler.start())
            
            # Запускаем сборщик новостей
            self.collection_task = asyncio.create_task(self._collection_loop())
            
            # Отправляем уведомление о запуске
            async with self.sender:
                await self.sender.send_status_message("News Agent запущен и готов к работе")
            
            logger.info("News agent started successfully")
            
        except Exception as e:
            logger.error(f"Error starting news agent: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Остановить агента."""
        if not self.is_running:
            return
        
        logger.info("Stopping news agent...")
        
        self.is_running = False
        
        # Останавливаем задачи
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Останавливаем планировщик
        await self.scheduler.stop()
        
        # Отправляем уведомление об остановке
        try:
            async with self.sender:
                await self.sender.send_status_message("News Agent остановлен")
        except:
            pass  # Игнорируем ошибки при остановке
        
        logger.info("News agent stopped")
    
    async def _test_connections(self):
        """Тестировать подключения."""
        logger.info("Testing connections...")
        
        # Тестируем Telegram
        async with self.sender:
            if not await self.sender.test_connection():
                raise Exception("Failed to connect to Telegram")
        
        # Тестируем LLM (если нужно)
        # Можно добавить тест LLM здесь
        
        logger.info("All connections tested successfully")
    
    async def _collection_loop(self):
        """Основной цикл сбора новостей."""
        logger.info("Starting news collection loop")
        
        while self.is_running:
            try:
                await self._collect_and_analyze_news()
                
                # Ждем до следующего сбора
                await asyncio.sleep(self.config.collection_interval_minutes * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                
                # Отправляем уведомление об ошибке
                try:
                    async with self.sender:
                        await self.sender.send_error_message(f"Collection error: {str(e)[:200]}")
                except:
                    pass
                
                # Ждем перед повтором
                await asyncio.sleep(60)
        
        logger.info("News collection loop stopped")
    
    async def _collect_and_analyze_news(self):
        """Собрать и проанализировать новости."""
        logger.info("Starting news collection and analysis")
        
        # Сбор новостей
        async with self.collector:
            raw_news = await self.collector.collect_from_all_sources()
        
        if not raw_news:
            logger.info("No new news collected")
            return
        
        # Сохранение сырых новостей (с дедупликацией)
        new_news = await self.storage.store_raw_news(raw_news)
        
        if not new_news:
            logger.info("No new unique news to analyze")
            return
        
        # Анализ новостей
        analyzed_news = await self.analyzer.analyze_news_batch(new_news)
        
        if analyzed_news:
            # Сохранение проанализированных новостей
            await self.storage.store_analyzed_news(analyzed_news)
            
            # Статистика
            critical_count = sum(1 for news in analyzed_news if news.importance.value == 'critical')
            high_count = sum(1 for news in analyzed_news if news.importance.value == 'high')
            
            logger.info(f"Analysis complete: {len(analyzed_news)} news analyzed "
                       f"({critical_count} critical, {high_count} high importance)")
        
        # Очистка старых данных (раз в день)
        now = datetime.now()
        if now.hour == 3 and now.minute < self.config.collection_interval_minutes:
            await self.storage.cleanup_old_data()
    
    async def _send_digest_callback(self, digest):
        """Callback для отправки дайджеста."""
        async with self.sender:
            success = await self.sender.send_digest(digest)
            
            if success:
                logger.info(f"Digest {digest.digest_id} sent successfully")
            else:
                logger.error(f"Failed to send digest {digest.digest_id}")
    
    async def create_manual_digest(self, hours_back: int = 6):
        """Создать и отправить дайджест вручную."""
        logger.info(f"Creating manual digest for last {hours_back} hours")
        
        digest = await self.scheduler.create_manual_digest(hours_back)
        
        if digest:
            async with self.sender:
                success = await self.sender.send_digest(digest)
                
                if success:
                    await self.storage.store_digest(digest)
                    logger.info(f"Manual digest sent successfully")
                    return True
                else:
                    logger.error(f"Failed to send manual digest")
                    return False
        else:
            logger.info("No relevant news for manual digest")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Получить статус агента."""
        stats = await self.storage.get_stats()
        
        status = {
            'is_running': self.is_running,
            'config': {
                'collection_interval_minutes': self.config.collection_interval_minutes,
                'digest_schedule': self.config.digest_schedule,
                'min_importance_for_digest': self.config.min_importance_for_digest.value,
                'max_news_per_digest': self.config.max_news_per_digest
            },
            'storage_stats': stats,
            'scheduler_info': self.scheduler.get_schedule_info()
        }
        
        return status
    
    async def run_forever(self):
        """Запустить агента и ждать завершения."""
        await self.start()
        
        try:
            # Ждем завершения основных задач
            if self.collection_task and self.scheduler_task:
                await asyncio.gather(self.collection_task, self.scheduler_task)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await self.stop()


# Функция для создания конфигурации по умолчанию
def create_default_config(
    telegram_bot_token: str,
    telegram_chat_id: str,
    llm_api_key: str = "",
    storage_path: str = "data/news_agent"
) -> AgentConfig:
    """Создать конфигурацию по умолчанию."""
    return AgentConfig(
        telegram_bot_token=telegram_bot_token,
        telegram_chat_id=telegram_chat_id,
        llm_api_key=llm_api_key,
        storage_path=storage_path
    )