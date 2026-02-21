"""
Автономный новостной агент для сбора и доставки украинских новостей.

Модули:
- news_collector: Сбор новостей из RSS и веб-источников
- news_analyzer: Анализ важности через LLM
- news_scheduler: Планировщик отправки дайджестов
- news_storage: Хранение данных и дедупликация
- news_sender: Отправка в Telegram
- news_agent: Главный оркестратор
"""

from .news_agent import NewsAgent
from .news_collector import NewsCollector
from .news_analyzer import NewsAnalyzer
from .news_scheduler import NewsScheduler
from .news_storage import NewsStorage
from .news_sender import NewsSender

__all__ = [
    'NewsAgent',
    'NewsCollector', 
    'NewsAnalyzer',
    'NewsScheduler',
    'NewsStorage',
    'NewsSender'
]