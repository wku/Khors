"""
Модели данных для новостного агента.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import hashlib


class NewsImportance(Enum):
    """Уровни важности новостей."""
    CRITICAL = "critical"      # Критически важные события
    HIGH = "high"             # Высокая важность
    MEDIUM = "medium"         # Средняя важность  
    LOW = "low"              # Низкая важность
    SKIP = "skip"            # Пропустить


class NewsCategory(Enum):
    """Категории новостей."""
    POLITICS = "politics"
    MILITARY = "military"
    ECONOMY = "economy"
    SOCIETY = "society"
    INTERNATIONAL = "international"
    TECHNOLOGY = "technology"
    CULTURE = "culture"
    SPORT = "sport"
    OTHER = "other"


@dataclass
class NewsSource:
    """Источник новостей."""
    name: str
    url: str
    source_type: str  # 'rss', 'web', 'api'
    language: str = 'uk'
    enabled: bool = True
    last_check: Optional[datetime] = None
    check_interval_minutes: int = 30
    selector_config: Optional[Dict[str, str]] = None  # Для веб-скрапинга
    
    
@dataclass
class RawNews:
    """Сырая новость из источника."""
    title: str
    content: str
    url: str
    source_name: str
    published_at: datetime
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def content_hash(self) -> str:
        """Хеш для дедупликации."""
        content = f"{self.title}|{self.content[:500]}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()


@dataclass
class AnalyzedNews:
    """Проанализированная новость."""
    raw_news: RawNews
    importance: NewsImportance
    category: NewsCategory
    summary: str
    key_points: List[str]
    social_impact_score: float  # 0.0 - 1.0
    analysis_reasoning: str
    analyzed_at: datetime
    
    @property
    def is_socially_critical(self) -> bool:
        """Является ли новость социально критичной."""
        return (
            self.importance in [NewsImportance.CRITICAL, NewsImportance.HIGH] and
            self.social_impact_score >= 0.7
        )


@dataclass
class NewsDigest:
    """Дайджест новостей для отправки."""
    created_at: datetime
    period_start: datetime
    period_end: datetime
    critical_news: List[AnalyzedNews]
    important_news: List[AnalyzedNews]
    summary_text: str
    total_news_processed: int
    digest_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M"))


@dataclass
class AgentConfig:
    """Конфигурация новостного агента."""
    # Telegram
    telegram_bot_token: str
    telegram_chat_id: str
    
    # LLM для анализа
    llm_model: str = "anthropic/claude-3-haiku"
    llm_api_key: str = ""
    
    # Расписание (часы UTC)
    digest_schedule: List[int] = field(default_factory=lambda: [6, 12, 18, 22])
    
    # Фильтры
    min_importance_for_digest: NewsImportance = NewsImportance.MEDIUM
    max_news_per_digest: int = 20
    max_critical_news_per_digest: int = 5
    
    # Хранение
    storage_path: str = "data/news_agent"
    max_storage_days: int = 30
    
    # Сбор
    collection_interval_minutes: int = 15
    max_news_per_collection: int = 100
    
    # Дедупликация
    duplicate_threshold_hours: int = 24