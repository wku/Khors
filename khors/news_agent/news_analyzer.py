"""
Анализатор новостей через LLM для определения важности и категоризации.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..llm import LLMClient
from .models import RawNews, AnalyzedNews, NewsImportance, NewsCategory, AgentConfig


logger = logging.getLogger(__name__)


class NewsAnalyzer:
    """Анализатор новостей через LLM."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = LLMClient()
        
        # Промпт для анализа новостей
        self.analysis_prompt = """
Ти — експерт-аналітик українських новин. Проаналізуй подану новину та надай структуровану оцінку.

НОВИНА:
Заголовок: {title}
Джерело: {source}
Контент: {content}

ЗАВДАННЯ:
1. Визнач рівень важливості (critical/high/medium/low/skip)
2. Визнач категорію (politics/military/economy/society/international/technology/culture/sport/other)
3. Оціни соціальний вплив (0.0-1.0, де 1.0 = максимальний вплив на суспільство)
4. Створи стислий саммарі (2-3 речення)
5. Виділи ключові пункти (3-5 пунктів)
6. Поясни своє рішення щодо важливості

КРИТЕРІЇ ВАЖЛИВОСТІ:
- CRITICAL: Війна, теракти, природні катастрофи, смерть важливих осіб, критичні політичні рішення
- HIGH: Важливі політичні події, економічні зміни, соціальні проблеми
- MEDIUM: Регіональні події, культурні новини, спорт
- LOW: Розваги, другорядні події
- SKIP: Реклама, спам, неактуальне

ФОРМАТ ВІДПОВІДІ (JSON):
{
    "importance": "critical|high|medium|low|skip",
    "category": "politics|military|economy|society|international|technology|culture|sport|other",
    "social_impact_score": 0.8,
    "summary": "Короткий саммарі новини",
    "key_points": [
        "Ключовий пункт 1",
        "Ключовий пункт 2",
        "Ключовий пункт 3"
    ],
    "reasoning": "Пояснення рішення щодо важливості"
}
"""
    
    async def analyze_news_batch(self, news_list: List[RawNews]) -> List[AnalyzedNews]:
        """Проаналізувати пакет новин."""
        if not news_list:
            return []
        
        logger.info(f"Analyzing {len(news_list)} news items")
        
        # Аналізуємо новини пакетами для оптимізації
        batch_size = 5
        analyzed_news = []
        
        for i in range(0, len(news_list), batch_size):
            batch = news_list[i:i + batch_size]
            batch_results = await self._analyze_batch(batch)
            analyzed_news.extend(batch_results)
            
            # Невелика пауза між пакетами
            if i + batch_size < len(news_list):
                await asyncio.sleep(1)
        
        # Фільтруємо новини, які потрібно пропустити
        filtered_news = [
            news for news in analyzed_news 
            if news.importance != NewsImportance.SKIP
        ]
        
        logger.info(f"Analysis complete: {len(filtered_news)} relevant news from {len(news_list)} total")
        return filtered_news
    
    async def _analyze_batch(self, batch: List[RawNews]) -> List[AnalyzedNews]:
        """Проаналізувати пакет новин."""
        tasks = [self._analyze_single_news(news) for news in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        analyzed_news = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error analyzing news '{batch[i].title}': {result}")
                # Створюємо fallback аналіз
                analyzed_news.append(self._create_fallback_analysis(batch[i]))
            else:
                analyzed_news.append(result)
        
        return analyzed_news
    
    async def _analyze_single_news(self, news: RawNews) -> AnalyzedNews:
        try:
            prompt = self.analysis_prompt.format(
                title=news.title,
                source=news.source_name,
                content=news.content[:1500]
            )
            loop = asyncio.get_event_loop()
            msg, _usage = await loop.run_in_executor(None, lambda: self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.llm_model,
                max_tokens=500,
            ))
            content = msg.get("content") or ""
            analysis_data = self._parse_llm_response(content)
            return AnalyzedNews(
                raw_news=news,
                importance=NewsImportance(analysis_data['importance']),
                category=NewsCategory(analysis_data['category']),
                summary=analysis_data['summary'],
                key_points=analysis_data['key_points'],
                social_impact_score=float(analysis_data['social_impact_score']),
                analysis_reasoning=analysis_data['reasoning'],
                analyzed_at=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error in LLM analysis for '{news.title}': {e}")
            return self._create_fallback_analysis(news)
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Парсинг відповіді LLM."""
        try:
            # Пробуємо знайти JSON в відповіді
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                
                # Валідація обов'язкових полів
                required_fields = ['importance', 'category', 'social_impact_score', 'summary', 'key_points', 'reasoning']
                for field in required_fields:
                    if field not in data:
                        raise ValueError(f"Missing required field: {field}")
                
                # Валідація значень
                if data['importance'] not in [e.value for e in NewsImportance]:
                    data['importance'] = 'medium'
                
                if data['category'] not in [e.value for e in NewsCategory]:
                    data['category'] = 'other'
                
                # Нормалізація social_impact_score
                score = float(data['social_impact_score'])
                data['social_impact_score'] = max(0.0, min(1.0, score))
                
                # Забезпечуємо, що key_points це список
                if not isinstance(data['key_points'], list):
                    data['key_points'] = [str(data['key_points'])]
                
                return data
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            raise
    
    def _create_fallback_analysis(self, news: RawNews) -> AnalyzedNews:
        """Створити fallback аналіз при помилці LLM."""
        # Простий евристичний аналіз
        importance = self._guess_importance(news)
        category = self._guess_category(news)
        
        return AnalyzedNews(
            raw_news=news,
            importance=importance,
            category=category,
            summary=news.content[:200] + "..." if len(news.content) > 200 else news.content,
            key_points=[news.title],
            social_impact_score=0.5,  # Нейтральний скор
            analysis_reasoning="Fallback analysis due to LLM error",
            analyzed_at=datetime.now()
        )
    
    def _guess_importance(self, news: RawNews) -> NewsImportance:
        """Евристичне визначення важливості."""
        title_lower = news.title.lower()
        content_lower = news.content.lower()
        
        # Критичні ключові слова
        critical_keywords = [
            'війна', 'вибух', 'теракт', 'загинув', 'померла', 'катастрофа',
            'надзвичайна ситуація', 'евакуація', 'обстріл'
        ]
        
        # Важливі ключові слова
        high_keywords = [
            'президент', 'уряд', 'парламент', 'закон', 'бюджет',
            'економіка', 'інфляція', 'курс', 'санкції'
        ]
        
        for keyword in critical_keywords:
            if keyword in title_lower or keyword in content_lower:
                return NewsImportance.CRITICAL
        
        for keyword in high_keywords:
            if keyword in title_lower or keyword in content_lower:
                return NewsImportance.HIGH
        
        return NewsImportance.MEDIUM
    
    def _guess_category(self, news: RawNews) -> NewsCategory:
        """Евристичне визначення категорії."""
        title_lower = news.title.lower()
        content_lower = news.content.lower()
        
        category_keywords = {
            NewsCategory.MILITARY: ['війна', 'армія', 'військов', 'обстріл', 'зброя'],
            NewsCategory.POLITICS: ['президент', 'уряд', 'парламент', 'політик', 'вибори'],
            NewsCategory.ECONOMY: ['економіка', 'бюджет', 'курс', 'інфляція', 'банк'],
            NewsCategory.INTERNATIONAL: ['світ', 'міжнародн', 'країн', 'дипломат'],
            NewsCategory.SOCIETY: ['суспільство', 'освіта', 'медицина', 'соціальн'],
            NewsCategory.SPORT: ['спорт', 'футбол', 'олімпіада', 'чемпіонат'],
            NewsCategory.CULTURE: ['культура', 'мистецтво', 'театр', 'кіно', 'музика']
        }
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in title_lower or keyword in content_lower:
                    return category
        
        return NewsCategory.OTHER