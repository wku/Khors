"""
Инструмент для управления новостным агентом из основного Khors.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional

from ..news_agent.news_agent import NewsAgent, create_default_config
from ..news_agent.models import AgentConfig


logger = logging.getLogger(__name__)


class NewsAgentManager:
    """Менеджер новостного агента для интеграции с Khors."""
    
    def __init__(self):
        self.agent: Optional[NewsAgent] = None
        self.config: Optional[AgentConfig] = None
    
    async def start_agent(
        self,
        telegram_bot_token: str,
        telegram_chat_id: str,
        llm_api_key: str = "",
        storage_path: str = "data/news_agent"
    ) -> Dict[str, Any]:
        """Запустить новостного агента."""
        try:
            if self.agent and self.agent.is_running:
                return {
                    "success": False,
                    "error": "News agent is already running"
                }
            
            # Создаем конфигурацию
            self.config = create_default_config(
                telegram_bot_token=telegram_bot_token,
                telegram_chat_id=telegram_chat_id,
                llm_api_key=llm_api_key,
                storage_path=storage_path
            )
            
            # Создаем и запускаем агента
            self.agent = NewsAgent(self.config)
            await self.agent.start()
            
            return {
                "success": True,
                "message": "News agent started successfully",
                "status": await self.agent.get_status()
            }
            
        except Exception as e:
            logger.error(f"Error starting news agent: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def stop_agent(self) -> Dict[str, Any]:
        """Остановить новостного агента."""
        try:
            if not self.agent or not self.agent.is_running:
                return {
                    "success": False,
                    "error": "News agent is not running"
                }
            
            await self.agent.stop()
            
            return {
                "success": True,
                "message": "News agent stopped successfully"
            }
            
        except Exception as e:
            logger.error(f"Error stopping news agent: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Получить статус новостного агента."""
        try:
            if not self.agent:
                return {
                    "success": True,
                    "status": "not_initialized"
                }
            
            status = await self.agent.get_status()
            
            return {
                "success": True,
                "status": status
            }
            
        except Exception as e:
            logger.error(f"Error getting news agent status: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_manual_digest(self, hours_back: int = 6) -> Dict[str, Any]:
        """Создать дайджест вручную."""
        try:
            if not self.agent or not self.agent.is_running:
                return {
                    "success": False,
                    "error": "News agent is not running"
                }
            
            success = await self.agent.create_manual_digest(hours_back)
            
            return {
                "success": True,
                "digest_sent": success,
                "message": f"Manual digest for last {hours_back} hours " + 
                          ("sent successfully" if success else "had no relevant news")
            }
            
        except Exception as e:
            logger.error(f"Error creating manual digest: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Глобальный экземпляр менеджера
_news_manager = NewsAgentManager()


from typing import Dict, Any, Optional, List
from .registry import ToolContext, ToolEntry

def get_tools() -> List[ToolEntry]:
    """Получить инструменты для работы с новостным агентом."""
    
    def _start_news_agent(
        ctx: ToolContext,
        telegram_bot_token: str,
        telegram_chat_id: str,
        llm_api_key: str = "",
        storage_path: str = "data/news_agent"
    ) -> str:
        # Для простоты вызываем асинхронный метод через run_coroutine_threadsafe
        # Но у нас _news_manager.start_agent - асинхронный
        # Быстрый хак: использовать asyncio.run() в отдельном потоке, 
        # или переписать ToolRegistry для поддержки async handler.
        # В Khors инструменты синхронные
        result = asyncio.run(_news_manager.start_agent(
            telegram_bot_token, telegram_chat_id, llm_api_key, storage_path
        ))
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def _stop_news_agent(ctx: ToolContext) -> str:
        result = asyncio.run(_news_manager.stop_agent())
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def _news_agent_status(ctx: ToolContext) -> str:
        result = asyncio.run(_news_manager.get_agent_status())
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def _create_news_digest(ctx: ToolContext, hours_back: int = 6) -> str:
        result = asyncio.run(_news_manager.create_manual_digest(hours_back))
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    return [
        ToolEntry("start_news_agent", {
            "name": "start_news_agent",
            "description": "Запустить автономного новостного агента для сбора украинских новостей",
            "parameters": {
                "type": "object",
                "properties": {
                    "telegram_bot_token": {"type": "string"},
                    "telegram_chat_id": {"type": "string"},
                    "llm_api_key": {"type": "string"},
                    "storage_path": {"type": "string", "default": "data/news_agent"}
                },
                "required": ["telegram_bot_token", "telegram_chat_id"]
            }
        }, _start_news_agent),
        
        ToolEntry("stop_news_agent", {
            "name": "stop_news_agent", 
            "description": "Остановить новостного агента",
            "parameters": {"type": "object", "properties": {}}
        }, _stop_news_agent),
        
        ToolEntry("news_agent_status", {
            "name": "news_agent_status",
            "description": "Получить статус новостного агента и статистику",
            "parameters": {"type": "object", "properties": {}}
        }, _news_agent_status),
        
        ToolEntry("create_news_digest", {
            "name": "create_news_digest",
            "description": "Создать дайджест новостей вручную",
            "parameters": {
                "type": "object",
                "properties": {
                    "hours_back": {"type": "integer", "default": 6}
                }
            }
        }, _create_news_digest)
    ]