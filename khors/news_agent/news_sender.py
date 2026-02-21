"""
Отправитель новостных дайджестов в Telegram.
"""

import asyncio
import aiohttp
import logging
from typing import Optional, Dict, Any
from urllib.parse import quote

from .models import NewsDigest, AgentConfig


logger = logging.getLogger(__name__)


class NewsSender:
    """Отправитель новостных дайджестов в Telegram."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Базовый URL Telegram Bot API
        self.base_url = f"https://api.telegram.org/bot{config.telegram_bot_token}"
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def send_digest(self, digest: NewsDigest) -> bool:
        """Отправить дайджест в Telegram."""
        try:
            # Разбиваем длинный дайджест на части, если нужно
            messages = self._split_digest_message(digest.summary_text)
            
            success = True
            for i, message in enumerate(messages):
                if i == 0:
                    # Первое сообщение с полным форматированием
                    sent = await self._send_message(message, parse_mode='Markdown')
                else:
                    # Продолжение
                    continuation = f"📰 Дайджест (продолження {i+1}):\n\n{message}"
                    sent = await self._send_message(continuation, parse_mode='Markdown')
                
                if not sent:
                    success = False
                    break
                
                # Небольшая пауза между сообщениями
                if i < len(messages) - 1:
                    await asyncio.sleep(1)
            
            if success:
                logger.info(f"Digest {digest.digest_id} sent successfully")
            else:
                logger.error(f"Failed to send digest {digest.digest_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending digest {digest.digest_id}: {e}")
            return False
    
    async def send_status_message(self, message: str) -> bool:
        """Отправить статусное сообщение."""
        return await self._send_message(f"🤖 News Agent: {message}")
    
    async def send_error_message(self, error: str) -> bool:
        """Отправить сообщение об ошибке."""
        return await self._send_message(f"❌ News Agent Error: {error}")
    
    async def _send_message(self, text: str, parse_mode: str = 'HTML') -> bool:
        """Отправить сообщение в Telegram."""
        if not self.session:
            logger.error("Session not initialized")
            return False
        
        url = f"{self.base_url}/sendMessage"
        
        data = {
            'chat_id': self.config.telegram_chat_id,
            'text': text,
            'parse_mode': parse_mode,
            'disable_web_page_preview': True
        }
        
        try:
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('ok'):
                        return True
                    else:
                        logger.error(f"Telegram API error: {result.get('description')}")
                        return False
                else:
                    logger.error(f"HTTP error {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def _split_digest_message(self, text: str, max_length: int = 4000) -> list[str]:
        """Разбить длинное сообщение на части."""
        if len(text) <= max_length:
            return [text]
        
        messages = []
        lines = text.split('\n')
        current_message = ""
        
        for line in lines:
            # Если добавление строки превысит лимит
            if len(current_message) + len(line) + 1 > max_length:
                if current_message:
                    messages.append(current_message.strip())
                    current_message = line + '\n'
                else:
                    # Строка сама по себе слишком длинная
                    if len(line) > max_length:
                        # Разбиваем строку по словам
                        words = line.split(' ')
                        current_line = ""
                        for word in words:
                            if len(current_line) + len(word) + 1 > max_length:
                                if current_line:
                                    messages.append(current_line.strip())
                                current_line = word + ' '
                            else:
                                current_line += word + ' '
                        if current_line:
                            current_message = current_line
                    else:
                        current_message = line + '\n'
            else:
                current_message += line + '\n'
        
        if current_message.strip():
            messages.append(current_message.strip())
        
        return messages
    
    async def test_connection(self) -> bool:
        """Тестировать подключение к Telegram."""
        if not self.session:
            return False
        
        url = f"{self.base_url}/getMe"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('ok'):
                        bot_info = result.get('result', {})
                        logger.info(f"Connected to Telegram bot: {bot_info.get('username')}")
                        return True
                    else:
                        logger.error(f"Telegram API error: {result.get('description')}")
                        return False
                else:
                    logger.error(f"HTTP error {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error testing connection: {e}")
            return False
    
    async def get_chat_info(self) -> Optional[Dict[str, Any]]:
        """Получить информацию о чате."""
        if not self.session:
            return None
        
        url = f"{self.base_url}/getChat"
        
        data = {
            'chat_id': self.config.telegram_chat_id
        }
        
        try:
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('ok'):
                        return result.get('result')
                    else:
                        logger.error(f"Telegram API error: {result.get('description')}")
                        return None
                else:
                    logger.error(f"HTTP error {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting chat info: {e}")
            return None