import pytest
from khors.llm import LLMClient
from unittest.mock import MagicMock

def test_llm_msg_cleaning():
    """Проверяет, что LLMClient.chat очищает метаданные OpenRouter (например, usage) из ответа."""
    client = LLMClient()
    # Подменяем внутренний _get_client, чтобы не делать реальный сетевой запрос
    mock_openai_client = MagicMock()
    
    # Имитируем ответ, содержащий "мусор" от OpenRouter
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "id": "gen_123",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.005},
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "some text",
                    "tool_calls": [{"id": "abc", "type": "function", "function": {"name": "test"}}],
                    "provider": "Anthropic", # Мусор 1
                    "usage": {"cost": 0.05}  # Мусор 2 (критично!)
                }
            }
        ]
    }
    mock_openai_client.chat.completions.create.return_value = mock_response
    client._get_client = lambda: mock_openai_client
    
    # Мок _fetch_generation_cost
    client._fetch_generation_cost = lambda x: 0.005
    
    msg, usage = client.chat([{"role": "user", "content": "hi"}], "test-model")
    
    assert "provider" not in msg
    assert "usage" not in msg
    assert msg["role"] == "assistant"
    assert msg["content"] == "some text"
    assert len(msg["tool_calls"]) == 1
