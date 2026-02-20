import os
import pytest
import asyncio
from dotenv import load_dotenv
from khors.llm import LLMClient

def test_dotenv_loading():
    """Проверяет загрузку переменных из .env."""
    if os.path.exists(".env"):
        load_dotenv(".env")
    
    key = os.environ.get("OPENROUTER_API_KEY")
    assert key is not None, "OPENROUTER_API_KEY не найден в окружении после load_dotenv"
    assert len(key) > 0, "OPENROUTER_API_KEY пустой"

def test_llm_client_init():
    """Проверяет инициализацию LLMClient и наличие апи-ключа."""
    if os.path.exists(".env"):
        load_dotenv(".env")
    client = LLMClient()
    assert client._api_key is not None, "LLMClient не смог получить API ключ"
    assert len(client._api_key) > 10, "API ключ подозрительно короткий"

def test_llm_connectivity():
    """Test LLMClient connectivity and basic chat."""
    if os.path.exists(".env"):
        load_dotenv(".env")
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not found in environment")
    
    client = LLMClient(api_key=key.strip())
    messages = [{"role": "user", "content": "Hello, say hi!"}]
    
    try:
        # LLMClient.chat is synchronous
        msg, usage = client.chat(messages, model="google/gemini-2.0-flash-001", max_tokens=10)
        assert "role" in msg
        assert "content" in msg
        print(f"\n[LLM TEST SUCCESS] Response: {msg.get('content')}")
        print(f"[LLM TEST USAGE] {usage}")
    except Exception as e:
        print(f"\n[LLM TEST FAILED] {type(e).__name__}: {e}")
        if "401" in str(e):
            print(f"[DEBUG] Key length: {len(key)}")
            print(f"[DEBUG] Key starts with: {key[:8]}...")
            print(f"[DEBUG] Key ends with: ...{key[-4:]}")
            if key != key.strip():
                print("[DEBUG] WARNING: Key contains trailing/leading whitespace!")
        raise
