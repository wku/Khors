import json

def test_msg_clearing():
    # Моделируем ответ OpenRouter API (он может инжектить всякое прямо в msg)
    msg = {
        "role": "assistant",
        "content": "some text",
        "tool_calls": [{"id": "abc"}],
        "provider": "Anthropic", # пример мусора
        "usage": {"cost": 0.05} # ВАЖНО: вот откуда брались токены! OpenRouter иногда дублирует usage внутрь msg
    }
    
    # Мы должны очистить msg от всего, кроме role, content, tool_calls, null's, refusal'ов.
    # Так как мы добавляем msg в историю сообщений, любой лишний ключ может
    # сбить LLM с толку, и она решит, что "usage" = это ответ на ее запрос!
    
    allowed_keys = {"role", "content", "refusal", "tool_calls"}
    cleaned_msg = {k: v for k, v in msg.items() if k in allowed_keys}
    
    print(json.dumps(cleaned_msg, indent=2))

test_msg_clearing()
