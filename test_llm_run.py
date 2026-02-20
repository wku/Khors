import os
import json
from khors.llm import LLMClient

# test OpenRouter parsing using environment API key
client = LLMClient()
model = "google/gemini-2.5-flash"

messages = [
    {"role": "user", "content": "read the file test.txt"},
]

tools = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
    }
}]

print(f"Testing basic chat with {model}...")
msg, usage = client.chat(messages=messages, model=model, tools=tools, reasoning_effort="low")
print("Response MSG:", json.dumps(msg, indent=2))

if "tool_calls" in msg:
    tool_call_id = msg["tool_calls"][0]["id"]
    print("\nProviding tool result...")
    messages.append(msg)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": "read_file",  # <--- Added name, testing if it works
        "content": "file content is BAZINGA. The reading is complete."
    })
    
    msg2, usage2 = client.chat(messages=messages, model=model, tools=tools, reasoning_effort="low")
    print("Final Response MSG:", json.dumps(msg2, indent=2))
