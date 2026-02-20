import urllib.request
import json
import os

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    print("NO API KEY")
    exit(1)

data = {
    "model": "google/gemini-2.5-flash",
    "messages": [
        {"role": "user", "content": "What is in test.txt?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "test.txt"}'}
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "read_file",
            "content": "Secret payload: BAZINGA 123"
        }
    ],
    "tools": [{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
        }
    }]
}

req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                             data=json.dumps(data).encode("utf-8"),
                             headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
                             
try:
    with urllib.request.urlopen(req) as response:
         resp_data = json.loads(response.read().decode())
         print("RESPONSE:", resp_data['choices'][0]['message']['content'])
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code}")
    print(e.read().decode())
        
