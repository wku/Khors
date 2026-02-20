import json
import os
from khors.llm import LLMClient

msg, cache = LLMClient().chat([{"role": "user", "content": "hello"}], model="google/gemini-2.5-flash")
print(msg.keys()) # должно быть только role, content

