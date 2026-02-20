import os
import json
from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))
models = client.models.list().data
print("Loaded models:", len(models))
for m in models[:3]:
    # see if it exposes pricing via model_dump
    d = getattr(m, "model_dump", lambda: vars(m))()
    print(json.dumps(d, indent=2))
