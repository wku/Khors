from openai import OpenAI
import json

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="dummy_key_not_checked_for_models")
try:
    page = client.models.list()
    if page.data:
        m = page.data[0]
        # model_dump is the pydantic method
        print(json.dumps(m.model_dump(), indent=2))
except Exception as e:
    print(e)
