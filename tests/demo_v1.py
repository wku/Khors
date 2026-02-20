import sys
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY = "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"
MODEL = "google/gemini-2.5-flash"


def describe_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
    client = OpenAI(base_url=OPENROUTER_URL, api_key=OPENROUTER_KEY, timeout=60)
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"Opishi chto delaet etot python kod, na russkom yazyke, kratko i strukturirovano:\n\n{code}"}],
        temperature=0.2
    )
    return r.choices[0].message.content or ""


if __name__ == "__main__":
    print("Starting...")
    result = describe_file("test_basic.py")
    print(result)