import sys
import logging
from openai import OpenAI
import pathlib
import queue as builtin_queue
import sys
import os

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from dotenv import load_dotenv

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
load_dotenv("./../.env")
sys.path.append(os.getcwd())

OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") 
print(OPENROUTER_API_KEY)
MODEL = "google/gemini-2.5-flash"


def describe_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
    client = OpenAI(
    base_url=OPENROUTER_URL,
    api_key=OPENROUTER_API_KEY,
    timeout=60,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "demo"
    }
)
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
