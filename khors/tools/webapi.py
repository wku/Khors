"""Web API tool for generic HTTP requests."""

import json
import logging
from typing import Any, Dict, List, Optional

import requests
from khors.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)

def _web_api(ctx: ToolContext, url: str, method: str = "GET", data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> str:
    try:
        response = requests.request(
            method=method,
            url=url,
            json=data,
            headers=headers,
            timeout=timeout
        )
        
        try:
            return f"Status: {response.status_code}\nResponse: {json.dumps(response.json(), indent=2, ensure_ascii=False)}"
        except Exception:
            text = response.text
            return f"Status: {response.status_code}\nResponse: {text[:5000]}"
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out."
    except Exception as e:
        return f"⚠️ Error making API request: {e}"

def _web_fetch(ctx: ToolContext, url: str, extract_text: bool = True) -> str:
    try:
        import re
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        if extract_text and 'text/html' in response.headers.get('Content-Type', ''):
            text = response.text
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return f"URL: {url}\n\nContent:\n{text[:10000]}..." if len(text) > 10000 else f"URL: {url}\n\nContent:\n{text}"
        
        return response.text[:20000]
    except Exception as e:
        return f"⚠️ Error fetching URL: {e}"

def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("web_api", {
            "name": "web_api",
            "description": "Make a generic HTTP API request (GET, POST, etc).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "method": {"type": "string", "default": "GET"},
                    "data": {"type": "object"},
                    "headers": {"type": "object"},
                    "timeout": {"type": "integer", "default": 30}
                },
            }
        }, _web_api),
        ToolEntry("web_fetch", {
            "name": "web_fetch",
            "description": "Fetch content from a URL. Can automatically extract text from HTML and remove tags/scripts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "extract_text": {"type": "boolean", "default": True}
                },
                "required": ["url"]
            }
        }, _web_fetch)
    ]
