"""
Khors - LLM client.

The only module that communicates with the LLM API (OpenRouter).
Contract: chat(), default_model(), available_models(), add_usage().
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

log = logging.getLogger(__name__)

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_dotenv_loaded = False

def _ensure_dotenv():
    global _dotenv_loaded
    if not _dotenv_loaded:
        env_path = _PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
        _dotenv_loaded = True

_ensure_dotenv()

_DEFAULT_MODEL = os.environ.get("KHORS_MODEL", "google/gemini-2.5-flash")


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def _get_pricing_prefixes() -> Tuple[str, ...]:
    raw = os.environ.get(
        "KHORS_PRICING_PREFIXES",
        "anthropic/,openai/,google/,meta-llama/,x-ai/,qwen/,deepseek/,mistralai/"
    )
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def fetch_openrouter_pricing() -> Dict[str, Tuple[float, float, float]]:
    """
    Fetch current pricing from OpenRouter API.
    Returns dict of {model_id: (input_per_1m, cached_per_1m, output_per_1m)}.
    """
    try:
        from openai import OpenAI
    except ImportError:
        log.warning("openai not installed, cannot fetch pricing")
        return {}

    try:
        _ensure_dotenv()
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            log.warning("OPENROUTER_API_KEY not set, cannot fetch pricing")
            return {}

        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        models = client.models.list().data
        prefixes = _get_pricing_prefixes()

        pricing_dict = {}
        for model in models:
            model_id = model.id
            if not model_id.startswith(prefixes):
                continue

            model_dict = getattr(model, "model_dump", lambda: vars(model))()
            pricing = model_dict.get("pricing", {})
            if not pricing or not pricing.get("prompt"):
                continue

            raw_prompt = float(pricing.get("prompt", 0))
            raw_completion = float(pricing.get("completion", 0))
            raw_cached_str = pricing.get("input_cache_read")
            raw_cached = float(raw_cached_str) if raw_cached_str else None

            prompt_price = round(raw_prompt * 1_000_000, 4)
            completion_price = round(raw_completion * 1_000_000, 4)
            if raw_cached is not None:
                cached_price = round(raw_cached * 1_000_000, 4)
            else:
                cached_price = round(prompt_price * 0.1, 4)

            if prompt_price > 1000 or completion_price > 1000:
                log.warning(f"Skipping {model_id}: prices seem wrong (prompt={prompt_price}, completion={completion_price})")
                continue

            pricing_dict[model_id] = (prompt_price, cached_price, completion_price)

        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict

    except Exception as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


class LLMClient:

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        _ensure_dotenv()
        self._api_key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            key_preview = f"{self._api_key[:8]}... (len={len(self._api_key)})" if self._api_key else "EMPTY"
            log.warning(f"[DEBUG_API_KEY] Init OpenAI client. Key: {key_preview}, base_url: {self._base_url}")
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                default_headers={"X-Title": "Khors"},
            )
        return self._client

    def _fetch_generation_cost_async(self, generation_id: str, usage: Dict[str, Any]) -> None:
        def _fetch():
            try:
                client = self._get_client()
                time.sleep(1.5)
                response = client.get(f"/generation?id={generation_id}", cast_to=object)
                data = response.get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    usage["cost"] = float(cost)
                    return
                time.sleep(1.0)
                response = client.get(f"/generation?id={generation_id}", cast_to=object)
                data = response.get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    usage["cost"] = float(cost)
            except Exception:
                log.debug("Failed to fetch generation cost from OpenRouter", exc_info=True)

        threading.Thread(target=_fetch, daemon=True).start()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 860384,
        tool_choice: str = "auto",
        temperature: float = 0.2,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        client = self._get_client()

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        # Dump messages for debugging
        try:
            dump_dir = _PROJECT_ROOT / "data" / "logs"
            dump_dir.mkdir(parents=True, exist_ok=True)
            dump_path = dump_dir / "messages_dump.json"
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
            log.warning(f"[DEBUG_LLM_DUMP] Messages dumped to {dump_path}")
        except Exception as e:
            log.debug(f"Failed to dump messages: {e}")

        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        if not usage.get("cached_tokens"):
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                usage["cached_tokens"] = int(prompt_details["cached_tokens"])

        if not usage.get("cache_write_tokens"):
            prompt_details_for_write = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details_for_write, dict):
                cache_write = (prompt_details_for_write.get("cache_write_tokens")
                              or prompt_details_for_write.get("cache_creation_tokens")
                              or prompt_details_for_write.get("cache_creation_input_tokens"))
                if cache_write:
                    usage["cache_write_tokens"] = int(cache_write)

        if not usage.get("cost"):
            gen_id = resp_dict.get("id") or ""
            if gen_id:
                self._fetch_generation_cost_async(gen_id, usage)

        if msg:
            allowed_keys = {"role", "content", "refusal", "tool_calls", "function_call"}
            msg = {k: v for k, v in msg.items() if k in allowed_keys}
            
            # Log finish reason
            finish_reason = choices[0].get("finish_reason") if choices else None
            if finish_reason:
                log.warning(f"[DEBUG_LLM_RESPONSE] Finish reason: {finish_reason}")

        return msg, usage

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Tuple[str, Dict[str, Any]]:
        if model is None:
            model = os.environ.get("KHORS_MODEL_VISION", self.default_model())

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img["url"]},
                })
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img['base64']}"},
                })
            else:
                log.warning("vision_query: skipping image with unknown format: %s", list(img.keys()))

        messages = [{"role": "user", "content": content}]
        response_msg, usage = self.chat(
            messages=messages,
            model=model,
            tools=None,
            max_tokens=max_tokens,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        return os.environ.get("KHORS_MODEL", _DEFAULT_MODEL)

    def available_models(self) -> List[str]:
        main = os.environ.get("KHORS_MODEL", _DEFAULT_MODEL)
        code = os.environ.get("KHORS_MODEL_CODE", "")
        light = os.environ.get("KHORS_MODEL_LIGHT", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)
        return models
