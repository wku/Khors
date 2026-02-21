"""
Khors — Loop Helpers (Budget, Self-check, Message Draining, LLM Call).
"""

from __future__ import annotations

import logging
import os
import pathlib
import queue
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from khors.llm import LLMClient, add_usage
from khors.pricing import estimate_cost, emit_llm_usage_event
from khors.context import compact_tool_history, compact_tool_history_llm
from khors.utils import (
    utc_now_iso, append_jsonl, estimate_tokens
)

log = logging.getLogger(__name__)


def check_budget_limits(
    budget_remaining_usd: Optional[float],
    accumulated_usage: Dict[str, Any],
    round_idx: int,
    messages: List[Dict[str, Any]],
    llm: LLMClient,
    active_model: str,
    max_retries: int,
    drive_logs: pathlib.Path,
    task_id: str,
    event_queue: Optional[queue.Queue],
    llm_trace: Dict[str, Any],
    task_type: str = "task",
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """
    Check budget limits and handle budget overrun.
    """
    if budget_remaining_usd is None:
        return None

    task_cost = accumulated_usage.get("cost", 0)
    budget_pct = task_cost / budget_remaining_usd if budget_remaining_usd > 0 else 1.0

    if budget_pct > 0.5:
        finish_reason = f"Task spent ${task_cost:.3f} (>50% of remaining ${budget_remaining_usd:.2f}). Budget exhausted."
        messages.append({"role": "system", "content": f"[BUDGET LIMIT] {finish_reason} Give your final response now."})
        try:
            final_msg, final_cost = call_llm_with_retry(
                llm, messages, active_model, None,
                max_retries, drive_logs, task_id, round_idx, event_queue, accumulated_usage, task_type
            )
            if final_msg:
                return (final_msg.get("content") or finish_reason), accumulated_usage, llm_trace
            return finish_reason, accumulated_usage, llm_trace
        except Exception:
            log.warning("Failed to get final response after budget limit", exc_info=True)
            return finish_reason, accumulated_usage, llm_trace
    elif budget_pct > 0.3 and round_idx % 10 == 0:
        messages.append({"role": "system", "content": f"[INFO] Task spent ${task_cost:.3f} of ${budget_remaining_usd:.2f}. Wrap up if possible."})

    return None


def maybe_inject_self_check(
    round_idx: int,
    max_rounds: int,
    messages: List[Dict[str, Any]],
    accumulated_usage: Dict[str, Any],
    emit_progress: Callable[[str], None],
) -> None:
    """Inject a soft self-check reminder every REMINDER_INTERVAL rounds."""
    REMINDER_INTERVAL = 50
    if round_idx <= 1 or round_idx % REMINDER_INTERVAL != 0:
        return
    ctx_tokens = sum(
        estimate_tokens(str(m.get("content", "")))
        if isinstance(m.get("content"), str)
        else sum(estimate_tokens(str(b.get("text", ""))) for b in m.get("content", []) if isinstance(b, dict))
        for m in messages
    )
    task_cost = accumulated_usage.get("cost", 0)
    checkpoint_num = round_idx // REMINDER_INTERVAL

    reminder = (
        f"[CHECKPOINT {checkpoint_num} — round {round_idx}/{max_rounds}]\n"
        f"📊 Context: ~{ctx_tokens} tokens | Cost so far: ${task_cost:.2f} | "
        f"Rounds remaining: {max_rounds - round_idx}\n\n"
        f"⏸️ PAUSE AND REFLECT before continuing:\n"
        f"1. Am I making real progress, or repeating the same actions?\n"
        f"2. Is my current strategy working? Should I try something different?\n"
        f"3. Is my context bloated with old tool results I no longer need?\n"
        f"   → If yes, call `compact_context` to summarize them selectively.\n"
        f"4. Have I been stuck on the same sub-problem for many rounds?\n"
        f"   → If yes, consider: simplify the approach, skip the sub-problem, or finish with what I have.\n"
        f"5. Should I just STOP and return my best result so far?\n\n"
        f"This is not a hard limit — you decide. But be honest with yourself."
    )
    messages.append({"role": "system", "content": reminder})
    emit_progress(f"🔄 Checkpoint {checkpoint_num} at round {round_idx}: ~{ctx_tokens} tokens, ${task_cost:.2f} spent")


def setup_dynamic_tools(tools_registry, tool_schemas, messages):
    """
    Wire tool-discovery handlers onto an existing tool_schemas list.
    """
    enabled_extra: set = set()

    def _handle_list_tools(ctx=None, **kwargs):
        non_core = tools_registry.list_non_core_tools()
        if not non_core:
            return "All tools are already in your active set."
        lines = [f"**{len(non_core)} additional tools available** (use `enable_tools` to activate):\n"]
        for t in non_core:
            lines.append(f"- **{t['name']}**: {t['description'][:120]}")
        return "\n".join(lines)

    def _handle_enable_tools(ctx=None, tools: str = "", **kwargs):
        names = [n.strip() for n in tools.split(",") if n.strip()]
        enabled, not_found = [], []
        for name in names:
            schema = tools_registry.get_schema_by_name(name)
            if schema and name not in enabled_extra:
                tool_schemas.append(schema)
                enabled_extra.add(name)
                enabled.append(name)
            elif name in enabled_extra:
                enabled.append(f"{name} (already active)")
            else:
                not_found.append(name)
        parts = []
        if enabled:
            parts.append(f"✅ Enabled: {', '.join(enabled)}")
        if not_found:
            parts.append(f"❌ Not found: {', '.join(not_found)}")
        return "\n".join(parts) if parts else "No tools specified."

    tools_registry.override_handler("list_available_tools", _handle_list_tools)
    tools_registry.override_handler("enable_tools", _handle_enable_tools)

    non_core = tools_registry.list_non_core_tools()
    non_core_count = len(non_core)
    if non_core_count > 0:
        names = ", ".join(f"`{t['name']}`" for t in non_core if t["name"] not in ("list_available_tools", "enable_tools"))
        prompt_addition = (
            f"\n\nNote: You have {len(tool_schemas)} core tools loaded. "
            f"Additional inactive tools: {names} "
            f"(use `enable_tools(tools=\"...\")` to activate their schemas). "
            f"Core tools cover most tasks. Enable extras only when needed."
        )

        injected = False
        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                msg["content"] += prompt_addition
                injected = True
                break
        
        if not injected:
            messages.insert(0, {
                "role": "system",
                "content": prompt_addition.lstrip(),
            })

    return tool_schemas, enabled_extra


def drain_incoming_messages(
    messages: List[Dict[str, Any]],
    incoming_messages: queue.Queue,
    drive_root: Optional[pathlib.Path],
    task_id: str,
    event_queue: Optional[queue.Queue],
    _owner_msg_seen: set,
) -> None:
    """
    Inject owner messages received during task execution.
    """
    while not incoming_messages.empty():
        try:
            injected = incoming_messages.get_nowait()
            messages.append({"role": "user", "content": injected})
        except queue.Empty:
            break

    if drive_root is not None and task_id:
        from khors.owner_inject import drain_owner_messages
        drive_msgs = drain_owner_messages(drive_root, task_id=task_id, seen_ids=_owner_msg_seen)
        for dmsg in drive_msgs:
            messages.append({
                "role": "user",
                "content": f"[Owner message during task]: {dmsg}",
            })
            if event_queue is not None:
                try:
                    event_queue.put_nowait({
                        "type": "owner_message_injected",
                        "task_id": task_id,
                        "text": dmsg[:200],
                    })
                except Exception:
                    pass


def call_llm_with_retry(
    llm: LLMClient,
    messages: List[Dict[str, Any]],
    model: str,
    tools: Optional[List[Dict[str, Any]]],
    max_retries: int,
    drive_logs: pathlib.Path,
    task_id: str,
    round_idx: int,
    event_queue: Optional[queue.Queue],
    accumulated_usage: Dict[str, Any],
    task_type: str = "",
) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Call LLM with retry logic, usage tracking, and event emission.
    """
    msg = None
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            kwargs = {"messages": messages, "model": model}
            if tools:
                kwargs["tools"] = tools
            resp_msg, usage = llm.chat(**kwargs)
            msg = resp_msg
            add_usage(accumulated_usage, usage)

            cost = float(usage.get("cost") or 0)
            if not cost:
                cost = estimate_cost(
                    model,
                    int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0),
                    int(usage.get("cached_tokens") or 0),
                    int(usage.get("cache_write_tokens") or 0),
                )

            category = task_type if task_type in ("evolution", "consciousness", "review", "summarize") else "task"
            emit_llm_usage_event(event_queue, task_id, model, usage, cost, category)

            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content")
            if not tool_calls and (not content or not content.strip()):
                log.warning("LLM returned empty response (no content, no tool_calls), attempt %d/%d", attempt + 1, max_retries)

                append_jsonl(drive_logs / "events.jsonl", {
                    "ts": utc_now_iso(), "type": "llm_empty_response",
                    "task_id": task_id,
                    "round": round_idx, "attempt": attempt + 1,
                    "model": model,
                    "raw_content": repr(content)[:500] if content else None,
                    "raw_tool_calls": repr(tool_calls)[:500] if tool_calls else None,
                    "finish_reason": msg.get("finish_reason") or msg.get("stop_reason"),
                })

                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None, cost

            accumulated_usage["rounds"] = accumulated_usage.get("rounds", 0) + 1

            _round_event = {
                "ts": utc_now_iso(), "type": "llm_round",
                "task_id": task_id,
                "round": round_idx, "model": model,
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "cached_tokens": int(usage.get("cached_tokens") or 0),
                "cache_write_tokens": int(usage.get("cache_write_tokens") or 0),
                "cost_usd": cost,
            }
            append_jsonl(drive_logs / "events.jsonl", _round_event)
            return msg, cost

        except Exception as e:
            print(f"\n[DEBUG_LLM_ERROR] Attempt {attempt + 1} failed with error: {e}")
            last_error = e
            append_jsonl(drive_logs / "events.jsonl", {
                "ts": utc_now_iso(), "type": "llm_api_error",
                "task_id": task_id,
                "round": round_idx, "attempt": attempt + 1,
                "model": model, "error": repr(e),
            })
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt * 2, 30))

    return None, 0.0


_TEXT_TOOL_CALL_RE = None

def has_text_tool_calls(content: str) -> bool:
    global _TEXT_TOOL_CALL_RE
    if _TEXT_TOOL_CALL_RE is None:
        import re
        _TEXT_TOOL_CALL_RE = re.compile(
            r'\b(?:read_file|repo_read|repo_list|drive_read|drive_write|run_shell|'
            r'repo_write_commit|write_to_file|replace_file_content|web_search|'
            r'browse_page|browser_action|analyze_code|update_scratchpad|'
            r'update_identity|send_owner_message|schedule_task|chat_history|'
            r'knowledge_read|knowledge_write|knowledge_list|codebase_digest|'
            r'archive_version|request_restart|switch_model|compact_context)'
            r'\s*\{',
            re.IGNORECASE
        )
    return bool(_TEXT_TOOL_CALL_RE.search(content or ""))
