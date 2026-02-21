"""
Khors — LLM Orchestration Loop.
"""

from __future__ import annotations

import logging
import pathlib
import queue
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from khors.llm import LLMClient
from khors.tools.registry import ToolRegistry
from khors.context import compact_tool_history, compact_tool_history_llm
from khors.utils import (
    utc_now_iso, append_jsonl
)
from khors.pricing import get_pricing
from khors.tool_executor import StatefulToolExecutor, handle_tool_calls
from khors.loop_helpers import (
    check_budget_limits, maybe_inject_self_check, setup_dynamic_tools,
    drain_incoming_messages, call_llm_with_retry, has_text_tool_calls
)

log = logging.getLogger(__name__)


def run_llm_loop(
    llm: LLMClient,
    messages: List[Dict[str, Any]],
    tools: ToolRegistry,
    model: str = "",
    max_rounds: int = 50,
    max_retries: int = 3,
    budget_remaining_usd: Optional[float] = None,
    drive_root: Optional[pathlib.Path] = None,
    task_id: str = "",
    event_queue: Optional[queue.Queue] = None,
    incoming_messages: Optional[queue.Queue] = None,
    emit_progress: Optional[Callable[[str], None]] = None,
    task_type: str = "task",
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Main LLM orchestration loop.
    """
    if not model:
        model = llm.default_model()
        
    if emit_progress is None:
        emit_progress = lambda x: None

    drive_logs = drive_root / "logs" if drive_root else pathlib.Path("data/logs")
    drive_logs.mkdir(parents=True, exist_ok=True)

    accumulated_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "rounds": 0}
    llm_trace = {"model": model, "rounds": [], "tool_calls": [], "assistant_notes": []}
    active_model = model
    stateful_executor = StatefulToolExecutor()
    _owner_msg_seen = set()
    _text_tc_retries = 0

    # Setup dynamic tool handlers
    tool_schemas = tools.schemas(core_only=True)
    tool_schemas, enabled_extra = setup_dynamic_tools(tools, tool_schemas, messages)

    try:
        for round_idx in range(1, max_rounds + 1):
            # --- Context Management ---
            # Compact tool history if it gets too long
            messages = compact_tool_history(messages, keep_recent=15)
            messages = compact_tool_history_llm(messages, keep_recent=6)

            # --- Owner Interrupts ---
            if incoming_messages:
                drain_incoming_messages(messages, incoming_messages, drive_root, task_id, event_queue, _owner_msg_seen)

            # --- Self-check ---
            maybe_inject_self_check(round_idx, max_rounds, messages, accumulated_usage, emit_progress)

            # --- LLM Call ---
            msg, cost = call_llm_with_retry(
                llm, messages, active_model, tool_schemas,
                max_retries, drive_logs, task_id, round_idx, event_queue, accumulated_usage, task_type
            )

            if not msg:
                finish_reason = f"Could not get response from LLM after {max_retries} retries."
                return finish_reason, accumulated_usage, llm_trace

            content = msg.get("content")
            tool_calls = msg.get("tool_calls") or []

            # --- Model Switching ---
            if not tool_calls and content and "switch_model" in content and "{" in content:
                import json
                import re
                m = re.search(r'switch_model\s*(\{[^}]+\})', content)
                if m:
                    try:
                        args = json.loads(m.group(1))
                        new_model = args.get("model")
                        if new_model:
                            active_model = new_model
                            emit_progress(f"🔄 Switched model to {active_model}")
                            messages.append({"role": "assistant", "content": content})
                            messages.append({"role": "system", "content": f"Model switched to {active_model}. Continue."})
                            continue
                    except Exception:
                        pass

            # --- Text-based Tool Call Detection ---
            if not tool_calls and has_text_tool_calls(content):
                _text_tc_retries += 1
                if _text_tc_retries <= 2:
                    emit_progress("⚠️ Model wrote tool calls as text. Re-prompting...")
                    messages.append({"role": "assistant", "content": content or ""})
                    messages.append({
                        "role": "system",
                        "content": (
                            "CRITICAL: You wrote tool calls as text. This is NOT executed. "
                            "You MUST use the tool_calls API field to execute tools. "
                            "Please repeat your last action using the proper tool_calls mechanism."
                        )
                    })
                    continue

            # --- Final Response ---
            if not tool_calls:
                return content or "", accumulated_usage, llm_trace

            # --- Tool Execution ---
            _text_tc_retries = 0
            messages.append({"role": "assistant", "content": content or "", "tool_calls": tool_calls})

            if content and content.strip():
                emit_progress(content.strip())
                llm_trace["assistant_notes"].append(content.strip()[:320])

            handle_tool_calls(
                tool_calls, tools, drive_logs, task_id, stateful_executor,
                messages, llm_trace, emit_progress
            )

            # --- Budget guard ---
            budget_result = check_budget_limits(
                budget_remaining_usd, accumulated_usage, round_idx, messages,
                llm, active_model, max_retries, drive_logs,
                task_id, event_queue, llm_trace, task_type
            )
            if budget_result is not None:
                return budget_result

        return "Max rounds reached.", accumulated_usage, llm_trace

    finally:
        if stateful_executor:
            stateful_executor.shutdown(wait=False, cancel_futures=True)
        if drive_root is not None and task_id:
            try:
                from khors.owner_inject import cleanup_task_mailbox
                cleanup_task_mailbox(drive_root, task_id)
            except Exception:
                log.debug("Failed to cleanup task mailbox", exc_info=True)
