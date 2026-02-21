"""
Task lifecycle management for Khors.

Handles task preparation, execution coordination, and result emission.
Extracted from agent.py to follow Single Responsibility Principle.
"""

from __future__ import annotations

import logging
import pathlib
import queue
import threading
import time
import traceback
from typing import Any, Dict, Optional

from khors.utils import utc_now_iso, append_jsonl, truncate_for_log, sanitize_task_for_event
from khors.llm import LLMClient
from khors.tools.registry import ToolContext, ToolRegistry
from khors.memory import Memory
from khors.context import build_llm_messages
from khors.loop import run_llm_loop


logger = logging.getLogger(__name__)


class TaskManager:
    """Manages task lifecycle: preparation, execution, and result emission."""

    def __init__(self, env: Any, llm_client: LLMClient, memory: Memory, tools: ToolRegistry, event_queue: Any = None):
        self.env = env
        self.llm_client = llm_client
        self.memory = memory
        self.tools = tools
        self.event_queue = event_queue
        self._heartbeat_threads = {}

    def handle_task(self, task: Dict[str, Any], result_queue: queue.Queue) -> None:
        """Execute a single task and emit results."""
        task_id = task["id"]
        task_type = str(task.get("type") or "task")

        try:
            self._emit_typing_start(task_id)
            self._start_task_heartbeat_loop(task_id)

            messages, _cap_info = build_llm_messages(
                env=self.env,
                memory=self.memory,
                task=task,
            )

            ctx = ToolContext(
                repo_dir=self.env.repo_dir,
                drive_root=self.env.drive_root,
                task_id=task_id,
                current_task_type=task_type,
                is_direct_chat=bool(task.get("_is_direct_chat")),
                event_queue=self.event_queue,
            )
            self.tools.set_context(ctx)

            content, usage, _trace = run_llm_loop(
                llm=self.llm_client,
                messages=messages,
                tools=self.tools,
                model=self.llm_client.default_model(),
                drive_root=self.env.drive_root,
                task_id=task_id,
                task_type=task_type,
                event_queue=self.event_queue,
            )

            for evt in ctx.pending_events:
                if self.event_queue is not None:
                    self.event_queue.put(evt)
            ctx.pending_events.clear()

            self._emit_task_results(task, {"content": content, "usage": usage}, result_queue)

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            logger.error(traceback.format_exc())
            self._emit_task_results(
                task,
                {"content": f"⚠️ Error during processing: {type(e).__name__}: {e}", "usage": {}},
                result_queue,
            )
        finally:
            if task_id in self._heartbeat_threads:
                self._heartbeat_threads[task_id] = None

    def _emit_task_results(self, task: Dict[str, Any], response: Dict[str, Any], result_queue: queue.Queue) -> None:
        task_id = task["id"]
        result_queue.put({
            "type": "task_complete",
            "task_id": task_id,
            "chat_id": task.get("chat_id"),
            "is_direct_chat": bool(task.get("_is_direct_chat")),
            "response": response,
        })
        event = {
            "timestamp": utc_now_iso(),
            "type": "task_complete",
            "task_id": task_id,
            "task": sanitize_task_for_event(task),
            "response_preview": truncate_for_log(response.get("content", ""), 200),
            "usage": response.get("usage", {}),
        }
        append_jsonl(self.env.drive_root / "logs" / "events.jsonl", event)

    def _emit_progress(self, task_id: str, content: str) -> None:
        progress_event = {"timestamp": utc_now_iso(), "type": "progress", "task_id": task_id, "content": content}
        append_jsonl(self.env.drive_root / "logs" / "progress.jsonl", progress_event)

    def _emit_typing_start(self, task_id: str) -> None:
        typing_event = {"timestamp": utc_now_iso(), "type": "typing_start", "task_id": task_id}
        append_jsonl(self.env.drive_root / "logs" / "events.jsonl", typing_event)

    def _emit_task_heartbeat(self, task_id: str) -> None:
        heartbeat_event = {"timestamp": utc_now_iso(), "type": "task_heartbeat", "task_id": task_id}
        append_jsonl(self.env.drive_root / "logs" / "events.jsonl", heartbeat_event)

    def _start_task_heartbeat_loop(self, task_id: str) -> None:
        def heartbeat_loop():
            while self._heartbeat_threads.get(task_id) is not None:
                time.sleep(30)
                if self._heartbeat_threads.get(task_id) is not None:
                    self._emit_task_heartbeat(task_id)
        self._heartbeat_threads[task_id] = True
        threading.Thread(target=heartbeat_loop, daemon=True).start()