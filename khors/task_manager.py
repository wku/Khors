"""
Task lifecycle management for Khors.

Handles task preparation, execution coordination, and result emission.
Extracted from agent.py to follow Single Responsibility Principle.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import traceback
from typing import Any, Dict, Optional

from khors.utils import utc_now_iso, append_jsonl, truncate_for_log, sanitize_task_for_event
from khors.llm import LLMClient, add_usage
from khors.tools.registry import ToolContext
from khors.memory import Memory
from khors.context import build_llm_messages
from khors.loop import run_llm_loop


logger = logging.getLogger(__name__)


class TaskManager:
    """Manages task lifecycle: preparation, execution, and result emission."""
    
    def __init__(self, repo_path: str, drive_path: str, llm_client: LLMClient, memory: Memory):
        self.repo_path = repo_path
        self.drive_path = drive_path
        self.llm_client = llm_client
        self.memory = memory
        self._heartbeat_threads = {}
    
    def prepare_task_context(self, task: Dict[str, Any]) -> ToolContext:
        """Prepare context for task execution."""
        return ToolContext(
            repo_root=self.repo_path,
            drive_root=self.drive_path,
            task_id=task["id"],
            task_type=task.get("type", "chat"),
            llm_client=self.llm_client,
            memory=self.memory
        )
    
    def handle_task(self, task: Dict[str, Any], result_queue: queue.Queue) -> None:
        """Execute a single task and emit results."""
        task_id = task["id"]
        
        try:
            self._emit_typing_start(task_id)
            self._start_task_heartbeat_loop(task_id)
            
            # Prepare context
            tool_context = self.prepare_task_context(task)
            
            # Build LLM messages
            messages = build_llm_messages(
                task=task,
                memory=self.memory,
                repo_root=self.repo_path,
                drive_root=self.drive_path
            )
            
            # Execute LLM loop
            final_response = run_llm_loop(
                messages=messages,
                llm_client=self.llm_client,
                tool_context=tool_context,
                task=task,
                repo_root=self.repo_path,
                drive_root=self.drive_path
            )
            
            # Emit results
            self._emit_task_results(task, final_response, result_queue)
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            logger.error(traceback.format_exc())
            
            error_response = {
                "content": f"⚠️ Error during processing: {type(e).__name__}: {e}",
                "usage": {"total_cost": 0.0}
            }
            self._emit_task_results(task, error_response, result_queue)
        
        finally:
            # Stop heartbeat
            if task_id in self._heartbeat_threads:
                self._heartbeat_threads[task_id] = None
    
    def _emit_task_results(self, task: Dict[str, Any], response: Dict[str, Any], result_queue: queue.Queue) -> None:
        """Emit task completion results."""
        task_id = task["id"]
        
        # Add usage tracking
        if "usage" in response:
            add_usage(response["usage"])
        
        # Emit to result queue
        result_queue.put({
            "type": "task_complete",
            "task_id": task_id,
            "response": response
        })
        
        # Log event
        event = {
            "timestamp": utc_now_iso(),
            "type": "task_complete",
            "task_id": task_id,
            "task": sanitize_task_for_event(task),
            "response_preview": truncate_for_log(response.get("content", ""), 200),
            "usage": response.get("usage", {})
        }
        
        events_path = f"{self.drive_path}/logs/events.jsonl"
        append_jsonl(events_path, event)
    
    def _emit_progress(self, task_id: str, content: str) -> None:
        """Emit progress message."""
        progress_event = {
            "timestamp": utc_now_iso(),
            "type": "progress",
            "task_id": task_id,
            "content": content
        }
        
        progress_path = f"{self.drive_path}/logs/progress.jsonl"
        append_jsonl(progress_path, progress_event)
    
    def _emit_typing_start(self, task_id: str) -> None:
        """Emit typing indicator."""
        typing_event = {
            "timestamp": utc_now_iso(),
            "type": "typing_start",
            "task_id": task_id
        }
        
        events_path = f"{self.drive_path}/logs/events.jsonl"
        append_jsonl(events_path, typing_event)
    
    def _emit_task_heartbeat(self, task_id: str) -> None:
        """Emit periodic heartbeat for long-running tasks."""
        heartbeat_event = {
            "timestamp": utc_now_iso(),
            "type": "task_heartbeat",
            "task_id": task_id
        }
        
        events_path = f"{self.drive_path}/logs/events.jsonl"
        append_jsonl(events_path, heartbeat_event)
    
    def _start_task_heartbeat_loop(self, task_id: str) -> None:
        """Start heartbeat thread for task."""
        def heartbeat_loop():
            while self._heartbeat_threads.get(task_id) is not None:
                time.sleep(30)  # Heartbeat every 30 seconds
                if self._heartbeat_threads.get(task_id) is not None:
                    self._emit_task_heartbeat(task_id)
        
        self._heartbeat_threads[task_id] = True
        thread = threading.Thread(target=heartbeat_loop, daemon=True)
        thread.start()