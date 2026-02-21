"""
Khors agent core — thin orchestrator.

Delegates to specialized modules:
- task_manager.py (task lifecycle)
- system_monitor.py (system state verification)
- loop.py (LLM tool loop)
- tools/ (tool schemas/execution)
- llm.py (LLM calls)
- memory.py (scratchpad/identity)
- context.py (context building)
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from khors.utils import (
    utc_now_iso, read_text, append_jsonl,
    safe_relpath, get_git_info
)
from khors.llm import LLMClient
from khors.tools import ToolRegistry
from khors.memory import Memory
from khors.task_manager import TaskManager
from khors.system_monitor import SystemMonitor

log = logging.getLogger(__name__)

# Module-level guard for one-time worker boot logging
_worker_boot_logged = False
_worker_boot_lock = threading.Lock()


@dataclass(frozen=True)
class Env:
    """Environment configuration for Khors agent."""
    repo_dir: pathlib.Path
    drive_root: pathlib.Path
    branch_dev: str = ""

    def __post_init__(self):
        if not self.branch_dev:
            object.__setattr__(self, "branch_dev", os.environ.get("KHORS_BRANCH_DEV", "main"))

    def repo_path(self, rel: str) -> pathlib.Path:
        return (self.repo_dir / safe_relpath(rel)).resolve()

    def drive_path(self, rel: str) -> pathlib.Path:
        return (self.drive_root / safe_relpath(rel)).resolve()


class KhorsAgent:
    """
    Main agent orchestrator.
    
    Coordinates between specialized modules rather than doing everything itself.
    Maintains minimal state - delegates to TaskManager and SystemMonitor.
    """

    def __init__(self, env: Env, event_queue: Any = None):
        self.env = env
        self._event_queue = event_queue
        self._current_chat_id: Optional[int] = None
        self._current_task_type: Optional[str] = None

        # Message injection: owner can send messages while agent is busy
        self._incoming_messages: queue.Queue = queue.Queue()
        self._busy = False

        # Core modules
        self.llm = LLMClient()
        self.tools = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
        self.memory = Memory(drive_root=env.drive_root, repo_dir=env.repo_dir)
        
        # Specialized managers
        self.task_manager = TaskManager(
            repo_path=str(env.repo_dir),
            drive_path=str(env.drive_root),
            llm_client=self.llm,
            memory=self.memory
        )
        
        self.system_monitor = SystemMonitor(
            repo_path=str(env.repo_dir),
            drive_path=str(env.drive_root)
        )

        self._log_worker_boot_once()

    def inject_message(self, text: str) -> None:
        """Thread-safe: inject owner message into the active conversation."""
        self._incoming_messages.put(text)

    def handle_task(self, task: Dict[str, Any], result_queue: queue.Queue) -> None:
        """Main task handler - delegates to TaskManager."""
        self._busy = True
        self._current_task_type = task.get("type")
        
        try:
            self.task_manager.handle_task(task, result_queue)
        finally:
            self._busy = False
            self._current_task_type = None

    def _log_worker_boot_once(self) -> None:
        """Log worker boot and verify system state (once per process)."""
        global _worker_boot_logged
        
        try:
            with _worker_boot_lock:
                if _worker_boot_logged:
                    return
                _worker_boot_logged = True
            
            # Log boot event
            git_branch, git_sha = get_git_info(self.env.repo_dir)
            append_jsonl(self.env.drive_path('logs') / 'events.jsonl', {
                'ts': utc_now_iso(), 
                'type': 'worker_boot',
                'pid': os.getpid(), 
                'git_branch': git_branch, 
                'git_sha': git_sha,
            })
            
            # Verify restart if needed
            self._verify_restart(git_sha)
            
            # Verify system state (with lock to avoid race conditions)
            verify_lock = self.env.drive_path("state") / "startup_verify.lock"
            fd = -1
            try:
                fd = os.open(str(verify_lock), os.O_CREAT | os.O_WRONLY | os.O_EXCL)
                with os.fdopen(fd, 'w') as f:
                    f.write(f"{os.getpid()} {utc_now_iso()}")
                fd = -1
                
                log.info(f"Process {os.getpid()} acquired startup lock, verifying system state...")
                self._verify_system_state(git_sha)
                
            except FileExistsError:
                log.debug(f"Process {os.getpid()} skipped verification: lock exists")
                return
            finally:
                if fd != -1:
                    os.close(fd)
                    
        except Exception:
            log.warning("Worker boot logging failed", exc_info=True)

    def _verify_restart(self, git_sha: str) -> None:
        """Verify restart was successful by checking expected git SHA."""
        try:
            pending_path = self.env.drive_path('state') / 'pending_restart_verify.json'
            claim_path = pending_path.with_name(f"pending_restart_verify.claimed.{os.getpid()}.json")
            
            try:
                os.rename(str(pending_path), str(claim_path))
            except (FileNotFoundError, Exception):
                return
            
            try:
                claim_data = json.loads(read_text(claim_path))
                expected_sha = str(claim_data.get("expected_sha", "")).strip()
                ok = bool(expected_sha and expected_sha == git_sha)
                
                append_jsonl(self.env.drive_path('logs') / 'events.jsonl', {
                    'ts': utc_now_iso(), 
                    'type': 'restart_verify',
                    'pid': os.getpid(), 
                    'ok': ok,
                    'expected_sha': expected_sha, 
                    'observed_sha': git_sha,
                })
            except Exception:
                log.debug("Failed to log restart verify event", exc_info=True)
            
            try:
                os.unlink(str(claim_path))
            except Exception:
                pass
                
        except Exception:
            log.debug("Restart verification failed", exc_info=True)

    def _verify_system_state(self, git_sha: str) -> None:
        """Verify system state using SystemMonitor."""
        try:
            is_healthy, warnings = self.system_monitor.verify_system_state()
            
            # Log verification result
            append_jsonl(self.env.drive_path('logs') / 'events.jsonl', {
                'ts': utc_now_iso(),
                'type': 'system_verify',
                'pid': os.getpid(),
                'git_sha': git_sha,
                'healthy': is_healthy,
                'warnings': warnings
            })
            
            if warnings:
                log.warning(f"System state warnings: {warnings}")
            else:
                log.info("System state verification passed")
                
        except Exception:
            log.warning("System state verification failed", exc_info=True)


def make_agent(repo_dir: str, drive_root: str, event_queue: Any = None) -> KhorsAgent:
    """Factory function to create KhorsAgent instance."""
    env = Env(
        repo_dir=pathlib.Path(repo_dir),
        drive_root=pathlib.Path(drive_root)
    )
    return KhorsAgent(env, event_queue)