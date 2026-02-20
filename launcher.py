"""
Khors Launcher.
Initializes the supervisor components and runs the main event loop.
"""
import logging
import os
import pathlib
import queue as builtin_queue
import sys
import threading
import time

import signal

from dotenv import load_dotenv

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")
sys.path.append(os.getcwd())

from supervisor import state, queue, workers, telegram
from supervisor.telegram import TelegramClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("launcher")

_PID_FILE = _PROJECT_ROOT / "data" / "launcher.pid"


def _kill_previous_instance():
    try:
        if not _PID_FILE.exists():
            return
        old_pid = int(_PID_FILE.read_text().strip())
        if old_pid == os.getpid():
            return
        log.info(f"Killing previous launcher (pid {old_pid})")
        os.kill(old_pid, signal.SIGTERM)
        for _ in range(30):
            time.sleep(0.1)
            try:
                os.kill(old_pid, 0)
            except OSError:
                break
        else:
            log.warning(f"Previous launcher (pid {old_pid}) did not exit, sending SIGKILL")
            try:
                os.kill(old_pid, signal.SIGKILL)
            except OSError:
                pass
    except (ValueError, OSError, FileNotFoundError):
        pass


def _write_pid():
    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()))

def process_events_loop():
    event_q = workers.get_event_q()
    while True:
        try:
            e = event_q.get(timeout=1.0)
            if e is None:
                continue
            e_type = e.get("type", "")
            chat_id = e.get("chat_id")
            
            if e_type == "send_message" and chat_id:
                telegram.send_with_budget(
                    chat_id, e.get("text", ""),
                    log_text=e.get("log_text"),
                    fmt=e.get("format", ""),
                    is_progress=e.get("is_progress", False)
                )
            elif e_type == "typing_start" and chat_id:
                telegram.get_tg().send_chat_action(chat_id, "typing")
            elif e_type == "task_done":
                state.update_budget_from_usage(e)
                # Remove from running queue if needed
                task_id = e.get("task_id")
                from supervisor.queue import RUNNING, persist_queue_snapshot, _queue_lock
                with _queue_lock:
                    if task_id in RUNNING:
                        del RUNNING[task_id]
                        persist_queue_snapshot(reason="task_done")
        except builtin_queue.Empty:
            continue
        except Exception as exc:
            log.error(f"Event processing error: {exc}", exc_info=True)
            time.sleep(1)

def main():
    _kill_previous_instance()
    _write_pid()
    log.info("Khors initialization started.")

    # 1. Configuration from Environment
    REPO_DIR = pathlib.Path(os.environ.get("REPO_DIR", os.getcwd()))
    DRIVE_ROOT = pathlib.Path(os.environ.get("DRIVE_ROOT", os.path.join(os.getcwd(), "data")))
    
    TOTAL_BUDGET = float(os.environ.get("TOTAL_BUDGET", "50.0"))
    TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    MAX_WORKERS = int(os.environ.get("KHORS_MAX_WORKERS", "5"))
    
    # Timeouts
    SOFT_TIMEOUT = 600
    HARD_TIMEOUT = 1800

    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN not found in environment.")
        sys.exit(1)

    # 2. Initialize Components
    
    # State
    log.info(f"Initializing state at {DRIVE_ROOT}")
    state.init(DRIVE_ROOT, total_budget_limit=TOTAL_BUDGET)
    
    # Queue
    log.info("Initializing queue")
    queue.init(DRIVE_ROOT, soft_timeout=SOFT_TIMEOUT, hard_timeout=HARD_TIMEOUT)
    
    # Client
    tg_client = TelegramClient(TELEGRAM_TOKEN)
    
    # Telegram module
    log.info("Initializing telegram module")
    telegram.init(
        drive_root=DRIVE_ROOT,
        total_budget_limit=TOTAL_BUDGET,
        budget_report_every=10,
        tg_client=tg_client
    )

    # Workers
    log.info("Initializing workers")
    workers.init(
        repo_dir=REPO_DIR,
        drive_root=DRIVE_ROOT,
        max_workers=MAX_WORKERS,
        soft_timeout=SOFT_TIMEOUT,
        hard_timeout=HARD_TIMEOUT,
        total_budget_limit=TOTAL_BUDGET
    )

    # 3. Startup Sequence
    log.info("Spawning initial workers...")
    workers.spawn_workers(n=0) # Spawns up to MAX_WORKERS if needed, or 0 to start
    # Actually workers.spawn_workers() spawns based on queue, but maybe we need some standby?
    # workers.py logic says spawn_workers(n) spawns n workers.
    # Let's check if we need to pre-spawn. Usually Khors spawns on demand or keeps a pool.
    # 4. Starting Background Event Processor
    log.info("Starting event processor...")
    threading.Thread(target=process_events_loop, daemon=True).start()

    # 5. Main Loop
    log.info("Entering main loop.")
    offset = 0
    
    try:
        while True:
            # 1. Check timeouts
            queue.enforce_task_timeouts()
            workers.ensure_workers_healthy()

            # 2. Process Telegram updates
            try:
                updates = tg_client.get_updates(offset=offset, timeout=2)
                for u in updates:
                    offset = u["update_id"] + 1
                    message = u.get("message")
                    if message:
                        # Handle message via workers.handle_chat_direct or queue
                        chat_id = message.get("chat", {}).get("id")
                        text = message.get("text", "")
                        # Simple routing logic: 
                        # This should match previous logic. 
                        # Likely delegation to workers.handle_chat_direct for direct messages
                        if chat_id and text:
                             threading.Thread(target=workers.handle_chat_direct, args=(chat_id, text), daemon=True).start()
            except Exception as e:
                log.error(f"Telegram update error: {e}", exc_info=True)
                time.sleep(5)

            # 3. Evolution/Background logic (if needed in main loop)
            # workers.py handles most of it via background threads or tasks

            time.sleep(0.5)

    except KeyboardInterrupt:
        log.info("Stopping...")
        workers.kill_workers()
        _PID_FILE.unlink(missing_ok=True)
        sys.exit(0)
    except Exception as e:
        log.critical(f"Critical crash: {e}", exc_info=True)
        workers.kill_workers()
        _PID_FILE.unlink(missing_ok=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
