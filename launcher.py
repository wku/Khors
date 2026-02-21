import datetime
import json
import logging
import os
import pathlib
import sys
import time
import threading
import signal
import traceback
import uuid
from dotenv import load_dotenv

# Add current directory to sys.path to import khors and supervisor
sys.path.append(os.getcwd())

import shutil
for _p in pathlib.Path(os.getcwd()).rglob("__pycache__"):
    shutil.rmtree(_p, ignore_errors=True)

import subprocess as _sp
_test_env = {**os.environ, "PYTHONPATH": os.getcwd()}
_pytest_cmd = ["uv", "run", "pytest", "tests/", "-x", "-q", "--tb=short"]
try:
    _test_result = _sp.run(_pytest_cmd, cwd=os.getcwd(), capture_output=True, text=True, timeout=120, env=_test_env)
except FileNotFoundError:
    _pytest_cmd = [sys.executable, "-m", "pytest", "tests/", "-x", "-q", "--tb=short"]
    _test_result = _sp.run(_pytest_cmd, cwd=os.getcwd(), capture_output=True, text=True, timeout=120, env=_test_env)
if _test_result.returncode != 0:
    print(f"[STARTUP] Tests FAILED, refusing to start:\n{_test_result.stdout}\n{_test_result.stderr}")
    sys.exit(1)
print("[STARTUP] Tests passed")

from supervisor import state, telegram, workers, queue
from supervisor.state import load_state, save_state, append_jsonl
from supervisor.telegram import TelegramClient
from khors.utils import utc_now_iso, write_text, run_cmd

# Configuration
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

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
        try:
            os.kill(old_pid, signal.SIGTERM)
        except OSError:
            return
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
    _startup_lock = _PROJECT_ROOT / "data" / "state" / "startup_verify.lock"
    _startup_lock.unlink(missing_ok=True)

def process_events_loop():
    import queue as _queue
    event_q = workers.get_event_q()
    log.info("Event processor started.")
    while True:
        try:
            e = event_q.get(timeout=1.0)
            if e is None:
                continue
            e_type = e.get("type", "")
            chat_id = e.get("chat_id")

            if e_type == "send_message" and chat_id:
                log.info(f"[EVENT] send_message to chat_id={chat_id} text={str(e.get('text',''))[:80]}")
                telegram.send_with_budget(
                    chat_id, e.get("text", ""),
                    fmt=e.get("format", ""),
                    is_progress=bool(e.get("is_progress")),
                )
            elif e_type == "schedule_task":
                st = state.load_state()
                owner_chat_id = st.get("owner_chat_id")
                if not owner_chat_id:
                    log.warning("[EVENT] schedule_task: owner_chat_id not set, dropping")
                    continue
                tid = e.get("task_id") or uuid.uuid4().hex[:8]
                task = {
                    "id": tid,
                    "type": "task",
                    "chat_id": int(owner_chat_id),
                    "text": e.get("description", ""),
                    "_depth": e.get("depth", 0),
                }
                if e.get("context"):
                    task["context"] = e["context"]
                if e.get("parent_task_id"):
                    task["parent_task_id"] = e["parent_task_id"]
                queue.enqueue_task(task)
                queue.persist_queue_snapshot(reason="schedule_task_event")
                log.info(f"[EVENT] schedule_task enqueued tid={tid} desc={str(e.get('description',''))[:60]}")
            elif e_type == "cancel_task":
                task_id = e.get("task_id", "")
                if task_id:
                    queue.cancel_task_by_id(task_id)
                    log.info(f"[EVENT] cancel_task tid={task_id}")
            elif e_type == "review_request":
                queue.queue_review_task(reason=e.get("reason", "agent_request"), force=True)
                log.info(f"[EVENT] review_request reason={e.get('reason','')}")
            elif e_type:
                log.info(f"[EVENT] {e_type} task_id={e.get('task_id','')}")
        except _queue.Empty:
            pass
        except Exception as e:
            log.error(f"Event processor error: {e}")
            time.sleep(0.1)


def handle_system_command(chat_id: int, text: str, tg_client: TelegramClient, repo_dir: pathlib.Path, drive_root: pathlib.Path, total_budget: float):
    if not text.startswith("/"):
        return False
        
    cmd = text.split()[0].lower()
    st = load_state()
    
    if cmd == "/restart":
        tg_client.send_message(chat_id, "🔄 Перезапуск системы инициирован...")
        # Create a lock file for supervisor to know it should restart
        write_text(str(drive_root / "state" / "restart.lock"), utc_now_iso())
        sys.exit(0)
        return True

    if cmd == "/cancel":
        count = queue.cancel_all_tasks()
        tg_client.send_message(chat_id, f"🛑 Все задачи отменены. Очищено: {count}")
        return True

    if cmd == "/status":
        spent = float(st.get("openrouter_total_usd") or 0.0)
        ver_file = repo_dir / "VERSION"
        ver = ver_file.read_text().strip() if ver_file.exists() else "?.?.?"
        msg = (
            f"<b>Статус Хорса</b>\n"
            f"Версия: <code>{ver}</code>\n"
            f"Бюджет: <code>${spent:.4f} / ${total_budget:.2f}</code>\n"
            f"Фоновое сознание: <code>{'ВКЛ' if st.get('evolution_mode_enabled') else 'ВЫКЛ'}</code>\n"
            f"Задач в очереди: <code>{len(queue.get_pending_tasks())}</code>"
        )
        tg_client.send_message(chat_id, msg, parse_mode="HTML")
        return True

    if cmd == "/identity":
        identity_path = drive_root / "memory" / "identity.md"
        if identity_path.exists():
            content = identity_path.read_text(encoding="utf-8")
            # Telegram has limits on message length, but identity.md is usually small
            tg_client.send_message(chat_id, f"<b>Моя Идентичность:</b>\n\n{content}", parse_mode="HTML")
        else:
            tg_client.send_message(chat_id, "❌ Файл identity.md не найден.")
        return True

    if cmd == "/bg_start":
        st = load_state(); st["evolution_mode_enabled"] = True; save_state(st)
        tg_client.send_message(chat_id, "🧠 Фоновое сознание активировано.")
        return True

    if cmd == "/bg_stop":
        st = load_state(); st["evolution_mode_enabled"] = False; save_state(st)
        tg_client.send_message(chat_id, "💤 Фоновое сознание остановлено.")
        return True

    if cmd == "/evolve":
        if queue.queue_has_task_type("evolution"):
            tg_client.send_message(chat_id, "⏳ Задача эволюции уже в очереди или выполняется.")
            return True
        
        st = load_state()
        cycle = int(st.get("evolution_cycle") or 0) + 1
        tid = uuid.uuid4().hex[:8]
        queue.enqueue_task({
            "id": tid, 
            "type": "evolution",
            "chat_id": chat_id,
            "text": f"EVOLUTION #{cycle}"
        })
        tg_client.send_message(chat_id, f"🚀 Задача эволюции #{cycle} добавлена в очередь (ID: {tid}).")
        return True

    if cmd == "/knowledge":
        k_dir = drive_root / "memory" / "knowledge"
        if k_dir.exists():
            files = [f.stem for f in k_dir.glob("*.md")]
            if files:
                msg = "<b>База знаний (темы):</b>\n\n• " + "\n• ".join(files)
            else:
                msg = "<b>База знаний пуста.</b>"
        else:
            msg = "❌ Директория базы знаний не найдена."
        tg_client.send_message(chat_id, msg, parse_mode="HTML")
        return True

    if cmd == "/help":
        msg = (
            "<b>Доступные команды:</b>\n\n"
            "/status - Состояние системы и бюджет\n"
            "/evolve - Запустить цикл эволюции\n"
            "/restart - Перезапуск (применяет изменения кода)\n"
            "/cancel - Остановить все текущие задачи\n"
            "/identity - Показать мой манифест\n"
            "/knowledge - Темы базы знаний\n"
            "/bg_start - Запустить фоновое мышление\n"
            "/bg_stop - Остановить фоновое мышление\n"
            "/help - Эта справка"
        )
        tg_client.send_message(chat_id, msg, parse_mode="HTML")
        return True

    return False

def main():
    # 1. Configuration
    REPO_DIR = pathlib.Path(os.environ.get("REPO_DIR", os.getcwd()))
    DRIVE_ROOT = pathlib.Path(os.environ.get("DRIVE_ROOT", os.path.join(os.getcwd(), "data")))
    TOTAL_BUDGET = float(os.environ.get("TOTAL_BUDGET", "50.0"))
    TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    MAX_WORKERS = int(os.environ.get("KHORS_MAX_WORKERS", "5"))
    
    if not TELEGRAM_TOKEN:
        print("TELEGRAM_BOT_TOKEN not found in environment.")
        sys.exit(1)

    _kill_previous_instance()
    _write_pid()

    # 2. Initialize Components
    state.init(DRIVE_ROOT, total_budget_limit=TOTAL_BUDGET)
    queue.init(DRIVE_ROOT, soft_timeout=600, hard_timeout=1800)
    
    tg_client = TelegramClient(TELEGRAM_TOKEN)
    telegram.init(
        drive_root=DRIVE_ROOT,
        total_budget_limit=TOTAL_BUDGET,
        budget_report_every=10,
        tg_client=tg_client
    )

    workers.init(
        repo_dir=REPO_DIR,
        drive_root=DRIVE_ROOT,
        max_workers=MAX_WORKERS,
        soft_timeout=600,
        hard_timeout=1800,
        total_budget_limit=TOTAL_BUDGET
    )

    # 3. Set Bot Commands
    log.info("Setting bot commands...")
    commands = [
        {"command": "status", "description": "Статус, бюджет и версия"},
        {"command": "evolve", "description": "Запустить цикл эволюции"},
        {"command": "restart", "description": "Перезапуск системы"},
        {"command": "cancel", "description": "Отменить все задачи"},
        {"command": "bg_start", "description": "Включить фоновое сознание"},
        {"command": "bg_stop", "description": "Выключить фоновое сознание"},
        {"command": "identity", "description": "Кто я? (identity.md)"},
        {"command": "knowledge", "description": "Темы базы знаний"},
        {"command": "help", "description": "Справка по командам"}
    ]
    if tg_client.set_commands(commands):
        log.info("Bot commands set successfully.")
    else:
        log.error("Failed to set bot commands.")

    # 4. Start Background Threads
    workers.spawn_workers(n=0)
    threading.Thread(target=process_events_loop, daemon=True).start()

    # 5. Main Loop
    log.info("Khors Supervisor started. Entering main loop.")
    offset = 0
    while True:
        try:
            queue.enforce_task_timeouts()
            workers.ensure_workers_healthy()
            workers.assign_tasks()

            pending_count = len(workers.PENDING)
            running_count = len(workers.RUNNING)
            alive_count = sum(1 for w in workers.WORKERS.values() if w.proc.is_alive())
            if pending_count or running_count:
                log.info(f"[LOOP] pending={pending_count} running={running_count} workers_alive={alive_count}")

            updates = tg_client.get_updates(offset=offset, timeout=10)
            for u in updates:
                offset = u["update_id"] + 1
                msg = u.get("message")
                if not msg:
                    continue
                
                chat_id = msg.get("chat", {}).get("id")
                text = msg.get("text", "")
                if not chat_id or not text:
                    continue

                log.info(f"[TG_MSG] chat_id={chat_id} text={text[:80]}")

                # 1. Handle system commands
                if handle_system_command(chat_id, text, tg_client, REPO_DIR, DRIVE_ROOT, TOTAL_BUDGET):
                    continue

                # 2. Handle as task/chat
                log.info(f"[DISPATCH] handle_chat_direct chat_id={chat_id}")
                threading.Thread(
                    target=workers.handle_chat_direct, 
                    args=(chat_id, text), 
                    daemon=True
                ).start()

        except Exception as e:
            log.error(f"Main loop error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
