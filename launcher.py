import datetime
import json
import logging
import os
import pathlib
import sys
import time
import traceback

from supervisor import state, telegram, workers, queue
from supervisor.state import load_state, save_state, append_jsonl
from khors.utils import utc_now_iso, write_text, run_cmd

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
    _startup_lock = _PROJECT_ROOT / "data" / "state" / "startup_verify.lock"
    _startup_lock.unlink(missing_ok=True)

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
def set_commands(repo_dir: pathlib.Path, drive_root: pathlib.Path):
    commands = [
        {"command": "start", "description": "Запуск и приветствие"},
        {"command": "status", "description": "Общий статус и бюджет"},
        {"command": "restart", "description": "Перезапуск системы"},
        {"command": "cancel", "description": "Отмена всех задач"},
        {"command": "identity", "description": "Кто я (манифест)"},
        {"command": "bg_start", "description": "Включить фоновое сознание"},
        {"command": "bg_stop", "description": "Выключить фоновое сознание"},
        {"command": "help", "description": "Справка по командам"}
    ]
    telegram.set_commands(commands)


def handle_system_command(chat_id, text, repo_dir, drive_root):
    cmd = text.split()[0].lower()
    st = load_state()
    
    if cmd == "/restart":
        sha = run_cmd(["git", "rev-parse", "HEAD"], cwd=str(repo_dir)).strip()
        verify_path = drive_root / "state" / "pending_restart_verify.json"
        write_text(str(verify_path), json.dumps({
            "ts": utc_now_iso(), "expected_sha": sha, "reason": "owner_command"
        }))
        append_jsonl(drive_root / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "restart_request", "reason": "owner_command"
        })
        telegram.send_with_budget(chat_id, "🔄 Перезапуск системы инициирован...")
        sys.exit(0)
        return True

    if cmd == "/cancel":
        count = queue.cancel_all_tasks()
        telegram.send_with_budget(chat_id, f"🛑 Все задачи отменены. Очищено: {count}")
        return True

    if cmd == "/status":
        budget_info = f"💰 Бюджет: {st.get('openrouter_daily_usd', 0):.4f}$ / {state.TOTAL_BUDGET_LIMIT}$"
        version_path = repo_dir / "VERSION"
        version = f"📦 Версия: {version_path.read_text().strip() if version_path.exists() else 'unknown'}"
def handle_system_command(chat_id: int, text: str, tg_client: telegram.TelegramClient):
    if not text.startswith("/"):
        return False
        
    cmd = text.split()[0].lower()
    
    if cmd == "/restart":
        tg_client.send_message(chat_id, "🔄 Запрашиваю перезапуск системы...")
        write_text(DRIVE_ROOT / "state" / "restart.lock", utc_now_iso())
        return True
        
    if cmd == "/cancel":
        tg_client.send_message(chat_id, "🛑 Отменяю все активные и ожидающие задачи...")
        queue.cancel_all_tasks()
        return True
        
    if cmd == "/identity":
        identity_path = DRIVE_ROOT / "memory" / "identity.md"
        if identity_path.exists():
            content = identity_path.read_text(encoding="utf-8")
            tg_client.send_message(chat_id, f"<b>Моя Идентичность:</b>\n\n{content}", parse_mode="HTML")
        else:
            tg_client.send_message(chat_id, "❌ Файл identity.md не найден.")
        return True
        
    if cmd == "/status":
        st = load_state(DRIVE_ROOT)
        spent = float(st.get("openrouter_total_usd") or 0.0)
        total = TOTAL_BUDGET
        ver_file = DRIVE_ROOT.parent / "VERSION"
        ver = ver_file.read_text().strip() if ver_file.exists() else "?.?.?"
        msg = (
            f"<b>Статус Хорса</b>\n"
            f"Версия: <code>{ver}</code>\n"
            f"Бюджет: <code>${spent:.4f} / ${total:.2f}</code>\n"
            f"Фоновое сознание: <code>{'ВКЛ' if st.get('evolution_mode_enabled') else 'ВЫКЛ'}</code>\n"
            f"Задач в очереди: <code>{len(queue.get_pending_tasks())}</code>"
        )
        tg_client.send_message(chat_id, msg, parse_mode="HTML")
        return True

    if cmd == "/help":
        msg = (
            "<b>Доступные команды:</b>\n\n"
            "/status - Состояние системы и бюджет\n"
            "/restart - Перезапуск (применяет изменения кода)\n"
            "/cancel - Остановить все текущие задачи\n"
            "/identity - Показать мой манифест\n"
            "/bg_start - Запустить фоновое мышление\n"
            "/bg_stop - Остановить фоновое мышление\n"
            "/help - Эта справка"
        )
        tg_client.send_message(chat_id, msg, parse_mode="HTML")
        return True

    return False

def main():
        telegram.send_with_budget(chat_id, f"🤖 *Статус Хорса*\n\n{version}\n{budget_info}\n{tasks}", parse_mode="Markdown")
        return True

    if cmd == "/identity":
        path = drive_root / "memory" / "identity.md"
        content = path.read_text() if path.exists() else "Идентичность не найдена."
        telegram.send_with_budget(chat_id, f"👤 *Мой Манифест*\n\n{content}", parse_mode="Markdown")
        return True

    return False


def main():
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
    
    # Set bot commands
    log.info("Setting bot commands...")
    tg_client.set_commands([
        {"command": "start", "description": "Запуск и приветствие"},
        {"command": "status", "description": "Статус, бюджет и версия"},
        {"command": "restart", "description": "Перезапуск системы"},
        {"command": "cancel", "description": "Отменить все задачи"},
        {"command": "bg_start", "description": "Включить фоновое сознание"},
        {"command": "bg_stop", "description": "Выключить фоновое сознание"},
        {"command": "identity", "description": "Кто я? (identity.md)"},
        {"command": "help", "description": "Справка по командам"}
    ])
    log.info("Initializing telegram module")
                        if chat_id and text:
                             if not handle_system_command(chat_id, text, tg_client):
                                 threading.Thread(target=workers.handle_chat_direct, args=(chat_id, text), daemon=True).start()
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