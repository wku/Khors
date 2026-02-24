import os
import time
import json
import threading
import pathlib
from datetime import datetime, timezone
from khors import utils

# Пути к файлам
DATA_DIR = os.environ.get("KHORS_DATA_DIR", "data")
STATE_FILE = pathlib.Path(DATA_DIR) / "state" / "state.json"
TIMERS_FILE = pathlib.Path(DATA_DIR) / "state" / "timers.json"
QUEUE_FILE = pathlib.Path(DATA_DIR) / "state" / "queue.jsonl"
LOG_FILE = pathlib.Path(DATA_DIR) / "logs" / "supervisor.jsonl"

def log_event(message, level="info"):
    utils.append_jsonl(LOG_FILE, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "planner_service",
        "level": level,
        "message": message
    })

def emit_supervisor_event(event):
    """Отправляет событие напрямую в очередь событий супервизора"""
    try:
        from supervisor import workers
        event_q = workers.get_event_q()
        if event_q:
            event_q.put(event)
            log_event(f"Emitted supervisor event: {event.get('type')}")
    except Exception as e:
        log_event(f"Failed to emit supervisor event: {e}", level="error")

def get_file_lock(path: pathlib.Path):
    """Простая блокировка файла через .lock файл"""
    lock_path = path.parent / f".{path.name}.lock"
    start_time = time.time()
    while time.time() - start_time < 5.0: # 5 секунд таймаут
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            return fd, lock_path
        except FileExistsError:
            # Проверка на протухший лок (10 сек)
            try:
                if time.time() - lock_path.stat().st_mtime > 10.0:
                    lock_path.unlink()
                    continue
            except:
                pass
            time.sleep(0.1)
    return None, None

def release_file_lock(fd, lock_path):
    if fd:
        os.close(fd)
    if lock_path and lock_path.exists():
        try:
            lock_path.unlink()
        except:
            pass

def should_trigger(timer_id, config, timers_state, now_ts):
    """Определяет, должен ли сработать таймер. Чистая функция для тестов."""
    t_type = config.get("type", "regular")
    if t_type == "regular":
        interval = config.get("interval", 3600)
        last_ts = timers_state.get(timer_id, 0)
        if timer_id not in timers_state:
            return False, False # Инициализация при первом запуске
        return (now_ts - last_ts >= interval), False
    
    elif t_type == "once":
        if "completed_at" in config:
            return False, False
        at_str = config.get("at")
        if not at_str:
            return False, False
        try:
            at_dt = datetime.fromisoformat(at_str.replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) >= at_dt), False
        except Exception:
            return False, True
    return False, False

def check_timers():
    try:
        if not TIMERS_FILE.exists():
            return

        # 1. Читаем таймеры и состояние атомарно
        fd, lock = get_file_lock(STATE_FILE)
        if not fd:
            log_event("Could not acquire state lock, skipping cycle", level="warning")
            return

        try:
            with open(TIMERS_FILE, "r") as f:
                timers = json.load(f)
            
            if STATE_FILE.exists():
                with open(STATE_FILE, "r") as f:
                    state = json.load(f)
            else:
                state = {}

            now_ts = time.time()
            now_iso = datetime.now(timezone.utc).isoformat()
            timers_state = state.get("timers_state", {})
            state_changed = False
            timers_changed = False
            
            # Проверяем очередь
            existing_tasks = []
            if QUEUE_FILE.exists():
                with open(QUEUE_FILE, "r") as f:
                    for line in f:
                        try:
                            t = json.loads(line)
                            existing_tasks.append(t)
                        except:
                            continue

            for timer_id, config in timers.items():
                t_type = config.get("type", "regular")
                
                # Инициализация для новых регулярных таймеров
                if t_type == "regular" and timer_id not in timers_state:
                    timers_state[timer_id] = now_ts
                    state_changed = True
                    continue

                triggered, error = should_trigger(timer_id, config, timers_state, now_ts)
                
                if error:
                    log_event(f"Invalid config for timer {timer_id}", level="error")
                    continue

                if triggered:
                    # Проверяем, нет ли уже такой задачи в очереди (защита от спама)
                    is_duplicate = any(t.get("id", "").startswith(f"autoplan-{timer_id}") for t in existing_tasks)
                    
                    if is_duplicate:
                        log_event(f"Timer {timer_id} triggered but task already in queue, skipping")
                        # Обновляем время, чтобы не долбиться каждую секунду
                        if t_type == "regular":
                            timers_state[timer_id] = now_ts
                            state_changed = True
                        elif t_type == "once":
                            config["completed_at"] = now_iso
                            timers_changed = True
                        continue

                    log_event(f"Timer {timer_id} ({t_type}) triggered")
                    
                    task_id = f"autoplan-{timer_id}-{int(now_ts)}"
                    emit_supervisor_event({
                        "type": "new_task",
                        "task": {
                            "id": task_id,
                            "type": "evolution",
                            "description": f"Autonomous trigger: {timer_id} ({config.get('description', '')})",
                            "context": {"is_wakeup": True},
                            "is_wakeup": True
                        }
                    })
                    
                    if t_type == "regular":
                        timers_state[timer_id] = now_ts
                        state_changed = True
                    elif t_type == "once":
                        config["completed_at"] = now_iso
                        timers_changed = True

            if timers_changed:
                new_timers = {k: v for k, v in timers.items() if v.get("type") != "once" or "completed_at" not in v}
                with open(TIMERS_FILE, "w") as f:
                    json.dump(new_timers, f, indent=2)
                log_event("timers.json updated")

            if state_changed:
                state["timers_state"] = timers_state
                state["last_autoplan_ts"] = now_ts
                state["last_autoplan_at"] = now_iso
                with open(STATE_FILE, "w") as f:
                    json.dump(state, f, indent=2)
                log_event("state.json updated")

        finally:
            release_file_lock(fd, lock)

    except Exception as e:
        log_event(f"Error in check_timers: {e}", level="error")

def planner_loop():
    log_event("Planner service loop started (v2: atomic, once/regular support)")
    while True:
        try:
            check_timers()
        except Exception as e:
            log_event(f"Loop error: {e}", level="error")
        time.sleep(15)

def start_planner_service():
    log_event("Starting planner service thread")
    thread = threading.Thread(target=planner_loop, daemon=True)
    thread.start()
    return thread
