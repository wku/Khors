"""
Planner Service Extension.
This module is designed to be imported by the launcher to provide background planning.
"""

import logging
import time
import threading
import datetime
import os
import json
import pathlib

# We use local imports or dynamic lookups to avoid circular dependencies
# since this might be imported early in the boot process.

log = logging.getLogger("planner_service")

def get_state_path():
    drive_root = os.environ.get("DRIVE_ROOT", os.path.join(os.getcwd(), "data"))
    return pathlib.Path(drive_root) / "state" / "state.json"

def get_queue_path():
    drive_root = os.environ.get("DRIVE_ROOT", os.path.join(os.getcwd(), "data"))
    return pathlib.Path(drive_root) / "state" / "queue.jsonl"

def planner_loop(interval_sec: int = 3600 * 4):
    """
    Background loop that checks if it's time to trigger an autonomous planning task.
    This version is standalone to be easily integrated into the launcher.
    """
    log.info(f"Planner service loop started (interval: {interval_sec}s).")
    state_path = get_state_path()
    queue_path = get_queue_path()
    
    while True:
        try:
            if not state_path.exists():
                time.sleep(60)
                continue
                
            with open(state_path, 'r') as f:
                st = json.load(f)
            
            # Only work if evolution mode is enabled
            if not st.get("evolution_mode_enabled"):
                time.sleep(60)
                continue
                
            last_at_str = st.get("last_autoplan_at", "")
            now = datetime.datetime.now(datetime.timezone.utc)
            
            should_plan = False
            if not last_at_str:
                should_plan = True
            else:
                try:
                    last_at = datetime.datetime.fromisoformat(last_at_str)
                    if (now - last_at).total_seconds() >= interval_sec:
                        should_plan = True
                except ValueError:
                    should_plan = True
            
            if should_plan:
                # Check queue for existing evolution tasks
                has_evo = False
                if queue_path.exists():
                    with open(queue_path, 'r') as f:
                        for line in f:
                            if '"type": "evolution"' in line:
                                has_evo = True
                                break
                
                if not has_evo:
                    log.info("Triggering autonomous planning task...")
                    
                    # Create the task
                    task = {
                        "id": f"autoplan-{int(time.time())}",
                        "type": "evolution",
                        "chat_id": st.get("owner_chat_id"),
                        "text": "AUTOPLAN: Strategic Review and System Analysis",
                        "context": "This is an autonomous planning task triggered by the Planner Service. Use 'generate_plan' tool to analyze state and decide on next steps.",
                        "created_at": now.isoformat()
                    }
                    
                    # Append to queue
                    with open(queue_path, 'a') as f:
                        f.write(json.dumps(task) + "\n")
                    
                    # Update state
                    st["last_autoplan_at"] = now.isoformat()
                    with open(state_path, 'w') as f:
                        json.dump(st, f, indent=2)
                        
                    log.info(f"Autonomous planning task enqueued. Next check in {interval_sec}s.")
                else:
                    log.debug("Evolution task already in queue, skipping autoplan trigger.")
                    
        except Exception as e:
            log.error(f"Error in planner_loop: {e}")
            
        time.sleep(60)

def start_planner_service(interval_sec: int = 3600 * 4):
    """Starts the planner loop in a daemon thread."""
    t = threading.Thread(target=planner_loop, args=(interval_sec,), daemon=True, name="PlannerService")
    t.start()
    return t
