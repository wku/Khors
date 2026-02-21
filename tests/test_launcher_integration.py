import os
import pathlib
import pytest
import shutil
import time

import sys
from unittest.mock import MagicMock

# Add project root to sys.path so we can import launcher.py even via pre-commit hooks
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the launcher module and supervisor components
import launcher
from supervisor import state, queue

class MockTelegramClient:
    def __init__(self, token="dummy"):
        self.sent_messages = []

    def send_message(self, chat_id, text, parse_mode=None):
        self.sent_messages.append({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode
        })

@pytest.fixture
def setup_env(tmp_path: pathlib.Path):
    """Setup isolated environment for testing launcher commands."""
    drive_root = tmp_path / "data"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    
    # Write a dummy version file
    version_file = repo_dir / "VERSION"
    version_file.write_text("1.0.0")

    # Initialize state system
    state.init(drive_root=drive_root, total_budget_limit=50.0)
    
    # Initialize queue system 
    queue.init(drive_root=drive_root, soft_timeout=60, hard_timeout=120)
    # Clear the queue since globals persist across tests
    queue.PENDING.clear()
    queue.RUNNING.clear()
    queue.QUEUE_SEQ_COUNTER_REF["value"] = 0

    return {
        "drive_root": drive_root,
        "repo_dir": repo_dir,
        "total_budget": 50.0,
        "chat_id": 123456
    }

def test_handle_status_command(setup_env):
    tg_client = MockTelegramClient()
    
    # Execute handle_system_command
    handled = launcher.handle_system_command(
        chat_id=setup_env["chat_id"],
        text="/status",
        tg_client=tg_client,
        repo_dir=setup_env["repo_dir"],
        drive_root=setup_env["drive_root"],
        total_budget=setup_env["total_budget"]
    )

    assert handled is True
    assert len(tg_client.sent_messages) == 1
    
    output = tg_client.sent_messages[0]["text"]
    assert "Статус Хорса" in output
    assert "1.0.0" in output
    assert "Бюджет" in output
    assert "Задач в очереди: <code>0</code>" in output


def test_handle_evolve_command(setup_env):
    tg_client = MockTelegramClient()
    
    handled = launcher.handle_system_command(
        chat_id=setup_env["chat_id"],
        text="/evolve",
        tg_client=tg_client,
        repo_dir=setup_env["repo_dir"],
        drive_root=setup_env["drive_root"],
        total_budget=setup_env["total_budget"]
    )

    assert handled is True
    assert len(tg_client.sent_messages) == 1
    
    output = tg_client.sent_messages[0]["text"]
    assert "Задача эволюции #1 добавлена в очередь" in output
    
    # Check if a task was actually queued
    assert len(queue.PENDING) == 1
    task = queue.PENDING[0]
    assert task["type"] == "evolution"
    assert task["chat_id"] == setup_env["chat_id"]


def test_handle_bg_commands(setup_env):
    tg_client = MockTelegramClient()
    
    # Test bg_start
    handled_start = launcher.handle_system_command(
        chat_id=setup_env["chat_id"],
        text="/bg_start",
        tg_client=tg_client,
        repo_dir=setup_env["repo_dir"],
        drive_root=setup_env["drive_root"],
        total_budget=setup_env["total_budget"]
    )
    
    assert handled_start is True
    st = state.load_state()
    assert st.get("evolution_mode_enabled") is True
    assert "Фоновое сознание активировано" in tg_client.sent_messages[-1]["text"]

    # Test bg_stop
    handled_stop = launcher.handle_system_command(
        chat_id=setup_env["chat_id"],
        text="/bg_stop",
        tg_client=tg_client,
        repo_dir=setup_env["repo_dir"],
        drive_root=setup_env["drive_root"],
        total_budget=setup_env["total_budget"]
    )
    
    assert handled_stop is True
    st = state.load_state()
    assert st.get("evolution_mode_enabled") is False
    assert "Фоновое сознание остановлено" in tg_client.sent_messages[-1]["text"]
