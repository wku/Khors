import pytest
import os
import json
import time
from khors.tools.extensions.planner_service import should_trigger

def test_memory_node_creation(tmp_path):
    # (keeps original logic)
    node_dir = tmp_path / "nodes"
    node_dir.mkdir()
    node_id = "test_node_1"
    content = "Test content for evolution"
    metadata = {"type": "test"}
    node_path = node_dir / f"{node_id}.json"
    node_data = {"id": node_id, "content": content, "metadata": metadata, "created_at": time.time()}
    with open(node_path, "w") as f:
        json.dump(node_data, f)
    assert node_path.exists()
    with open(node_path, "r") as f:
        loaded = json.load(f)
        assert loaded["id"] == node_id

def test_planner_timer_logic():
    # Тестируем логику срабатывания таймера через should_trigger
    config = {"type": "regular", "interval": 100}
    timers_state = {"test_timer": 880}
    
    # Срабатывает через 120 сек (1000 - 880 = 120 > 100)
    triggered, err = should_trigger("test_timer", config, timers_state, 1000)
    assert triggered is True
    assert err is False
    
    # Не срабатывает через 50 сек (930 - 880 = 50 < 100)
    triggered2, err2 = should_trigger("test_timer", config, timers_state, 930)
    assert triggered2 is False
    assert err2 is False

def test_planner_once_timer():
    from datetime import datetime, timedelta, timezone
    at_past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    config = {"type": "once", "at": at_past}
    
    triggered, err = should_trigger("once_timer", config, {}, time.time())
    assert triggered is True
    assert err is False
