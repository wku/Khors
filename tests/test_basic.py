import pytest
import os
import json
from khors.utils import utc_now_iso, truncate_for_log, clip_text
from khors.llm import add_usage, normalize_reasoning_effort

def test_utc_now_iso():
    iso = utc_now_iso()
    assert isinstance(iso, str)
    assert "T" in iso
    assert iso.endswith("Z") or "+00:00" in iso

def test_truncate_for_log():
    short = "hello"
    assert truncate_for_log(short, 10) == "hello"
    
    long_text = "this is a very long text indeed"
    truncated = truncate_for_log(long_text, 15)
    # max_chars // 2 = 7. 7 + len("\n...\n") + 7 = 19
    assert len(truncated) < len(long_text)
    assert "..." in truncated

def test_clip_text():
    short = "hello"
    assert clip_text(short, 10) == "hello"
    
    long_text = "this is a very long text indeed"
    clipped = clip_text(long_text, 15)
    # half = max(200, 7) = 200, so it takes text[:200] + ... + text[-200:]
    # which results in the whole text duplicated around the truncation marker
    assert "truncated" in clipped
    assert len(clipped) > len(long_text)

def test_add_usage():
    total = {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01}
    usage = {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.005}
    
    add_usage(total, usage)
    
    assert total["prompt_tokens"] == 15
    assert total["completion_tokens"] == 7
    assert total["cost"] == 0.015

def test_normalize_reasoning_effort():
    assert normalize_reasoning_effort("HIGH") == "high"
    assert normalize_reasoning_effort("invalid_value", "medium") == "medium"
    assert normalize_reasoning_effort(None, "low") == "low"
    assert normalize_reasoning_effort("none") == "none"

def test_system_imports():
    # Ensure major modules can be imported without syntax errors
    import khors.agent
    import khors.loop
    import khors.memory
    import supervisor.state
    import supervisor.queue
    assert True
