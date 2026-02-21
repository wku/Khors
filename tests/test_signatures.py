import pytest
import inspect
import importlib


def _get_params(func):
    sig = inspect.signature(func)
    return {name: p.kind for name, p in sig.parameters.items()}


def _assert_accepts(func, *param_names):
    params = _get_params(func)
    has_kwargs = any(v == inspect.Parameter.VAR_KEYWORD for v in params.values())
    for name in param_names:
        if not has_kwargs:
            assert name in params, (
                f"{func.__qualname__}() does not accept parameter '{name}'. "
                f"Available: {list(params.keys())}"
            )


def _assert_rejects(func, *param_names):
    params = _get_params(func)
    has_kwargs = any(v == inspect.Parameter.VAR_KEYWORD for v in params.values())
    if has_kwargs:
        return
    for name in param_names:
        assert name not in params, (
            f"{func.__qualname__}() should NOT have parameter '{name}' "
            f"but it does. Likely stale after refactoring."
        )


def _assert_has_method(cls, method_name):
    assert hasattr(cls, method_name), (
        f"{cls.__name__} missing method '{method_name}'"
    )


def _assert_no_method(cls, method_name):
    assert not hasattr(cls, method_name), (
        f"{cls.__name__} has unexpected method '{method_name}' - "
        f"callers may reference a stale name"
    )


class TestRunLlmLoopSignature:

    def test_accepts_drive_root(self):
        from khors.loop import run_llm_loop
        _assert_accepts(run_llm_loop, "drive_root")

    def test_rejects_drive_logs(self):
        from khors.loop import run_llm_loop
        _assert_rejects(run_llm_loop, "drive_logs")

    def test_accepts_required_params(self):
        from khors.loop import run_llm_loop
        _assert_accepts(
            run_llm_loop,
            "llm", "messages", "tools", "model",
            "task_id", "event_queue", "emit_progress",
            "incoming_messages", "task_type",
            "budget_remaining_usd", "drive_root",
        )


class TestMemorySignatures:

    def test_summarize_chat_accepts_entries_only(self):
        from khors.memory import Memory
        _assert_accepts(Memory.summarize_chat, "entries")
        _assert_rejects(Memory.summarize_chat, "task_id")

    def test_summarize_progress_signature(self):
        from khors.memory import Memory
        _assert_accepts(Memory.summarize_progress, "entries", "limit")


class TestCompactToolHistory:

    def test_compact_tool_history_accepts_keep_recent(self):
        from khors.context import compact_tool_history
        _assert_accepts(compact_tool_history, "messages", "keep_recent")
        _assert_rejects(compact_tool_history, "max_tool_results", "max_tokens")

    def test_compact_tool_history_llm_accepts_keep_recent(self):
        from khors.context import compact_tool_history_llm
        _assert_accepts(compact_tool_history_llm, "messages", "keep_recent")
        _assert_rejects(compact_tool_history_llm, "max_tokens")


class TestToolRegistryMethods:

    def test_has_schemas(self):
        from khors.tools.registry import ToolRegistry
        _assert_has_method(ToolRegistry, "schemas")

    def test_no_get_schemas(self):
        from khors.tools.registry import ToolRegistry
        _assert_no_method(ToolRegistry, "get_schemas")

    def test_has_list_non_core_tools(self):
        from khors.tools.registry import ToolRegistry
        _assert_has_method(ToolRegistry, "list_non_core_tools")

    def test_has_set_context(self):
        from khors.tools.registry import ToolRegistry
        _assert_has_method(ToolRegistry, "set_context")


class TestSendWithBudgetSignature:

    def test_accepts_fmt(self):
        from supervisor.telegram import send_with_budget
        _assert_accepts(send_with_budget, "chat_id", "text", "fmt", "is_progress")

    def test_rejects_parse_mode(self):
        from supervisor.telegram import send_with_budget
        _assert_rejects(send_with_budget, "parse_mode")


class TestHandleToolCallsSignature:

    def test_signature(self):
        from khors.tool_executor import handle_tool_calls
        _assert_accepts(
            handle_tool_calls,
            "tool_calls", "tools", "drive_logs", "task_id",
            "stateful_executor", "messages", "llm_trace", "emit_progress",
        )


class TestAgentCallsMatchLoop:

    def test_agent_handle_task_exists(self):
        from khors.agent import KhorsAgent
        _assert_has_method(KhorsAgent, "handle_task")

    def test_agent_has_no_stale_run_llm_params(self):
        src = inspect.getsource(importlib.import_module("khors.agent"))
        assert "drive_logs=drive_logs" not in src or "run_llm_loop" not in src, (
            "agent.py passes drive_logs to run_llm_loop which does not accept it"
        )


class TestCallLlmWithRetrySignature:

    def test_signature(self):
        from khors.loop_helpers import call_llm_with_retry
        _assert_accepts(
            call_llm_with_retry,
            "llm", "messages", "model", "tools",
            "max_retries", "drive_logs", "task_id",
            "round_idx", "event_queue", "accumulated_usage",
        )


class TestEstimateCostSignature:

    def test_signature(self):
        from khors.pricing import estimate_cost
        _assert_accepts(
            estimate_cost,
            "model", "prompt_tokens", "completion_tokens",
        )
