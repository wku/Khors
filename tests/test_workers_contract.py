import inspect
import pathlib
import queue
import importlib


def _get_params(func):
    return dict(inspect.signature(func).parameters)


class TestHandleTaskSignature:

    def test_handle_task_requires_result_queue(self):
        from khors.agent import KhorsAgent
        params = _get_params(KhorsAgent.handle_task)
        assert "result_queue" in params, (
            "KhorsAgent.handle_task() must accept 'result_queue'. "
            "workers.py passes it."
        )

    def test_handle_task_task_manager_requires_result_queue(self):
        from khors.task_manager import TaskManager
        params = _get_params(TaskManager.handle_task)
        assert "result_queue" in params, (
            "TaskManager.handle_task() must accept 'result_queue'."
        )

    def test_workers_calls_handle_task_with_result_queue(self):
        src = importlib.import_module("supervisor.workers")
        source = inspect.getsource(src)
        assert "agent.handle_task(task, result_q)" in source, (
            "supervisor/workers.py handle_chat_direct must call "
            "agent.handle_task(task, result_q)."
        )

    def test_worker_main_calls_handle_task_with_result_queue(self):
        src = inspect.getsource(importlib.import_module("supervisor.workers"))
        count = src.count("agent.handle_task(task, result_q)")
        assert count >= 2, (
            "supervisor/workers.py must call agent.handle_task(task, result_q) "
            "in both handle_chat_direct and worker_main."
        )


class TestReadTextAcceptsPath:

    def test_read_text_accepts_path_object(self, tmp_path):
        from khors.utils import read_text
        f = tmp_path / "test.txt"
        f.write_text("hello", encoding="utf-8")
        result = read_text(f)
        assert result == "hello"

    def test_read_text_fails_on_str(self, tmp_path):
        from khors.utils import read_text
        f = tmp_path / "test.txt"
        f.write_text("hello", encoding="utf-8")
        try:
            read_text(str(f))
            raise AssertionError(
                "read_text(str) should raise AttributeError: "
                "'str' object has no attribute 'read_text'. "
                "Always pass pathlib.Path, not str."
            )
        except AttributeError:
            pass

    def test_system_monitor_passes_path_not_str(self):
        src = inspect.getsource(importlib.import_module("khors.system_monitor"))
        assert "read_text(str(" not in src, (
            "system_monitor.py must not wrap Path in str() before passing to read_text(). "
            "read_text() accepts pathlib.Path only."
        )


class TestSanitizeTaskForEvent:

    def test_accepts_without_drive_logs(self):
        from khors.utils import sanitize_task_for_event
        params = _get_params(sanitize_task_for_event)
        p = params.get("drive_logs")
        assert p is not None, "drive_logs parameter missing"
        assert p.default is not inspect.Parameter.empty, (
            "sanitize_task_for_event() drive_logs must have a default value "
            "(called without it from task_manager.py)."
        )

    def test_call_without_drive_logs_does_not_raise(self):
        from khors.utils import sanitize_task_for_event
        task = {"id": "abc123", "text": "test message"}
        result = sanitize_task_for_event(task)
        assert "id" in result

    def test_call_with_large_text_without_drive_logs(self):
        from khors.utils import sanitize_task_for_event
        task = {"id": "abc123", "text": "x" * 5000}
        result = sanitize_task_for_event(task)
        assert result.get("text_truncated") is True
        assert "text_full_path" not in result


class TestTaskManagerPaths:

    def test_no_fstring_paths_in_task_manager(self):
        src = inspect.getsource(importlib.import_module("khors.task_manager"))
        assert 'f"{self.drive_path}' not in src, (
            "task_manager.py must not use f-string paths. "
            "append_jsonl() requires pathlib.Path, not str."
        )

    def test_add_usage_not_called_with_one_arg(self):
        src = inspect.getsource(importlib.import_module("khors.task_manager"))
        assert 'add_usage(response["usage"])' not in src, (
            "task_manager.py calls add_usage() with one argument. "
            "add_usage(total, usage) requires two arguments."
        )


class TestTaskManagerConstructor:

    def test_task_manager_accepts_env_not_repo_path(self):
        from khors.task_manager import TaskManager
        params = _get_params(TaskManager.__init__)
        assert "env" in params, (
            "TaskManager.__init__() must accept 'env'. "
            "agent.py passes Env object."
        )
        assert "repo_path" not in params, (
            "TaskManager.__init__() must not have 'repo_path'. "
            "Use env.repo_dir instead."
        )

    def test_task_manager_accepts_tools(self):
        from khors.task_manager import TaskManager
        params = _get_params(TaskManager.__init__)
        assert "tools" in params, (
            "TaskManager.__init__() must accept 'tools' (ToolRegistry). "
            "Required for run_llm_loop."
        )

    def test_run_llm_loop_called_with_llm_not_llm_client(self):
        src = inspect.getsource(importlib.import_module("khors.task_manager"))
        assert "llm_client=self.llm_client" not in src, (
            "task_manager.py must call run_llm_loop with llm=, not llm_client=."
        )

    def test_build_llm_messages_called_with_env(self):
        src = inspect.getsource(importlib.import_module("khors.task_manager"))
        assert "env=self.env" in src, (
            "task_manager.py must call build_llm_messages with env=self.env."
        )

