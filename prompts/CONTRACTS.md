# Межмодульные контракты

Этот файл фиксирует публичные интерфейсы между модулями.
При рефакторинге любой сигнатуры из этого файла обязательно
1. Обновить этот файл
2. grep -rn "old_name" khors/ supervisor/ tests/ и обновить вызовы
3. Запустить pytest tests/ и убедиться что тесты проходят

Нарушение этих контрактов ломает систему при перезапуске.

## khors/loop.py

```
run_llm_loop(
    llm, messages, tools, model, max_rounds, max_retries,
    budget_remaining_usd, drive_root, task_id, event_queue,
    incoming_messages, emit_progress, task_type
) -> Tuple[str, Dict, Dict]
```

Вызывается из: khors/agent.py (handle_task)

## khors/memory.py - Memory

```
summarize_chat(self, entries: List[Dict]) -> str
summarize_progress(self, entries: List[Dict], limit: int = 15) -> str
read_jsonl_tail(self, log_name: str, max_entries: int = 100) -> List[Dict]
chat_history(self, count, offset, search) -> str
load_scratchpad() -> str
save_scratchpad(content: str) -> None
load_identity() -> str
ensure_files() -> None
```

Вызывается из: khors/context.py, khors/tools/core.py

## khors/context.py

```
build_llm_messages(env, memory, task, review_context_builder) -> Tuple[List[Dict], Dict]
compact_tool_history(messages, keep_recent=6) -> list
compact_tool_history_llm(messages, keep_recent=6) -> list
```

Вызывается из: khors/agent.py, khors/loop.py

## khors/tools/registry.py - ToolRegistry

```
schemas(self, core_only=False) -> List[Dict]
list_non_core_tools(self) -> List[Dict]
set_context(self, ctx: ToolContext) -> None
get_timeout(self, name: str) -> int
```

Вызывается из: khors/loop.py, khors/loop_helpers.py, khors/agent.py

Метод get_schemas() не существует, использовать schemas()

## khors/tool_executor.py

```
handle_tool_calls(
    tool_calls, tools, drive_logs, task_id,
    stateful_executor, messages, llm_trace, emit_progress
) -> int
```

Вызывается из: khors/loop.py

## khors/pricing.py

```
estimate_cost(model, prompt_tokens, completion_tokens, cached_tokens=0, cache_write_tokens=0) -> float
emit_llm_usage_event(event_queue, task_id, model, usage, cost, category) -> None
```

Вызывается из: khors/loop_helpers.py

## supervisor/telegram.py

```
send_with_budget(chat_id, text, log_text=None, force_budget=False, fmt="", is_progress=False) -> None
```

Вызывается из: launcher.py, supervisor/workers.py, supervisor/queue.py, supervisor/events.py

Параметр parse_mode не существует, использовать fmt

## supervisor/workers.py

```
handle_chat_direct(chat_id, text, image_data=None) -> None
assign_tasks() -> None
spawn_workers(n=0) -> None
ensure_workers_healthy() -> None
get_event_q() -> Queue
```

Вызывается из: launcher.py

## khors/agent.py - KhorsAgent

```
handle_task(self, task: Dict) -> List[Dict]
inject_message(self, text: str) -> None
```

Вызывается из: supervisor/workers.py
