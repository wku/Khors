"""
Khors — self-modifying AI agent.

Philosophy: BIBLE.md
Architecture: agent.py (orchestrator), tools/ (plugin tools),
              llm.py (LLM client), memory.py (memory), review.py (deep review),
              utils.py (shared utilities).
"""

# IMPORTANT: Do NOT import agent/loop/llm/etc here!
# launcher.py imports khors.apply_patch, which triggers __init__.py.
# Any eager imports here get loaded into supervisor's memory and persist
# in forked worker processes as stale code, preventing hot-reload.
# Workers import make_agent directly from khors.agent.

__all__ = ['agent', 'tools', 'llm', 'memory', 'review', 'utils']

from pathlib import Path as _Path
__version__ = (_Path(__file__).resolve().parent.parent / 'VERSION').read_text(encoding='utf-8').strip()
