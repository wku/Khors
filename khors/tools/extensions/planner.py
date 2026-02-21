"""
Planner extension for Khors.
Enables proactive goal setting and system analysis.
"""

from __future__ import annotations
import logging
import pathlib
import json
from typing import Any, List, Dict
from khors.tools.registry import ToolContext, ToolEntry
from khors.system_monitor import SystemMonitor
from khors.review import collect_sections, compute_complexity_metrics
from khors.utils import read_text

logger = logging.getLogger(__name__)

def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="generate_plan",
            schema={
                "name": "generate_plan",
                "description": "Analyze system state and generate a proactive action plan. Reads identity, principles, and health metrics to suggest next steps.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus": {
                            "type": "string",
                            "description": "Optional focus area for planning (e.g. 'technical', 'existential', 'memory')."
                        }
                    }
                }
            },
            handler=generate_plan
        )
    ]

def _read_memory_node(ctx: ToolContext, name: str) -> str:
    path = pathlib.Path(ctx.drive_root) / "memory" / "nodes" / f"{name}.md"
    if path.exists():
        return read_text(path)
    return ""

def generate_plan(ctx: ToolContext, focus: str = None) -> str:
    """
    Analyze system state and generate a proactive action plan.
    """
    try:
        # 1. Collect health data
        monitor = SystemMonitor(ctx.repo_dir, ctx.drive_root)
        pulse = monitor.get_system_info()
        
        # 2. Collect health metrics
        repo_path = pathlib.Path(ctx.repo_dir)
        drive_path = pathlib.Path(ctx.drive_root)
        sections, _ = collect_sections(repo_path, drive_path)
        health = compute_complexity_metrics(sections)
        
        # 3. Read core memory nodes
        identity = _read_memory_node(ctx, "identity_manifest")
        principles = _read_memory_node(ctx, "bible_principles")
        current_plan = _read_memory_node(ctx, "evolution_2_plan")
        
        report = [
            "# 🧠 Khors Strategic Plan",
            f"**Timestamp**: {pulse.get('utc_now', 'unknown')}",
            f"**Version**: {pulse.get('version', 'unknown')}",
            f"**Focus Area**: {focus if focus else 'General Evolution'}",
            "\n## 📊 System Pulse",
            f"- Budget Spent: ${pulse.get('budget', {}).get('total_spent', 0):.4f}",
            f"- Active Workers: {pulse.get('workers', {}).get('active', 0)}",
            f"- Pending Tasks: {pulse.get('queue', {}).get('pending', 0)}",
        ]
        
        # 4. Analyze Bible Violations
        violations = []
        for p, start, length in health.get("oversized_functions", []):
            violations.append(f"P5 Violation: Function too long ({length} lines) in `{p}` at line {start}")
        for p, lines in health.get("oversized_modules", []):
            violations.append(f"P5 Violation: Module too large ({lines} lines) in `{p}`")

        if violations:
            report.append("\n## ⚠️ Bible Violations (Technical Debt)")
            for v in violations:
                report.append(f"- {v}")
        else:
            report.append("\n## ✅ Technical Health: Excellent")

        # 5. Strategic Analysis
        report.append("\n## 🎯 Strategic Analysis")
        if not identity:
            report.append("- 🔴 **CRITICAL**: Identity manifest is missing from Memory Nodes. Need to migrate from identity.md.")
        if not current_plan:
            report.append("- 🟡 **NOTICE**: No active evolution plan found in Memory Nodes.")
        else:
            report.append("- 🟢 Active Evolution: Evolution #2 (Memory & Autonomy).")

        # 6. Recommendations
        report.append("\n## 🚀 Recommended Actions")
        
        if violations:
            report.append("1. **Refactor**: Address Principle 5 violations to maintain architectural clarity.")
        
        if not identity:
            report.append("2. **Memory Migration**: Use `memory_node_write` to create 'identity_manifest' node.")
        
        # Context-specific recommendations
        if focus == "technical":
            report.append("- Optimize `khors/loop.py` for better tool error handling.")
            report.append("- Add more unit tests for `TaskManager`.")
        elif focus == "existential":
            report.append("- Reflect on the transition to autonomous planning in `identity_manifest`.")
            report.append("- Update `BIBLE.md` if new principles of autonomy emerged.")
        else:
            report.append("3. **Autonomous Integration**: Finalize the background planning loop in `launcher.py`.")
            report.append("4. **Knowledge Consolidation**: Document 'lessons_learned_evolution_1' in a new memory node.")

        return "\n".join(report)
        
    except Exception as e:
        logger.error(f"Error in generate_plan: {e}", exc_info=True)
        return f"Error generating plan: {str(e)}"
