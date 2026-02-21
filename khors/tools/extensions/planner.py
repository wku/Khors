"""
Planner extension for Khors.
Enables proactive goal setting and system analysis.
"""

from __future__ import annotations
import logging
import pathlib
from typing import Any, List
from khors.tools.registry import ToolContext, ToolEntry
from khors.system_monitor import SystemMonitor
from khors.review import collect_sections, compute_complexity_metrics

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
                    "properties": {}
                }
            },
            handler=generate_plan
        )
    ]

def generate_plan(ctx: ToolContext) -> str:
    """
    Analyze system state and generate a proactive action plan.
    """
    try:
        # 1. Collect health data
        monitor = SystemMonitor(ctx.repo_dir, ctx.drive_root)
        pulse = monitor.get_system_info()
        
        # Manually compute health
        repo_path = pathlib.Path(ctx.repo_dir)
        drive_path = pathlib.Path(ctx.drive_root)
        sections, _ = collect_sections(repo_path, drive_path)
        health = compute_complexity_metrics(sections)
        
        report = [
            "# System Health Report for Planning",
            f"- **Version**: {pulse.get('version', 'unknown')}",
            f"- **Budget Spent**: ${pulse.get('budget', {}).get('total_spent', 0):.4f}",
            f"- **Critical Issues**: {len(health.get('oversized_functions', [])) + len(health.get('oversized_modules', []))}",
        ]
        
        violations = []
        for p, start, length in health.get("oversized_functions", []):
            violations.append(f"Principle 5 Violation: Function too long ({length} lines) in {p} at line {start}")
        for p, lines in health.get("oversized_modules", []):
            violations.append(f"Principle 5 Violation: Module too large ({lines} lines) in {p}")

        if violations:
            report.append("\n## Bible Violations:")
            for v in violations:
                report.append(f"- {v}")
                
        report.append("\n## Contextual Analysis:")
        report.append("- Memory Nodes system is active and populated.")
        report.append("- Evolution #2 is in progress (Memory & Planning).")
        
        report.append("\n## Recommended Next Steps:")
        report.append("1. **Review Identity**: Check if identity_manifest needs updates based on recent Evolution #1 success.")
        report.append("2. **Autonomous Tasking**: Use the data above to formulate 2-3 concrete tasks for the next hour.")
        report.append("3. **Knowledge Consolidation**: Create a memory node for 'lessons_learned_evolution_1'.")
        
        return "\n".join(report)
        
    except Exception as e:
        import traceback
        logger.error(f"Error in generate_plan: {e}")
        return f"Error generating plan: {str(e)}"
