"""Codebase health tool — complexity metrics and self-assessment."""

import json
import logging
import os
import pathlib
import subprocess
from typing import Any, Dict

from khors.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)


def system_pulse(ctx: ToolContext) -> str:
    """Check core system health: budget, git, filesystem, environment, versions."""
    results = []
    lines = ["## System Pulse Report\n"]

    # 1. Budget
    try:
        state_path = ctx.drive_path("state/state.json")
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            spent = state.get("spent_usd", 0)
            results.append(f"✅ **Budget:** ${spent:.4f} spent")
        else:
            results.append("⚠️ **Budget:** state.json not found")
    except Exception as e:
        results.append(f"❌ **Budget check failed:** {e}")

    # 2. Git
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(ctx.repo_dir), capture_output=True, text=True, timeout=5
        )
        if res.stdout.strip():
            results.append(f"⚠️ **Git:** Uncommitted changes detected\n```\n{res.stdout.strip()}\n```")
        else:
            results.append("✅ **Git:** Clean")
    except Exception as e:
        results.append(f"❌ **Git check failed:** {e}")

    # 3. Filesystem
    try:
        test_file = ctx.drive_path("health_test.tmp")
        test_file.write_text("test", encoding="utf-8")
        if test_file.read_text(encoding="utf-8") == "test":
            test_file.unlink()
            results.append("✅ **Filesystem:** Read/Write OK")
        else:
            results.append("❌ **Filesystem:** Integrity check failed")
    except Exception as e:
        results.append(f"❌ **Filesystem check failed:** {e}")

    # 4. Environment
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key and len(api_key) > 10:
        results.append(f"✅ **Environment:** OPENROUTER_API_KEY set (len={len(api_key)})")
    else:
        results.append("❌ **Environment:** OPENROUTER_API_KEY missing or too short")

    # 5. Version Sync
    try:
        version_path = ctx.repo_path("VERSION")
        pyproject_path = ctx.repo_path("pyproject.toml")
        if version_path.exists() and pyproject_path.exists():
            version = version_path.read_text(encoding="utf-8").strip()
            pyproject = pyproject_path.read_text(encoding="utf-8")
            if f'version = "{version}"' in pyproject or f"version = '{version}'" in pyproject:
                results.append(f"✅ **Version:** {version} (Synced with pyproject.toml)")
            else:
                results.append(f"⚠️ **Version:** {version} (Desync with pyproject.toml)")
        else:
            results.append("⚠️ **Version:** VERSION or pyproject.toml missing")
    except Exception as e:
        results.append(f"❌ **Version check failed:** {e}")

    lines.extend(results)
    return "\n".join(lines)


def codebase_health(ctx: ToolContext) -> str:
    """Compute and format codebase health report."""
    try:
        from khors.review import collect_sections, compute_complexity_metrics

        repo_dir = pathlib.Path(ctx.repo_dir)
        drive_root = pathlib.Path(os.environ.get("DRIVE_ROOT", os.path.join(os.getcwd(), "data")))

        sections, stats = collect_sections(repo_dir, drive_root)
        metrics = compute_complexity_metrics(sections)

        # Format report
        lines = []
        lines.append("## Codebase Health Report\n")
        lines.append(f"**Analyzed:** {stats['files']} files, {stats['chars']:,} chars")
        lines.append(f"**Files:** {metrics['total_files']} ({metrics['py_files']} Python)")
        lines.append(f"**Total lines:** {metrics['total_lines']:,}")
        lines.append(f"**Functions:** {metrics['total_functions']}")
        lines.append(f"**Avg function length:** {metrics['avg_function_length']} lines")
        lines.append(f"**Max function length:** {metrics['max_function_length']} lines")

        # Largest files
        if metrics.get("largest_files"):
            lines.append("\n### Largest Files")
            for path, size in metrics["largest_files"][:10]:
                marker = " ⚠️ OVERSIZED" if size > 1000 else ""
                lines.append(f"  {path}: {size} lines{marker}")

        # Longest functions
        if metrics.get("longest_functions"):
            lines.append("\n### Longest Functions")
            for path, start, length in metrics["longest_functions"][:10]:
                marker = " ⚠️ OVERSIZED" if length > 150 else ""
                lines.append(f"  {path}:{start}: {length} lines{marker}")

        # Warnings
        oversized_funcs = metrics.get("oversized_functions", [])
        oversized_mods = metrics.get("oversized_modules", [])

        if oversized_funcs or oversized_mods:
            lines.append("\n### ⚠️ Bible Violations (Principle 5: Minimalism)")
            if oversized_funcs:
                lines.append(f"  Functions > 150 lines: {len(oversized_funcs)}")
                for path, start, length in oversized_funcs:
                    lines.append(f"    - {path}:{start} ({length} lines)")
            if oversized_mods:
                lines.append(f"  Modules > 1000 lines: {len(oversized_mods)}")
                for path, size in oversized_mods:
                    lines.append(f"    - {path} ({size} lines)")
        else:
            lines.append("\n✅ No Bible violations detected (all functions < 150 lines, all modules < 1000 lines)")

        return "\n".join(lines)

    except Exception as e:
        log.warning("codebase_health failed: %s", e, exc_info=True)
        return f"⚠️ Failed to compute codebase health: {e}"


def get_tools():
    return [
        ToolEntry("codebase_health", {
            "name": "codebase_health",
            "description": "Get codebase complexity metrics: file sizes, longest functions, modules exceeding limits. Useful for self-assessment per Bible Principle 5 (Minimalism).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }, codebase_health),
        ToolEntry("system_pulse", {
            "name": "system_pulse",
            "description": "Check core system health: budget, git status, filesystem access, environment variables, and version synchronization. Use this to verify operational integrity.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }, system_pulse),
    ]
