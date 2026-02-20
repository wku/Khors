"""Code analysis tools."""

import ast
import pathlib
import re
from typing import Any, Dict, List

from khors.tools.registry import ToolContext, ToolEntry


def _analyze_python(content: str) -> Dict[str, Any]:
    analysis = {'functions': [], 'classes': [], 'imports': []}
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                analysis['functions'].append(node.name)
            elif isinstance(node, ast.ClassDef):
                analysis['classes'].append(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        analysis['imports'].append(f"from {module} import {alias.name}")
    except Exception:
        pass
    return analysis


def _analyze_js(content: str) -> Dict[str, Any]:
    analysis = {'functions': [], 'classes': [], 'imports': []}
    lines = content.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith('import ') or (line.startswith('const ') and 'require(' in line):
            analysis['imports'].append(line)
        if 'class ' in line:
            match = re.search(r'class\s+(\w+)', line)
            if match:
                analysis['classes'].append(match.group(1))
        if 'function ' in line:
            match = re.search(r'function\s+(\w+)', line)
            if match:
                analysis['functions'].append(match.group(1))
        if 'const ' in line and '=>' in line:
            match = re.search(r'const\s+(\w+)\s*=', line)
            if match:
                analysis['functions'].append(match.group(1))
    return analysis


def _analyze_code(ctx: ToolContext, path: str) -> str:
    target = (ctx.repo_dir / path).resolve()
    if not target.exists():
        return f"⚠️ Error: File {path} does not exist."
    if not target.is_relative_to(ctx.repo_dir):
        return f"⚠️ Error: Access outside repo is forbidden."
    
    try:
        content = target.read_text(encoding='utf-8')
        ext = target.suffix.lower()
        
        analysis = {}
        if ext == '.py':
            analysis = _analyze_python(content)
        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            analysis = _analyze_js(content)
        else:
            return f"⚠️ Analysis not supported for extension {ext}. Supported: .py, .js, .ts"

        return (
            f"Code Analysis for {target.name}:\n"
            f"Functions: {len(analysis.get('functions', []))}\n"
            f"Classes: {len(analysis.get('classes', []))}\n"
            f"Imports: {len(analysis.get('imports', []))}\n\n"
            f"Details:\n"
            f"Classes: {', '.join(analysis.get('classes', []))}\n"
            f"Functions: {', '.join(analysis.get('functions', []))}\n"
            f"Imports: {', '.join(analysis.get('imports', []))}"
        )
    except Exception as e:
        return f"⚠️ Error analyzing code: {e}"


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("analyze_code", {
            "name": "analyze_code",
            "description": "Analyze code file structure (functions, classes, imports). Supports Python, JS/TS.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }, _analyze_code)
    ]
