"""File tools: read, write, edit, delete, move, list, file_info, search."""

import os
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from khors.tools.registry import ToolContext, ToolEntry
from khors.utils import safe_relpath

log = logging.getLogger(__name__)

def _assert_write_permission(ctx: ToolContext, path: Path) -> None:
    """Ensure bots cannot overwrite core engine files."""
    try:
        rel = path.relative_to(ctx.repo_dir)
        parts = rel.parts
        
        if not parts:
            return
            
        # Core is khors/ except khors/tools/extensions/
        is_khors_core = (
            parts[0] == "khors" and 
            not (len(parts) >= 3 and parts[1] == "tools" and parts[2] == "extensions")
        )
        
        # supervisor/ and launcher.py are also core
        is_supervisor = parts[0] == "supervisor"
        is_launcher = parts[0] == "launcher.py"
        
        if is_khors_core or is_supervisor or is_launcher:
            raise PermissionError(
                f"Access denied: modifying core file '{rel}' is disabled for safety. "
                "Use 'khors/tools/extensions/' for new tools."
            )
    except ValueError:
        # Not relative to repo_dir (e.g., drive_root or absolute path outside). Let it pass.
        pass

def _resolve_safe_path(ctx: ToolContext, path: str, require_write: bool = False) -> Optional[Path]:
    """Resolve path prioritizing repo_dir, fallback to drive_root."""
    repo_target = (ctx.repo_dir / safe_relpath(path)).resolve()
    
    if require_write:
        _assert_write_permission(ctx, repo_target)
        
    if repo_target.exists():
        return repo_target
    drive_target = (ctx.drive_root / safe_relpath(path)).resolve()
    if drive_target.exists():
        if require_write:
             _assert_write_permission(ctx, drive_target)
        return drive_target
    
    # If it doesn't exist, default to repo_dir for writing
    return repo_target

def _read_file(ctx: ToolContext, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    target = _resolve_safe_path(ctx, path)
    if not target.exists():
        return f"⚠️ Error: File {path} does not exist."
    try:
        content = target.read_text(encoding="utf-8")
        if start_line is None and end_line is None:
            return f"### File: {path}\n\n<file_content path=\"{path}\">\n{content}\n</file_content>"
        
        lines = content.splitlines()
        start = (start_line - 1) if start_line else 0
        end = end_line if end_line else len(lines)
        selected_lines = lines[start:end]
        joined = "\n".join(selected_lines)
        return f"### File: {path} (lines {start_line or 1}-{end_line or len(lines)})\n\n<file_content path=\"{path}\">\n{joined}\n</file_content>"
    except Exception as e:
        return f"⚠️ Error reading file: {e}"

def _write_file(ctx: ToolContext, path: str, content: str) -> str:
    try:
        target = _resolve_safe_path(ctx, path, require_write=True)
    except PermissionError as pe:
        return str(pe)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"OK: Successfully wrote to {path}"
    except Exception as e:
        return f"⚠️ Error writing file: {e}"

def _edit_file(ctx: ToolContext, path: str, old_text: str, new_text: str) -> str:
    try:
        target = _resolve_safe_path(ctx, path, require_write=True)
    except PermissionError as pe:
        return str(pe)
    if not target.exists():
        return f"⚠️ Error: File {path} does not exist."
    try:
        content = target.read_text(encoding="utf-8")
        if old_text not in content:
            return f"⚠️ Error: old_text not found in {path}"
        new_content = content.replace(old_text, new_text)
        target.write_text(new_content, encoding="utf-8")
        return f"OK: Successfully edited {path}"
    except Exception as e:
        return f"⚠️ Error editing file: {e}"

def _edit_file_by_lines(ctx: ToolContext, path: str, start_line: int, end_line: int, new_content: str) -> str:
    try:
        target = _resolve_safe_path(ctx, path, require_write=True)
    except PermissionError as pe:
        return str(pe)
    if not target.exists():
        return f"⚠️ Error: File {path} does not exist."
    try:
        content = target.read_text(encoding="utf-8")
        lines = content.splitlines()
        total_lines = len(lines)
        if start_line < 1 or end_line > total_lines or start_line > end_line:
            return f"⚠️ Error: Invalid line range {start_line}-{end_line} (File has {total_lines} lines)"
        
        new_lines_content = new_content.splitlines()
        result_lines = lines[:start_line-1] + new_lines_content + lines[end_line:]
        target.write_text("\n".join(result_lines), encoding="utf-8")
        return f"OK: Successfully edited lines {start_line}-{end_line} in {path}"
    except Exception as e:
        return f"⚠️ Error editing lines: {e}"

def _multi_edit_file(ctx: ToolContext, path: str, edits: List[Dict[str, str]]) -> str:
    try:
        target = _resolve_safe_path(ctx, path, require_write=True)
    except PermissionError as pe:
        return str(pe)
    if not target.exists():
        return f"⚠️ Error: File {path} does not exist."
    try:
        content = target.read_text(encoding="utf-8")
        original_content = content
        
        for i, edit in enumerate(edits):
            old_t = edit.get("old_text")
            new_t = edit.get("new_text")
            if old_t not in content:
                return f"⚠️ Error: old_text '{old_t}' from edit index {i} not found (aborted all edits)"
            content = content.replace(old_t, new_t)
            
        if content == original_content:
            return "⚠️ Warning: No changes made"
            
        target.write_text(content, encoding="utf-8")
        return f"OK: Successfully applied {len(edits)} edits to {path}"
    except Exception as e:
        return f"⚠️ Error applying multi-edits: {e}"

def _delete_file(ctx: ToolContext, path: str, recursive: bool = False) -> str:
    try:
        target = _resolve_safe_path(ctx, path, require_write=True)
    except PermissionError as pe:
        return str(pe)
    if not target.exists():
        return f"⚠️ Error: Path {path} does not exist."
    try:
        if target.is_file():
            target.unlink()
            return f"OK: Successfully deleted file: {path}"
        elif target.is_dir():
            if recursive:
                shutil.rmtree(target)
                return f"OK: Successfully deleted directory (recursive): {path}"
            else:
                target.rmdir()
                return f"OK: Successfully deleted directory: {path}"
        return "⚠️ Error: Unknown path type"
    except Exception as e:
        return f"⚠️ Error deleting: {e}"

def _move_file(ctx: ToolContext, source: str, destination: str) -> str:
    try:
        src = _resolve_safe_path(ctx, source, require_write=True)
        dst = _resolve_safe_path(ctx, destination, require_write=True)
    except PermissionError as pe:
        return str(pe)
        
    if not src.exists():
        return f"⚠️ Error: Source {source} does not exist."
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"OK: Successfully moved {source} to {destination}"
    except Exception as e:
        return f"⚠️ Error moving file: {e}"

def _list_directory(ctx: ToolContext, directory: str = ".", recursive: bool = False) -> str:
    root = (ctx.repo_dir / safe_relpath(directory)).resolve()
    if not root.exists():
        return f"⚠️ Error: Directory {directory} does not exist."
        
    IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "build", "dist"}
    try:
        entries = []
        if recursive:
            count = 0
            for path in root.rglob("*"):
                try:
                    rel_path = path.relative_to(root)
                    if any(part in IGNORED_DIRS for part in rel_path.parts):
                        continue
                except ValueError:
                    continue 

                prefix = "📁 " if path.is_dir() else "📄 "
                indent = "    " * (len(rel_path.parts) - 1)
                entries.append(f"{indent}{prefix}{path.name}")
                
                count += 1
                if count > 1000:
                    entries.append("... (truncated: too many files)")
                    break
        else:
            for path in sorted(root.iterdir()):
                if path.name in IGNORED_DIRS:
                    continue
                prefix = "📁 " if path.is_dir() else "📄 "
                entries.append(f"{prefix}{path.name}")
        
        return "\n".join(entries) if entries else "(empty directory)"
    except Exception as e:
        return f"⚠️ Error listing directory: {e}"

def _get_file_info(ctx: ToolContext, path: str) -> str:
    target = _resolve_safe_path(ctx, path)
    if not target.exists():
        return f"⚠️ Error: Path {path} does not exist."
    try:
        stats = target.stat()
        type_str = "Directory" if target.is_dir() else "File"
        size_str = f"{stats.st_size} bytes"
        return f"Path: {target}\nType: {type_str}\nSize: {size_str}\nPermissions: {oct(stats.st_mode)[-3:]}"
    except Exception as e:
        return f"⚠️ Error getting info: {e}"

def _search_files(ctx: ToolContext, directory: str, pattern: str, recursive: bool = True) -> str:
    root = (ctx.repo_dir / safe_relpath(directory)).resolve()
    if not root.exists():
        return f"⚠️ Error: Directory {directory} does not exist."
    try:
        results = []
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        for p in iterator:
            results.append(str(p.relative_to(ctx.repo_dir)))
        if not results:
            return "No files found."
        return "\n".join(results[:50]) + ("\n... (truncated)" if len(results) > 50 else "")
    except Exception as e:
        return f"⚠️ Error searching files: {e}"

def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("read_file", {
            "name": "read_file",
            "description": "Read content from a file. Can read specific lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"}
                },
                "required": ["path"]
            }
        }, _read_file),
        ToolEntry("write_file", {
            "name": "write_file",
            "description": "Write or overwrite content strictly to a file. Creates directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }, _write_file),
        ToolEntry("edit_file", {
            "name": "edit_file",
            "description": "Edit a file by replacing precise text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"}
                },
                "required": ["path", "old_text", "new_text"]
            }
        }, _edit_file, is_code_tool=True),
        ToolEntry("edit_file_by_lines", {
            "name": "edit_file_by_lines",
            "description": "Replace a range of lines in a file with new content. start_line and end_line are inclusive limits. For example, to replace line 4 with something else: start_line=4, end_line=4.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "new_content": {"type": "string"}
                },
                "required": ["path", "start_line", "end_line", "new_content"]
            }
        }, _edit_file_by_lines, is_code_tool=True),
        ToolEntry("multi_edit_file", {
            "name": "multi_edit_file",
            "description": "Perform multiple text replacements in a single file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_text": {"type": "string"},
                                "new_text": {"type": "string"}
                            },
                            "required": ["old_text", "new_text"]
                        }
                    }
                },
                "required": ["path", "edits"]
            }
        }, _multi_edit_file, is_code_tool=True),
        ToolEntry("delete_file", {
            "name": "delete_file",
            "description": "Delete a file or directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean", "default": False}
                },
                "required": ["path"]
            }
        }, _delete_file),
        ToolEntry("move_file", {
            "name": "move_file",
            "description": "Move or rename a file or directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "destination": {"type": "string"}
                },
                "required": ["source", "destination"]
            }
        }, _move_file),
        ToolEntry("list_directory", {
            "name": "list_directory",
            "description": "List files and directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "default": "."},
                    "recursive": {"type": "boolean", "default": False}
                },
                "required": []
            }
        }, _list_directory),
        ToolEntry("get_file_info", {
            "name": "get_file_info",
            "description": "Get fundamental information about a file or directory (size, type, permissions).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }, _get_file_info),
        ToolEntry("search_files", {
            "name": "search_files",
            "description": "Search for files strictly matching a glob pattern (e.g. *.py) in the name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "default": "."},
                    "pattern": {"type": "string"},
                    "recursive": {"type": "boolean", "default": True}
                },
                "required": ["directory", "pattern"]
            }
        }, _search_files)
    ]
