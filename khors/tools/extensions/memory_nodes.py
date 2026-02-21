"""
Khors — Memory Nodes Tools (Extension).
Graph-like long-term memory using Markdown files with YAML frontmatter.
"""

from __future__ import annotations

import os
import yaml
import pathlib
from typing import Any, Dict, List, Optional
from khors.tools.registry import ToolContext, ToolEntry
from khors.utils import read_text, write_text, utc_now_iso

def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="memory_node_write",
            schema={
                "name": "memory_node_write",
                "description": "Create or update a memory node (Markdown with YAML frontmatter).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Node name (alphanumeric, hyphens, underscores)."},
                        "content": {"type": "string", "description": "Markdown content of the node."},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "List of tags."},
                        "links": {"type": "array", "items": {"type": "string"}, "description": "List of related node names."}
                    },
                    "required": ["name", "content"]
                }
            },
            handler=memory_node_write
        ),
        ToolEntry(
            name="memory_node_read",
            schema={
                "name": "memory_node_read",
                "description": "Read a memory node's content and metadata.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Node name."}
                    },
                    "required": ["name"]
                }
            },
            handler=memory_node_read
        ),
        ToolEntry(
            name="memory_node_list",
            schema={
                "name": "memory_node_list",
                "description": "List all memory nodes with their tags and summaries.",
                "parameters": {"type": "object", "properties": {}}
            },
            handler=memory_node_list
        )
    ]

def _get_node_path(ctx: ToolContext, name: str) -> pathlib.Path:
    # Ensure name is safe
    safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")).lower()
    return ctx.drive_root / "memory" / "nodes" / f"{safe_name}.md"

def memory_node_write(ctx: ToolContext, name: str, content: str, tags: List[str] = None, links: List[str] = None) -> str:
    path = _get_node_path(ctx, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "name": name,
        "updated_at": utc_now_iso(),
        "tags": tags or [],
        "links": links or []
    }
    
    # If file exists, preserve created_at
    if path.exists():
        try:
            existing = read_text(path)
            if "---" in existing:
                parts = existing.split("---", 2)
                if len(parts) >= 3:
                    old_meta = yaml.safe_load(parts[1])
                    if "created_at" in old_meta:
                        metadata["created_at"] = old_meta["created_at"]
        except Exception:
            pass
            
    if "created_at" not in metadata:
        metadata["created_at"] = metadata["updated_at"]
        
    frontmatter = yaml.dump(metadata, sort_keys=False, allow_unicode=True)
    full_content = f"---\n{frontmatter}---\n\n{content}"
    
    write_text(path, full_content)
    return f"✅ Memory node '{name}' saved to {path.name}"

def memory_node_read(ctx: ToolContext, name: str) -> str:
    path = _get_node_path(ctx, name)
    if not path.exists():
        return f"❌ Memory node '{name}' not found."
    return read_text(path)

def memory_node_list(ctx: ToolContext) -> str:
    nodes_dir = ctx.drive_root / "memory" / "nodes"
    if not nodes_dir.exists():
        return "Memory nodes directory is empty."
    
    files = list(nodes_dir.glob("*.md"))
    if not files:
        return "No memory nodes found."
    
    lines = [f"Found {len(files)} memory nodes:"]
    for f in files:
        try:
            content = read_text(f)
            if "---" in content:
                parts = content.split("---", 2)
                meta = yaml.safe_load(parts[1])
                tags = ", ".join(meta.get("tags", []))
                links = ", ".join(meta.get("links", []))
                lines.append(f"- **{meta.get('name', f.stem)}**: tags=[{tags}], links=[{links}]")
            else:
                lines.append(f"- {f.stem} (no metadata)")
        except Exception as e:
            lines.append(f"- {f.stem} (error reading: {e})")
            
    return "\n".join(lines)
