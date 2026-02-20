"""
Archive Manager Tool.
Creates tar.gz archives of the codebase in /app/data/archive/ instead of git commits.
"""
import os
import tarfile
import pathlib
import datetime
from typing import List, Dict, Any
from khors.tools.registry import ToolContext, ToolEntry

def _archive_version(ctx: ToolContext, version: str, description: str) -> str:
    """Create a new version archive of the current codebase."""
    repo_dir = ctx.repo_dir
    archive_dir = ctx.drive_root / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"khors_v{version}_{ts}.tar.gz"
    filepath = archive_dir / filename
    
    # Create the archive
    try:
        with tarfile.open(filepath, "w:gz") as tar:
            tar.add(repo_dir, arcname=f"khors-v{version}", filter=_archive_filter)
        
        return f"OK: Archived version {version} to {filepath}\nDescription: {description}"
    except Exception as e:
        return f"ERROR: Failed to create archive: {e}"

def _archive_filter(tarinfo):
    """Filter out unwanted files from the archive."""
    exclude = {
        ".git", "__pycache__", ".venv", "env", "venv", 
        "node_modules", ".DS_Store", "data" # Exclude data dir if it's inside repo
    }
    # Check if any part of the path is in exclude list
    path_parts = pathlib.Path(tarinfo.name).parts
    for part in path_parts:
        if part in exclude:
            return None
    return tarinfo

def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("archive_version", {
            "name": "archive_version",
            "description": "Create a new version archive of the codebase. Use this INSTEAD of git commit.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "version": {"type": "string", "description": "Version number (e.g. 1.0.1)"},
                    "description": {"type": "string", "description": "Description of changes in this version"}
                }, 
                "required": ["version", "description"]
            },
        }, _archive_version)
    ]
