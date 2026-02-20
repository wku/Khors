"""Sequential thinking and meta-cognitive tools."""

import json
from typing import Any, Dict, List, Optional

from khors.tools.registry import ToolContext, ToolEntry

def _sequential_thinking(ctx: ToolContext, thought: str, thought_number: int, total_thoughts: int, next_thought_needed: bool, is_revision: bool = False, revises_thought: Optional[int] = None, branch_from_thought: Optional[int] = None, branch_id: Optional[str] = None, needs_more_thoughts: Optional[bool] = None) -> str:
    output = [
        f"THOUGHT {thought_number}/{total_thoughts}",
        f"Content: {thought}"
    ]
    
    if is_revision and revises_thought is not None:
        output.append(f"(Revision of thought {revises_thought})")
    
    if branch_from_thought is not None:
        output.append(f"(Branch from thought {branch_from_thought}, ID: {branch_id or 'unknown'})")
        
    if next_thought_needed:
        output.append("Status: Thinking process continues...")
    else:
        output.append("Status: Thinking process complete.")
        
    return "\n".join(output)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("sequential_thinking", {
            "name": "sequential_thinking",
            "description": "A meta-cognitive tool for complex problem solving. Allows breaking down problems into thoughts, revising them, and branching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "thought_number": {"type": "integer"},
                    "total_thoughts": {"type": "integer"},
                    "next_thought_needed": {"type": "boolean"},
                    "is_revision": {"type": "boolean", "default": False},
                    "revises_thought": {"type": "integer"},
                    "branch_from_thought": {"type": "integer"},
                    "branch_id": {"type": "string"},
                    "needs_more_thoughts": {"type": "boolean"}
                },
                "required": ["thought", "thought_number", "total_thoughts", "next_thought_needed"]
            }
        }, _sequential_thinking)
    ]
