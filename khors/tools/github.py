"""GitHub tools (REST API version)."""

from __future__ import annotations

import os
import requests
import logging
import subprocess
from typing import List, Dict, Any, Optional

from khors.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)

GITHUB_API_URL = "https://api.github.com"

def _get_headers():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return None
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Khors-AI-Agent"
    }

def _get_default_repo(ctx: ToolContext) -> str:
    """Tries to detect repository from git config, fallback to wku/Khors."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(ctx.repo_dir),
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            if url.startswith("git@github.com:"):
                return url.replace("git@github.com:", "").replace(".git", "")
            elif url.startswith("https://github.com/"):
                return url.replace("https://github.com/", "").replace(".git", "")
    except Exception:
        pass
    return "wku/Khors"

def list_github_issues(ctx: ToolContext, repo: Optional[str] = None, state: str = "open") -> str:
    """List issues and pull requests in a GitHub repository."""
    headers = _get_headers()
    if not headers:
        return "⚠️ GITHUB_TOKEN not found in environment."
    
    repo = repo or _get_default_repo(ctx)
    url = f"{GITHUB_API_URL}/repos/{repo}/issues"
    params = {"state": state}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        issues = response.json()
        
        if not issues:
            return f"No {state} issues found in {repo}."
        
        result = [f"Issues in {repo} ({state}):"]
        for issue in issues:
            type_str = "[PR]" if "pull_request" in issue else "[Issue]"
            result.append(f"{type_str} #{issue['number']}: {issue['title']} (by {issue['user']['login']})")
        
        return "\n".join(result)
    except Exception as e:
        return f"⚠️ Error listing issues: {e}"

def get_github_issue(ctx: ToolContext, number: int, repo: Optional[str] = None) -> str:
    """Get full details of a GitHub issue including body and comments."""
    headers = _get_headers()
    if not headers:
        return "⚠️ GITHUB_TOKEN not found."
    
    repo = repo or _get_default_repo(ctx)
    url = f"{GITHUB_API_URL}/repos/{repo}/issues/{number}"
    
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        issue = response.json()
        
        result = [
            f"## Issue #{issue['number']}: {issue['title']}",
            f"**State:** {issue['state']}  |  **Author:** @{issue['user']['login']}",
            f"**Created at:** {issue['created_at']}",
            "\n**Body:**\n",
            issue.get('body', 'No body content.'),
            "\n**Comments:**"
        ]
        
        comments_url = issue['comments_url']
        comments_resp = requests.get(comments_url, headers=headers, timeout=20)
        if comments_resp.status_code == 200:
            comments = comments_resp.json()
            for comment in comments:
                result.append(f"\n@{comment['user']['login']} at {comment['created_at']}:\n{comment['body']}")
        
        return "\n".join(result)
    except Exception as e:
        return f"⚠️ Error getting issue: {e}"

def create_github_issue(ctx: ToolContext, title: str, body: str, repo: Optional[str] = None) -> str:
    """Create a new issue in a GitHub repository."""
    headers = _get_headers()
    if not headers:
        return "⚠️ GITHUB_TOKEN not found."
    
    repo = repo or _get_default_repo(ctx)
    url = f"{GITHUB_API_URL}/repos/{repo}/issues"
    data = {"title": title, "body": body}
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=20)
        response.raise_for_status()
        issue = response.json()
        return f"OK: Created issue #{issue['number']} in {repo}: {issue['html_url']}"
    except Exception as e:
        return f"⚠️ Error creating issue: {e}"

def comment_on_issue(ctx: ToolContext, number: int, body: str, repo: Optional[str] = None) -> str:
    """Add a comment to a GitHub issue or pull request."""
    headers = _get_headers()
    if not headers:
        return "⚠️ GITHUB_TOKEN not found."
    
    repo = repo or _get_default_repo(ctx)
    url = f"{GITHUB_API_URL}/repos/{repo}/issues/{number}/comments"
    data = {"body": body}
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=20)
        response.raise_for_status()
        comment = response.json()
        return f"OK: Added comment to #{number} in {repo}: {comment['html_url']}"
    except Exception as e:
        return f"⚠️ Error adding comment: {e}"

def close_github_issue(ctx: ToolContext, number: int, repo: Optional[str] = None) -> str:
    """Close a GitHub issue or pull request."""
    headers = _get_headers()
    if not headers:
        return "⚠️ GITHUB_TOKEN not found."
    
    repo = repo or _get_default_repo(ctx)
    url = f"{GITHUB_API_URL}/repos/{repo}/issues/{number}"
    data = {"state": "closed"}
    
    try:
        response = requests.patch(url, headers=headers, json=data, timeout=20)
        response.raise_for_status()
        return f"OK: Closed #{number} in {repo}."
    except Exception as e:
        return f"⚠️ Error closing issue: {e}"

def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="list_github_issues",
            handler=list_github_issues,
            schema={
                "name": "list_github_issues",
                "description": "List issues and pull requests in a GitHub repository using REST API.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "description": "Repository name (owner/repo)."},
                        "state": {"type": "string", "enum": ["open", "closed", "all"], "default": "open"}
                    }
                }
            }
        ),
        ToolEntry(
            name="get_github_issue",
            handler=get_github_issue,
            schema={
                "name": "get_github_issue",
                "description": "Get details of a GitHub issue/PR.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "integer", "description": "Issue number."},
                        "repo": {"type": "string", "description": "Repository name."}
                    },
                    "required": ["number"]
                }
            }
        ),
        ToolEntry(
            name="create_github_issue",
            handler=create_github_issue,
            schema={
                "name": "create_github_issue",
                "description": "Create a GitHub issue.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "repo": {"type": "string"}
                    },
                    "required": ["title", "body"]
                }
            }
        ),
        ToolEntry(
            name="comment_on_issue",
            handler=comment_on_issue,
            schema={
                "name": "comment_on_issue",
                "description": "Comment on GitHub issue/PR.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "integer"},
                        "body": {"type": "string"},
                        "repo": {"type": "string"}
                    },
                    "required": ["number", "body"]
                }
            }
        ),
        ToolEntry(
            name="close_github_issue",
            handler=close_github_issue,
            schema={
                "name": "close_github_issue",
                "description": "Close a GitHub issue/PR.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "integer"},
                        "repo": {"type": "string"}
                    },
                    "required": ["number"]
                }
            }
        )
    ]
