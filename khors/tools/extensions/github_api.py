import os
import requests
import logging
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

def _get_default_repo():
    # Пытаемся определить репозиторий из текущей папки через git
    try:
        import subprocess
        result = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True)
        if result.returncode == 0:
            url = result.stdout.strip()
            if url.startswith("git@github.com:"):
                return url.replace("git@github.com:", "").replace(".git", "")
            elif url.startswith("https://github.com/"):
                return url.replace("https://github.com/", "").replace(".git", "")
    except:
        pass
    return "wku/Khors" # Fallback

def list_github_issues(ctx: ToolContext, repo: Optional[str] = None, state: str = "open") -> str:
    """List issues in a GitHub repository."""
    headers = _get_headers()
    if not headers:
        return "⚠️ GITHUB_TOKEN not found in environment. Please add it to .env and restart."
    
    repo = repo or _get_default_repo()
    url = f"{GITHUB_API_URL}/repos/{repo}/issues"
    params = {"state": state}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        issues = response.json()
        
        if not issues:
            return f"No {state} issues found in {repo}."
        
        result = [f"Issues in {repo} ({state}):"]
        for issue in issues:
            # GitHub API returns both issues and pull requests, PRs have 'pull_request' key
            type_str = "[PR]" if "pull_request" in issue else "[Issue]"
            result.append(f"{type_str} #{issue['number']}: {issue['title']} (by {issue['user']['login']})")
        
        return "\n".join(result)
    except Exception as e:
        return f"⚠️ Error listing issues: {e}"

def create_github_issue(ctx: ToolContext, title: str, body: str, repo: Optional[str] = None) -> str:
    """Create a new issue in a GitHub repository."""
    headers = _get_headers()
    if not headers:
        return "⚠️ GITHUB_TOKEN not found in environment. Please add it to .env and restart."
    
    repo = repo or _get_default_repo()
    url = f"{GITHUB_API_URL}/repos/{repo}/issues"
    data = {"title": title, "body": body}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        issue = response.json()
        return f"OK: Created issue #{issue['number']} in {repo}: {issue['html_url']}"
    except Exception as e:
        return f"⚠️ Error creating issue: {e}"

def comment_on_issue(ctx: ToolContext, issue_number: int, body: str, repo: Optional[str] = None) -> str:
    """Add a comment to a GitHub issue or pull request."""
    headers = _get_headers()
    if not headers:
        return "⚠️ GITHUB_TOKEN not found in environment. Please add it to .env and restart."
    
    repo = repo or _get_default_repo()
    url = f"{GITHUB_API_URL}/repos/{repo}/issues/{issue_number}/comments"
    data = {"body": body}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        comment = response.json()
        return f"OK: Added comment to #{issue_number} in {repo}: {comment['html_url']}"
    except Exception as e:
        return f"⚠️ Error adding comment: {e}"

def close_github_issue(ctx: ToolContext, issue_number: int, repo: Optional[str] = None) -> str:
    """Close a GitHub issue or pull request."""
    headers = _get_headers()
    if not headers:
        return "⚠️ GITHUB_TOKEN not found in environment. Please add it to .env and restart."
    
    repo = repo or _get_default_repo()
    url = f"{GITHUB_API_URL}/repos/{repo}/issues/{issue_number}"
    data = {"state": "closed"}
    
    try:
        response = requests.patch(url, headers=headers, json=data)
        response.raise_for_status()
        return f"OK: Closed #{issue_number} in {repo}."
    except Exception as e:
        return f"⚠️ Error closing issue: {e}"

def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="list_github_issues",
            handler=list_github_issues,
            schema={
                "name": "list_github_issues",
                "description": "List issues and pull requests in a GitHub repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "description": "Repository name (e.g., 'owner/repo'). Defaults to current repo."},
                        "state": {"type": "string", "enum": ["open", "closed", "all"], "default": "open", "description": "State of issues to list."}
                    }
                }
            }
        ),
        ToolEntry(
            name="create_github_issue",
            handler=create_github_issue,
            schema={
                "name": "create_github_issue",
                "description": "Create a new issue in a GitHub repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Issue title."},
                        "body": {"type": "string", "description": "Issue body (markdown supported)."},
                        "repo": {"type": "string", "description": "Repository name (e.g., 'owner/repo'). Defaults to current repo."}
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
                "description": "Add a comment to a GitHub issue or pull request.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "issue_number": {"type": "integer", "description": "Issue or PR number."},
                        "body": {"type": "string", "description": "Comment body (markdown supported)."},
                        "repo": {"type": "string", "description": "Repository name (e.g., 'owner/repo'). Defaults to current repo."}
                    },
                    "required": ["issue_number", "body"]
                }
            }
        ),
        ToolEntry(
            name="close_github_issue",
            handler=close_github_issue,
            schema={
                "name": "close_github_issue",
                "description": "Close a GitHub issue or pull request.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "issue_number": {"type": "integer", "description": "Issue or PR number."},
                        "repo": {"type": "string", "description": "Repository name (e.g., 'owner/repo'). Defaults to current repo."}
                    },
                    "required": ["issue_number"]
                }
            }
        )
    ]
