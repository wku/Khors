"""
System state monitoring for Khors.

Handles verification of git state, budget, versions, and other system invariants.
Extracted from agent.py to follow Single Responsibility Principle.
"""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for supervisor import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from khors.utils import read_text, get_git_info
from supervisor.state import load_state


logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitors system state and verifies invariants."""
    
    def __init__(self, repo_path: str, drive_path: str):
        self.repo_path = repo_path
        self.drive_path = drive_path
    
    def verify_system_state(self) -> Tuple[bool, List[str]]:
        """Verify all system invariants. Returns (is_healthy, warnings)."""
        warnings = []
        
        # Check git state
        git_warnings = self._check_uncommitted_changes()
        warnings.extend(git_warnings)
        
        # Check version sync
        version_warnings = self._check_version_sync()
        warnings.extend(version_warnings)
        
        # Check budget
        budget_warnings = self._check_budget()
        warnings.extend(budget_warnings)
        
        # System is healthy if no critical warnings
        is_healthy = len(warnings) == 0
        
        return is_healthy, warnings
    
    def _check_uncommitted_changes(self) -> List[str]:
        """Check for uncommitted changes in git."""
        warnings = []
        
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                warnings.append("UNCOMMITTED_CHANGES: Git repository has uncommitted changes")
                
        except Exception as e:
            warnings.append(f"GIT_CHECK_FAILED: {e}")
        
        return warnings
    
    def _check_version_sync(self) -> List[str]:
        """Check version synchronization across files."""
        warnings = []
        
        try:
            # Read VERSION file
            version_file = pathlib.Path(self.repo_path) / "VERSION"
            if not version_file.exists():
                warnings.append("VERSION_FILE_MISSING: VERSION file not found")
                return warnings
            
            version_content = read_text(version_file).strip()
            
            # Read pyproject.toml version
            pyproject_file = pathlib.Path(self.repo_path) / "pyproject.toml"
            if pyproject_file.exists():
                pyproject_content = read_text(pyproject_file)
                
                # Extract version from pyproject.toml
                for line in pyproject_content.split('\n'):
                    if line.strip().startswith('version = '):
                        pyproject_version = line.split('=')[1].strip().strip('"\'')
                        if pyproject_version != version_content:
                            warnings.append(f"VERSION_MISMATCH: VERSION={version_content}, pyproject.toml={pyproject_version}")
                        break
            
        except Exception as e:
            warnings.append(f"VERSION_CHECK_FAILED: {e}")
        
        return warnings
    
    def _check_budget(self) -> List[str]:
        """Check budget status and drift."""
        warnings = []
        
        try:
            state = load_state()
            
            # Check budget drift
            if state.get("budget_drift_alert", False):
                drift_pct = state.get("budget_drift_pct", 0)
                warnings.append(f"BUDGET_DRIFT: {drift_pct:.1f}% drift detected")
            
            # Check for high-cost tasks
            total_usd = state.get("openrouter_total_usd", 0)
            if total_usd > 5.0:  # Arbitrary threshold for "expensive" session
                warnings.append(f"HIGH_COST_SESSION: ${total_usd:.2f} spent")
                
        except Exception as e:
            warnings.append(f"BUDGET_CHECK_FAILED: {e}")
        
        return warnings
    
    def get_system_info(self) -> Dict[str, any]:
        """Get comprehensive system information."""
        info = {}
        
        # Git info
        try:
            git_info = get_git_info(self.repo_path)
            info["git"] = git_info
        except Exception as e:
            info["git"] = {"error": str(e)}
        
        # Budget info
        try:
            state = load_state()
            info["budget"] = {
                "total_spent": state.get("openrouter_total_usd", 0),
                "daily_spent": state.get("openrouter_daily_usd", 0),
                "drift_pct": state.get("budget_drift_pct"),
                "drift_alert": state.get("budget_drift_alert", False)
            }
        except Exception as e:
            info["budget"] = {"error": str(e)}
        
        # Version info
        try:
            version_file = pathlib.Path(self.repo_path) / "VERSION"
            if version_file.exists():
                info["version"] = read_text(version_file).strip()
            else:
                info["version"] = "unknown"
        except Exception as e:
            info["version"] = {"error": str(e)}
        
        # Environment info
        info["environment"] = {
            "repo_path": self.repo_path,
            "drive_path": self.drive_path,
            "openrouter_key_set": bool(os.getenv("OPENROUTER_API_KEY"))
        }
        
        return info