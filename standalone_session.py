"""
Session management for the standalone orchestrator.

Handles persistence of task state, progress tracking, and session continuity.
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

from standalone_models import TaskState, ExecutionPhase

import logging
logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages session state persistence and progress tracking.

    Key files:
    - PROGRESS.md: Human-readable progress log
    - .agents/state.json: Machine-readable task state
    """

    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
        self.state_file = working_dir / ".agents" / "state.json"
        self.progress_file = working_dir / "PROGRESS.md"

        # Ensure directories exist
        for d in ["plans", "reports", "logs"]:
            (working_dir / ".agents" / d).mkdir(parents=True, exist_ok=True)

    def has_existing_session(self) -> bool:
        return self.state_file.exists()

    def save_state(self, state: TaskState):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(state.to_json())
        logger.debug(f"State saved: iteration={state.iteration}, phase={state.phase.value}")

    def load_state(self) -> TaskState:
        if not self.state_file.exists():
            raise FileNotFoundError(f"No state file found: {self.state_file}")
        state = TaskState.from_json(self.state_file.read_text())
        logger.debug(f"State loaded: iteration={state.iteration}, phase={state.phase.value}")
        return state

    def update_progress(self, state: TaskState, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.progress_file.exists():
            content = self.progress_file.read_text()
        else:
            content = f"# Progress Log\n\n**Task:** {state.goal}\n**Task ID:** {state.task_id}\n**Started:** {state.started_at}\n\n---\n\n"

        entry = f"## [{timestamp}] Iteration {state.iteration} — {state.phase.value.upper()}\n\n{message}\n\n---\n\n"

        parts = content.split("---\n\n", 1)
        if len(parts) == 2:
            content = parts[0] + "---\n\n" + entry + parts[1]
        else:
            content += entry

        self.progress_file.write_text(content)

    def mark_feature_complete(self, task_id: str):
        """No-op if feature list doesn't exist."""
        feature_file = self.working_dir / "feature_list.json"
        if not feature_file.exists():
            return
        try:
            data = json.loads(feature_file.read_text())
            for f in data.get("features", []):
                if f.get("assigned_task_id") == task_id:
                    f["passes"] = True
                    f["last_tested"] = datetime.now().isoformat()
            feature_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Could not update feature list: {e}")

    def get_git_status(self) -> str:
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.working_dir, capture_output=True, text=True, timeout=10
            )
            return result.stdout if result.returncode == 0 else "Not a git repository"
        except Exception as e:
            return f"Error: {e}"

    def get_recent_commits(self, count: int = 10) -> str:
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", f"-{count}"],
                cwd=self.working_dir, capture_output=True, text=True, timeout=10
            )
            return result.stdout if result.returncode == 0 else "No git history"
        except Exception as e:
            return f"Error: {e}"

    def build_session_context(self, state: TaskState) -> str:
        parts = [
            f"## Session Context",
            f"**Task ID:** {state.task_id}",
            f"**Goal:** {state.goal}",
            f"**Iteration:** {state.iteration}",
            f"**Phase:** {state.phase.value}",
            "",
            "## Git Status", "```", self.get_git_status(), "```", "",
            "## Recent Commits", "```", self.get_recent_commits(5), "```", "",
        ]

        if state.failure_history:
            parts.append("## Failure History")
            for failure in state.failure_history[-3:]:
                parts.append(f"- Iteration {failure.get('iteration')}: {failure.get('phase')} — {failure.get('error', '?')}")
            parts.append("")

        if state.current_plan:
            parts.append("## Current Plan")
            plan = state.current_plan[:1500]
            if len(state.current_plan) > 1500:
                plan += "\n... (truncated)"
            parts.append(plan)
            parts.append("")

        if state.dod:
            parts.append(state.dod.to_markdown())
            parts.append("")

        return "\n".join(parts)
