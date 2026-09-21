"""Conservative cross-run loop memory backed by the v2 event store.

This is deliberately advisory.  A prior result is relevant only when the
contract, workspace identity, hypothesis, and relevant source state match.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .state import StateStore, Task, TaskContract, source_hash, task_scope


def _digest(value: Any, size: int = 16) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:size]


def contract_fingerprint(contract: TaskContract) -> str:
    return _digest({
        "goal": contract.goal.strip(),
        "permitted_files": sorted(contract.permitted_files),
        "permitted_actions": sorted(contract.permitted_actions),
        "criteria": [(c.key, c.description, c.required) for c in contract.criteria],
        "version": contract.version,
        "test_command": contract.test_command,
        "visual_required": contract.visual_required,
    })


def workspace_identity(root: Path) -> str:
    return _digest({"path": str(root.resolve())})


@dataclass(frozen=True)
class CrossRunFinding:
    kind: str
    relevant: bool
    prior_task_id: str | None = None
    prior_event_id: int | None = None
    state_changed: bool = False
    reason: str = ""


class LoopMemory:
    """Search prior factual attempts without permanently blacklisting work."""

    def __init__(self, store: StateStore):
        self.store = store

    def identity(self, task: Task, root: Path) -> dict[str, str]:
        return {
            "contract_fingerprint": contract_fingerprint(task.contract),
            "goal_fingerprint": _digest(task.contract.goal.strip()),
            "workspace_identity": workspace_identity(root),
            "state_fingerprint": source_hash(root, task_scope(task)),
        }

    def find(self, task: Task, root: Path, *, hypothesis: str, action: str) -> list[CrossRunFinding]:
        current = self.identity(task, root)
        out: list[CrossRunFinding] = []
        for row in self.store.events_all("cross_run_attempt"):
            payload = row["payload"]
            if payload.get("task_id") == task.task_id:
                continue
            if any(payload.get(k) != current[k] for k in ("contract_fingerprint", "goal_fingerprint", "workspace_identity")):
                continue
            same_state = payload.get("state_fingerprint") == current["state_fingerprint"]
            same_hypothesis = payload.get("hypothesis") == hypothesis
            same_action = payload.get("action") == action
            outcome = str(payload.get("outcome") or "").upper()
            if same_state and same_hypothesis and same_action:
                out.append(CrossRunFinding("SAME_TASK_SAME_STATE_REPEAT", True, payload.get("task_id"), row["id"], False, "same contract, workspace, state, hypothesis, and action"))
            elif same_state and same_hypothesis and outcome in {"REJECTED", "NO_PROGRESS", "STUCK_LOOP"}:
                out.append(CrossRunFinding("SAME_HYPOTHESIS_ALREADY_REJECTED", True, payload.get("task_id"), row["id"], False, "same hypothesis was rejected without state change"))
            elif same_state and outcome == "NO_PROGRESS":
                out.append(CrossRunFinding("PRIOR_NO_PROGRESS", True, payload.get("task_id"), row["id"], False, "prior attempt made no progress"))
            elif same_state and outcome == "STUCK_LOOP":
                out.append(CrossRunFinding("PRIOR_STUCK_LOOP", True, payload.get("task_id"), row["id"], False, "prior attempt entered a stuck loop"))
            elif same_hypothesis and not same_state:
                out.append(CrossRunFinding("STATE_CHANGED_SINCE_PRIOR_ATTEMPT", False, payload.get("task_id"), row["id"], True, "hypothesis is old but relevant source state changed"))
        return out

    def record_attempt(self, task: Task, root: Path, *, hypothesis: str, action: str, outcome: str, evidence: list[str] | None = None, files_changed: bool = False, tests_changed: bool = False, new_evidence: bool = False) -> None:
        payload = self.identity(task, root)
        payload.update({
            "task_id": task.task_id,
            "hypothesis": hypothesis,
            "action": action,
            "outcome": outcome,
            "evidence": list(evidence or []),
            "files_changed": bool(files_changed),
            "tests_changed": bool(tests_changed),
            "new_evidence": bool(new_evidence),
        })
        self.store.event(task.task_id, "cross_run_attempt", payload)

    @staticmethod
    def should_suppress_read_only(task: Task, findings: list[CrossRunFinding]) -> bool:
        return "write_file" not in task.contract.permitted_actions and "browser" not in task.contract.permitted_actions and any(
            f.relevant and f.kind in {"SAME_TASK_SAME_STATE_REPEAT", "SAME_HYPOTHESIS_ALREADY_REJECTED", "PRIOR_NO_PROGRESS", "PRIOR_STUCK_LOOP"}
            for f in findings
        )
