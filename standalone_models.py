"""
Data models for the standalone orchestrator.

Zero external dependencies beyond Python stdlib.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json


class ExecutionPhase(Enum):
    INITIALIZING = "initializing"
    EXPLORE = "explore"
    PLAN = "plan"
    BUILD = "build"
    TEST = "test"
    COMPLETE = "complete"
    ESCALATED = "escalated"
    FAILED = "failed"


@dataclass
class DoDCriterion:
    """A single Definition of Done criterion."""
    id: str
    description: str
    verification_method: str = "test"
    verification_command: Optional[str] = None
    passed: bool = False
    evidence: Optional[str] = None


@dataclass
class DoD:
    """Definition of Done — collection of success criteria."""
    criteria: List[DoDCriterion] = field(default_factory=list)

    def add(self, description: str, method: str = "test", command: str = None) -> str:
        cid = f"criterion-{len(self.criteria)}"
        self.criteria.append(DoDCriterion(
            id=cid, description=description,
            verification_method=method, verification_command=command
        ))
        return cid

    def mark_passed(self, criterion_id: str, evidence: str = None):
        for c in self.criteria:
            if c.id == criterion_id:
                c.passed = True
                c.evidence = evidence
                return
        raise ValueError(f"Unknown criterion: {criterion_id}")

    def all_passed(self) -> bool:
        return all(c.passed for c in self.criteria)

    def failed_criteria(self) -> List[DoDCriterion]:
        return [c for c in self.criteria if not c.passed]

    def to_markdown(self) -> str:
        lines = ["### Definition of Done", ""]
        for c in self.criteria:
            status = "✅" if c.passed else "❌"
            lines.append(f"- [{status}] {c.description}")
            if c.evidence:
                lines.append(f"  Evidence: {c.evidence}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "criteria": [
                {
                    "id": c.id, "description": c.description,
                    "verification_method": c.verification_method,
                    "verification_command": c.verification_command,
                    "passed": c.passed, "evidence": c.evidence
                }
                for c in self.criteria
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DoD":
        dod = cls()
        for c_data in data.get("criteria", []):
            dod.criteria.append(DoDCriterion(**c_data))
        return dod


@dataclass
class IterationResult:
    success: bool
    phase: ExecutionPhase
    error: Optional[str] = None
    dod_results: Optional[Dict[str, bool]] = None
    unrecoverable: bool = False
    rca: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success, "phase": self.phase.value,
            "error": self.error, "dod_results": self.dod_results,
            "unrecoverable": self.unrecoverable, "rca": self.rca
        }


@dataclass
class AgentResult:
    """Result from running a single agent."""
    success: bool
    output: str
    error: Optional[str] = None
    exit_code: int = 0
    duration_seconds: float = 0.0
    tokens_used: Optional[int] = None
    dod: Optional[DoD] = None
    dod_results: Optional[Dict[str, bool]] = None
    plan: Optional[str] = None
    exploration_summary: Optional[str] = None
    test_report: Optional[dict] = None


@dataclass
class TaskState:
    task_id: str
    goal: str
    iteration: int = 1
    phase: ExecutionPhase = ExecutionPhase.INITIALIZING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    exploration_context: Optional[str] = None
    needs_re_exploration: bool = False
    current_plan: Optional[str] = None
    dod: Optional[DoD] = None
    failure_history: List[Dict[str, Any]] = field(default_factory=list)
    escalation_reason: Optional[str] = None

    def add_failure(self, result: IterationResult):
        self.failure_history.append({
            "iteration": self.iteration,
            "phase": result.phase.value,
            "error": result.error,
            "dod_results": result.dod_results,
            "rca": result.rca,
        })

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "goal": self.goal,
            "iteration": self.iteration, "phase": self.phase.value,
            "started_at": self.started_at, "completed_at": self.completed_at,
            "exploration_context": self.exploration_context,
            "needs_re_exploration": self.needs_re_exploration,
            "current_plan": self.current_plan,
            "dod": self.dod.to_dict() if self.dod else None,
            "failure_history": self.failure_history,
            "escalation_reason": self.escalation_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskState":
        state = cls(
            task_id=data["task_id"], goal=data["goal"],
            iteration=data.get("iteration", 1),
            phase=ExecutionPhase(data.get("phase", "initializing")),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            exploration_context=data.get("exploration_context"),
            needs_re_exploration=data.get("needs_re_exploration", False),
            current_plan=data.get("current_plan"),
            failure_history=data.get("failure_history", []),
            escalation_reason=data.get("escalation_reason"),
        )
        if data.get("dod"):
            state.dod = DoD.from_dict(data["dod"])
        return state

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TaskState":
        return cls.from_dict(json.loads(json_str))
