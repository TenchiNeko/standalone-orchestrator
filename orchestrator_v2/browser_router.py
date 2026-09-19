"""Candidate-only browser routing; execution remains deterministic."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .decisions import DecisionProvider, DeterministicDecisionProvider


@dataclass(frozen=True)
class BrowserCandidate:
    candidate_id: str
    kind: str
    label: str
    operation: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrowserRoute:
    selected: str | None
    distribution: dict[str, float]
    executed: bool
    verification: dict[str, Any] | None = None
    reason: str = ""


class BrowserCandidateRouter:
    """The provider chooses among supplied candidates, never arbitrary code."""

    def __init__(self, provider: DecisionProvider | None = None, mode: str = "shadow"):
        if mode not in {"shadow", "active"}:
            raise ValueError("browser routing mode must be shadow or active")
        self.provider = provider or DeterministicDecisionProvider()
        self.mode = mode

    def route(self, goal: str, browser_state: dict[str, Any], candidates: list[BrowserCandidate], execute: Callable[[BrowserCandidate], dict[str, Any]], verify: Callable[[], dict[str, Any]]) -> BrowserRoute:
        if not candidates:
            return BrowserRoute(None, {}, False, reason="no compatible candidate control")
        criteria = {c.candidate_id: f"{c.kind} {c.label}; permitted operation is {c.operation}" for c in candidates}
        question = {"control": {"type": "choice", "instructions": "Select one supplied browser control for the bounded operation. Do not invent selectors, JavaScript, coordinates, or shell commands.", "criteria": criteria}}
        selected = None; distribution: dict[str, float] = {}
        try:
            batch = self.provider.decide(json.dumps({"goal": goal, "browser": browser_state, "candidates": criteria}, sort_keys=True), question)
            if batch.decisions:
                selected = batch.decisions[0].selected if batch.decisions[0].selected in criteria else None
                distribution = batch.decisions[0].distribution
        except Exception:
            pass
        if self.mode != "active" or selected is None:
            return BrowserRoute(selected, distribution, False, reason="shadow-only or no valid selection")
        chosen = next(c for c in candidates if c.candidate_id == selected)
        result = execute(chosen)
        verification = verify()
        return BrowserRoute(selected, distribution, True, verification, reason="deterministic execution and post-action verification")
