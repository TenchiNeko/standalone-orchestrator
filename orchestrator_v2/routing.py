"""Future model-routing labels; shadow-only until measured on real tasks."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .decisions import DecisionProvider, DeterministicDecisionProvider

ROUTINE_QWEN = "ROUTINE_QWEN"
HARD_QWEN = "HARD_QWEN"
ESCALATE_STRONG_MODEL = "ESCALATE_STRONG_MODEL"
DETERMINISTIC_ONLY = "DETERMINISTIC_ONLY"
ROUTES = (ROUTINE_QWEN, HARD_QWEN, ESCALATE_STRONG_MODEL, DETERMINISTIC_ONLY)


@dataclass
class RouteJudgment:
    selected: str
    distribution: dict[str, float] = field(default_factory=dict)
    provider: str = "off"
    shadow: bool = True
    reason: str = ""


class ModelRouter:
    """Record routing suggestions while deterministic policy remains in charge."""

    def __init__(self, provider: DecisionProvider | None = None, mode: str = "shadow", default: str = ROUTINE_QWEN):
        if mode not in {"shadow", "advisory"}:
            raise ValueError("routing mode must be shadow or advisory")
        if default not in ROUTES:
            raise ValueError("unknown default route")
        self.provider = provider or DeterministicDecisionProvider()
        self.mode = mode
        self.default = default

    def classify(self, goal: str, facts: dict[str, Any]) -> RouteJudgment:
        criteria = {
            ROUTINE_QWEN: "Use local Qwen for bounded routine work with clear evidence.",
            HARD_QWEN: "Use local Qwen with its full reasoning/tool budget for difficult work.",
            ESCALATE_STRONG_MODEL: "A stronger model or human review is needed; do not silently continue.",
            DETERMINISTIC_ONLY: "Use code and tests only; no generative model is needed.",
        }
        question = {"route": {"type": "choice", "instructions": "Suggest a route; this suggestion cannot change scope, safety, tests, or completion.", "criteria": criteria}}
        try:
            batch = self.provider.decide(json.dumps({"goal": goal, "facts": facts}, sort_keys=True), question)
            if batch.decisions:
                d = batch.decisions[0]
                suggestion = d.selected if d.selected in ROUTES else self.default
                return RouteJudgment(suggestion, d.distribution, batch.provider, True, "shadow-only provider suggestion")
        except Exception:
            pass
        return RouteJudgment(self.default, {self.default: 1.0}, "off", True, "deterministic default")

    def effective_route(self, judgment: RouteJudgment) -> str:
        """Routing authority is intentionally deterministic until evaluation promotes it."""
        return self.default
