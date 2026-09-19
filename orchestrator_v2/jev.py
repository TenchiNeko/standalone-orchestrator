from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class JevJudgment:
    mode: str
    question: str
    answer: dict[str, Any] | None
    model: str | None
    raw: dict[str, Any] | None = None


class JevAdapter:
    """Optional TypeSafe System One adapter; never authorizes mutations."""

    def __init__(self, mode: str = "off", timeout: float = 20):
        if mode not in {"off", "shadow", "advisory"}: raise ValueError("jev mode must be off, shadow, or advisory")
        self.mode = mode; self.timeout = timeout

    def classify_failure(self, excerpt: str) -> JevJudgment:
        q = "Classify this failure as one of: test_failure, tool_failure, timeout, unknown. Return unknown when evidence is insufficient."
        if self.mode == "off" or not os.environ.get("TYPESAFE_API_KEY"):
            return JevJudgment(self.mode, q, None, None)
        from typesafe_sdk import Choice, TypeSafeClient
        with TypeSafeClient(timeout=self.timeout, model=os.environ.get("JEV_MODEL", "jev-1.13.0")) as client:
            result = client.system_one(state={"failure_excerpt": excerpt[:5000]}, questions={"category": Choice(instructions=q, criteria={"test_failure": None, "tool_failure": None, "timeout": None, "unknown": None})})
        answer = result.choices["category"]
        payload = {"choice": answer.choice, "probabilities": answer.probabilities, "confidence": getattr(answer, "confidence", None)}
        return JevJudgment(self.mode, q, payload, getattr(result, "model", None), raw=getattr(result, "model_dump", lambda: None)())
