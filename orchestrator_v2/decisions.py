from __future__ import annotations

import importlib.util
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Decision:
    question_id: str
    allowed_candidates: list[str]
    selected: str | None
    distribution: dict[str, float]
    model: str
    latency: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionBatch:
    provider: str
    decisions: list[Decision]
    latency: float
    metadata: dict[str, Any] = field(default_factory=dict)


class DecisionProvider:
    name = "off"

    def decide(self, state: str, questions: dict[str, dict[str, Any]]) -> DecisionBatch:
        raise NotImplementedError


class DeterministicDecisionProvider(DecisionProvider):
    name = "off"

    def decide(self, state: str, questions: dict[str, dict[str, Any]]) -> DecisionBatch:
        return DecisionBatch(self.name, [], 0.0, {"authority": "none"})


class MockDecisionProvider(DecisionProvider):
    name = "mock"

    def __init__(self, selected: str | None = None):
        self.selected = selected

    def decide(self, state: str, questions: dict[str, dict[str, Any]]) -> DecisionBatch:
        start = time.perf_counter(); out = []
        for qid, question in questions.items():
            candidates = list(question["criteria"])
            selected = self.selected if self.selected in candidates else candidates[0]
            distribution = {candidate: (1.0 if candidate == selected else 0.0) for candidate in candidates}
            out.append(Decision(qid, candidates, selected, distribution, "mock", 0.0, {"authority": "none"}))
        return DecisionBatch(self.name, out, time.perf_counter() - start, {"authority": "none"})


class NanoJevProvider(DecisionProvider):
    name = "nanojev"

    def __init__(self, checkpoint: Path, upstream_predictor: Path | None = None, threads: int = 4, max_length: int = 256):
        self.checkpoint = Path(checkpoint).resolve()
        self.upstream_predictor = (upstream_predictor or Path(__file__).resolve().parents[1] / "vendor" / "NanoJev" / "scripts" / "predict_toy_decisions.py").resolve()
        self.threads = threads
        self.max_length = max_length
        self._engine = None
        self._module = None
        self.calls = 0

    def _load(self):
        if self._engine is not None:
            return
        if not self.upstream_predictor.is_file():
            raise FileNotFoundError(f"NanoJev predictor not found: {self.upstream_predictor}")
        spec = importlib.util.spec_from_file_location("standalone_orchestrator_v2_nanojev_predictor", self.upstream_predictor)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load the pinned NanoJev predictor")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        import torch
        torch.set_num_threads(max(1, int(self.threads)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self._module = module
        self._engine = module.DecisionPredictor(str(self.checkpoint), max_length=self.max_length, device_name="cpu", precision="fp32")

    def decide(self, state: str, questions: dict[str, dict[str, Any]]) -> DecisionBatch:
        self._load()
        payload = {"states": [{"id": "orchestrator", "state": state if isinstance(state, str) else json.dumps(state, sort_keys=True), "questions": questions}]}
        start = time.perf_counter(); result = self._engine.predict(payload); elapsed = time.perf_counter() - start; self.calls += 1
        answers = result["states"][0]["answers"]
        decisions = []
        for qid, question in questions.items():
            answer = answers[qid]; distribution = {str(k): float(v) for k, v in answer["probabilities"].items()}
            decisions.append(Decision(qid, list(question["criteria"]), answer.get("value"), distribution, "C-Tianyu/NanoJev@4a19595eada0857133c0d2be024f879a4077054b", elapsed, {"execution": result["execution"], "authority": "shadow-only"}))
        return DecisionBatch(self.name, decisions, elapsed, {"authority": "shadow-only", "calls": self.calls, "execution": result["execution"]})


def make_decision_provider(name: str, checkpoint: Path | None = None) -> DecisionProvider:
    if name == "off":
        return DeterministicDecisionProvider()
    if name == "mock":
        return MockDecisionProvider()
    if name == "nanojev":
        if checkpoint is None:
            checkpoint = Path(os.environ.get("NANOJEV_CHECKPOINT", "/home/brandon/models/nanojev"))
        return NanoJevProvider(checkpoint)
    raise ValueError(f"unknown decision provider: {name}")
