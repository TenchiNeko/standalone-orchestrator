"""Reversible, shadow-first context block selection.

The model may suggest a disposition, but the coordinator owns pinning,
artifact retention, and whether a filtered representation is shown to Qwen.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .decisions import DecisionProvider, DeterministicDecisionProvider
from .state import StateStore

KEEP_VERBATIM = "KEEP_VERBATIM"
HIDE_BUT_RECALLABLE = "HIDE_BUT_RECALLABLE"
TRUNCATE_WITH_POINTER = "TRUNCATE_WITH_POINTER"
PIN = "PIN"
DISPOSITIONS = (KEEP_VERBATIM, HIDE_BUT_RECALLABLE, TRUNCATE_WITH_POINTER, PIN)


@dataclass(frozen=True)
class ContextBlock:
    block_id: str
    kind: str
    text: str
    source: str = ""
    pinned: bool = False
    error: bool = False
    recent_mutation: bool = False
    state_version: str = ""


@dataclass
class ContextDecision:
    block_id: str
    disposition: str
    distribution: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    shadow: bool = True
    artifact: str | None = None


@dataclass
class ContextResult:
    blocks: list[ContextBlock]
    rendered: list[str]
    decisions: list[ContextDecision]
    original_chars: int
    rendered_chars: int
    retrievals: int = 0


class ContextGC:
    """Classify old evidence while preserving every byte for recall."""

    def __init__(self, state_dir: Path, store: StateStore, provider: DecisionProvider | None = None, mode: str = "shadow", hide_probability: float = 0.9):
        if mode not in {"shadow", "active"}:
            raise ValueError("context mode must be shadow or active")
        self.state_dir = Path(state_dir)
        self.store = store
        self.provider = provider or DeterministicDecisionProvider()
        self.mode = mode
        self.hide_probability = hide_probability

    def process(self, task_id: str, root: Path, blocks: list[ContextBlock], state: str) -> ContextResult:
        artifact_dir = self.state_dir / "context" / task_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        rendered: list[str] = []
        decisions: list[ContextDecision] = []
        for block in blocks:
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", block.block_id)[:120] or "block"
            artifact = artifact_dir / f"{safe_id}.txt"
            artifact.write_text(block.text, encoding="utf-8")
            ref = str(artifact.relative_to(self.state_dir))
            decision = self._decide(block, state)
            decision.artifact = ref
            if block.pinned or block.error or block.recent_mutation:
                decision.disposition = PIN
                decision.reason = "pinned safety evidence"
            safe = decision.disposition in DISPOSITIONS and self._confident(decision)
            decision.shadow = self.mode == "shadow" or not safe
            if decision.shadow or decision.disposition in {KEEP_VERBATIM, PIN}:
                rendered.append(block.text)
            elif decision.disposition == HIDE_BUT_RECALLABLE:
                rendered.append(self._pointer(block, ref))
            else:
                rendered.append(self._truncate(block, ref))
            decisions.append(decision)
        payload = {
            "mode": self.mode,
            "provider": self.provider.name,
            "blocks": [{"id": b.block_id, "kind": b.kind, "source": b.source, "chars": len(b.text), "sha256": hashlib.sha256(b.text.encode()).hexdigest(), "state_version": b.state_version, "decision": d.__dict__} for b, d in zip(blocks, decisions)],
            "original_chars": sum(len(b.text) for b in blocks),
            "rendered_chars": sum(len(x) for x in rendered),
        }
        self.store.evidence(task_id, "context_gc", root, payload)
        self.store.event(task_id, "context_gc", {"mode": self.mode, "provider": self.provider.name, "blocks": len(blocks), "shadow": self.mode == "shadow"})
        return ContextResult(blocks, rendered, decisions, payload["original_chars"], payload["rendered_chars"])

    def recall(self, ref: str) -> str:
        path = (self.state_dir / ref).resolve()
        root = (self.state_dir / "context").resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("invalid context evidence reference")
        return path.read_text(encoding="utf-8")

    def _decide(self, block: ContextBlock, state: str) -> ContextDecision:
        if block.pinned or block.error or block.recent_mutation:
            return ContextDecision(block.block_id, PIN, {PIN: 1.0}, "pinned safety evidence")
        question = {"disposition": {"type": "choice", "instructions": "Choose a reversible context disposition. Never hide errors, recent mutations, unresolved side effects, or exact evidence needed for verification.", "criteria": {
            KEEP_VERBATIM: "Keep the complete block in the model context.",
            HIDE_BUT_RECALLABLE: "Hide it behind a local pointer while retaining exact text for explicit recall.",
            TRUNCATE_WITH_POINTER: "Show a short head and a local pointer to the exact retained block.",
            PIN: "Pin the complete block as safety-critical evidence.",
        }}}
        try:
            batch = self.provider.decide(json.dumps({"task": state, "block": {"kind": block.kind, "source": block.source, "text": block.text[:12000]}}, sort_keys=True), question)
            if batch.decisions:
                d = batch.decisions[0]
                return ContextDecision(block.block_id, d.selected if d.selected in DISPOSITIONS else KEEP_VERBATIM, d.distribution, "provider suggestion")
        except Exception:
            pass
        return ContextDecision(block.block_id, KEEP_VERBATIM, {KEEP_VERBATIM: 1.0}, "provider unavailable")

    def _confident(self, decision: ContextDecision) -> bool:
        return self.mode == "active" and decision.distribution.get(decision.disposition, 0.0) >= self.hide_probability

    @staticmethod
    def _pointer(block: ContextBlock, ref: str) -> str:
        return f"[hidden evidence {ref}: {block.kind} from {block.source or 'unknown source'}, {len(block.text)} chars; recall explicitly to inspect exact text]"

    @staticmethod
    def _truncate(block: ContextBlock, ref: str) -> str:
        head = block.text[:600]
        return f"{head}\n[truncated evidence {ref}: {max(0, len(block.text) - len(head))} chars omitted; recall explicitly for the unchanged block]"
