"""Outcome-grounded supervisory traces; no model-generated reasoning labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .features import FEATURE_SCHEMA
from .state import StateStore


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("<redacted>" if any(word in k.lower() for word in ("secret", "token", "password", "authorization", "api_key")) else _redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


class TraceCollector:
    def __init__(self, store: StateStore):
        self.store = store

    def record(self, task_id: str, root: Path, *, run_id: str, decision_family: str, phase: str, before: dict[str, Any], deterministic_choice: str | None, provider_choice: str | None, after: dict[str, Any], hard_labels: dict[str, Any] | None = None) -> str:
        before = {key: before.get(key) for key in FEATURE_SCHEMA} if all(key in before for key in FEATURE_SCHEMA) else before
        payload = {
            "trace_id": "trace-" + hashlib.sha256(_stable([task_id, run_id, decision_family, phase, before, after]).encode()).hexdigest()[:20],
            "task_id": task_id,
            "run_id": run_id,
            "decision_family": decision_family,
            "controller_phase": phase,
            "before": _redact(before),
            "decision": {"deterministic": deterministic_choice, "provider": provider_choice},
            "after": _redact(after),
            "labels": _redact(hard_labels or {"status": "UNKNOWN", "source": "not deterministically established"}),
        }
        ref = self.store.evidence(task_id, "decision_trace", root, payload)
        self.store.event(task_id, "decision_trace", {"ref": ref, "trace_id": payload["trace_id"], "decision_family": decision_family})
        return ref

    @staticmethod
    def export(store: StateStore, output: Path) -> int:
        rows = store.evidence_rows(kind="decision_trace")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                payload = row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"])
                handle.write(_stable(payload) + "\n")
        return len(rows)
