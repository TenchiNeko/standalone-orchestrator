"""Stable, hard-evidence feature vectors for future routing evaluation."""
from __future__ import annotations

import json
from typing import Any
from pathlib import Path

from .state import StateStore, Task, file_manifest, source_hash, task_scope


FEATURE_SCHEMA = (
    "read_only_task", "permitted_file_count", "changed_file_count", "repo_dirty",
    "prior_stuck_loop", "same_task_prior_failure", "same_hypothesis_count",
    "same_action_count", "new_evidence_count", "error_signature_changed",
    "test_has_run", "test_passed", "test_stale", "unresolved_mutation_count",
    "current_step", "qwen_calls_so_far", "remaining_budget", "ocr_findings_count",
    "supported_findings_count", "candidate_file_count", "last_action_was_read",
    "last_action_was_write", "last_action_was_test", "last_action_failed",
    "last_action_uncertain",
)


def _bool(value: Any) -> bool:
    return bool(value)


def feature_vector(task: Task, store: StateStore, root: Path, *, cross_run_findings: list[Any] | None = None, candidate_file_count: int = 0) -> dict[str, Any]:
    events = store.events(task.task_id)
    intake = next((e for e in events if e["kind"] == "intake"), None)
    intake_payload = json.loads(intake["payload"]) if intake else {}
    before = intake_payload.get("file_manifest", {})
    scope = task_scope(task)
    current = file_manifest(root, scope)
    changed = sum(1 for key in set(before) | set(current) if before.get(key) != current.get(key))
    tests = [e for e in events if e["kind"] == "evidence_recorded" and json.loads(e["payload"]).get("kind") in {"test", "final_test"}]
    evidence = [e for e in events if e["kind"] == "evidence_recorded"]
    tool_events = [e for e in events if e["kind"] == "tool_call"]
    last = json.loads(tool_events[-1]["payload"]) if tool_events else {}
    usages = [e for e in events if e["kind"] == "usage"]
    test_passed = False
    test_stale = False
    for event in tests:
        ref = json.loads(event["payload"]).get("ref")
        record = store.evidence_record(ref) if ref else None
        if record:
            payload = record.get("payload", {})
            if payload.get("exit_code") == 0 and not payload.get("timed_out"):
                test_passed = True
            if record.get("source_hash") != source_hash(root, scope):
                test_stale = True
    payload = {
        "read_only_task": not any(a in task.contract.permitted_actions for a in ("write_file", "browser")),
        "permitted_file_count": len(scope or []),
        "changed_file_count": changed,
        "repo_dirty": changed > 0,
        "prior_stuck_loop": any(getattr(f, "kind", "") == "PRIOR_STUCK_LOOP" for f in (cross_run_findings or [])),
        "same_task_prior_failure": any(getattr(f, "kind", "") == "SAME_TASK_SAME_STATE_REPEAT" for f in (cross_run_findings or [])),
        "same_hypothesis_count": sum(1 for e in events if e["kind"] == "cross_run_attempt"),
        "same_action_count": sum(1 for e in tool_events if json.loads(e["payload"]).get("status") == "ok"),
        "new_evidence_count": len(evidence),
        "error_signature_changed": False,
        "test_has_run": bool(tests),
        "test_passed": test_passed,
        "test_stale": test_stale,
        "unresolved_mutation_count": len(store.unresolved_mutations(task.task_id)),
        "current_step": len([e for e in events if e["kind"] == "phase"]),
        "qwen_calls_so_far": len(usages),
        "remaining_budget": max(0, task.budget_limit - task.budget_calls),
        "ocr_findings_count": sum(1 for e in evidence if json.loads(e["payload"]).get("kind") == "ocr_review"),
        "supported_findings_count": 0,
        "candidate_file_count": int(candidate_file_count),
        "last_action_was_read": last.get("name") == "read_file",
        "last_action_was_write": last.get("name") == "write_file",
        "last_action_was_test": last.get("name") == "run_tests",
        "last_action_failed": last.get("status") in {"rejected", "failed"},
        "last_action_uncertain": last.get("status") == "unknown",
    }
    return {key: payload.get(key, None) for key in FEATURE_SCHEMA}


def serialize_features(features: dict[str, Any]) -> str:
    return json.dumps({key: features.get(key) for key in FEATURE_SCHEMA}, sort_keys=True, separators=(",", ":"))
