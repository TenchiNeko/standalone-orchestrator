"""Usage and budget accounting over actual v2 events."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .state import StateStore


def _num(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def usage_from_response(usage: dict[str, Any] | None, *, elapsed: float, prompt_chars: int = 0, output_chars: int = 0) -> dict[str, Any]:
    data = usage or {}
    prompt = _num(data.get("prompt_tokens"))
    completion = _num(data.get("completion_tokens"))
    cached = None
    details = data.get("prompt_tokens_details") or data.get("prompt_token_details") or {}
    if isinstance(details, dict):
        cached = _num(details.get("cached_tokens"))
    exact = prompt is not None or completion is not None
    return {"prompt_tokens": prompt, "completion_tokens": completion, "cached_prompt_tokens": cached, "token_source": "api" if exact else "unavailable", "prompt_chars": prompt_chars, "output_chars": output_chars, "elapsed": float(elapsed)}


def usage_summary(store: StateStore, task_id: str) -> dict[str, Any]:
    rows = store.events(task_id)
    model = [__import__("json").loads(row["payload"]) for row in rows if row["kind"] == "usage"]
    tools = [__import__("json").loads(row["payload"]) for row in rows if row["kind"] == "tool_call"]
    decisions = [__import__("json").loads(row["payload"]) for row in rows if row["kind"] == "decision_usage"]
    exact_prompt = sum(v.get("prompt_tokens") or 0 for v in model if v.get("prompt_tokens") is not None)
    exact_output = sum(v.get("completion_tokens") or 0 for v in model if v.get("completion_tokens") is not None)
    sources = {v.get("token_source") for v in model}
    measurement = "exact_api_fields" if "api" in sources else ("exact_opencode_event_fields" if "opencode_event" in sources else "unavailable")
    return {
        "qwen_requests": len(model),
        "prompt_tokens": exact_prompt,
        "output_tokens": exact_output,
        "cached_prompt_tokens": sum(v.get("cached_prompt_tokens") or 0 for v in model if v.get("cached_prompt_tokens") is not None),
        "token_measurement": measurement,
        "model_seconds": round(sum(float(v.get("elapsed", 0)) for v in model), 6),
        "tool_calls": len(tools),
        "reads": sum(v.get("name") == "read_file" for v in tools),
        "writes": sum(v.get("name") == "write_file" for v in tools),
        "tests": sum(v.get("name") == "run_tests" for v in tools),
        "failed_tool_calls": sum(v.get("status") in {"rejected", "failed"} for v in tools),
        "uncertain_tool_calls": sum(v.get("status") == "unknown" for v in tools),
        "decision_provider_calls": len(decisions),
        "decision_provider_seconds": round(sum(float(v.get("latency", 0)) for v in decisions), 6),
    }
