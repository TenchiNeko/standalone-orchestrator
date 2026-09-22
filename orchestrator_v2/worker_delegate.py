#!/usr/bin/env python3
"""Bounded, read-only Sharp-Spark delegation harness for OpenCode v2.

The process owns the worker conversation and returns only a compact report on
stdout.  It deliberately has no general shell or network tool: its only
network operation is the configured OpenAI-compatible worker endpoint.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MAX_STEPS = 14
MAX_WALL_SECONDS = 120.0
MAX_MODEL_TOKENS = 900
MAX_TOOL_OUTPUT = 12_000
MAX_TOTAL_TOOL_OUTPUT = 48_000
# Cumulative prompt+completion accounting is intentionally finite.  A worker
# conversation repeats its bounded tool context on each turn, so 40K leaves
# room for a focused final report without allowing an unbounded investigation.
MAX_TOTAL_MODEL_TOKENS = 40_000
MODEL_REQUEST_TIMEOUT = 60
MAX_RESULT_BYTES = 3_200
MAX_EVIDENCE = 8
MAX_FOCUS_PATHS = 12
WORKER_URL = os.environ.get("SHARP_SPARK_WORKER_URL", "http://100.81.200.82:18090/v1").rstrip("/")
WORKER_KEY_FILE = Path(os.environ.get(
    "SHARP_SPARK_WORKER_KEY_FILE",
    "/home/brandon/.config/opencode/secrets/sharp-spark-worker.key",
))
STATE_DIR = Path(os.environ.get(
    "SHARP_SPARK_WORKER_STATE",
    "/home/brandon/.local/state/opencode/sharp-spark-worker",
))
MODEL = os.environ.get("SHARP_SPARK_WORKER_MODEL", "sharp-spark-worker")

ROLE_PROMPTS = {
    "scout": "Locate the relevant implementation and trace the concrete call/data path. Return exact files/functions/lines and the smallest useful finding. Do not propose unrelated refactors.",
    "debugger": "Investigate the supplied failure or symptom. Separate observed evidence from hypothesis. Identify the likeliest root cause and the next discriminating check. Do not edit.",
    "reviewer": "Review the supplied/current diff against the stated objective. Look for correctness errors, regressions, security/data-loss issues, and missing verification. Return concrete findings only.",
}

SECRET_PARTS = {
    ".ssh", ".aws", ".gnupg", ".env", "credentials", "cookies", "secrets",
    "id_rsa", "id_ed25519", "authorized_keys",
}
SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".secret")
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".tox", "dist", "build"}


class WorkerError(Exception):
    pass


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "\n…[bounded]"


def _json_line(value: Any, limit: int = MAX_TOOL_OUTPUT) -> str:
    return _bounded_text(json.dumps(value, ensure_ascii=False, sort_keys=True), limit)


def _workspace_root(value: Any) -> Path:
    if not isinstance(value, str) or not value.startswith("/"):
        raise WorkerError("workspace must be an absolute path")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise WorkerError("workspace is not a directory")
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        raise WorkerError("workspace is not a Git repository")
    if Path(top).resolve() != root:
        raise WorkerError("workspace must be the active repository root")
    return root


def _is_secret_path(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    for part in parts:
        lower = part.lower()
        if lower in SECRET_PARTS or lower.endswith(SECRET_SUFFIXES):
            return True
        if lower.startswith(".env") or "credential" in lower or "token" in lower:
            return True
    return False


def _safe_path(root: Path, value: Any, *, allow_directory: bool = False) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise WorkerError("path must be a non-empty project-relative string")
    candidate_text = Path(value)
    if candidate_text.is_absolute() or ".." in candidate_text.parts:
        raise WorkerError("path must not be absolute or contain '..'")
    candidate = (root / candidate_text).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise WorkerError("path escapes the repository root")
    if _is_secret_path(candidate, root):
        raise WorkerError("secret-like paths are not available to workers")
    if candidate != root and ".git" in candidate.relative_to(root).parts:
        raise WorkerError(".git internals are not available to workers")
    if not candidate.exists():
        raise WorkerError("path does not exist")
    if not allow_directory and not candidate.is_file():
        raise WorkerError("path is not a regular file")
    if allow_directory and not candidate.is_dir():
        raise WorkerError("path is not a directory")
    return candidate


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _safe_any_path(root: Path, value: Any) -> Path:
    """Allow a project-relative file or directory while retaining all guards."""
    try:
        return _safe_path(root, value)
    except WorkerError as first:
        try:
            return _safe_path(root, value, allow_directory=True)
        except WorkerError:
            raise first


def _read_file(root: Path, args: dict[str, Any], examined: set[str]) -> dict[str, Any]:
    path = _safe_path(root, args.get("path"))
    start = max(1, int(args.get("start_line", 1)))
    end = min(start + 239, int(args.get("end_line", start + 239)))
    if end < start:
        raise WorkerError("end_line must not precede start_line")
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    examined.add(_relative(root, path))
    selected = data[start - 1:end]
    text = "\n".join(f"{number}: {line}" for number, line in enumerate(selected, start))
    return {"path": _relative(root, path), "lines": f"{start}-{min(end, len(data))}", "content": _bounded_text(text, MAX_TOOL_OUTPUT)}


def _walk_files(root: Path):
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS and not _is_secret_path(Path(base) / name, root)]
        for name in files:
            path = Path(base) / name
            if not _is_secret_path(path, root):
                yield path


def _search(root: Path, args: dict[str, Any], examined: set[str]) -> dict[str, Any]:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern or len(pattern) > 300:
        raise WorkerError("pattern must be a bounded non-empty string")
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise WorkerError(f"invalid search pattern: {exc}")
    base = root if not args.get("path") else _safe_path(root, args["path"], allow_directory=True)
    max_results = min(40, max(1, int(args.get("max_results", 20))))
    results: list[dict[str, Any]] = []
    files_seen = 0
    for path in _walk_files(base):
        if files_seen >= 200 or len(results) >= max_results:
            break
        files_seen += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = _relative(root, path)
        examined.add(rel)
        for line_no, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                results.append({"path": rel, "line": line_no, "text": _bounded_text(line.strip(), 300)})
                if len(results) >= max_results:
                    break
    return {"matches": results, "files_scanned": files_seen}


def _glob(root: Path, args: dict[str, Any], examined: set[str]) -> dict[str, Any]:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern or len(pattern) > 200 or Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise WorkerError("glob must be a bounded project-relative pattern")
    max_results = min(100, max(1, int(args.get("max_results", 50))))
    matches: list[str] = []
    for path in _walk_files(root):
        rel = _relative(root, path)
        if fnmatch.fnmatch(rel, pattern):
            matches.append(rel)
            examined.add(rel)
            if len(matches) >= max_results:
                break
    return {"matches": matches}


def _git(root: Path, args: dict[str, Any], examined: set[str]) -> dict[str, Any]:
    action = args.get("action")
    if action not in {"status", "diff", "log"}:
        raise WorkerError("git action must be status, diff, or log")
    paths = args.get("paths", [])
    if not isinstance(paths, list) or len(paths) > 12:
        raise WorkerError("git paths must be a bounded list")
    safe_paths: list[str] = []
    for value in paths:
        path = _safe_any_path(root, value)
        safe_paths.append(_relative(root, path))
        examined.add(_relative(root, path))
    if action == "status":
        argv = ["git", "-C", str(root), "status", "--short", "--branch"]
    elif action == "diff":
        if not safe_paths:
            raise WorkerError("git diff requires explicit non-secret paths")
        argv = ["git", "-C", str(root), "diff", "--no-ext-diff", "--unified=2", "--"] + safe_paths
    else:
        argv = ["git", "-C", str(root), "log", "-n", "12", "--oneline", "--decorate", "--"] + safe_paths
    try:
        completed = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorkerError(f"git inspection failed: {exc}")
    output = completed.stdout
    if action == "status":
        output = "\n".join(line for line in output.splitlines() if not _is_secret_path(root / line[3:].strip(), root))
    return {"action": action, "exit_code": completed.returncode, "output": _bounded_text(output, MAX_TOOL_OUTPUT)}


def dispatch_tool(root: Path, name: Any, args: Any, examined: set[str]) -> dict[str, Any]:
    if not isinstance(name, str) or not isinstance(args, dict):
        raise WorkerError("malformed structured worker tool call")
    if name == "worker_read_file":
        return _read_file(root, args, examined)
    if name == "worker_search":
        return _search(root, args, examined)
    if name == "worker_glob":
        return _glob(root, args, examined)
    if name == "worker_git":
        return _git(root, args, examined)
    raise WorkerError("unknown worker tool")


def _tool_schemas() -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": "worker_read_file", "description": "Read bounded lines from a project file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "worker_search", "description": "Search project text with a bounded regular expression.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["pattern"]}}},
        {"type": "function", "function": {"name": "worker_glob", "description": "List bounded project-relative glob matches.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["pattern"]}}},
        {"type": "function", "function": {"name": "worker_git", "description": "Inspect Git status, diff, or recent log only.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["status", "diff", "log"]}, "paths": {"type": "array", "items": {"type": "string"}}}, "required": ["action"]}}},
    ]


def _api_call(messages: list[dict[str, Any]], key: str) -> dict[str, Any]:
    payload = {"model": MODEL, "messages": messages, "tools": _tool_schemas(), "tool_choice": "auto", "temperature": 0.2, "max_tokens": MAX_MODEL_TOKENS, "stream": False}
    request = Request(f"{WORKER_URL}/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, method="POST")
    try:
        with urlopen(request, timeout=MODEL_REQUEST_TIMEOUT) as response:
            body = response.read(MAX_TOOL_OUTPUT * 4)
    except HTTPError as exc:
        raise WorkerError(f"worker HTTP {exc.code}")
    except (URLError, TimeoutError, OSError) as exc:
        raise WorkerError(f"worker unavailable: {type(exc).__name__}")
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        raise WorkerError("worker returned malformed JSON")
    if not isinstance(value, dict) or not isinstance(value.get("choices"), list) or not value["choices"]:
        raise WorkerError("worker returned no completion choice")
    return value


def _key() -> str:
    try:
        value = WORKER_KEY_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        raise WorkerError("worker API key is unavailable")
    if not value or len(value) > 512 or any(char.isspace() for char in value):
        raise WorkerError("worker API key file is invalid")
    return value


def _evidence_from_text(text: str) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    # Markdown fallback is intentionally conservative: only file-like paths
    # are promoted to evidence.  This avoids turning prose such as
    # "security/data-loss" into a fabricated repository path.
    path_re = re.compile(
        r"(?<![A-Za-z0-9_./-])"
        r"([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.[A-Za-z0-9][A-Za-z0-9_-]{0,9})"
        r"(?:\s*(?::|,?\s+lines?\s+)([0-9]+(?:\s*[-–]\s*[0-9]+)?))?",
        re.IGNORECASE,
    )
    global_line = re.search(r"(?:exact\s+)?lines?\s*[:#`* ]+([0-9]+(?:\s*[-–]\s*[0-9]+)?)", text, re.IGNORECASE)
    seen: set[str] = set()
    for match in path_re.finditer(text):
        path = match.group(1)
        if "://" in text[max(0, match.start() - 8):match.start() + 3] or path in seen:
            continue
        seen.add(path)
        line_value = match.group(2) or (global_line.group(1) if global_line else "")
        evidence.append({"path": path, "lines": line_value.replace("–", "-").replace(" ", ""), "note": "Concrete file path/line cited by the worker's bounded report."})
        if len(evidence) >= MAX_EVIDENCE:
            break
    return evidence


def _parse_report(content: Any) -> dict[str, Any]:
    text = str(content or "").strip()
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
    if not isinstance(parsed, dict):
        evidence = _evidence_from_text(text)
        next_match = re.search(r"(?:recommended\s+next action|smallest correct next action|next action)\**\s*[:\-]?\s*(.*?)(?=\n\s*\**(?:uncertainty|finding|observed|impact|$))", text, re.IGNORECASE | re.DOTALL)
        uncertainty_match = re.search(r"uncertainty\**\s*[:\-]?\s*(.*?)(?=\n\s*\**(?:recommended|finding|$))", text, re.IGNORECASE | re.DOTALL)
        next_text = re.sub(r"^[\s:*\-]+", "", next_match.group(1)).strip() if next_match else ""
        if not next_text:
            action_match = re.search(r"\b(?:change|restore|run|inspect|use)\b[^\n]{8,}", text, re.IGNORECASE)
            next_text = action_match.group(0).strip() if action_match else "Use the cited file/line evidence to perform the smallest verified next check."
        uncertainty_text = re.sub(r"^[\s:*\-]+", "", uncertainty_match.group(1)).strip() if uncertainty_match else ""
        if not uncertainty_text:
            marker = re.search(r"uncertainty", text, re.IGNORECASE)
            uncertainty_text = re.sub(r"^[\s:*\-]+", "", text[marker.end():].splitlines()[0] if marker else "").strip() or "Worker returned bounded markdown rather than strict JSON."
        parsed = {
            "finding": _bounded_text(text, 900),
            "evidence": evidence,
            "recommended_next_action": _bounded_text(next_text, 450),
            "uncertainty": _bounded_text(uncertainty_text, 300),
        }
    elif not isinstance(parsed.get("evidence"), list) or not parsed.get("evidence"):
        parsed["evidence"] = _evidence_from_text(text)
    return parsed


def _validate_evidence(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Keep only concrete, confined files that exist in the active repo."""
    evidence = result.get("evidence") if isinstance(result.get("evidence"), list) else []
    valid: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        raw_path = item["path"].strip()
        try:
            path = _safe_path(root, raw_path)
        except WorkerError:
            continue
        if _relative(root, path) != raw_path:
            continue
        valid.append(item)
        if len(valid) >= MAX_EVIDENCE:
            break
    result["evidence"] = valid
    return result


def bound_result(result: dict[str, Any], role: str, metrics: dict[str, Any]) -> dict[str, Any]:
    evidence = result.get("evidence") if isinstance(result.get("evidence"), list) else []
    bounded_evidence = []
    for item in evidence[:MAX_EVIDENCE]:
        if not isinstance(item, dict):
            continue
        bounded_evidence.append({"path": _bounded_text(item.get("path"), 180), "lines": _bounded_text(item.get("lines"), 80), "note": _bounded_text(item.get("note"), 300)})
    output: dict[str, Any] = {
        "status": result.get("status") if result.get("status") in {"complete", "NEEDS_MORE_SCOPE", "ERROR"} else "complete",
        "role": role,
        "finding": _bounded_text(result.get("finding"), 900),
        "evidence": bounded_evidence,
        "recommended_next_action": _bounded_text(result.get("recommended_next_action"), 450),
        "uncertainty": _bounded_text(result.get("uncertainty"), 300),
        **metrics,
    }
    encoded = json.dumps(output, ensure_ascii=False, separators=(",", ":")).encode()
    if len(encoded) <= MAX_RESULT_BYTES:
        return output
    output["evidence"] = bounded_evidence[:4]
    output["finding"] = _bounded_text(output["finding"], 600)
    output["recommended_next_action"] = _bounded_text(output["recommended_next_action"], 250)
    output["uncertainty"] = _bounded_text(output["uncertainty"], 180)
    encoded = json.dumps(output, ensure_ascii=False, separators=(",", ":")).encode()
    if len(encoded) <= MAX_RESULT_BYTES:
        return output
    output["evidence"] = []
    output["finding"] = _bounded_text(output["finding"], 450)
    return output


def _save_metadata(role: str, result: dict[str, Any], examined: set[str], started: float) -> None:
    try:
        STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(STATE_DIR, 0o700)
        metadata = {"run_id": uuid.uuid4().hex, "role": role, "status": result.get("status"), "files_examined": len(examined), "elapsed_seconds": round(time.monotonic() - started, 2), "paths": sorted(examined)[:100]}
        target = STATE_DIR / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{metadata['run_id'][:8]}.json"
        target.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(target, 0o600)
    except OSError:
        pass


def run(request: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    role = request.get("role")
    if role not in ROLE_PROMPTS:
        return {"status": "ERROR", "role": str(role), "finding": "role must be scout, debugger, or reviewer"}
    try:
        root = _workspace_root(request.get("workspace"))
        objective = request.get("objective")
        if not isinstance(objective, str) or not objective.strip() or len(objective) > 1200:
            raise WorkerError("objective must be a bounded non-empty string")
        focus = request.get("focus_paths", [])
        if focus is None:
            focus = []
        if not isinstance(focus, list) or len(focus) > MAX_FOCUS_PATHS or not all(isinstance(item, str) for item in focus):
            raise WorkerError("focus_paths must be a bounded list of strings")
        for item in focus:
            _safe_any_path(root, item)
        key = _key()
    except WorkerError as exc:
        return {"status": "ERROR", "role": role, "finding": str(exc), "evidence": []}
    examined: set[str] = set()
    total_tool_output = 0
    steps = 0
    invalid_calls = 0
    input_tokens = 0
    output_tokens = 0
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a bounded read-only repository worker. Use only the supplied structured tools. Never edit, run shell, use network, inspect secrets, or claim unobserved facts. Finish with JSON containing finding, evidence (path/lines/note), recommended_next_action, and uncertainty. Evidence must be concrete and concise."},
        {"role": "user", "content": f"Role: {role}.\nRole instruction: {ROLE_PROMPTS[role]}\nObjective: {objective.strip()}\nFocus paths: {', '.join(focus) if focus else '(none; discover narrowly)'}\nRepository root is already fixed by the host; use project-relative paths only."},
    ]
    final: dict[str, Any] | None = None
    try:
        while steps < MAX_STEPS and time.monotonic() - started < MAX_WALL_SECONDS:
            steps += 1
            response = _api_call(messages, key)
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            input_tokens += int(usage.get("prompt_tokens") or 0)
            output_tokens += int(usage.get("completion_tokens") or 0)
            if input_tokens + output_tokens > MAX_TOTAL_MODEL_TOKENS:
                raise WorkerError("worker exceeded cumulative model-token budget")
            message = response["choices"][0].get("message") if isinstance(response["choices"][0], dict) else None
            if not isinstance(message, dict):
                raise WorkerError("worker returned malformed message")
            calls = message.get("tool_calls")
            if isinstance(calls, list) and calls:
                messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
                for call in calls[:3]:
                    call_id = str(call.get("id") or f"invalid-{steps}") if isinstance(call, dict) else f"invalid-{steps}"
                    function = call.get("function") if isinstance(call, dict) else None
                    name = function.get("name") if isinstance(function, dict) else None
                    raw_args = function.get("arguments") if isinstance(function, dict) else None
                    try:
                        arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        tool_result = dispatch_tool(root, name, arguments, examined)
                        rendered = _json_line(tool_result)
                    except (WorkerError, json.JSONDecodeError, TypeError, ValueError) as exc:
                        invalid_calls += 1
                        rendered = _json_line({"status": "INVALID_TOOL_CALL", "error": str(exc)})
                    total_tool_output += len(rendered)
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": rendered})
                    if invalid_calls > 1 or total_tool_output > MAX_TOTAL_TOOL_OUTPUT:
                        raise WorkerError("worker exceeded safe tool-call or output budget")
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise WorkerError("worker returned an empty final report")
            final = _parse_report(content)
            if not str(final.get("finding") or "").strip():
                raise WorkerError("worker returned a final report without a finding")
            break
        if final is None:
            final = {"status": "NEEDS_MORE_SCOPE", "finding": "Investigation budget ended before a bounded final report.", "recommended_next_action": "Narrow the objective and inspect the smallest missing path.", "uncertainty": "step, wall-time, or model-token budget reached."}
    except WorkerError as exc:
        message = str(exc)
        budget_stop = message.startswith("worker exceeded cumulative model-token budget") or message.startswith("worker exceeded safe tool-call or output budget")
        final = {
            "status": "NEEDS_MORE_SCOPE" if budget_stop else "ERROR",
            "finding": "Investigation budget ended before a bounded final report." if budget_stop else message,
            "recommended_next_action": "Narrow the objective and inspect the smallest missing path." if budget_stop else "Continue with local Qwen investigation; worker failure is non-fatal.",
            "uncertainty": "step, wall-time, model-token, or tool-output budget reached." if budget_stop else "remote worker did not produce a complete report.",
        }
    metrics = {"files_examined": len(examined), "worker_steps": steps, "worker_input_tokens": input_tokens, "worker_output_tokens": output_tokens, "elapsed_seconds": round(time.monotonic() - started, 2)}
    result = bound_result(_validate_evidence(root, final), role, metrics)
    _save_metadata(role, result, examined, started)
    return result


def main() -> int:
    try:
        request = json.loads(sys.stdin.readline())
        if not isinstance(request, dict):
            raise WorkerError("request must be a JSON object")
        result = run(request)
    except (json.JSONDecodeError, WorkerError) as exc:
        result = {"status": "ERROR", "finding": str(exc), "evidence": []}
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
