"""Local JSONL bridge used by the project-local OpenCode plugin.

The bridge is intentionally a small adapter around the existing Python core;
it does not run a second agent loop or expose a network listener.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from orchestrator_v2.controller import Controller
    from orchestrator_v2.cross_run import LoopMemory
    from orchestrator_v2.state import Criterion, Phase, Task, TaskContract, source_hash
    from orchestrator_v2.symbols import SymbolIndex
    from orchestrator_v2.usage import usage_summary
else:
    from .controller import Controller
    from .cross_run import LoopMemory
    from .state import Criterion, Phase, Task, TaskContract, source_hash
    from .symbols import SymbolIndex
    from .usage import usage_summary


FILE_KEYS = ("filePath", "file_path", "path", "file")
READ_TOOLS = {"read", "grep", "glob", "ls", "list"}
WRITE_TOOLS = {"write", "edit", "apply_patch", "patch"}
TEST_WORDS = ("pytest", "unittest", "test", "check", "lint", "typecheck", "pyright", "mypy")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()[:24]


def _output_hash(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", "replace")).hexdigest()


def _same_file_state(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    """Compare observable file facts without treating absolute/relative paths as state changes."""
    if not left or not right:
        return left == right
    return all(left.get(key) == right.get(key) for key in ("exists", "sha256", "bytes"))


def _tool_kind(tool: str, args: dict[str, Any]) -> str:
    name = str(tool or "").lower()
    if name.startswith("orchestrator_"):
        return "supervisor"
    if name in READ_TOOLS:
        return "read_file"
    if name in WRITE_TOOLS:
        return "write_file"
    if name in {"bash", "shell", "terminal", "run"}:
        command = str(args.get("command") or args.get("cmd") or "").lower()
        if any(word in command for word in TEST_WORDS):
            return "run_tests"
        return "shell"
    return name or "unknown"


def _path_arg(args: dict[str, Any]) -> str | None:
    for key in FILE_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return None


class SupervisorBridge:
    def __init__(self, state_root: Path | None = None):
        self.default_state_root = Path(state_root or os.environ.get("V2_SUPERVISOR_STATE", "~/.local/state/opencode/orchestrator-v2")).expanduser().resolve()
        self.sessions: dict[str, str] = {}
        self.tasks: dict[str, Controller] = {}
        self.pending: dict[str, dict[str, Any]] = {}

    def _controller(self, workspace: Path, state_dir: Path | None = None) -> Controller:
        if state_dir is None:
            digest = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:16]
            state_dir = self.default_state_root / digest
        key = str(state_dir.resolve())
        if key not in self.tasks:
            self.tasks[key] = Controller(Path(key))
        return self.tasks[key]

    def _task(self, session_id: str | None = None, task_id: str | None = None) -> tuple[Controller, Task] | None:
        if task_id:
            for controller in self.tasks.values():
                try:
                    return controller, controller.store.load_task(task_id)
                except KeyError:
                    continue
        if session_id and session_id in self.sessions:
            task_id = self.sessions[session_id]
            return self._task(task_id=task_id)
        return None

    @staticmethod
    def _permitted(task: Task, path: str | None) -> bool:
        if not path:
            return True
        root = Path(task.workspace).resolve()
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = root / target
        try:
            rel = target.resolve().relative_to(root).as_posix()
        except ValueError:
            return False
        permitted = task.contract.permitted_files
        return "*" in permitted or rel in {Path(item).as_posix().lstrip("./") for item in permitted}

    @staticmethod
    def _file_state(task: Task, path: str | None) -> dict[str, Any] | None:
        if not path:
            return None
        root = Path(task.workspace).resolve()
        target = (root / path).resolve()
        if target != root and root not in target.parents:
            return {"path": path, "error": "path escapes workspace"}
        if not target.exists():
            return {"path": path, "exists": False, "sha256": None, "bytes": 0}
        data = target.read_bytes()
        return {"path": path, "exists": True, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    def start(self, args: dict[str, Any], session_id: str, workspace: str) -> dict[str, Any]:
        root = Path(workspace).resolve()
        permitted_files = list(args.get("permitted_files") or [])
        if not permitted_files:
            return {"status": "BLOCKED", "reason": "permitted_files is required; v2 will not grant implicit repository-wide write access"}
        actions = list(args.get("permitted_actions") or ["read_file", "write_file", "run_tests"])
        criteria = [Criterion(str(item["key"]), str(item["description"]), bool(item.get("required", True))) for item in args.get("criteria", []) if isinstance(item, dict) and item.get("key") and item.get("description")]
        try:
            contract = TaskContract(str(args.get("goal") or ""), permitted_files, actions, criteria, int(args.get("version", 1)), args.get("test_command"), bool(args.get("visual_required", False)))
            contract.validate()
        except (KeyError, TypeError, ValueError) as exc:
            return {"status": "BLOCKED", "reason": f"invalid task contract: {exc}"}
        state_dir = Path(args["state_dir"]).resolve() if args.get("state_dir") else None
        controller = self._controller(root, state_dir)
        task = controller.intake(contract, root, int(args.get("budget_limit", 12)))
        task.max_model_seconds = float(args["max_model_seconds"]) if args.get("max_model_seconds") is not None else None
        controller.store.save_task(task)
        self.sessions[session_id] = task.task_id
        controller.store.event(task.task_id, "opencode_session", {"session_id": session_id, "workspace": str(root), "host": "opencode"})
        return {"status": "STARTED", "task_id": task.task_id, "workspace": str(root), "phase": task.phase.value}

    def before(self, args: dict[str, Any]) -> dict[str, Any]:
        found = self._task(args.get("session_id"), args.get("task_id"))
        if not found:
            return {"decision": "WARN", "reason": "no v2 task is associated with this OpenCode session"}
        controller, task = found
        tool = str(args.get("tool") or "")
        tool_args = args.get("args") if isinstance(args.get("args"), dict) else {}
        kind = _tool_kind(tool, tool_args)
        path = _path_arg(tool_args)
        # Supervisor tools are the control plane itself; they are not project
        # actions and must remain callable after a task starts.
        if kind == "supervisor":
            controller.store.event(task.task_id, "opencode_before", {"tool": tool, "kind": kind, "decision": "ALLOW"})
            return {"decision": "ALLOW"}
        if kind == "shell" or kind not in set(task.contract.permitted_actions) | READ_TOOLS | WRITE_TOOLS:
            controller.store.event(task.task_id, "opencode_before", {"tool": tool, "kind": kind, "decision": "BLOCK", "reason": "tool/action is not in the task contract"})
            return {"decision": "BLOCK", "reason": f"{tool} is not an allowed bounded v2 action"}
        if kind in {"read_file", "write_file"} and not self._permitted(task, path):
            controller.store.event(task.task_id, "opencode_before", {"tool": tool, "kind": kind, "decision": "BLOCK", "path": path, "reason": "path is outside permitted_files"})
            return {"decision": "BLOCK", "reason": f"path is not permitted: {path}"}
        if kind == "write_file" and controller.store.unresolved_mutations(task.task_id):
            return {"decision": "BLOCK", "reason": "an earlier mutation outcome is unresolved; reconcile it before another write"}
        state = source_hash(Path(task.workspace), task.contract.permitted_files)
        fingerprint = _json_hash({"tool": tool, "kind": kind, "args": tool_args, "state": state})
        prior = [json.loads(row["payload"]) for row in controller.store.events(task.task_id) if row["kind"] == "opencode_before" and json.loads(row["payload"]).get("action_fingerprint") == fingerprint]
        duplicate = bool(prior)
        if duplicate and kind == "write_file":
            # A confirmed failure after readback is a new decision point; an
            # acknowledged write at the same state remains a redundant retry.
            results = [json.loads(row["payload"]) for row in controller.store.events(task.task_id) if row["kind"] == "mutation_result" and json.loads(row["payload"]).get("idempotency_key") == fingerprint]
            duplicate = not any(result.get("status") == "failed" for result in results)
        if duplicate:
            decision = "BLOCK" if kind == "write_file" else "WARN"
            return {"decision": decision, "reason": "same action and relevant source state already occurred without a new state fingerprint", "action_fingerprint": fingerprint}
        before_state = self._file_state(task, path) if kind == "write_file" else None
        if kind == "write_file":
            controller.store.mutation_intent(task.task_id, "opencode:" + tool, fingerprint, {"path": path, "before": before_state, "session_id": args.get("session_id")})
        controller.store.event(task.task_id, "opencode_before", {"tool": tool, "kind": kind, "decision": "ALLOW", "path": path, "action_fingerprint": fingerprint, "state_fingerprint": state})
        self.pending[str(args.get("call_id") or fingerprint)] = {"task_id": task.task_id, "controller": controller, "kind": kind, "path": path, "fingerprint": fingerprint, "before": before_state}
        return {"decision": "ALLOW", "action_fingerprint": fingerprint}

    def after(self, args: dict[str, Any]) -> dict[str, Any]:
        key = str(args.get("call_id") or "")
        pending = self.pending.pop(key, None)
        found = self._task(args.get("session_id"), args.get("task_id"))
        if not found and not pending:
            return {"status": "UNTRACKED"}
        controller, task = found or (pending["controller"], pending["controller"].store.load_task(pending["task_id"]))
        output = args.get("output") if isinstance(args.get("output"), dict) else {}
        text = output.get("output", output.get("text", ""))
        metadata = output.get("metadata") if isinstance(output.get("metadata"), dict) else {}
        exit_code = metadata.get("exitCode", metadata.get("exit_code", metadata.get("exit")))
        status = "failed" if exit_code not in (0, None) or metadata.get("error") else "acknowledged"
        if args.get("ambiguous"):
            status = "unknown"
        result: dict[str, Any] = {"tool": args.get("tool"), "status": status, "output_sha256": _output_hash(text), "output_bytes": len(str(text).encode("utf-8", "replace")), "metadata": {k: v for k, v in metadata.items() if k not in {"authorization", "token", "secret"}}}
        if pending and pending["kind"] == "write_file":
            after_state = self._file_state(task, pending["path"])
            result.update({"path": pending["path"], "before": pending["before"], "after": after_state})
            if status != "unknown":
                controller.store.mutation_ack(task.task_id, pending["fingerprint"], status, result)
        if pending and pending["kind"] == "run_tests":
            controller.store.evidence(task.task_id, "test", Path(task.workspace), {"command": args.get("args", {}).get("command"), "exit_code": exit_code, "timed_out": bool(metadata.get("timed_out")), "output_sha256": result["output_sha256"], "source": "opencode"}, task.contract.permitted_files)
        controller.store.event(task.task_id, "tool_call", {"name": pending["kind"] if pending else _tool_kind(str(args.get("tool") or ""), args.get("args") if isinstance(args.get("args"), dict) else {}), "status": status, "source": "opencode"})
        controller.store.event(task.task_id, "opencode_after", result)
        return result

    def reconcile(self, args: dict[str, Any]) -> dict[str, Any]:
        found = self._task(args.get("session_id"), args.get("task_id"))
        if not found:
            return {"status": "BLOCKED", "reason": "unknown task"}
        controller, task = found
        path = str(args.get("path") or "")
        current = self._file_state(task, path)
        unresolved = controller.store.unresolved_mutations(task.task_id)
        target_path = (Path(task.workspace) / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        for item in unresolved:
            intent = item.get("intent", {})
            details = intent.get("details", {})
            detail_path = details.get("path")
            if not detail_path or Path(detail_path).resolve() != target_path or not current:
                continue
            before = details.get("before")
            # A readback is authoritative: a changed hash confirms the
            # mutation, while an unchanged hash confirms it did not occur.
            # Either outcome closes the UNKNOWN state without replaying a
            # potentially non-idempotent write.
            status = "acknowledged" if not _same_file_state(current, before) else "failed"
            controller.store.mutation_ack(task.task_id, item["idempotency_key"], status, {"reconciled": True, "after": current, "before": before})
        return {"status": "RECONCILED", "path": path, "state": current, "remaining": controller.store.unresolved_mutations(task.task_id)}

    def evidence(self, args: dict[str, Any]) -> dict[str, Any]:
        found = self._task(args.get("session_id"), args.get("task_id"))
        if not found:
            return {"status": "BLOCKED", "reason": "unknown task"}
        controller, task = found
        kind = str(args.get("kind") or "")
        if kind not in {"visual_verify", "test", "reconcile"}:
            return {"status": "BLOCKED", "reason": "only deterministic evidence kinds are accepted"}
        if kind == "test":
            # Test evidence is created only by an observed run_tests tool
            # result in `after`; accepting a model-supplied exit code would
            # turn prose into completion authority.
            return {"status": "BLOCKED", "reason": "test evidence is recorded only from an observed run_tests result"}
        payload = dict(args.get("payload") or {})
        payload["source"] = "opencode"
        ref = controller.store.evidence(task.task_id, kind, Path(task.workspace), payload, task.contract.permitted_files)
        return {"status": "RECORDED", "ref": ref, "kind": kind}

    def finalize(self, args: dict[str, Any]) -> dict[str, Any]:
        found = self._task(args.get("session_id"), args.get("task_id"))
        if not found:
            return {"status": "BLOCKED", "reason": "no task is associated with this session"}
        controller, task = found
        current = controller.store.current_successful_check(task.task_id, Path(task.workspace), task.contract.permitted_files)
        if current:
            for criterion in task.contract.criteria:
                if criterion.required and criterion.key.lower() in {"test", "tests", "required_tests", "test_command"} and criterion.status != "verified":
                    criterion.status = "verified"; criterion.evidence.append(current["ref"]); criterion.verified_source_hash = current["source_hash"]
            controller.store.save_task(task)
        gate = controller.completion_gate(task, Path(task.workspace), review_ok=True)
        pending = [criterion.key for criterion in task.contract.criteria if criterion.required and criterion.status != "verified"]
        if pending:
            gate.update({"allowed": False, "reason": "required criteria remain unverified", "pending": pending})
        status = "COMPLETE" if gate.get("allowed") else ("NEEDS_REVIEW" if gate.get("needs_review") else "INCOMPLETE")
        if status == "COMPLETE":
            task.phase = Phase.COMPLETE; controller.store.save_task(task)
        controller.store.event(task.task_id, "opencode_finalize", {"status": status, "gate": gate})
        return {"status": status, "task_id": task.task_id, "gate": gate, "usage": usage_summary(controller.store, task.task_id)}

    def status(self, args: dict[str, Any]) -> dict[str, Any]:
        found = self._task(args.get("session_id"), args.get("task_id"))
        if not found:
            return {"status": "BLOCKED", "reason": "no task is associated with this session"}
        controller, task = found
        return controller.status(task.task_id)

    def symbols(self, args: dict[str, Any]) -> dict[str, Any]:
        found = self._task(args.get("session_id"), args.get("task_id"))
        if not found:
            return {"status": "BLOCKED", "reason": "no task is associated with this session"}
        _, task = found
        return {"status": "OK", "candidates": SymbolIndex(Path(task.workspace), task.contract.permitted_files).render(str(args.get("query") or task.contract.goal), limit=min(50, max(1, int(args.get("limit", 20)))))}

    def event(self, args: dict[str, Any]) -> dict[str, Any]:
        found = self._task(args.get("session_id"), args.get("task_id"))
        if found:
            controller, task = found
            properties = args.get("properties", {}) if isinstance(args.get("properties"), dict) else {}
            controller.store.event(task.task_id, "opencode_event", {"type": args.get("type"), "properties": properties})
            usage = properties.get("usage") if isinstance(properties.get("usage"), dict) else None
            if usage and any(isinstance(usage.get(key), (int, float)) for key in ("prompt_tokens", "completion_tokens")):
                controller.store.event(task.task_id, "usage", {"phase": "opencode", "model": usage.get("model", "unknown"), "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "cached_prompt_tokens": usage.get("cached_prompt_tokens"), "token_source": "opencode_event", "elapsed": usage.get("elapsed", 0)})
        return {"status": "RECORDED"}

    def summary(self, args: dict[str, Any]) -> dict[str, Any]:
        result = self.status(args)
        if result.get("status") == "BLOCKED":
            return result
        return {"task_id": result.get("task_id"), "phase": result.get("phase"), "goal": next((json.loads(e["payload"]).get("goal") for e in result.get("events", []) if e["kind"] == "intake"), None), "completion": result.get("completion_facts"), "usage": result.get("usage"), "unresolved": result.get("unresolved")}

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        if op == "start": return self.start(request, str(request.get("session_id") or "unknown"), str(request.get("workspace") or os.getcwd()))
        if op == "before": return self.before(request)
        if op == "after": return self.after(request)
        if op == "reconcile": return self.reconcile(request)
        if op == "evidence": return self.evidence(request)
        if op == "finalize": return self.finalize(request)
        if op == "status": return self.status(request)
        if op == "symbols": return self.symbols(request)
        if op == "event": return self.event(request)
        if op == "summary": return self.summary(request)
        return {"status": "BLOCKED", "reason": f"unknown bridge operation: {op}"}


def main() -> int:
    bridge = SupervisorBridge()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = bridge.handle(request)
        except Exception as exc:  # bridge must never kill the host agent
            response = {"status": "ERROR", "reason": f"{type(exc).__name__}: {exc}"}
        if "id" in request:
            response["id"] = request["id"]
        sys.stdout.write(json.dumps(response, sort_keys=True, default=str) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
