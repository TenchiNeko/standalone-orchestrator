"""Local JSONL bridge used by the project-local OpenCode plugin.

The bridge is intentionally a small adapter around the existing Python core;
it does not run a second agent loop or expose a network listener.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from orchestrator_v2.controller import Controller
    from orchestrator_v2.cross_run import LoopMemory
    from orchestrator_v2.state import Criterion, Phase, Task, TaskContract, source_hash, task_scope
    from orchestrator_v2.symbols import SymbolIndex
    from orchestrator_v2.usage import usage_summary
else:
    from .controller import Controller
    from .cross_run import LoopMemory
    from .state import Criterion, Phase, Task, TaskContract, source_hash, task_scope
    from .symbols import SymbolIndex
    from .usage import usage_summary


FILE_KEYS = ("filePath", "file_path", "path", "file")
# These host tools inspect repository/session context but do not mutate the
# task workspace.  Keep discovery usable before a contract exists; unknown
# tools remain fail-closed in strict mode.
READ_TOOLS = {"read", "grep", "glob", "ls", "list", "skill", "webfetch", "question", "todoread", "todowrite"}
# AgentMemory's local policy exposes these retrieval operations as read-only.
# Writes, lesson saves, compression, exports, and other memory operations stay
# blocked because the bridge cannot treat them as project evidence.
MEMORY_READ_TOOLS = {"memory_recall", "memory_smart_search", "memory_sessions", "memory_lesson_recall"}
WRITE_TOOLS = {"write", "edit", "apply_patch", "patch"}
SHELL_TOOLS = {"bash", "shell", "terminal", "run"}
SHELL_META = re.compile(r"[;&|<>$`(){}*?\[\]\n\r]")
BRIDGE_VERSION = "opencode-supervisor-v1"
MODEL_SCOPE_SAMPLE_LIMIT = 40
MODEL_SCOPE_EXPANSION_LIMIT = 20
MODEL_BLOCKER_LIMIT = 12
MODEL_GOAL_LIMIT = 2000


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()[:24]


def _output_hash(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", "replace")).hexdigest()


def _same_file_state(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    """Compare observable file facts without treating absolute/relative paths as state changes."""
    if not left or not right:
        return left == right
    return all(left.get(key) == right.get(key) for key in ("exists", "sha256", "bytes"))


def _command_args(value: Any) -> list[str] | None:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    if not isinstance(value, str) or not value.strip() or SHELL_META.search(value):
        return None
    try:
        return shlex.split(value, posix=True)
    except ValueError:
        return None


def _normal_command_args(value: Any) -> list[str] | None:
    """Parse one argv-shaped command; reject shell composition entirely."""
    args = _command_args(value)
    if not args:
        return None
    if SHELL_META.search(" ".join(args)):
        return None
    args[0] = Path(args[0]).name
    return args


def _exact_test_command(command: Any, configured: list[str] | None) -> bool:
    actual = _normal_command_args(command)
    expected = _normal_command_args(configured)
    return bool(actual and expected and actual == expected)


def _looks_like_verification(command: Any) -> bool:
    """Recognize likely verification commands without restricting light mode.

    This is evidence classification only.  The host remains free to execute
    any normal command in light mode; an exit-0 `ls` or `echo` is not a test.
    """
    args = _normal_command_args(command) if isinstance(command, (str, list)) else None
    if not args:
        return False
    exe, rest = args[0], args[1:]
    if exe in {"pytest", "unittest", "go"} and (exe != "go" or "test" in rest):
        return True
    if exe in {"npm", "npx", "yarn", "pnpm", "cargo", "mvn", "gradle", "make"}:
        return any(part in {"test", "check", "verify", "lint", "typecheck"} for part in rest)
    if exe in {"python", "python3"}:
        return any(part in {"pytest", "unittest"} or "test" in Path(part).name.lower() for part in rest)
    return exe in {"ruff", "mypy", "pyright", "flake8", "eslint", "tsc", "cmake"} and any(part in {"check", "test", "lint", "build"} for part in rest)


def _safe_read_path(root: Path, value: str, base: Path | None = None) -> bool:
    """Resolve a shell read path and require it to remain inside root."""
    try:
        anchor = (base or root).resolve()
        candidate = (anchor / value).resolve() if not Path(value).expanduser().is_absolute() else Path(value).expanduser().resolve()
        candidate.relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _readonly_shell_command(command: Any, root: Path, base: Path | None = None) -> bool:
    """Recognize a deliberately small, single-argv inspection subset.

    This is intentionally an allowlist, not a shell-safety heuristic.  The
    host still executes the command; v2 only authorizes these exact shapes.
    """
    args = _normal_command_args(command)
    if not args:
        return False
    command_name, rest = args[0], args[1:]
    if command_name == "pwd":
        return rest in ([], ["-P"])
    if command_name == "ls":
        options = {"-a", "-l", "-la", "-al"}
        if rest and rest[0] in options:
            rest = rest[1:]
        return len(rest) <= 1 and (not rest or _safe_read_path(root, rest[0], base))
    if command_name == "stat":
        return len(rest) == 1 and _safe_read_path(root, rest[0], base)
    if command_name in {"head", "tail"}:
        return len(rest) == 1 and _safe_read_path(root, rest[0], base)
    if command_name == "wc":
        return len(rest) == 2 and rest[0] in {"-l", "-c", "-w"} and _safe_read_path(root, rest[1], base)
    return False


def _command_from_args(args: dict[str, Any]) -> Any:
    return args.get("command") if "command" in args else args.get("cmd")


def _tool_kind(tool: str, args: dict[str, Any], task: Task | None = None) -> str:
    name = str(tool or "").lower()
    if name.startswith("orchestrator_"):
        return "supervisor"
    if name in MEMORY_READ_TOOLS:
        return "memory_read"
    if name in READ_TOOLS:
        return "read_file"
    if name in WRITE_TOOLS:
        return "write_file"
    if name in SHELL_TOOLS:
        if task and _exact_test_command(_command_from_args(args), task.contract.test_command):
            return "run_tests"
        if task:
            root = Path(task.workspace).resolve()
            workdir = Path(str(args.get("workdir") or args.get("cwd") or root)).expanduser()
            base = workdir if workdir.is_absolute() else root / workdir
            if _readonly_shell_command(_command_from_args(args), root, base):
                return "shell_readonly"
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
        self.policy_mode = os.environ.get("V2_POLICY_MODE", "strict").strip().lower() or "strict"
        if self.policy_mode not in {"strict", "light"}:
            self.policy_mode = "strict"
        self.light_mode = self.policy_mode == "light"
        self.require_contract = os.environ.get("V2_SUPERVISOR_REQUIRE_CONTRACT") == "1"
        self.allow_broad_workspace = os.environ.get("V2_SUPERVISOR_ALLOW_BROAD_WORKSPACE") == "1"
        self.sessions: dict[str, str] = {}
        self.tasks: dict[str, Controller] = {}
        self.pending: dict[str, dict[str, Any]] = {}
        self.usage_seen: set[str] = set()

    def _controller(self, workspace: Path, state_dir: Path | None = None) -> Controller:
        if state_dir is None:
            digest = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:16]
            state_dir = self.default_state_root / digest
        key = str(state_dir.resolve())
        if key not in self.tasks:
            self.tasks[key] = Controller(Path(key))
        return self.tasks[key]

    @staticmethod
    def _scope(task: Task) -> list[str] | None:
        return task_scope(task)

    @staticmethod
    def _model_scope(scope: list[str]) -> tuple[list[str], int, bool]:
        """Return a bounded model-facing scope while retaining full state internally."""
        normalized = [Path(item).as_posix() for item in scope]
        if len(normalized) <= MODEL_SCOPE_SAMPLE_LIMIT:
            return normalized, len(normalized), False
        head = MODEL_SCOPE_SAMPLE_LIMIT // 2
        tail = MODEL_SCOPE_SAMPLE_LIMIT - head
        return normalized[:head] + normalized[-tail:], len(normalized), True

    def _track_path(self, task: Task, path: str | None, reason: str) -> str | None:
        if not path:
            return None
        root = Path(task.workspace).resolve()
        rel = self._relative_path(root, path)
        if rel is None:
            return None
        if "*" not in task.contract.permitted_files:
            return rel
        if rel not in task.scope_files:
            advisory = None
            if task.scope_files:
                advisory = "new path is outside the initial task hint; confirm its relationship to the objective"
            task.scope_files.append(rel)
            found = self._task(task_id=task.task_id)
            controller = found[0] if found else self._controller(root)
            controller.store.save_task(task)
            payload = {"path": rel, "reason": reason, "mode": "light"}
            if advisory:
                payload["advisory"] = advisory
            controller.store.event(task.task_id, "scope_expanded", payload)
        return rel

    def _seed_light_scope(self, task: Task, controller: Controller) -> None:
        """Capture a one-time fallback scope before an unobserved verification.

        Normal sessions populate scope through reads/writes first.  This narrow
        fallback keeps a test run fresh-safe when the model verifies immediately
        after intake, while excluding the bridge's own state database and common
        generated trees from the fingerprint.
        """
        if not self.light_mode or task.scope_files or "*" not in task.contract.permitted_files:
            return
        root = Path(task.workspace).resolve()
        state_dir = controller.state_dir.resolve()
        ignored = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"}
        task.scope_files = [
            p.relative_to(root).as_posix()
            for p in sorted(root.rglob("*"))
            if p.is_file() and not any(part in ignored for part in p.relative_to(root).parts) and state_dir not in p.parents and not p.name.endswith((".sqlite3", "-wal", "-shm")) and p.suffix not in {".db", ".sqlite"}
        ]
        controller.store.save_task(task)
        controller.store.event(task.task_id, "scope_seeded", {"count": len(task.scope_files), "reason": "verification started before a path was observed", "mode": "light"})

    @staticmethod
    def _git_status(root: Path) -> list[str] | None:
        if not (root / ".git").exists():
            return None
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "-c", "core.quotepath=false", "status", "--porcelain=v1", "-z"],
                check=False, capture_output=True, text=False,
                env={"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1", "GIT_OPTIONAL_LOCKS": "0"},
                timeout=3,
            )
            if result.returncode != 0:
                return None
            entries = [item for item in result.stdout.decode("utf-8", "replace").split("\0") if item]
            return [entry[3:] if len(entry) >= 3 else entry for entry in entries]
        except (OSError, subprocess.SubprocessError):
            return None

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
    def _workspace_path(root: Path, path: str) -> Path:
        candidate = Path(path).expanduser()
        return (candidate if candidate.is_absolute() else root / candidate).resolve()

    @staticmethod
    def _relative_path(root: Path, path: str) -> str | None:
        try:
            return SupervisorBridge._workspace_path(root, path).relative_to(root).as_posix()
        except ValueError:
            return None

    @staticmethod
    def _test_or_harness(path: str) -> bool:
        normalized = Path(path).as_posix().lstrip("./")
        name = Path(normalized).name.lower()
        return normalized.startswith(("tests/", "test/", ".tests/", "__tests__/")) or name.startswith("test_") or name.endswith("_test.py") or name in {"conftest.py", "pytest.ini", "tox.ini"}

    @staticmethod
    def _permitted(task: Task, path: str | None) -> bool:
        if not path:
            return True
        root = Path(task.workspace).resolve()
        target = SupervisorBridge._workspace_path(root, path)
        try:
            rel = target.relative_to(root).as_posix()
        except ValueError:
            return False
        permitted = task.contract.permitted_files
        return "*" in permitted or rel in {Path(item).as_posix().lstrip("./") for item in permitted}

    @staticmethod
    def _file_state(task: Task, path: str | None) -> dict[str, Any] | None:
        if not path:
            return None
        root = Path(task.workspace).resolve()
        target = SupervisorBridge._workspace_path(root, path)
        if target != root and root not in target.parents:
            return {"path": path, "error": "path escapes workspace"}
        if not target.exists():
            return {"path": path, "exists": False, "sha256": None, "bytes": 0}
        data = target.read_bytes()
        return {"path": path, "exists": True, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    def start(self, args: dict[str, Any], session_id: str, workspace: str) -> dict[str, Any]:
        root = Path(workspace).resolve()
        broad_reason = self._broad_workspace_reason(root)
        if broad_reason:
            return {"status": "BLOCKED", "reason": broad_reason}
        existing = self._task(session_id=session_id)
        if existing:
            controller, active = existing
            if self.light_mode and active.phase == Phase.COMPLETE:
                controller.store.event(active.task_id, "task_released", {"reason": "new explicit orchestrator_start after COMPLETE", "mode": "light"})
                self.sessions.pop(session_id, None)
            else:
                return {"status": "BLOCKED", "reason": f"session already has active task {active.task_id}; call orchestrator_end before starting another"}
        raw_files = list(args.get("permitted_files") or [])
        if self.light_mode and not raw_files:
            raw_files = ["*"]
        permitted_files = raw_files
        if not permitted_files:
            return {"status": "BLOCKED", "reason": "permitted_files is required; v2 will not grant implicit repository-wide write access"}
        if self.require_contract and "*" in permitted_files:
            return {"status": "BLOCKED", "reason": "strict supervision rejects repository-wide wildcard permissions; list exact source files"}
        normalized_files: list[str] = []
        initial_scope: list[str] = []
        for item in permitted_files:
            if not isinstance(item, str) or not item.strip():
                return {"status": "BLOCKED", "reason": "permitted_files must contain non-empty paths"}
            if item == "*":
                normalized_files.append(item)
                continue
            rel = self._relative_path(root, item)
            if rel is None:
                return {"status": "BLOCKED", "reason": f"permitted file is outside workspace: {item}"}
            if self.require_contract and self._test_or_harness(rel):
                return {"status": "BLOCKED", "reason": f"strict supervision does not grant write access to test/harness file: {rel}"}
            normalized_files.append(rel)
            initial_scope.append(rel)
        permitted_files = ["*"] if self.light_mode else normalized_files
        if self.light_mode:
            # Actions are telemetry in light mode, never an authorization ACL.
            # Keep the stable vocabulary in persisted state even when a model
            # supplies an incomplete or host-specific list.
            actions = ["read_file", "write_file", "run_tests", "browser"]
        else:
            actions = list(args.get("permitted_actions") or ["read_file", "write_file", "run_tests"])
        criteria = [Criterion(str(item["key"]), str(item["description"]), bool(item.get("required", True))) for item in args.get("criteria", []) if isinstance(item, dict) and item.get("key") and item.get("description")]
        try:
            test_command = args.get("test_command")
            test_command = _normal_command_args(test_command) if test_command is not None else None
            if args.get("test_command") is not None and not test_command:
                raise ValueError("test_command must be one shell-free argv command")
            contract = TaskContract(str(args.get("goal") or ""), permitted_files, actions, criteria, int(args.get("version", 1)), test_command, bool(args.get("visual_required", False)))
            contract.validate()
        except (KeyError, TypeError, ValueError) as exc:
            return {"status": "BLOCKED", "reason": f"invalid task contract: {exc}"}
        state_dir = Path(args["state_dir"]).resolve() if args.get("state_dir") else None
        controller = self._controller(root, state_dir)
        task = controller.intake(contract, root, int(args.get("budget_limit", 48 if self.light_mode else 12)))
        task.scope_files = initial_scope if self.light_mode else []
        task.max_model_seconds = float(args["max_model_seconds"]) if args.get("max_model_seconds") is not None else None
        controller.store.save_task(task)
        self.sessions[session_id] = task.task_id
        controller.store.event(task.task_id, "opencode_session", {"session_id": session_id, "workspace": str(root), "host": "opencode", "policy_mode": self.policy_mode})
        return {"status": "STARTED", "task_id": task.task_id, "workspace": str(root), "phase": task.phase.value, "policy_mode": self.policy_mode}

    def _broad_workspace_reason(self, root: Path) -> str | None:
        if self.allow_broad_workspace:
            return None
        home = Path.home().resolve()
        if root == Path("/"):
            return "strict supervision refuses / as a coding workspace; start from the intended project directory"
        if root == home:
            return f"strict supervision refuses {home} as a coding workspace; start from the intended project directory"
        return None

    def before(self, args: dict[str, Any]) -> dict[str, Any]:
        found = self._task(args.get("session_id"), args.get("task_id"))
        if not found:
            tool = str(args.get("tool") or "")
            kind = _tool_kind(tool, args.get("args") if isinstance(args.get("args"), dict) else {})
            if self.light_mode:
                return {"decision": "ALLOW", "reason": "light supervision observes normal coding actions without preauthorization"}
            if kind == "supervisor" or kind in {"read_file", "memory_read"} or tool.lower() in READ_TOOLS:
                return {"decision": "ALLOW", "reason": "read-only discovery is allowed before a contract"}
            if self.require_contract:
                return {"decision": "BLOCK", "reason": "strict supervision requires orchestrator_start before mutations, tests, or shell commands"}
            return {"decision": "WARN", "reason": "no v2 task is associated with this OpenCode session"}
        controller, task = found
        tool = str(args.get("tool") or "")
        tool_args = args.get("args") if isinstance(args.get("args"), dict) else {}
        kind = _tool_kind(tool, tool_args, task)
        path = _path_arg(tool_args)
        # Supervisor tools are the control plane itself; they are not project
        # actions and must remain callable after a task starts.
        if kind == "supervisor":
            event_kind = "supervisor_refresh" if tool == "orchestrator_status" else "opencode_before"
            controller.store.event(task.task_id, event_kind, {"tool": tool, "kind": kind, "decision": "ALLOW"})
            return {"decision": "ALLOW"}
        if task.phase == Phase.COMPLETE and kind not in {"read_file", "memory_read"}:
            controller.store.event(task.task_id, "opencode_before", {"tool": tool, "kind": kind, "decision": "BLOCK", "reason": "task is already COMPLETE"})
            return {"decision": "BLOCK", "reason": "task is COMPLETE; start a new contract before another test or mutation"}
        workdir = tool_args.get("workdir") or tool_args.get("cwd")
        if workdir:
            root = Path(task.workspace).resolve()
            candidate = Path(str(workdir)).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            try:
                candidate.resolve().relative_to(root)
            except ValueError:
                controller.store.event(task.task_id, "opencode_before", {"tool": tool, "kind": kind, "decision": "BLOCK", "reason": "working directory is outside workspace"})
                return {"decision": "BLOCK", "reason": "working directory is outside the task workspace"}
        root = Path(task.workspace).resolve()
        if self.light_mode and kind in {"shell", "run_tests"}:
            self._seed_light_scope(task, controller)
        if not self.light_mode and kind == "shell_readonly":
            # Safe shell inspection uses the existing read_file contract
            # action; it never becomes a mutation or test action.
            base = Path(str(tool_args.get("workdir") or tool_args.get("cwd") or root)).expanduser()
            if not base.is_absolute():
                base = root / base
            if not _readonly_shell_command(_command_from_args(tool_args), root, base) or "read_file" not in task.contract.permitted_actions:
                controller.store.event(task.task_id, "opencode_before", {"tool": tool, "kind": kind, "decision": "BLOCK", "reason": "shell command is not an audited read-only inspection"})
                return {"decision": "BLOCK", "reason": "shell command is not an audited read-only inspection"}
        if not self.light_mode and (kind == "shell" or kind not in set(task.contract.permitted_actions) | READ_TOOLS | WRITE_TOOLS | {"memory_read", "shell_readonly"}):
            controller.store.event(task.task_id, "opencode_before", {"tool": tool, "kind": kind, "decision": "BLOCK", "reason": "tool/action is not in the task contract"})
            return {"decision": "BLOCK", "reason": f"{tool} is not an allowed bounded v2 action"}
        if not self.light_mode and kind == "run_tests" and ("run_tests" not in task.contract.permitted_actions or not _exact_test_command(_command_from_args(tool_args), task.contract.test_command)):
            controller.store.event(task.task_id, "opencode_before", {"tool": tool, "kind": kind, "decision": "BLOCK", "reason": "only the exact configured test command is evidence-producing"})
            return {"decision": "BLOCK", "reason": "test command does not exactly match the contract"}
        if not self.light_mode and kind == "write_file" and not path:
            return {"decision": "BLOCK", "reason": "state-changing tool did not provide a file path"}
        if path and self._relative_path(Path(task.workspace).resolve(), path) is None:
            controller.store.event(task.task_id, "opencode_before", {"tool": tool, "kind": kind, "decision": "BLOCK", "path": path, "reason": "path escapes workspace"})
            return {"decision": "BLOCK", "reason": "path escapes the task workspace"}
        # Strict contracts constrain mutations.  Light mode records every
        # project-relative path and lets normal task-related work proceed.
        rel_path = self._track_path(task, path, "observed tool path") if self.light_mode else None
        if not self.light_mode and kind == "write_file" and not self._permitted(task, path):
            controller.store.event(task.task_id, "opencode_before", {"tool": tool, "kind": kind, "decision": "BLOCK", "path": path, "reason": "path is outside permitted_files"})
            return {"decision": "BLOCK", "reason": f"path is not permitted: {path}"}
        if not self.light_mode and kind == "write_file" and controller.store.unresolved_mutations(task.task_id):
            return {"decision": "BLOCK", "reason": "an earlier mutation outcome is unresolved; reconcile it before another write"}
        state = source_hash(root, self._scope(task))
        fingerprint = _json_hash({"tool": tool, "kind": kind, "args": tool_args, "state": state})
        event_rows = controller.store.events(task.task_id)
        last_refresh = max((row["id"] for row in event_rows if row["kind"] == "supervisor_refresh"), default=0)
        prior = [json.loads(row["payload"]) for row in event_rows if row["id"] > last_refresh and row["kind"] == "opencode_before" and json.loads(row["payload"]).get("action_fingerprint") == fingerprint]
        duplicate = bool(prior)
        if duplicate and kind == "write_file":
            # A confirmed failure after readback is a new decision point; an
            # acknowledged write at the same state remains a redundant retry.
            results = [json.loads(row["payload"]) for row in controller.store.events(task.task_id) if row["kind"] == "mutation_result" and json.loads(row["payload"]).get("idempotency_key") == fingerprint]
            duplicate = not any(result.get("status") == "failed" for result in results)
        if duplicate:
            reason = "same action and relevant source state already occurred without a new state fingerprint"
            if self.light_mode:
                repeat_count = len(prior)
                if repeat_count >= 3:
                    controller.store.event(task.task_id, "no_progress", {"tool": tool, "action_fingerprint": fingerprint, "stage": "circuit_breaker", "reason": reason})
                    return {"decision": "BLOCK", "reason": "light supervisor circuit breaker: repeated action made no progress; refresh the plan before retrying", "action_fingerprint": fingerprint}
                stage = "nudge" if repeat_count == 1 else "plan_refresh"
                warning = "The previous approach has repeated without new evidence; re-evaluate the plan before retrying the same action."
                controller.store.event(task.task_id, "no_progress", {"tool": tool, "action_fingerprint": fingerprint, "stage": stage, "reason": reason})
                controller.store.event(task.task_id, "opencode_before", {"tool": tool, "kind": kind, "decision": "ALLOW", "reason": reason, "action_fingerprint": fingerprint, "state_fingerprint": state})
                before_state = self._file_state(task, path) if kind == "write_file" and path else None
                self.pending[str(args.get("call_id") or fingerprint)] = {"task_id": task.task_id, "controller": controller, "kind": kind, "path": path, "fingerprint": fingerprint, "before": before_state, "loop_warning": warning}
                return {"decision": "ALLOW", "action_fingerprint": fingerprint, "warning": warning, "loop_stage": stage}
            decision = "BLOCK" if kind == "write_file" else "WARN"
            return {"decision": decision, "reason": reason, "action_fingerprint": fingerprint}
        before_state = self._file_state(task, path) if kind == "write_file" else None
        workspace_before = self._git_status(root) if self.light_mode and kind == "shell" else None
        if kind == "write_file":
            controller.store.mutation_intent(task.task_id, "opencode:" + tool, fingerprint, {"path": path, "before": before_state, "session_id": args.get("session_id")})
        controller.store.event(task.task_id, "opencode_before", {"tool": tool, "kind": kind, "decision": "ALLOW", "path": path, "action_fingerprint": fingerprint, "state_fingerprint": state})
        self.pending[str(args.get("call_id") or fingerprint)] = {"task_id": task.task_id, "controller": controller, "kind": kind, "path": path, "fingerprint": fingerprint, "before": before_state, "workspace_before": workspace_before}
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
            if status == "unknown" and self.light_mode:
                status = "acknowledged" if not _same_file_state(after_state, pending["before"]) else "failed"
                result["status"] = status
                result["reconciled"] = True
            controller.store.mutation_ack(task.task_id, pending["fingerprint"], status, result)
            if self.light_mode:
                self._track_path(task, pending["path"], "observed mutation")
        if pending and pending["kind"] == "shell" and self.light_mode:
            root = Path(task.workspace).resolve()
            after_files = self._git_status(root)
            before_files = pending.get("workspace_before")
            changed_files = sorted(set(after_files or []) - set(before_files or [])) if after_files is not None and before_files is not None else []
            for changed in changed_files:
                self._track_path(task, changed, "shell command changed repository state")
            result["command"] = _command_from_args(args.get("args", {}) if isinstance(args.get("args"), dict) else {})
            result["changed_files"] = changed_files
            result["workspace_state_observed"] = after_files is not None
            command = result["command"]
            if exit_code is not None and exit_code == 0 and _looks_like_verification(command):
                ref = controller.store.evidence(task.task_id, "test", root, {"command": command, "exit_code": exit_code, "timed_out": bool(metadata.get("timed_out")), "output_sha256": result["output_sha256"], "source": "opencode", "classification": "light-observed-verification"}, self._scope(task))
                result.update({"evidence": "recorded", "evidence_ref": ref})
        if pending and pending["kind"] == "run_tests":
            command = _command_from_args(args.get("args", {}) if isinstance(args.get("args"), dict) else {})
            evidence_allowed = _exact_test_command(command, task.contract.test_command) if not self.light_mode else (exit_code is not None and (_exact_test_command(command, task.contract.test_command) or _looks_like_verification(command)))
            if evidence_allowed and exit_code is not None:
                ref = controller.store.evidence(task.task_id, "test", Path(task.workspace), {"command": command, "exit_code": exit_code, "timed_out": bool(metadata.get("timed_out")), "output_sha256": result["output_sha256"], "source": "opencode"}, self._scope(task))
                result.update({"evidence": "recorded", "evidence_ref": ref})
            else:
                result["evidence"] = "not_recorded: command or exit status was not authoritative"
        if pending and pending.get("loop_warning"):
            result["loop_warning"] = pending["loop_warning"]
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
        target_path = self._workspace_path(Path(task.workspace).resolve(), path)
        for item in unresolved:
            intent = item.get("intent", {})
            details = intent.get("details", {})
            detail_path = details.get("path")
            if not detail_path or self._workspace_path(Path(task.workspace).resolve(), str(detail_path)) != target_path or not current or current.get("error"):
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
        ref = controller.store.evidence(task.task_id, kind, Path(task.workspace), payload, self._scope(task))
        return {"status": "RECORDED", "ref": ref, "kind": kind}

    def finalize(self, args: dict[str, Any]) -> dict[str, Any]:
        found = self._task(args.get("session_id"), args.get("task_id"))
        if not found:
            return {"status": "BLOCKED", "reason": "no task is associated with this session"}
        controller, task = found
        current = controller.store.current_successful_check(task.task_id, Path(task.workspace), self._scope(task))
        if current:
            configured_test = _normal_command_args(task.contract.test_command) or []
            configured_text = " ".join(configured_test)
            for criterion in task.contract.criteria:
                # A criterion explicitly named as a test criterion is
                # satisfied only by the observed exact-command evidence.
                # Other criteria remain pending until their own evidence is
                # recorded; model prose cannot broaden this mapping.
                description = str(criterion.description or "")
                test_named = criterion.key.lower() in {"test", "tests", "required_tests", "test_command"} or criterion.key.lower().startswith("test")
                # A non-test-shaped key is still eligible only when its
                # description contains the exact configured argv.  This lets
                # ordinary contracts use names such as ``all_tests_pass``
                # without turning arbitrary prose into evidence authority.
                test_described = bool(configured_text and configured_text in description)
                if criterion.required and ((test_named or test_described) or (self.light_mode and current.get("kind") == "test")) and criterion.status != "verified":
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
        return {"status": "OK", "candidates": SymbolIndex(Path(task.workspace), self._scope(task)).render(str(args.get("query") or task.contract.goal), limit=min(50, max(1, int(args.get("limit", 20)))))}

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
        found = self._task(args.get("session_id"), args.get("task_id"))
        if not found:
            return {"status": "NO_TASK", "strict_mode": self.require_contract, "policy_mode": self.policy_mode}
        controller, task = found
        root = Path(task.workspace)
        gate = controller.completion_gate(task, root, review_ok=True)
        usage = usage_summary(controller.store, task.task_id)
        events = controller.store.events(task.task_id)
        counts = {"reads": 0, "writes": 0, "tests": 0, "shell": 0, "repeated": 0, "no_progress": 0}
        goal = task.contract.goal
        loop_warnings: list[dict[str, Any]] = []
        scope_expansions: list[str] = []
        drift_warnings: list[dict[str, Any]] = []
        for row in events:
            try:
                payload = json.loads(row["payload"])
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            if row["kind"] == "tool_call":
                name = payload.get("name")
                if name in {"read_file", "shell_readonly", "memory_read"}: counts["reads"] += 1
                elif name == "write_file": counts["writes"] += 1
                elif name == "run_tests": counts["tests"] += 1
                elif name == "shell": counts["shell"] += 1
            elif row["kind"] == "no_progress": counts["no_progress"] += 1
            elif row["kind"] == "opencode_before" and payload.get("reason", "").startswith("same action"):
                counts["repeated"] += 1
            elif row["kind"] == "intake": goal = payload.get("goal", goal)
            elif row["kind"] == "scope_expanded" and payload.get("path"):
                scope_expansions.append(str(payload["path"]))
                if payload.get("advisory"):
                    drift_warnings.append({"path": payload["path"], "warning": payload["advisory"]})
            elif row["kind"] == "opencode_after" and payload.get("loop_warning"):
                loop_warnings.append({"stage": "nudge", "warning": payload["loop_warning"]})
        blockers = []
        if not gate.get("allowed"): blockers.append(gate.get("reason"))
        pending = [c.key for c in task.contract.criteria if c.required and c.status != "verified"]
        if pending:
            blockers.append("required criteria remain unverified")
            gate = {**gate, "allowed": False, "reason": "required criteria remain unverified", "pending": pending}
        blockers.extend(task.unresolved)
        remaining_budget = max(0, task.budget_limit - task.budget_calls)
        budget_warning = []
        if self.light_mode and remaining_budget <= max(3, task.budget_limit // 8):
            budget_warning.append("light supervisor budget is nearing exhaustion; summarize progress before another retry")
        raw_scope = self._scope(task) or []
        display_scope, scope_count, scope_truncated = self._model_scope(raw_scope)
        unique_blockers = list(dict.fromkeys(str(x) for x in blockers if x))
        return {
            "status": "OK",
            "task_id": task.task_id,
            "workspace": str(Path(task.workspace).resolve()),
            "phase": task.phase.value,
            "goal": str(goal)[:MODEL_GOAL_LIMIT],
            # Strict mode exposes its immutable ACL; light mode exposes the
            # observed scope as telemetry rather than an authorization list.
            "permitted_files": display_scope,
            "permitted_files_count": scope_count,
            "permitted_files_truncated": scope_truncated,
            "scope_mode": "dynamic" if self.light_mode else "immutable",
            "scope_expansions": sorted(dict.fromkeys(scope_expansions))[-MODEL_SCOPE_EXPANSION_LIMIT:],
            "scope_expansions_count": len(set(scope_expansions)),
            "drift_warnings": drift_warnings[-5:],
            "permitted_actions": list(task.contract.permitted_actions),
            "test_command": list(task.contract.test_command or []),
            "visual_required": task.contract.visual_required,
            "budget_limit": task.budget_limit,
            "budget_calls": task.budget_calls,
            "remaining_budget": remaining_budget,
            "criteria": [{"key": c.key, "status": c.status} for c in task.contract.criteria],
            "completion": {"allowed": gate.get("allowed"), "reason": gate.get("reason"), "pending": gate.get("pending", []), "unresolved_mutations": len(gate.get("unresolved_mutations", []))},
            "blockers": unique_blockers[:MODEL_BLOCKER_LIMIT],
            "blockers_truncated": len(unique_blockers) > MODEL_BLOCKER_LIMIT,
            "counts": counts,
            "loop_warnings": loop_warnings[-5:],
            "warnings": budget_warning,
            "usage": usage,
            "strict_mode": self.require_contract,
            "policy_mode": self.policy_mode,
        }

    def health(self, args: dict[str, Any]) -> dict[str, Any]:
        session_id = str(args.get("session_id") or "")
        found = self._task(session_id=session_id)
        workspace = str(Path(str(args.get("workspace") or (found[1].workspace if found else os.getcwd()))).resolve())
        broad_reason = self._broad_workspace_reason(Path(workspace))
        if broad_reason:
            return {"status": "BLOCKED", "bridge_alive": True, "version": BRIDGE_VERSION, "session_associated": bool(found), "workspace": workspace, "state_db_reachable": False, "strict_mode": self.require_contract, "policy_mode": self.policy_mode, "reason": broad_reason}
        digest = hashlib.sha256(workspace.encode()).hexdigest()[:16]
        state_dir = self.default_state_root / digest
        reachable = True
        try:
            state_dir.mkdir(parents=True, exist_ok=True)
            controller = self._controller(Path(workspace))
            with controller.store._db() as db:
                db.execute("SELECT 1").fetchone()
        except Exception:
            reachable = False
        return {"status": "OK" if reachable else "ERROR", "bridge_alive": True, "version": BRIDGE_VERSION, "session_associated": bool(found), "task_id": found[1].task_id if found else None, "workspace": workspace, "state_db_reachable": reachable, "strict_mode": self.require_contract, "policy_mode": self.policy_mode}

    def end(self, args: dict[str, Any]) -> dict[str, Any]:
        session_id = str(args.get("session_id") or "")
        found = self._task(session_id=session_id, task_id=args.get("task_id"))
        if not found:
            return {"status": "BLOCKED", "reason": "no task is associated with this session"}
        controller, task = found
        reason = str(args.get("reason") or "explicit hosted task end")
        controller.store.event(task.task_id, "opencode_session_end", {"session_id": session_id, "reason": reason, "phase": task.phase.value})
        if session_id:
            self.sessions.pop(session_id, None)
        return {"status": "ENDED", "task_id": task.task_id, "phase": task.phase.value, "reason": reason}

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
        if op == "health": return self.health(request)
        if op in {"end", "cancel"}: return self.end(request)
        return {"status": "BLOCKED", "reason": f"unknown bridge operation: {op}"}


def main() -> int:
    bridge = SupervisorBridge()
    for line in sys.stdin:
        request: dict[str, Any] = {}
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
