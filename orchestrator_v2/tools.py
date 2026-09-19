from __future__ import annotations

import os
import signal
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    required: tuple[str, ...]


class ToolRegistry:
    """Validate native tool calls before they reach the workspace boundary."""
    SPECS = {
        "read_file": ToolSpec("read_file", ("path",)),
        "write_file": ToolSpec("write_file", ("path", "content")),
        "run_tests": ToolSpec("run_tests", ()),
    }

    @classmethod
    def schemas(cls) -> list[dict]:
        """Return the only native tools the model is allowed to call."""
        return [
            {"type": "function", "function": {"name": "read_file", "description": "Read one permitted UTF-8 source file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "write_file", "description": "Write one permitted UTF-8 source file after inspecting it.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "run_tests", "description": "Run the coordinator-selected bounded test command.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
        ]

    def dispatch(self, tools: "WorkspaceTools", name: str, arguments: dict):
        spec = self.SPECS.get(name)
        if not spec or any(k not in arguments for k in spec.required):
            raise ToolError(f"unsupported or malformed tool call: {name}")
        if name == "read_file": return tools.read_file(arguments["path"])
        if name == "write_file": return tools.write_file(arguments["path"], arguments["content"])
        if arguments:
            raise ToolError("run_tests takes no model-supplied command")
        return tools.run_tests()


class WorkspaceTools:
    """Small allowlisted tool boundary for a disposable workspace."""

    def __init__(self, root: Path, permitted_files: list[str] | None = None, permitted_actions: list[str] | None = None, test_command: list[str] | None = None):
        self.root = root.resolve()
        self.permitted_files = {Path(p).as_posix() for p in (permitted_files or [])}
        self.permitted_actions = set(permitted_actions or {"read_file", "write_file", "run_tests"})
        self.test_command = list(test_command) if test_command else None

    def _action(self, name: str) -> None:
        if name not in self.permitted_actions:
            raise ToolError(f"action is not permitted: {name}")

    def _file(self, rel: str, action: str = "read_file") -> Path:
        self._action(action)
        path = self._path(rel)
        normalized = path.relative_to(self.root).as_posix()
        if self.permitted_files and normalized not in self.permitted_files:
            raise ToolError(f"file is not permitted: {normalized}")
        return path

    def _path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError("path escapes workspace")
        return p

    def read_file(self, rel: str, max_bytes: int = 100_000) -> str:
        p = self._file(rel)
        data = p.read_bytes()
        return data[:max_bytes].decode("utf-8", "replace")

    def write_file(self, rel: str, content: str) -> dict:
        p = self._file(rel, "write_file"); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return {"path": rel, "bytes": len(content.encode())}

    def run_tests(self, command: list[str] | None = None, timeout: int = 60) -> dict:
        self._action("run_tests")
        command = command or self.test_command or ["python", "-m", "unittest", "discover", "-v"]
        requested_command = list(command)
        allowed = {"python", "python3", "pytest"}
        if not command or Path(command[0]).name not in allowed:
            raise ToolError("only python/pytest commands are allowed")
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONPATH": "/workspace"}
        sandbox = shutil.which("bwrap")
        if sandbox:
            command = [sandbox, "--die-with-parent", "--unshare-all", "--ro-bind", "/usr", "/usr", "--ro-bind", "/usr/local", "/usr/local", "--ro-bind", "/bin", "/bin", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64", "--ro-bind", "/etc", "/etc", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--bind", str(self.root), "/workspace", "--chdir", "/workspace", "--clearenv", "--setenv", "PATH", env["PATH"], "--setenv", "PYTHONPATH", env["PYTHONPATH"], *command]
        proc = subprocess.Popen(command, cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM); out, _ = proc.communicate()
            return {"command": requested_command, "exit_code": None, "timed_out": True, "sandboxed": bool(sandbox), "output": out[-20_000:]}
        return {"command": requested_command, "exit_code": proc.returncode, "timed_out": False, "sandboxed": bool(sandbox), "output": out[-20_000:]}
