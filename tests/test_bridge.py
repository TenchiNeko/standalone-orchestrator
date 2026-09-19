import tempfile
import unittest
from pathlib import Path

from orchestrator_v2.bridge import SupervisorBridge


class BridgeTests(unittest.TestCase):
    def _start(self, root: Path) -> tuple[SupervisorBridge, str]:
        bridge = SupervisorBridge(root / "state")
        result = bridge.start(
            {
                "state_dir": str(root / "state"),
                "goal": "fix greeting",
                "permitted_files": ["hello.py", "test_hello.py"],
                "permitted_actions": ["read_file", "write_file", "run_tests"],
                "test_command": ["python3", "-m", "unittest", "test_hello.py"],
                "criteria": [{"key": "tests", "description": "tests pass", "required": True}],
            },
            "session",
            str(root),
        )
        self.assertEqual(result["status"], "STARTED")
        return bridge, "session"

    def test_absolute_paths_and_stale_finalize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hello.py").write_text("def greeting(name):\n    return 'hello ' + name\n")
            (root / "test_hello.py").write_text("assert True\n")
            bridge, session = self._start(root)
            read = bridge.before({"session_id": session, "call_id": "r", "tool": "read", "args": {"filePath": str(root / "hello.py")}})
            self.assertEqual(read["decision"], "ALLOW")
            bridge.after({"session_id": session, "call_id": "r", "tool": "read", "args": {}, "output": {"output": "ok", "metadata": {}}})
            before = bridge.before({"session_id": session, "call_id": "w", "tool": "edit", "args": {"filePath": str(root / "hello.py")}})
            self.assertEqual(before["decision"], "ALLOW")
            (root / "hello.py").write_text("def greeting(name):\n    return 'hello, ' + name\n")
            bridge.after({"session_id": session, "call_id": "w", "tool": "edit", "args": {}, "output": {"output": "ok", "metadata": {}}})
            self.assertEqual(bridge.before({"session_id": session, "call_id": "t", "tool": "bash", "args": {"command": "python3 -m unittest test_hello.py"}})["decision"], "ALLOW")
            bridge.after({"session_id": session, "call_id": "t", "tool": "bash", "args": {"command": "python3 -m unittest test_hello.py"}, "output": {"output": "OK", "metadata": {"exit": 0}}})
            self.assertEqual(bridge.finalize({"session_id": session})["status"], "COMPLETE")
            (root / "hello.py").write_text("def greeting(name):\n    return 'changed again'\n")
            stale = bridge.finalize({"session_id": session})
            self.assertEqual(stale["status"], "INCOMPLETE")
            self.assertIn("no current successful check", stale["gate"]["reason"])

    def test_unknown_mutation_requires_readback_before_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hello.py").write_text("x = 1\n")
            (root / "test_hello.py").write_text("assert True\n")
            bridge, session = self._start(root)
            args = {"filePath": str(root / "hello.py"), "content": "x = 2\n"}
            self.assertEqual(bridge.before({"session_id": session, "call_id": "u", "tool": "edit", "args": args})["decision"], "ALLOW")
            unknown = bridge.after({"session_id": session, "call_id": "u", "tool": "edit", "args": args, "ambiguous": True, "output": {"output": "lost", "metadata": {}}})
            self.assertEqual(unknown["status"], "unknown")
            retry = bridge.before({"session_id": session, "call_id": "u2", "tool": "edit", "args": args})
            self.assertEqual(retry["decision"], "BLOCK")
            reconciled = bridge.reconcile({"session_id": session, "path": "hello.py"})
            self.assertEqual(reconciled["remaining"], [])
            self.assertEqual(bridge.before({"session_id": session, "call_id": "u3", "tool": "edit", "args": args})["decision"], "ALLOW")

    def test_supervisor_tools_and_forbidden_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hello.py").write_text("x = 1\n")
            (root / "test_hello.py").write_text("assert True\n")
            bridge, session = self._start(root)
            self.assertEqual(bridge.before({"session_id": session, "tool": "orchestrator_status", "args": {}})["decision"], "ALLOW")
            self.assertEqual(bridge.before({"session_id": session, "tool": "read", "args": {"filePath": str(root / "secret.txt")}})["decision"], "BLOCK")
            self.assertEqual(bridge.before({"session_id": session, "tool": "bash", "args": {"command": "python3 -m unittest", "workdir": "/tmp"}})["decision"], "BLOCK")
            self.assertEqual(bridge.evidence({"session_id": session, "kind": "test", "payload": {"exit_code": 0}})["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
