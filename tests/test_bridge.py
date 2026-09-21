import tempfile
import time
import json
import subprocess
import os
import unittest
from unittest.mock import patch
from pathlib import Path

from orchestrator_v2.bridge import SupervisorBridge
from orchestrator_v2.controller import Controller
from orchestrator_v2.state import Criterion, TaskContract, source_hash


class BridgeTests(unittest.TestCase):
    def _start(self, root: Path) -> tuple[SupervisorBridge, str]:
        bridge = SupervisorBridge(root / "state")
        result = bridge.start(
            {
                "state_dir": str(root / "state"),
                "goal": "fix greeting",
                "permitted_files": ["hello.py"],
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
            self.assertEqual(bridge.before({"session_id": session, "tool": "read", "args": {"filePath": str(root / "test_hello.py")}})["decision"], "ALLOW")
            self.assertEqual(bridge.before({"session_id": session, "tool": "read", "args": {"filePath": "/tmp/secret.txt"}})["decision"], "BLOCK")
            self.assertEqual(bridge.before({"session_id": session, "tool": "bash", "args": {"command": "python3 -m unittest", "workdir": "/tmp"}})["decision"], "BLOCK")
            self.assertEqual(bridge.evidence({"session_id": session, "kind": "test", "payload": {"exit_code": 0}})["status"], "BLOCKED")

    def test_strict_discovery_requires_exact_contract_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"V2_SUPERVISOR_REQUIRE_CONTRACT": "1"}, clear=False):
            root = Path(directory); (root / "hello.py").write_text("x = 1\n")
            bridge = SupervisorBridge(root / "state")
            self.assertEqual(bridge.before({"session_id": "new", "tool": "read", "args": {"filePath": "hello.py"}})["decision"], "ALLOW")
            self.assertEqual(bridge.before({"session_id": "new", "tool": "edit", "args": {"filePath": "hello.py"}})["decision"], "BLOCK")
            wildcard = bridge.start({"goal": "x", "permitted_files": ["*"], "permitted_actions": ["write_file"], "criteria": [{"key": "x", "description": "x"}]}, "other", str(root))
            self.assertEqual(wildcard["status"], "BLOCKED")

    def test_only_exact_test_command_creates_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "hello.py").write_text("x = 1\n")
            bridge, session = self._start(root)
            for command in ("echo test", "python3 -m unittest test_hello.py && echo extra", "python3 -m unittest test_hello.py; echo extra"):
                self.assertEqual(bridge.before({"session_id": session, "call_id": command, "tool": "bash", "args": {"command": command}})["decision"], "BLOCK")
            allowed = bridge.before({"session_id": session, "call_id": "t", "tool": "bash", "args": {"command": "python3 -m unittest test_hello.py"}})
            self.assertEqual(allowed["decision"], "ALLOW")
            bridge.after({"session_id": session, "call_id": "t", "tool": "bash", "args": {"command": "python3 -m unittest test_hello.py"}, "output": {"output": "OK", "metadata": {"exit": 0}}})
            self.assertEqual(len(bridge._task(session)[0].store.evidence_rows(bridge._task(session)[1].task_id, "test")), 1)
            outside = bridge.before({"session_id": session, "call_id": "outside", "tool": "bash", "args": {"command": "python3 -m unittest test_hello.py", "workdir": "/tmp"}})
            self.assertEqual(outside["decision"], "BLOCK")

    def test_test_description_can_bind_exact_command_to_named_criterion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "hello.py").write_text("x = 1\n")
            bridge = SupervisorBridge(root / "state")
            result = bridge.start({
                "goal": "fix",
                "permitted_files": ["hello.py"],
                "permitted_actions": ["run_tests"],
                "test_command": ["python3", "-m", "unittest", "test_hello.py"],
                "criteria": [{"key": "all_tests_pass", "description": "python3 -m unittest test_hello.py passes", "required": True}],
            }, "named", str(root))
            self.assertEqual(result["status"], "STARTED")
            bridge.before({"session_id": "named", "call_id": "t", "tool": "bash", "args": {"command": "python3 -m unittest test_hello.py"}})
            bridge.after({"session_id": "named", "call_id": "t", "tool": "bash", "args": {"command": "python3 -m unittest test_hello.py"}, "output": {"output": "OK", "metadata": {"exit": 0}}})
            self.assertEqual(bridge.finalize({"session_id": "named"})["status"], "COMPLETE")

    def test_relative_absolute_reconciliation_and_changed_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); target = root / "hello.py"; target.write_text("x = 1\n")
            bridge, session = self._start(root)
            args = {"filePath": "hello.py", "content": "x = 2\n"}
            bridge.before({"session_id": session, "call_id": "u", "tool": "edit", "args": args})
            target.write_text("x = 2\n")
            bridge.after({"session_id": session, "call_id": "u", "tool": "edit", "args": args, "ambiguous": True, "output": {"output": "lost", "metadata": {}}})
            reconciled = bridge.reconcile({"session_id": session, "path": str(target)})
            self.assertEqual(reconciled["remaining"], [])
            # A second ambiguity with unchanged bytes is a confirmed failure,
            # and relative lookup must resolve against the task workspace.
            target.write_text("x = 2\n")
            args2 = {"filePath": str(target), "content": "x = 3\n"}
            bridge.before({"session_id": session, "call_id": "u2", "tool": "edit", "args": args2})
            bridge.after({"session_id": session, "call_id": "u2", "tool": "edit", "args": args2, "ambiguous": True, "output": {"output": "lost", "metadata": {}}})
            reconciled2 = bridge.reconcile({"session_id": session, "path": "hello.py"})
            self.assertEqual(reconciled2["remaining"], [])
            self.assertEqual(bridge.before({"session_id": session, "call_id": "outside", "tool": "edit", "args": {"filePath": "/tmp/hello.py"}})["decision"], "BLOCK")

    def test_complete_blocks_followup_mutation_and_end_releases_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "hello.py").write_text("x = 1\n")
            bridge, session = self._start(root)
            bridge.before({"session_id": session, "call_id": "t", "tool": "bash", "args": {"command": "python3 -m unittest test_hello.py"}})
            bridge.after({"session_id": session, "call_id": "t", "tool": "bash", "args": {"command": "python3 -m unittest test_hello.py"}, "output": {"output": "OK", "metadata": {"exit": 0}}})
            self.assertEqual(bridge.finalize({"session_id": session})["status"], "COMPLETE")
            self.assertEqual(bridge.before({"session_id": session, "call_id": "w", "tool": "edit", "args": {"filePath": "hello.py"}})["decision"], "BLOCK")
            self.assertEqual(bridge.end({"session_id": session})["status"], "ENDED")
            self.assertEqual(bridge.health({"session_id": session, "workspace": str(root)})["session_associated"], False)

    def test_intake_hash_is_scoped_to_permitted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "project_source.py").write_text("x = 1\n")
            unrelated = root / "huge-unrelated"
            unrelated.mkdir()
            for index in range(200):
                (unrelated / f"file-{index}.txt").write_text("unrelated\n")
            controller = Controller(root / "state")
            contract = TaskContract("inspect", ["project_source.py"], ["read_file"], [Criterion("inspect", "inspect")])
            calls = []

            def spy(path, permitted=None):
                calls.append((Path(path), permitted))
                return source_hash(path, permitted)

            with patch("orchestrator_v2.controller.source_hash", side_effect=spy):
                started = time.perf_counter()
                controller.intake(contract, root)
                elapsed = time.perf_counter() - started
            self.assertEqual(calls, [(root, ["project_source.py"])])
            self.assertLess(elapsed, 1.0)

    def test_strict_broad_workspace_is_rejected_without_intake_scan(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"V2_SUPERVISOR_REQUIRE_CONTRACT": "1"}, clear=False):
            root = Path(directory)
            bridge = SupervisorBridge(root / "state")
            with patch("orchestrator_v2.bridge.Path.home", return_value=root):
                result = bridge.start({"goal": "x", "permitted_files": ["source.py"], "permitted_actions": ["read_file"], "criteria": [{"key": "x", "description": "x"}]}, "broad", str(root))
                health = bridge.health({"workspace": str(root)})
            self.assertEqual(result["status"], "BLOCKED")
            self.assertIn("refuses", result["reason"])
            self.assertEqual(health["status"], "BLOCKED")

    def test_audited_readonly_shell_is_allowed_but_not_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "hello.py").write_text("x = 1\n")
            bridge, session = self._start(root)
            for index, command in enumerate(("pwd", "ls -la", "ls hello.py", "stat hello.py", "wc -l hello.py")):
                call_id = f"ro-{index}"
                allowed = bridge.before({"session_id": session, "call_id": call_id, "tool": "bash", "args": {"command": command, "workdir": str(root)}})
                self.assertEqual(allowed["decision"], "ALLOW", command)
                bridge.after({"session_id": session, "call_id": call_id, "tool": "bash", "args": {"command": command}, "output": {"output": "inspection", "metadata": {"exit": 0}}})
            controller, task = bridge._task(session)
            self.assertEqual(controller.store.evidence_rows(task.task_id, "test"), [])

    def test_shell_allowlist_rejects_mutation_composition_and_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "hello.py").write_text("x = 1\n")
            bridge, session = self._start(root)
            blocked = ("echo test", "ls *", "ls /tmp", "rm hello.py", "python3 script.py", "ls hello.py && pwd", "git status", "git checkout -- hello.py", "git -c core.pager=cat status", "find . -exec cat {} \\")
            for index, command in enumerate(blocked):
                result = bridge.before({"session_id": session, "call_id": f"blocked-{index}", "tool": "bash", "args": {"command": command, "workdir": str(root)}})
                self.assertEqual(result["decision"], "BLOCK", command)

    def test_agentmemory_retrieval_is_advisory_and_writes_stay_blocked(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"V2_SUPERVISOR_REQUIRE_CONTRACT": "1"}, clear=False):
            root = Path(directory); (root / "hello.py").write_text("x = 1\n")
            bridge = SupervisorBridge(root / "state")
            self.assertEqual(bridge.before({"session_id": "memory", "tool": "memory_recall", "args": {"query": "recent v2 decision"}})["decision"], "ALLOW")
            self.assertEqual(bridge.before({"session_id": "memory", "tool": "memory_save", "args": {"content": "do not authorize"}})["decision"], "BLOCK")
            _, session = self._start(root)
            self.assertEqual(bridge.before({"session_id": session, "tool": "memory_smart_search", "args": {"query": "v2"}})["decision"], "ALLOW")
            self.assertEqual(bridge.before({"session_id": session, "tool": "memory_compress_file", "args": {"path": "hello.py"}})["decision"], "BLOCK")

    def test_status_exposes_active_contract_scope_without_expanding_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "src").mkdir(); (root / "src" / "a.py").write_text("x = 1\n")
            (root / ".scratch").mkdir(); (root / ".scratch" / "helper.py").write_text("x = 2\n")
            bridge = SupervisorBridge(root / "state")
            started = bridge.start({
                "goal": "update the source",
                "permitted_files": ["src/a.py"],
                "permitted_actions": ["read_file", "write_file"],
                "criteria": [{"key": "done", "description": "source is updated"}],
            }, "scope", str(root))
            self.assertEqual(started["status"], "STARTED")
            blocked = bridge.before({"session_id": "scope", "call_id": "bad", "tool": "edit", "args": {"filePath": ".scratch/helper.py"}})
            self.assertEqual(blocked["decision"], "BLOCK")
            status = bridge.summary({"session_id": "scope"})
            self.assertEqual(status["permitted_files"], ["src/a.py"])
            self.assertEqual(status["permitted_actions"], ["read_file", "write_file"])
            self.assertEqual(status["test_command"], [])
            compact = json.dumps({key: status[key] for key in ("task_id", "workspace", "phase", "permitted_files", "permitted_actions", "test_command", "visual_required", "budget_limit", "budget_calls", "remaining_budget", "criteria", "completion", "blockers", "counts")}, separators=(",", ":"))
            self.assertLess(len(compact), 1800)
            self.assertEqual(bridge.summary({"session_id": "scope"})["permitted_files"], ["src/a.py"])
            self.assertEqual(bridge.end({"session_id": "scope"})["status"], "ENDED")
            restarted = bridge.start({
                "goal": "add helper",
                "permitted_files": [".scratch/helper.py"],
                "permitted_actions": ["read_file", "write_file"],
                "criteria": [{"key": "done", "description": "helper is added"}],
            }, "scope", str(root))
            self.assertEqual(restarted["status"], "STARTED")
            self.assertEqual(bridge.summary({"session_id": "scope"})["permitted_files"], [".scratch/helper.py"])

    def test_light_status_bounds_seeded_scope_but_keeps_internal_count(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"V2_POLICY_MODE": "light", "V2_SUPERVISOR_REQUIRE_CONTRACT": "0"}, clear=False):
            root = Path(directory)
            for index in range(120):
                (root / f"module-{index:03}.py").write_text(f"value = {index}\n")
            bridge = SupervisorBridge(root / "state")
            self.assertEqual(bridge.start({"goal": "inspect this project", "criteria": [{"key": "seen", "description": "the project was inspected"}]}, "bounded", str(root))["status"], "STARTED")
            bridge.before({"session_id": "bounded", "call_id": "inspect", "tool": "bash", "args": {"command": "git status", "workdir": str(root)}})
            status = bridge.summary({"session_id": "bounded"})
            self.assertEqual(status["scope_mode"], "dynamic")
            self.assertEqual(status["permitted_files_count"], 120)
            self.assertTrue(status["permitted_files_truncated"])
            self.assertEqual(len(status["permitted_files"]), 40)
            self.assertLess(len(json.dumps(status, separators=(",", ":"))), 6000)

    def test_light_broad_reads_dynamic_writes_and_test_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"V2_POLICY_MODE": "light", "V2_SUPERVISOR_REQUIRE_CONTRACT": "0"}, clear=False):
            root = Path(directory); (root / "app.py").write_text("x = 1\n"); (root / "config.yaml").write_text("x: 1\n")
            (root / "test_app.py").write_text("assert True\n")
            bridge = SupervisorBridge(root / "state")
            self.assertEqual(bridge.before({"session_id": "light", "tool": "read", "args": {"filePath": "config.yaml"}})["decision"], "ALLOW")
            started = bridge.start({"goal": "repair app", "permitted_files": ["app.py"], "criteria": [{"key": "tests", "description": "verification passes"}]}, "light", str(root))
            self.assertEqual(started["status"], "STARTED")
            for call_id, path in (("write", ".scratch/helper.py"), ("test-write", "test_app.py")):
                self.assertEqual(bridge.before({"session_id": "light", "call_id": call_id, "tool": "edit", "args": {"filePath": path}})["decision"], "ALLOW")
                target = root / path; target.parent.mkdir(parents=True, exist_ok=True); target.write_text("assert True\n")
                bridge.after({"session_id": "light", "call_id": call_id, "tool": "edit", "args": {"filePath": path}, "output": {"output": "ok", "metadata": {}}})
            summary = bridge.summary({"session_id": "light"})
            self.assertEqual(summary["scope_mode"], "dynamic")
            self.assertIn(".scratch/helper.py", summary["scope_expansions"])
            self.assertIn("test_app.py", summary["scope_expansions"])
            self.assertTrue(summary["drift_warnings"])

    def test_light_normal_shell_git_and_observed_verification(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"V2_POLICY_MODE": "light", "V2_SUPERVISOR_REQUIRE_CONTRACT": "0"}, clear=False):
            root = Path(directory); (root / "app.py").write_text("x = 1\n")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            bridge = SupervisorBridge(root / "state")
            bridge.start({"goal": "inspect and verify", "criteria": [{"key": "tests", "description": "tests pass"}]}, "shell", str(root))
            for index, command in enumerate(("git status --short", "rg x app.py", "python3 -m unittest")):
                call_id = f"shell-{index}"
                self.assertEqual(bridge.before({"session_id": "shell", "call_id": call_id, "tool": "bash", "args": {"command": command, "workdir": str(root)}})["decision"], "ALLOW")
                result = bridge.after({"session_id": "shell", "call_id": call_id, "tool": "bash", "args": {"command": command}, "output": {"output": "ok", "metadata": {"exit": 0}}})
                if command.startswith("python3"):
                    self.assertEqual(result["evidence"], "recorded")
            self.assertEqual(bridge.finalize({"session_id": "shell"})["status"], "COMPLETE")

    def test_light_git_write_is_observed_without_preauthorization(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"V2_POLICY_MODE": "light", "V2_SUPERVISOR_REQUIRE_CONTRACT": "0"}, clear=False):
            root = Path(directory); (root / "app.py").write_text("x = 1\n")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            env = {**os.environ, "GIT_AUTHOR_NAME": "v2-test", "GIT_AUTHOR_EMAIL": "v2-test@example.invalid", "GIT_COMMITTER_NAME": "v2-test", "GIT_COMMITTER_EMAIL": "v2-test@example.invalid"}
            subprocess.run(["git", "-C", str(root), "add", "app.py"], check=True, env=env)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True, env=env)
            bridge = SupervisorBridge(root / "state")
            bridge.start({"goal": "make and record a repository change", "criteria": [{"key": "recorded", "description": "the repository change is observed"}]}, "git-write", str(root))
            (root / "app.py").write_text("x = 2\n")
            for index, command in enumerate(("git add app.py", "git commit -qm changed")):
                call_id = f"git-write-{index}"
                before = bridge.before({"session_id": "git-write", "call_id": call_id, "tool": "bash", "args": {"command": command, "workdir": str(root)}})
                self.assertEqual(before["decision"], "ALLOW", command)
                subprocess.run(["git", "-C", str(root), *command.split()[1:]], check=True, env=env)
                bridge.after({"session_id": "git-write", "call_id": call_id, "tool": "bash", "args": {"command": command}, "output": {"output": "ok", "metadata": {"exit": 0}}})
            events = bridge._task(session_id="git-write")[0].store.events(bridge._task(session_id="git-write")[1].task_id)
            self.assertTrue(any(row["kind"] == "tool_call" for row in events))

    def test_light_unknown_write_reconciles_and_complete_starts_new_task(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"V2_POLICY_MODE": "light", "V2_SUPERVISOR_REQUIRE_CONTRACT": "0"}, clear=False):
            root = Path(directory); (root / "app.py").write_text("x = 1\n")
            bridge = SupervisorBridge(root / "state")
            bridge.start({"goal": "repair", "criteria": [{"key": "tests", "description": "tests pass"}]}, "task", str(root))
            args = {"filePath": "helper.py", "content": "x = 2\n"}
            self.assertEqual(bridge.before({"session_id": "task", "call_id": "unknown", "tool": "edit", "args": args})["decision"], "ALLOW")
            (root / "helper.py").write_text(args["content"])
            result = bridge.after({"session_id": "task", "call_id": "unknown", "tool": "edit", "args": args, "ambiguous": True, "output": {"output": "uncertain", "metadata": {}}})
            self.assertEqual(result["status"], "acknowledged")
            self.assertEqual(bridge.summary({"session_id": "task"})["completion"]["unresolved_mutations"], 0)
            bridge.before({"session_id": "task", "call_id": "test", "tool": "bash", "args": {"command": "python3 -m unittest"}})
            bridge.after({"session_id": "task", "call_id": "test", "tool": "bash", "args": {"command": "python3 -m unittest"}, "output": {"output": "OK", "metadata": {"exit": 0}}})
            self.assertEqual(bridge.finalize({"session_id": "task"})["status"], "COMPLETE")
            self.assertEqual(bridge.before({"session_id": "task", "call_id": "old", "tool": "edit", "args": {"filePath": "app.py"}})["decision"], "BLOCK")
            self.assertEqual(bridge.start({"goal": "new task", "criteria": [{"key": "tests", "description": "tests pass"}]}, "task", str(root))["status"], "STARTED")

    def test_light_stale_evidence_without_prior_path_observation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"V2_POLICY_MODE": "light", "V2_SUPERVISOR_REQUIRE_CONTRACT": "0"}, clear=False):
            root = Path(directory); (root / "app.py").write_text("x = 1\n")
            bridge = SupervisorBridge(root / "state")
            bridge.start({"goal": "verify", "criteria": [{"key": "tests", "description": "tests pass"}]}, "stale-light", str(root))
            command = {"command": "python3 -m unittest"}
            bridge.before({"session_id": "stale-light", "call_id": "test", "tool": "bash", "args": command})
            bridge.after({"session_id": "stale-light", "call_id": "test", "tool": "bash", "args": command, "output": {"output": "OK", "metadata": {"exit": 0}}})
            self.assertEqual(bridge.finalize({"session_id": "stale-light"})["status"], "COMPLETE")
            (root / "app.py").write_text("x = 2\n")
            self.assertNotEqual(bridge.finalize({"session_id": "stale-light"})["status"], "COMPLETE")

    def test_light_loop_warning_then_circuit_breaker_and_legitimate_iteration(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"V2_POLICY_MODE": "light", "V2_SUPERVISOR_REQUIRE_CONTRACT": "0"}, clear=False):
            root = Path(directory); (root / "app.py").write_text("x = 1\n")
            bridge = SupervisorBridge(root / "state")
            bridge.start({"goal": "repair", "criteria": [{"key": "tests", "description": "tests pass"}]}, "loop", str(root))
            bridge.before({"session_id": "loop", "call_id": "read-app", "tool": "read", "args": {"filePath": "app.py"}})
            results = []
            for index in range(4):
                call_id = f"loop-{index}"
                result = bridge.before({"session_id": "loop", "call_id": call_id, "tool": "bash", "args": {"command": "python3 -m unittest"}})
                results.append(result["decision"])
                if result["decision"] == "ALLOW":
                    bridge.after({"session_id": "loop", "call_id": call_id, "tool": "bash", "args": {"command": "python3 -m unittest"}, "output": {"output": "fail", "metadata": {"exit": 1}}})
            self.assertEqual(results, ["ALLOW", "ALLOW", "ALLOW", "BLOCK"])
            # A changed source fingerprint makes a real iteration a new action.
            (root / "app.py").write_text("x = 2\n")
            self.assertEqual(bridge.before({"session_id": "loop", "call_id": "new", "tool": "bash", "args": {"command": "python3 -m unittest"}})["decision"], "ALLOW")

            refreshed = SupervisorBridge(root / "refresh-state")
            refreshed.start({"goal": "repair", "criteria": [{"key": "tests", "description": "tests pass"}]}, "refresh", str(root))
            for index in range(3):
                call_id = f"refresh-{index}"
                self.assertEqual(refreshed.before({"session_id": "refresh", "call_id": call_id, "tool": "bash", "args": {"command": "python3 -m unittest"}})["decision"], "ALLOW")
                refreshed.after({"session_id": "refresh", "call_id": call_id, "tool": "bash", "args": {"command": "python3 -m unittest"}, "output": {"output": "fail", "metadata": {"exit": 1}}})
            self.assertEqual(refreshed.before({"session_id": "refresh", "call_id": "status", "tool": "orchestrator_status", "args": {}})["decision"], "ALLOW")
            refreshed.summary({"session_id": "refresh"})
            self.assertEqual(refreshed.before({"session_id": "refresh", "call_id": "after-refresh", "tool": "bash", "args": {"command": "python3 -m unittest"}})["decision"], "ALLOW")

    def test_light_standalone_reads_are_not_coupled_to_write_scope(self):
        from orchestrator_v2.tools import WorkspaceTools
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "dependency.py").write_text("x = 1\n")
            tools = WorkspaceTools(root, ["app.py"], ["read_file", "write_file"], allow_broad_reads=True)
            self.assertIn("x = 1", tools.read_file("dependency.py"))


if __name__ == "__main__":
    unittest.main()
