import json
import tempfile
import unittest
from pathlib import Path

from orchestrator_v2.state import Criterion, Phase, StateStore, TaskContract, new_task, source_hash
from orchestrator_v2.tools import ToolError, ToolRegistry, WorkspaceTools
from orchestrator_v2.controller import Controller
from orchestrator_v2.qwen import QwenResponse


class V2Tests(unittest.TestCase):
    def test_contract_rejects_empty_required_criteria(self):
        with self.assertRaises(ValueError):
            TaskContract("goal", ["a.py"], ["read_file"], []).validate()

    def test_round_trip_preserves_criteria_and_counters(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "a.py").write_text("x=1")
            contract = TaskContract("goal", ["a.py"], ["read_file"], [Criterion("c", "a required fact")])
            task = new_task(contract, root); task.phase = Phase.REPAIR; task.repair_cycles = 2; task.unresolved = ["pending"]
            store = StateStore(root / "state.sqlite3"); store.save_task(task)
            got = store.load_task(task.task_id)
            self.assertEqual(got.phase, Phase.REPAIR); self.assertEqual(got.repair_cycles, 2); self.assertEqual(got.unresolved, ["pending"])

    def test_evidence_becomes_stale_after_edit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "a.py").write_text("x=1")
            store = StateStore(root / "state.sqlite3"); ref = store.evidence("t", "test", root, {"exit": 0}, ["a.py"])
            self.assertTrue(store.evidence_is_current(ref, root, ["a.py"]))
            (root / "a.py").write_text("x=2")
            self.assertFalse(store.evidence_is_current(ref, root, ["a.py"]))

    def test_tools_reject_escape_and_arbitrary_command(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "a.py").write_text("x=1"); tools = WorkspaceTools(root, ["a.py"], ["read_file", "run_tests"])
            with self.assertRaises(ToolError): tools.read_file("../secret")
            with self.assertRaises(ToolError): tools.read_file("other.py")
            with self.assertRaises(ToolError): tools.write_file("a.py", "x=2")
            with self.assertRaises(ToolError): tools.run_tests(["bash", "-lc", "id"])
            with self.assertRaises(ToolError): tools.run_tests(["git", "commit", "-am", "bad"])

    def test_native_tool_schema_is_allowlisted(self):
        with tempfile.TemporaryDirectory() as d:
            tools = WorkspaceTools(Path(d)); registry = ToolRegistry()
            with self.assertRaises(ToolError): registry.dispatch(tools, "shell", {"command": "id"})
            with self.assertRaises(ToolError): registry.dispatch(tools, "run_tests", {"command": ["python", "-c", "print(1)"]})

    def test_repeated_action_without_new_state_stops_worker(self):
        class FakeQwen:
            def __init__(self): self.calls = 0
            def chat(self, messages, tools=None, max_tokens=0):
                self.calls += 1
                return QwenResponse("", [{"id": str(self.calls), "function": {"name": "read_file", "arguments": '{"path":"a.py"}'}}], {}, "tool_calls", 0.0)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "a.py").write_text("x=1")
            contract = TaskContract("inspect", ["a.py"], ["read_file"], [Criterion("c", "inspect a")])
            task = new_task(contract, root); ctl = Controller(root / "state", qwen=FakeQwen()); ctl.store.save_task(task)
            ctl._bounded_worker(task, root, WorkspaceTools(root, ["a.py"], ["read_file"]))
            self.assertTrue(any(e["kind"] == "no_progress" for e in ctl.store.events(task.task_id)))

    def test_advisory_only_contract_cannot_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "test_ok.py").write_text("assert True\n")
            contract = TaskContract("check", ["test_ok.py"], ["run_tests"], [Criterion("hint", "an advisory observation", required=False)])
            ctl = Controller(root / "state"); task = ctl.intake(contract, root, 1); result = ctl.run(task, review=False)
            self.assertEqual(result.phase, Phase.BLOCKED)


if __name__ == "__main__": unittest.main()
