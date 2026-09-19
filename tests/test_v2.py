import json
import tempfile
import unittest
from pathlib import Path

from orchestrator_v2.state import Criterion, Phase, StateStore, TaskContract, new_task, source_hash
from orchestrator_v2.tools import ToolError, ToolRegistry, WorkspaceTools
from orchestrator_v2.controller import Controller
from orchestrator_v2.qwen import QwenResponse
from orchestrator_v2.context_gc import ContextBlock, ContextGC, HIDE_BUT_RECALLABLE
from orchestrator_v2.decisions import Decision, DecisionBatch, DecisionProvider
from orchestrator_v2.browser_router import BrowserCandidate, BrowserCandidateRouter
from orchestrator_v2.routing import ModelRouter, ROUTINE_QWEN


class FixedDecisionProvider(DecisionProvider):
    name = "fixed"

    def __init__(self, selected):
        self.selected = selected

    def decide(self, state, questions):
        qid, question = next(iter(questions.items()))
        candidates = list(question["criteria"])
        return DecisionBatch(self.name, [Decision(qid, candidates, self.selected, {c: (1.0 if c == self.selected else 0.0) for c in candidates}, "fixed", 0.0)], 0.0)


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

    def test_completion_gate_rejects_passing_check_before_latest_edit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "a.py").write_text("x=1")
            contract = TaskContract("repair", ["a.py"], ["run_tests"], [Criterion("c", "current check")])
            ctl = Controller(root / "state"); task = ctl.intake(contract, root)
            intake = ctl.store.events(task.task_id)[0]
            self.assertIn("file_manifest", intake["payload"])
            ctl.store.evidence(task.task_id, "test", root, {"command": ["python", "-m", "unittest"], "exit_code": 0, "timed_out": False}, ["a.py"])
            (root / "a.py").write_text("x=2")
            gate = ctl.completion_gate(task, root)
            self.assertFalse(gate["allowed"])
            self.assertIn("no current successful check", gate["reason"])

    def test_completion_gate_blocks_unknown_mutation_and_missing_visual(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "a.py").write_text("x=1")
            contract = TaskContract("repair", ["a.py"], ["run_tests"], [Criterion("c", "current check")], visual_required=True)
            ctl = Controller(root / "state"); task = ctl.intake(contract, root)
            ctl.store.evidence(task.task_id, "final_test", root, {"command": ["python", "-m", "unittest"], "exit_code": 0, "timed_out": False}, ["a.py"])
            ctl.store.mutation_intent(task.task_id, "write", "u1"); ctl.store.mutation_ack(task.task_id, "u1", "unknown")
            gate = ctl.completion_gate(task, root)
            self.assertFalse(gate["allowed"])
            self.assertIn("visual verification", gate["reason"])
            ctl.store.evidence(task.task_id, "visual_verify", root, {"screenshot": "fixture.png"}, ["a.py"])
            gate = ctl.completion_gate(task, root)
            self.assertFalse(gate["allowed"])
            self.assertIn("unresolved mutation", gate["reason"])

    def test_context_shadow_retains_exact_block_and_recall(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "a.py").write_text("x=1")
            ctl = Controller(root / "state"); task = ctl.intake(TaskContract("inspect", ["a.py"], ["read_file"], [Criterion("c", "inspect")]), root)
            gc = ContextGC(root / "state", ctl.store, FixedDecisionProvider(HIDE_BUT_RECALLABLE), mode="shadow")
            original = "verbose output " * 300
            result = gc.process(task.task_id, root, [ContextBlock("b1", "read", original, "a.py")], "goal")
            self.assertEqual(result.rendered[0], original)
            self.assertEqual(gc.recall(result.decisions[0].artifact), original)
            self.assertTrue(result.decisions[0].shadow)

    def test_browser_router_never_executes_in_shadow_and_validates_candidates(self):
        router = BrowserCandidateRouter(FixedDecisionProvider("save"), mode="shadow")
        candidates = [BrowserCandidate("save", "button", "Save", "click")]
        called = []
        route = router.route("save", {"url": "http://127.0.0.1"}, candidates, lambda c: called.append(c) or {"ok": True}, lambda: {"ok": True})
        self.assertEqual(route.selected, "save"); self.assertFalse(route.executed); self.assertEqual(called, [])

    def test_model_router_is_shadow_only(self):
        router = ModelRouter(FixedDecisionProvider("ESCALATE_STRONG_MODEL"), mode="advisory")
        judgment = router.classify("routine", {"tests": "pass"})
        self.assertEqual(judgment.selected, "ESCALATE_STRONG_MODEL")
        self.assertEqual(router.effective_route(judgment), ROUTINE_QWEN)

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
