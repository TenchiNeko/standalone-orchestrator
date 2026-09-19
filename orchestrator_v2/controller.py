from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .jev import JevAdapter
from .ocr import OCRDelegation
from .decisions import DecisionProvider, DeterministicDecisionProvider
from .context_gc import ContextBlock, ContextGC, ContextResult
from .state import Criterion, Phase, StateStore, Task, TaskContract, file_manifest, new_task, source_hash
from .tools import ToolError, ToolRegistry, WorkspaceTools


class Controller:
    def __init__(self, state_dir: Path, qwen=None, jev_mode: str = "off", decision_provider: DecisionProvider | None = None, decision_mode: str = "shadow"):
        if decision_mode not in {"shadow", "advisory"}: raise ValueError("decision_mode must be shadow or advisory")
        self.state_dir = state_dir; self.store = StateStore(state_dir / "state.sqlite3"); self.qwen = qwen; self.jev = JevAdapter(jev_mode); self.ocr = OCRDelegation(); self.decision_provider = decision_provider or DeterministicDecisionProvider(); self.decision_mode = decision_mode

    def intake(self, contract: TaskContract, workspace: Path, budget: int = 12) -> Task:
        task = new_task(contract, workspace, budget); self.store.save_task(task); self.store.event(task.task_id, "intake", {"phase": task.phase.value, "source_hash": source_hash(workspace), "file_manifest": file_manifest(workspace, contract.permitted_files), "visual_required": contract.visual_required})
        return task

    def _phase(self, task: Task, phase: Phase):
        task.phase = phase; self.store.save_task(task); self.store.event(task.task_id, "phase", {"phase": phase.value})

    def run(self, task: Task, review: bool = True) -> Task:
        root = Path(task.workspace); tools = WorkspaceTools(root, task.contract.permitted_files, task.contract.permitted_actions, task.contract.test_command)
        review_ok = not review
        self._phase(task, Phase.INSPECT); self.store.evidence(task.task_id, "inspect", root, {"files": sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and ".git" not in p.parts)})
        self._phase(task, Phase.PLAN); self.store.evidence(task.task_id, "plan", root, {"criteria": [c.description for c in task.contract.criteria]}, task.contract.permitted_files)
        self._decision_probe(task, root)
        if self.qwen and task.budget_calls < task.budget_limit:
            response = self.qwen.chat([{"role": "system", "content": "Return a concise implementation plan. Do not claim execution or completion."}, {"role": "user", "content": json.dumps({"goal": task.contract.goal, "criteria": [c.description for c in task.contract.criteria]})}], max_tokens=350)
            task.budget_calls += 1; self.store.evidence(task.task_id, "qwen_plan", root, {"content": response.content, "usage": response.usage, "finish_reason": response.finish_reason, "elapsed": response.elapsed}, task.contract.permitted_files)
        self._phase(task, Phase.IMPLEMENT)
        if self.qwen and task.budget_calls < task.budget_limit and "read_file" in task.contract.permitted_actions:
            self._bounded_worker(task, root, tools)
        else:
            self.store.event(task.task_id, "implementation", {"mode": "deterministic", "note": "model worker skipped by budget or contract"})
        self._phase(task, Phase.TEST); result = tools.run_tests(); ref = self.store.evidence(task.task_id, "test", root, result, task.contract.permitted_files); self.store.event(task.task_id, "test_result", {"ref": ref, "exit_code": result["exit_code"]})
        if result["exit_code"] != 0:
            task.phase = Phase.BLOCKED; task.unresolved.append("required tests failed"); self.store.save_task(task); return task
        if review:
            self._phase(task, Phase.REVIEW)
            try:
                preview = self.ocr.preview(root); self.store.evidence(task.task_id, "ocr_preview", root, preview, task.contract.permitted_files)
                candidates = preview.get("reviewable_files", preview.get("files", []))
                files = [x.get("path") for x in candidates if isinstance(x, dict) and x.get("path")]
                if files: self.store.evidence(task.task_id, "ocr_rules", root, {"files": files, "rules": self.ocr.rules(root, files)}, files)
                review_ok = True
            except Exception as exc:
                task.unresolved.append(f"OCR unavailable: {exc}")
        self._phase(task, Phase.FINAL_VERIFY)
        fresh = tools.run_tests(); ref = self.store.evidence(task.task_id, "final_test", root, fresh, task.contract.permitted_files)
        gate = self.completion_gate(task, root, review_ok)
        if not gate["allowed"]:
            task.phase = Phase.NEEDS_REVIEW if gate["needs_review"] else Phase.BLOCKED
            task.unresolved.append(gate["reason"])
        else:
            for c in task.contract.criteria:
                if c.required:
                    c.status = "verified"; c.evidence.append(ref); c.verified_source_hash = source_hash(root, task.contract.permitted_files)
                elif c.status == "pending":
                    c.status = "advisory"
            task.phase = Phase.COMPLETE
        self.store.save_task(task); return task

    def completion_gate(self, task: Task, root: Path, review_ok: bool = True) -> dict[str, object]:
        """Deterministic completion facts; model prose cannot make this pass."""
        current = self.store.current_successful_check(task.task_id, root, task.contract.permitted_files)
        unresolved_mutations = self.store.unresolved_mutations(task.task_id)
        visual = None
        if task.contract.visual_required:
            with self.store._db() as db:
                rows = db.execute("SELECT id, source_hash FROM evidence WHERE task_id=? AND kind='visual_verify' ORDER BY id DESC", (task.task_id,)).fetchall()
            digest = source_hash(root, task.contract.permitted_files)
            visual = next((f"evidence:{row['id']}" for row in rows if row["source_hash"] == digest), None)
        required_pending = [criterion.key for criterion in task.contract.criteria if criterion.required and criterion.status != "verified"]
        if not current:
            return {"allowed": False, "needs_review": False, "reason": "required code changes have no current successful check after the latest relevant edit", "check": None, "visual": visual, "unresolved_mutations": unresolved_mutations, "pending": required_pending}
        if not review_ok:
            return {"allowed": False, "needs_review": True, "reason": "required review evidence is unavailable", "check": current, "visual": visual, "unresolved_mutations": unresolved_mutations, "pending": required_pending}
        if task.contract.visual_required and not visual:
            return {"allowed": False, "needs_review": False, "reason": "required visual verification is missing or stale", "check": current, "visual": None, "unresolved_mutations": unresolved_mutations, "pending": required_pending}
        if unresolved_mutations:
            return {"allowed": False, "needs_review": True, "reason": "an unresolved mutation outcome requires reconciliation", "check": current, "visual": visual, "unresolved_mutations": unresolved_mutations, "pending": required_pending}
        if not task.contract.criteria:
            return {"allowed": False, "needs_review": False, "reason": "contract has no required criteria", "check": current, "visual": visual, "unresolved_mutations": [], "pending": []}
        return {"allowed": True, "needs_review": False, "reason": "current successful check and required evidence are present", "check": current, "visual": visual, "unresolved_mutations": [], "pending": required_pending}

    def compact_context(self, task: Task, root: Path, blocks: list[ContextBlock], state: str, mode: str = "shadow") -> ContextResult:
        """Expose reversible context selection without changing the worker loop by default."""
        return ContextGC(self.state_dir, self.store, self.decision_provider, mode=mode).process(task.task_id, root, blocks, state)

    def _decision_probe(self, task: Task, root: Path) -> None:
        questions = {"next_read_only": {"type": "choice", "instructions": "Choose the next low-risk read-only supervisory direction. Do not authorize edits, retries, test skipping, scope changes, or completion.", "criteria": {"VERIFY": "Check a concrete missing fact or current test result.", "DIAGNOSE": "Classify an observed failure before choosing another check.", "REVIEW": "Inspect existing evidence or a bounded review finding.", "ESCALATE": "Evidence is insufficient and a human or explicit blocker is needed."}}}
        state = json.dumps({"goal": task.contract.goal, "phase": task.phase.value, "unresolved": task.unresolved, "required_criteria": [c.description for c in task.contract.criteria if c.required]}, sort_keys=True)
        try:
            batch = self.decision_provider.decide(state, questions)
        except Exception as exc:
            self.store.event(task.task_id, "decision_unavailable", {"provider": self.decision_provider.name, "error": type(exc).__name__})
            return
        if not batch.decisions:
            return
        payload = {"provider": batch.provider, "mode": self.decision_mode, "authority": "shadow-only", "latency": batch.latency, "decisions": [d.__dict__ for d in batch.decisions], "metadata": batch.metadata}
        ref = self.store.evidence(task.task_id, "decision", root, payload, task.contract.permitted_files)
        self.store.event(task.task_id, "decision_judgment", {"ref": ref, "provider": batch.provider, "mode": self.decision_mode, "authority": "shadow-only"})

    def _bounded_worker(self, task: Task, root: Path, tools: WorkspaceTools) -> None:
        """Run a short native-tool loop; state and filesystem remain coordinator-owned."""
        messages = [
            {"role": "system", "content": "You are a bounded implementation worker. Inspect before editing. Use only the supplied native tools and only permitted files. Never claim a test or edit happened unless a tool result proves it. Stop when the contract is satisfied or report the concrete blocker."},
            {"role": "user", "content": json.dumps({"goal": task.contract.goal, "criteria": [c.description for c in task.contract.criteria], "permitted_files": task.contract.permitted_files, "permitted_actions": task.contract.permitted_actions})},
        ]
        registry = ToolRegistry()
        seen_actions: set[tuple[str, str, str]] = set()
        while task.budget_calls < task.budget_limit:
            response = self.qwen.chat(messages, tools=registry.schemas(), max_tokens=1000)
            task.budget_calls += 1
            self.store.evidence(task.task_id, "qwen_worker", root, {"content": response.content, "usage": response.usage, "finish_reason": response.finish_reason, "elapsed": response.elapsed}, task.contract.permitted_files)
            if not response.tool_calls:
                self.store.event(task.task_id, "worker_stopped", {"reason": "no_tool_call", "content": response.content[:2000]})
                return
            assistant = {"role": "assistant", "content": response.content, "tool_calls": response.tool_calls}
            messages.append(assistant)
            for call in response.tool_calls:
                fn = call.get("function") or call
                name = fn.get("name")
                raw = fn.get("arguments", {})
                args = {}
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                    if not isinstance(args, dict): raise ValueError("arguments must be an object")
                    action_key = (str(name), json.dumps(args, sort_keys=True), source_hash(root, task.contract.permitted_files))
                    if action_key in seen_actions:
                        self.store.event(task.task_id, "no_progress", {"name": name, "reason": "same action and source version repeated"})
                        return
                    seen_actions.add(action_key)
                    result = registry.dispatch(tools, name, args)
                    status = "ok"
                except (ToolError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    result = {"error": str(exc)}; status = "rejected"
                self.store.event(task.task_id, "tool_call", {"name": name, "status": status, "arguments": {k: ("<content>" if k == "content" else v) for k, v in (args.items() if isinstance(args, dict) else [])}})
                call_id = call.get("id", f"tool-{task.budget_calls}")
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name or "unknown", "content": json.dumps(result)[:20_000]})

    def status(self, task_id: str) -> dict:
        task = self.store.load_task(task_id); root = Path(task.workspace); return {"task_id": task.task_id, "phase": task.phase.value, "criteria": [c.__dict__ for c in task.contract.criteria], "unresolved": task.unresolved, "completion_facts": self.completion_gate(task, root), "events": self.store.events(task_id)}
