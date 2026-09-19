"""Small local shadow benchmark for reversible context selection."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator_v2.context_gc import ContextBlock, ContextGC
from orchestrator_v2.decisions import make_decision_provider
from orchestrator_v2.state import StateStore


def main():
    with tempfile.TemporaryDirectory(prefix="orchestrator-context-") as d:
        root = Path(d); (root / "fixture.py").write_text("VALUE = 42\n")
        state = root / "state"; store = StateStore(state / "state.sqlite3")
        task_id = "context-shadow"
        blocks = [
            ContextBlock("read-relevant", "read", "Current source and exact ERROR_CODE_42 reproduction\n" * 40, "fixture.py"),
            ContextBlock("grep-old", "grep", "old unrelated matches\n" * 80, "logs/old.txt"),
            ContextBlock("bash-error", "bash", "Traceback: required evidence is unresolved\n" * 30, "pytest", error=True),
            ContextBlock("recent-write", "write", "mutation result requires reconciliation\n" * 20, "fixture.py", recent_mutation=True),
        ]
        provider = make_decision_provider("nanojev")
        gc = ContextGC(state, store, provider=provider, mode="shadow")
        result = gc.process(task_id, root, blocks, "Repair fixture.py and verify the exact current failure.")
        recalled = sum(gc.recall(d.artifact) == block.text for d, block in zip(result.decisions, blocks))
        print(json.dumps({
            "mode": "shadow",
            "original_chars": result.original_chars,
            "model_facing_chars": result.rendered_chars,
            "would_hide": [d.block_id for d in result.decisions if d.disposition in {"HIDE_BUT_RECALLABLE", "TRUNCATE_WITH_POINTER"}],
            "pinned": [d.block_id for d in result.decisions if d.disposition == "PIN"],
            "exact_recall_pass": recalled == len(blocks),
            "retrievals": 0,
            "qwen_prefill_comparison": "not run: shadow mode leaves the prompt unchanged",
        }, indent=2))


if __name__ == "__main__":
    main()
