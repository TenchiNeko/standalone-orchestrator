from __future__ import annotations

import argparse, json, os
from pathlib import Path

from .controller import Controller
from .qwen import QwenAdapter
from .state import Criterion, Phase, TaskContract


def main(argv=None):
    ap = argparse.ArgumentParser(prog="orchestrator-v2"); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    run = sub.add_parser("run"); run.add_argument("--task", required=True); run.add_argument("--jev", choices=["off", "shadow", "advisory"], default="off")
    for name in ("status", "resume", "cancel", "report"):
        p = sub.add_parser(name); p.add_argument("task_id")
    args = ap.parse_args(argv); root = Path(__file__).resolve().parents[1]; state_dir = root / ".orchestrator-v2"
    key = os.environ.get("QWEN_API_KEY")
    key_file = os.environ.get("QWEN_API_KEY_FILE")
    if not key and key_file:
        key = Path(key_file).read_text().strip()
    if args.cmd == "doctor":
        q = QwenAdapter(base_url=os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:18089/v1"), model=os.environ.get("QWEN_MODEL", "orca27b-ultra-q6-mtp"), api_key=key); print(json.dumps({"version": "2.0.0", "qwen": q.health(), "state": str(state_dir)}, indent=2)); return 0
    ctl = Controller(state_dir, qwen=QwenAdapter(base_url=os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:18089/v1"), model=os.environ.get("QWEN_MODEL", "orca27b-ultra-q6-mtp"), api_key=key), jev_mode=getattr(args, "jev", "off"))
    if args.cmd == "run":
        spec = json.loads(Path(args.task).read_text()); contract = TaskContract(spec["goal"], spec["permitted_files"], spec["permitted_actions"], [Criterion(**c) for c in spec["criteria"]], spec.get("version", 1), spec.get("test_command")); task = ctl.intake(contract, Path(spec["workspace"]).resolve(), spec.get("budget_limit", 12)); task = ctl.run(task); print(json.dumps(ctl.status(task.task_id), indent=2)); return 0 if task.phase.value == "complete" else 2
    if args.cmd == "resume":
        task = ctl.store.load_task(args.task_id); task = ctl.run(task); print(json.dumps(ctl.status(task.task_id), indent=2)); return 0 if task.phase.value == "complete" else 2
    if args.cmd == "cancel":
        task = ctl.store.load_task(args.task_id); task.phase = Phase.BLOCKED; task.unresolved.append("cancelled by operator"); ctl.store.save_task(task); ctl.store.event(task.task_id, "cancel", {}); print(json.dumps(ctl.status(task.task_id), indent=2)); return 0
    print(json.dumps(ctl.status(args.task_id), indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
