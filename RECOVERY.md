# Recovery and v2 boundary

## Sources inspected

- `/home/brandon/standalone-orchestrator` — local fork, `bf50a6041d686d272d592cacf8891987721286f4`, clean Git status at inspection but containing private databases/assets; retained unchanged and not executed.
- `/home/brandon/link-public-baseline` — public-style local baseline, `0e358e0e1496455ebd3120473f811e321491b2bb`, clean Git status.
- `/home/brandon/francesca-growth-lab/tools/standalone_orchestrator` — nested copy in a dirty private repository; not selected.
- `/home/brandon/francesca_orchestrator_full_quarantine_20260516` — quarantine copy containing private chatbot assets; not selected.
- `/home/brandon/link` — later/private automation tree, `a2daea9`, four dirty entries at inspection; not selected.
- `/home/brandon/projects/standalone-orchestrator-v2` — fresh public clone at `0e358e0e1496455ebd3120473f811e321491b2bb`, feature work is local only.

NAS was not accessed: the local/public source was sufficient and no archive extraction was necessary.

## Reuse decision

The old planner/build loop, librarian, subconscious daemon, sync scripts, model-training artifacts, and large benchmark harness remain legacy. v2 reuses the design ideas (phases, AST/evidence orientation, bounded repair) but uses a small new controller, SQLite event store, allowlisted tools, direct Qwen adapter, and OCR delegation adapter.

## Legacy issues confirmed

- Historical aggregate benchmark labels treated partial test results as passing.
- The public `DoD.all_passed()` returns `True` for an empty criteria list; v2 rejects empty contracts before execution.
- The public `TaskState.to_dict()` omitted `blamed_source_files` and `rca_history`; v2 persists its complete task contract, counters, unresolved items, events, and evidence transactionally.
- Persistence was JSON-only and did not provide transactional event/evidence history.
- Legacy configuration hard-coded Ollama ports and included daemon/sync side effects.
- v2 requires a nonempty contract, records every evidence hash, and never treats model prose as authoritative completion.
