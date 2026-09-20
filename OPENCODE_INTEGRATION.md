# OpenCode + v2 supervisor

The preferred interactive path is the installed `opencode-v2` launcher. OpenCode
owns the session, model transport, UI, context, and normal tools; the existing
Python v2 core owns contracts, policy, evidence, mutation reconciliation, stale
tests, and deterministic finalization. It never starts a second agent loop or
model server.

## Run and rollback

```bash
cd /path/to/project
opencode-v2
opencode-v2 run "fix the failing tests"
```

The launcher sets `V2_SUPERVISOR_REQUIRE_CONTRACT=1`, uses an isolated
`~/.config/opencode-v2` host config by default, and loads only the project-local
supervisor plugin plus the configured local Qwen provider. Ordinary `opencode`
and the global config remain unchanged. Set `V2_OPENCODE_CONFIG_HOME` for a
disposable config. Roll back by using ordinary `opencode`, or remove/rename the
reversible `~/.local/bin/opencode-v2` symlink; v2 SQLite state is retained.

## Verified host API

Verified on OpenCode **1.18.31** and `@opencode-ai/plugin` **1.14.48**:

- `tool.execute.before` sees tool, session, call ID, and mutable arguments;
- `tool.execute.after` sees arguments and result metadata;
- `event` records bounded lifecycle and token-part facts;
- `experimental.chat.system.transform` adds the short supervision contract;
- `experimental.session.compacting` receives a bounded v2 fact summary;
- `experimental.compaction.autocontinue` disables automatic continuation after
  deterministic completion;
- `tool({description,args,execute})` supplies custom tools.

The installed hook has no typed deny return and no trusted stop-generation API.
A deterministic BLOCK throws before a built-in tool runs (proven in the
disposable fixture); a fork would be needed for a stronger universal host gate.
After v2 records `COMPLETE`, every later project mutation, shell command, or test
is blocked until a new contract is started.

## Daily contract workflow

Read-only discovery is allowed before a task exists. Before the first edit,
write, shell command, or test, the model must call `orchestrator_start` with a
non-empty goal, exact mutation files (never `*` in strict mode), permitted
actions, a shell-free exact test argv, and a criterion that the test can prove.
Reads may inspect tests and neighboring source inside the workspace; the file
list constrains mutations. `orchestrator_health`, `orchestrator_status`,
`orchestrator_symbols`, `orchestrator_end`, and `orchestrator_cancel` are
available as bounded local tools.

Only an observed execution of the exact configured command with an observed
exit status creates test evidence. `echo test`, compound commands, appended
commands, model prose, and model-supplied exit codes never do. A successful
check is tied to the current permitted-file hash; a later edit makes it stale.
Unknown mutation results block another write until authoritative readback via
`orchestrator_evidence(kind="reconcile")` resolves them. Finalization returns
`COMPLETE`, `INCOMPLETE`, `BLOCKED`, or `NEEDS_REVIEW`; it never trusts prose.

## Bridge and state

The TypeScript adapter keeps one persistent local child:

```text
python3 -u orchestrator_v2/bridge.py
```

JSONL travels over stdin/stdout, with an 8-second request timeout, dead-child
detection, bounded diagnostic stderr, restart on a later request, and fail-closed
write/shell authorization. State is SQLite under a workspace hash. No cloud
service or second Qwen process is used. The verified host currently requires
absolute imports of its installed plugin SDK; this is recorded as a portability
limitation rather than changing the global installation.

## Compaction and usage

Only an active task receives a stable compact JSON fact summary (capped at 1800
characters): task/phase/goal, criteria, evidence blockers, counts, and bounded
usage. No event history or hidden evidence is injected. Token parts are forwarded
with stable IDs where available and exact host fields are stored; the host does
not expose a separate guaranteed usage API.

## Standalone mode

The standalone runner remains available for controlled experiments:

```bash
python3 -m unittest discover -s tests -q
PYTHONPATH=. .venv/bin/python -m orchestrator_v2.cli run --task examples/repair-task.json --jev off
```

The plugin is optional and can be disabled with `V2_SUPERVISOR_DISABLED=1`.
