# OpenCode + v2 supervisor

The preferred interactive path is the installed `opencode-v2` launcher. OpenCode
owns the session, model transport, UI, context, and normal tools; the existing
Python v2 core owns contracts, policy, evidence, mutation reconciliation, stale
tests, and deterministic finalization. It never starts a second agent loop or
model server.

## Run and rollback

```bash
cd /path/to/project
opencode-v2 --auto
opencode-v2 --auto run "fix the failing tests"
```

Run from the intended project directory. Strict mode refuses `/` and the
user's home directory before OpenCode starts, so an accidental `cd ~` cannot
trigger a broad source scan. An explicit human-only exception is available
for unusual maintenance work with `V2_SUPERVISOR_ALLOW_BROAD_WORKSPACE=1`;
the model cannot set this through `orchestrator_start`. Ordinary `opencode`
and unrelated global providers remain available unchanged.

The launcher sets `V2_SUPERVISOR_REQUIRE_CONTRACT=1`, uses an isolated
`~/.config/opencode-v2` host config by default, and loads only the project-local
supervisor plugin plus the configured local Qwen provider. Ordinary `opencode`
and unrelated global providers remain available unchanged. Set
`V2_OPENCODE_CONFIG_HOME` for a
disposable config. Roll back by using ordinary `opencode`, or remove/rename the
reversible `~/.local/bin/opencode-v2` symlink; v2 SQLite state is retained.

The promoted local Qwen backend has a **40960-token physical llama.cpp
context** while the generated supervised OpenCode model definition pins the
logical budget to `context=32768`, `input=28000`, and `output=2048`. This
headroom is intentional: OpenCode compacts at its logical budget while
transitional requests in the old 33--34K range can still reach the backend.
The backend uses native q8_0-K/q5_1-V Flash Attention and the validated
per-tensor CUDA0 overrides recorded in the shared workspace run.

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

`orchestrator_status` reports the active workspace, exact project-relative
`permitted_files`, permitted actions, exact test argv, visual requirement,
budget counters, criterion states, phase, blockers, and compact usage counts.
It does not change the contract. If a path is rejected, inspect this scope
once; end the task and start a new exact contract when the scope must change.

Only an observed execution of the exact configured command with an observed
exit status creates test evidence. `echo test`, compound commands, appended
commands, model prose, and model-supplied exit codes never do. A successful
check is tied to the current permitted-file hash; a later edit makes it stale.
Unknown mutation results block another write until authoritative readback via
`orchestrator_evidence(kind="reconcile")` resolves them. Finalization returns
`COMPLETE`, `INCOMPLETE`, `BLOCKED`, or `NEEDS_REVIEW`; it never trusts prose.

After a contract exists, a small audited read-only shell subset may be used for
inspection: `pwd`/`pwd -P`, `ls` with no more than one workspace path and the
`-a`/`-l`/`-al`/`-la` flags, `stat`, `head`, `tail`, and `wc -l|-c|-w`. Git is
intentionally not in this shell allowlist because repository configuration can
invoke external helpers even for queries; use OpenCode's native repository
tools instead. Each command is one argv command with no
composition, globbing, redirection, config injection, or outside-workspace
path. It is recorded as read-only inspection and can never satisfy a test
criterion. General Bash, mutating Git commands, interpreters, and shell
composition remain blocked. Exact AgentMemory retrieval tools (`memory_recall`,
`memory_smart_search`, `memory_sessions`, `memory_lesson_recall`) are advisory
context only; memory writes and unknown memory tools remain blocked.

After an observed successful exact test, the plugin reuses the same
deterministic finalizer and appends its bounded result to the test output. This
helps a model that naturally stops after verification receive `COMPLETE` without
adding a model call or granting any new authority.

## Bridge and state

The TypeScript adapter keeps one persistent local child:

```text
python3 -u orchestrator_v2/bridge.py
```

JSONL travels over stdin/stdout, with an 8-second request timeout, dead-child
detection, operation-specific timeout diagnostics, bounded diagnostic stderr,
restart on a later request, and fail-closed write/shell authorization. State is
SQLite under a workspace hash. Intake hashes only the contract's permitted
files, not the entire host workspace. No cloud
service or second Qwen process is used. The verified host currently requires
absolute imports of its installed plugin SDK; this is recorded as a portability
limitation rather than changing the global installation.

## Compaction and usage

Only an active task receives a stable compact JSON fact summary (capped at 1800
characters): task/phase/goal, project-relative permitted files/actions, exact
test argv, visual requirement, budget, criteria, evidence blockers, counts, and
bounded usage. No event history or hidden evidence is injected. Token parts are forwarded
with stable IDs where available and exact host fields are stored; the host does
not expose a separate guaranteed usage API.

## Standalone mode

The standalone runner remains available for controlled experiments:

```bash
python3 -m unittest discover -s tests -q
PYTHONPATH=. .venv/bin/python -m orchestrator_v2.cli run --task examples/repair-task.json --jev off
```

The plugin is optional and can be disabled with `V2_SUPERVISOR_DISABLED=1`.
