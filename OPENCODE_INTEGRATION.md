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

Run from the intended project directory. Both policies refuse `/` and the
user's home directory before OpenCode starts, so an accidental `cd ~` cannot
trigger a broad source scan. An explicit human-only exception is available
for unusual maintenance work with `V2_SUPERVISOR_ALLOW_BROAD_WORKSPACE=1`;
the model cannot set this through `orchestrator_start`. Ordinary `opencode`
and unrelated global providers remain available unchanged.

Normal `opencode-v2` uses the light supervisor: it observes ordinary coding
actions and leaves repository discovery, shell, Git, and task-relevant file
changes available. `opencode-v2-strict` (or `V2_POLICY_MODE=strict`) retains
the older contract/allowlist behavior for intentionally constrained work;
`opencode-v2-light` selects light explicitly. The light launcher leaves useful
project plugins enabled by default; set `V2_SUPERVISOR_ISOLATE_PLUGINS=1` for
a disposable plugin-isolated run. Set `V2_OPENCODE_CONFIG_HOME` for a
disposable config. Roll back to strict with `opencode-v2-strict`; v2 SQLite
state is retained.

The promoted local Qwen backend has a **40960-token physical llama.cpp
context** while the generated supervised OpenCode model definition pins the
logical budget to `context=32768`, `input=28000`, and `output=2048`. This
headroom is intentional: OpenCode compacts at its logical budget while
transitional requests in the old 33--34K range can still reach the backend.
The backend uses native q8_0-K/q5_1-V Flash Attention and the validated
per-tensor CUDA0 overrides recorded in the shared workspace run.

The v2 generator also sets a local compaction policy of `auto=true`,
`prune=true`, and `reserved=4000`. OpenCode 1.18.31 computes the automatic
threshold as `limit.input - reserved` and includes cached prompt tokens in the
current total; inheriting the user's ordinary 12000 reserve therefore caused
this 28K-input model to compact at 16K and repeatedly re-enter above the same
threshold. The v2-specific 4K reserve yields a 24K threshold while leaving
headroom below the logical input cap. `V2_COMPACTION_RESERVED` is available as
a bounded per-launch experiment/rollback override. Ordinary `opencode` keeps
the user's global compaction settings unchanged.

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
After v2 records `COMPLETE`, later mutation, shell, or test actions for that
closed task are blocked. In light mode a new explicit `orchestrator_start`
releases the closed task naturally; strict mode retains the explicit end/new
contract workflow.

## Daily contract workflow

In light mode, Qwen works like a normal coding agent: it may discover and read
the repository broadly, use normal shell and Git, and create task-relevant
files without restarting a contract. `orchestrator_start` records the objective
and verification plan when known; observed paths and repository expansion are
journaled as `SCOPE_EXPANDED` telemetry, not treated as an ACL. Strict mode
retains the older workflow: before mutation/shell/test, start with exact
mutation files, permitted actions, a shell-free exact test argv, and a criterion
that the test can prove. `orchestrator_health`, `orchestrator_status`,
`orchestrator_symbols`, `orchestrator_end`, and `orchestrator_cancel` are
available as bounded local tools.

`orchestrator_status` reports the active workspace, exact project-relative
`permitted_files`, permitted actions, exact test argv, visual requirement,
budget counters, criterion states, phase, blockers, and compact usage counts.
It does not change the contract. If a path is rejected, inspect this scope
once; end the task and start a new exact contract when the scope must change.

Only observed execution with an observed exit status creates test evidence.
Strict mode requires the exact configured command; light mode classifies
observed verification commands while never treating `ls`, `echo`, prose, or a
model-supplied exit code as evidence. Evidence is tied to the current observed
file state and becomes stale after relevant edits. Unknown light-mode writes
are read back and reconciled automatically when their outcome is knowable;
strict mode retains the explicit reconcile step. Finalization returns
`COMPLETE`, `INCOMPLETE`, `BLOCKED`, or `NEEDS_REVIEW`; it never trusts prose.

Strict mode keeps the small audited read-only shell subset and blocks arbitrary
shell/Git. Light mode observes normal host shell use, including Git, while
journaling commands, exit status, repository changes, and which successful
commands were classified as verification. Neither mode lets shell inspection
alone satisfy completion. AgentMemory retrieval remains advisory context; it
is not completion evidence.

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
restart on a later request, and strict-mode fail-closed authorization. State is
SQLite under a workspace hash. Intake hashes only strict contract files or the
bounded observed light-mode scope, not the entire host workspace. No cloud
service or second Qwen process is used. The verified host currently requires
absolute imports of its installed plugin SDK; this is recorded as a portability
limitation rather than changing the global installation.

## Compaction and usage

Only an active task receives a stable compact JSON fact summary (capped at 1800
characters): task/phase/goal, policy and scope mode, observed project-relative
files/expansions/actions, test argv, visual requirement, budget, criteria,
evidence blockers, loop warnings, counts, and bounded usage. No event history
or hidden evidence is injected. Token parts are forwarded
with stable IDs where available and exact host fields are stored; the host does
not expose a separate guaranteed usage API.

## Standalone mode

The standalone runner remains available for controlled experiments:

```bash
python3 -m unittest discover -s tests -q
PYTHONPATH=. .venv/bin/python -m orchestrator_v2.cli run --task examples/repair-task.json --jev off
```

The plugin is optional and can be disabled with `V2_SUPERVISOR_DISABLED=1`.
