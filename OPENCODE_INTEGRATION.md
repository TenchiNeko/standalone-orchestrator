# OpenCode hosted mode

This project-local integration keeps OpenCode as the interactive host and
uses the existing Python v2 core for policy, evidence, mutation state, and
deterministic finalization. It does not replace the standalone CLI.

## Verified host API

The installed host is OpenCode **1.18.31** with
`@opencode-ai/plugin` **1.14.48**. The plugin uses these installed hooks:

- `tool.execute.before` — receives `tool`, `sessionID`, and `callID`; the
  mutable `output.args` is inspected before execution.
- `tool.execute.after` — receives the tool arguments and result metadata.
- `event` — records bounded session/message lifecycle facts.
- `experimental.chat.system.transform` — adds the short supervision
  contract to the model system messages.
- `experimental.session.compacting` — injects a compact v2 fact summary when
  OpenCode compacts context.

The custom-tool mechanism is the installed `tool({description, args,
execute})` API. The project plugin exposes `orchestrator_start`,
`orchestrator_status`, `orchestrator_evidence`, `orchestrator_finalize`, and
`orchestrator_symbols`.

`tool.execute.before` has no typed reject result in this host API. The plugin
throws a hook error for deterministic BLOCK decisions; OpenCode reports the
tool as failed and, in the disposable test, did not execute the blocked read.
This is a host limitation, not a claim that every built-in tool can be
perfectly intercepted. A fork would be required for a stronger universal
pre-execution gate.

The plugin sees tool arguments/results and session IDs. When an OpenCode
event includes numeric `part.tokens` fields, the bridge records those exact
values; there is no separate guaranteed typed token-usage hook. JSON event
output can still be used for an external report. There is no plugin hook here
that authoritatively declares a final assistant completion.

## Bridge and state

The TypeScript plugin starts one persistent local child:

```text
python3 -u orchestrator_v2/bridge.py
```

It uses JSONL over stdin/stdout, never a network listener. Each project is
mapped to its own v2 SQLite state directory under
`$V2_SUPERVISOR_STATE/<workspace-hash>/`. The bridge forwards calls to the
existing `Controller`, `StateStore`, mutation journal, symbol index, and
completion gate. No cloud service or second model is started.

The active OpenCode project directory (`ToolContext.directory`) is used as
the workspace root. Absolute OpenCode file paths are normalized under that
root before checking the contract.

## Disposable configuration

The proof configuration is
`integration/opencode-disposable/opencode.json`. It is not installed in the
global OpenCode configuration. To run a disposable session without loading
global plugins:

```bash
export OPENCODE_DISABLE_DEFAULT_PLUGINS=1
export OPENCODE_DISABLE_PROJECT_CONFIG=1
export OPENCODE_CONFIG_CONTENT="$(python3 -c 'import json; print(json.dumps(json.load(open("integration/opencode-disposable/opencode.json")) ))')"
export V2_SUPERVISOR_STATE=/tmp/orchestrator-v2-opencode-state
/home/brandon/.opencode/bin/opencode run --format json --dir /path/to/fixture \
  --model orca-q6/orca27b-ultra-q6-mtp 'bounded task prompt'
```

The config references the existing local Qwen key file through OpenCode's
file-reference syntax; no key is copied into this repository.

## Safe workflow

1. Call `orchestrator_start` with a non-empty goal, permitted files/actions,
   required criteria, and (when applicable) the exact test command.
2. Use normal OpenCode reads/edits/tests. Before/after hooks record hashes,
   outputs, and mutation acknowledgements in SQLite.
3. If a mutation result is uncertain, the bridge records `unknown` and blocks
   another write until `orchestrator_evidence(kind="reconcile", path=...)`
   performs an authoritative readback. An unchanged hash is recorded as a
   confirmed failed mutation; a changed hash is acknowledged as applied.
4. Call `orchestrator_status` when state is unclear.
5. Call `orchestrator_finalize` before claiming completion. A successful
   check must match the latest permitted-file hash; edits make prior checks
   stale. Required criteria, visual evidence, and unresolved mutations are
   deterministic blockers.

Standalone equivalents remain available:

```bash
python3 -m unittest discover -s tests -q
PYTHONPATH=. .venv/bin/python -m orchestrator_v2.cli run --task examples/repair-task.json --jev off
```

## Disable / rollback

The integration is project-local. Remove the plugin path from the disposable
config (or set `V2_SUPERVISOR_DISABLED=1`) to disable it. No global
OpenCode configuration, model service, AgentMemory, OCR, or Flash fallback
was changed. The bridge state can be retained for audit or removed only from
the disposable state directory selected by `V2_SUPERVISOR_STATE`.

## Known limits

- OpenCode's typed hook cannot return a native deny object for arbitrary
  built-in tools; the throwing before-hook is best effort and must remain
  covered by fixture tests.
- Token usage is not exposed to this plugin hook.
- Session idle/end events are observable as generic events, not a trusted
  completion signal; explicit `orchestrator_finalize` remains required.
- Context compaction can receive a fact summary, but the plugin cannot take
  ownership of OpenCode's compaction algorithm.
- The current Qwen model can still loop or stop without finishing a task;
  hosted mode records that outcome rather than treating it as completion.
