# Standalone Orchestrator v2

This is a bounded local supervisor built alongside the historical project. It does not start daemons, synchronize sessions, train models, or replace OpenCode.

```bash
cd /home/brandon/projects/standalone-orchestrator-v2
.venv/bin/pip install -e .
.venv/bin/python -m unittest -v tests.test_v2
QWEN_API_KEY_FILE=/home/brandon/.config/opencode/secrets/orca-q6.key .venv/bin/python -m orchestrator_v2.cli doctor
PYTHONPATH=. QWEN_API_KEY_FILE=/home/brandon/.config/opencode/secrets/orca-q6.key .venv/bin/python -m orchestrator_v2.cli run --task examples/task.json --jev off
PYTHONPATH=. QWEN_API_KEY_FILE=/home/brandon/.config/opencode/secrets/orca-q6.key .venv/bin/orchestrator-v2 run --task examples/repair-task.json --jev off
```

The example writes only `.orchestrator-v2/state.sqlite3` and uses the existing local Qwen endpoint. For SSH-safe execution use `scripts/run-local.sh`; it starts one tmux session and never starts a model backend.

The project-local TypeSafe SDK is installed in `.venv`; no key is required for `--jev off`. OCR remains delegation-only and is called only for deterministic file/rule selection. The implementation worker uses only native `read_file`, `write_file`, and coordinator-selected `run_tests`; tests are sandboxed with bubblewrap when available.

## OpenCode hosted mode

`opencode-plugin/orchestrator-supervisor.ts` and
`orchestrator_v2/bridge.py` provide a project-local OpenCode adapter,
originally verified on 1.18.31 and regression-tested on the current 1.18.32.
OpenCode owns the session/model/tool loop while v2 records policy, evidence,
mutation uncertainty, stale tests, traces, and deterministic finalization.
The disposable proof configuration is under
`integration/opencode-disposable/`; see `OPENCODE_INTEGRATION.md`. The
preferred interactive path is `opencode-v2`: OpenCode supplies the normal
session/model/tool loop while the Python core supplies deterministic contracts,
mutation/evidence tracking, stale-test invalidation, and finalization. Standalone
mode remains available for controlled experiments and as the fallback path.

Rollback/removal is scoped to this directory: `scripts/stop-local.sh` stops only the v2 tmux session; deleting `.orchestrator-v2/` removes v2 runtime state, while deleting this v2 directory removes the v2 installation. Historical source directories, the Qwen service, and global OpenCode configuration are not touched.
