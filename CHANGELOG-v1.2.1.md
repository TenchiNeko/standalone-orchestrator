# v1.2.1 Patch Notes (2026-02-16)

## Bug Fixes

### 1. httpx Migration (requests → httpx)
**Files:** `standalone_agents.py`, `librarian.py`, `requirements.txt`

The requirements.txt listed `httpx` as the dependency but code imported `requests`.
Clean installs via `pip install -r requirements.txt` would fail at runtime.

Migrated all HTTP code to httpx:
- `requests.Session()` → `httpx.Client()`
- `requests.exceptions.ConnectionError` → `httpx.ConnectError`
- `requests.exceptions.Timeout` → `httpx.TimeoutException`
- Per-request timeouts preserved (900s Ollama, 300s Anthropic, 120s Librarian)

### 2. Path Traversal Guard
**File:** `standalone_agents.py`

Added `ToolExecutor._validate_path()` that resolves paths and verifies they
stay within `working_dir`. Called by `_write_file`, `_read_file`,
`_list_directory`, and `_edit_file`. Prevents LLM-generated paths like
`../../etc/passwd` from escaping the sandbox.

### 3. Per-Task Max Iterations (already applied in v1.2)
**File:** `benchmark.py`

Scaled `max_iterations` by task difficulty:
- Level 2-3: 3 iterations (always passes in 1-2)
- Level 4: 5 iterations (was hitting 36/37 at iter 3)
- Level 5: 5 iterations (high variance, 44-70 tests)
- Level 6: 5 iterations (passes at 3 but margins tight)

CLI `--max-iterations` still overrides per-task defaults.

## Documentation

### BENCHMARKS.md Updates
- Added variance analysis table from 3 benchmark runs
- Added v1.2.1 changelog section
- Fixed hardware spec (RAM was listed as 256GB, actually 64GB)

## Install

```bash
# From orchestrator root:
cp standalone_agents.py librarian.py requirements.txt benchmark.py BENCHMARKS.md ~/standalone-orchestrator/
cd ~/standalone-orchestrator
pip install -r requirements.txt
```
