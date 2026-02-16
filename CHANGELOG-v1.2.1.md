# v1.2.1 Patch Notes (2026-02-16)

## Bug Fixes

### 1. httpx Migration (requests → httpx)
**Files:** `standalone_agents.py`, `librarian.py`, `requirements.txt`

Migrated all HTTP code to httpx (was listed in requirements.txt but code used requests):
- `requests.Session()` → `httpx.Client()`
- `requests.exceptions.ConnectionError` → `httpx.ConnectError`
- `requests.exceptions.Timeout` → `httpx.TimeoutException`

### 2. Path Traversal Guard
**File:** `standalone_agents.py`

Added `ToolExecutor._validate_path()` — resolves paths and blocks escapes
from working_dir. Applied to `_write_file`, `_read_file`, `_list_directory`, `_edit_file`.

### 3. Undefined `is_flask` Bug (F821)
**File:** `standalone_agents.py`

`is_flask` was used before being defined in `_generate_test_template()`.
Moved detection code above its first use. This was a real runtime bug that
would crash test template generation for Flask projects.

### 4. Lint Cleanup (54 errors → 0)
**Files:** `standalone_agents.py`, `librarian.py`

- Removed 4 unused imports (Any, asdict, DoDCriterion, shutil)
- Removed ~47 needless f-string prefixes on strings without placeholders
- Removed whitespace from blank lines
- All files now pass `ruff check` clean

### 5. Per-Task Max Iterations
**File:** `benchmark.py`

Level 2-3: 3 iterations | Level 4-6: 5 iterations

### 6. BENCHMARKS.md Updates
- Added variance analysis from 3 runs
- Added v1.2.1 changelog
- Fixed RAM spec (256GB → 64GB)

## Install

```bash
cd ~/standalone-orchestrator
tar xzf v1.2.1-patch.tar.gz
pip install -r requirements.txt
```
