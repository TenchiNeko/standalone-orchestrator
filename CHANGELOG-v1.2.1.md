# v1.2.1 Patch Notes (2026-02-16)

## Bug Fixes

### 1. httpx Migration (requests → httpx)
**Files:** `standalone_agents.py`, `librarian.py`, `requirements.txt`

Migrated all HTTP code to httpx (was listed in requirements.txt but code used requests).

### 2. Path Traversal Guard
**File:** `standalone_agents.py`

Added `ToolExecutor._validate_path()` — resolves paths and blocks escapes from working_dir.

### 3. Undefined `is_flask` Bug (F821)
**File:** `standalone_agents.py`

`is_flask` was used before being defined in `_generate_test_template()`.
Moved detection code above first use. Runtime crash on Flask projects.

### 4. Lint Cleanup (54 ruff errors → 0)
- Removed 4 unused imports (Any, asdict, DoDCriterion, shutil)
- Removed ~47 needless f-string prefixes
- Removed whitespace from blank lines

### 5. mypy Type Fixes (23 errors → 0)
**Files:** `standalone_agents.py`, `librarian.py`

- Added `Optional[list]` for implicit optional parameter (PEP 484)
- Type-annotated `per_file_results`, `test_report`, `individual_tests`,
  `structured_report`, `all_evidence`, `unmapped_criteria_indices`
- Fixed `best_sim` float/int mismatch
- Cast `storage_class` and `cls_name` to `str` for re.escape/`.lower()`
- Typed `init_params` extraction to fix unpacking errors
- Added `str()` wrappers for dict `.get()` returns used as strings
- `type: ignore` for Anthropic tool_result mixed-type dict, dod_results
  dict shape, and join() on mixed list
- Added `run_edit_repair_structured` alias for orchestrator compat

### 6. Per-Task Max Iterations
Level 2-3: 3 iterations | Level 4-6: 5 iterations

## Install
```bash
cd ~/standalone-orchestrator
tar xzf v1.2.1-patch.tar.gz
pip install -r requirements.txt
```
