# v1.2.1 Patch Notes (2026-02-16)

## Bug Fixes

### 1. httpx Migration (requests → httpx)
**Files:** `standalone_agents.py`, `librarian.py`, `requirements.txt`

### 2. Path Traversal Guard
**File:** `standalone_agents.py` — `ToolExecutor._validate_path()`

### 3. Undefined `is_flask` Bug (F821)
**File:** `standalone_agents.py` — moved detection before first use

### 4. Lint Cleanup (54 ruff errors → 0)
- Removed unused imports (Any, asdict, DoDCriterion, shutil)
- Removed ~47 needless f-string prefixes
- Removed whitespace from blank lines

### 5. mypy Type Fixes (23+13 errors → 0)
**Files:** `standalone_agents.py`, `librarian.py`

Round 1 (23 errors):
- `Optional[list]` for implicit optional (PEP 484)
- Type-annotated `per_file_results`, `test_report`, `individual_tests`,
  `structured_report`, `all_evidence`, `unmapped_criteria_indices`
- Fixed `best_sim` float/int, `storage_class`/`cls_name` str casts
- `type: ignore` for Anthropic dict shape, dod_results, join()

Round 2 (13 errors):
- Fixed 5 no-redef errors (removed duplicate type annotations in else branches,
  renamed `init_params` → `cls_params` in generic template)
- `run_edit_repair_structured` returns parsed `list` (not `AgentResult`)
  so orchestrator `apply_search_replace(filepath, result)` type-checks

### 6. Per-Task Max Iterations
Level 2-3: 3 iterations | Level 4-6: 5 iterations

## Install
```bash
cd ~/standalone-orchestrator && tar xzf v1.2.1-patch.tar.gz
```
