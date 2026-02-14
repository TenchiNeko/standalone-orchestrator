# Benchmark Results

All benchmarks run on the same hardware and task to track orchestrator improvements across versions.

## Hardware

- **GPU 0:** RTX 5060 Ti 16GB → Ollama Instance 2 (port 11436) → Qwen 2.5 Coder 7B
- **GPU 1–3:** RTX 3090 24GB × 3 → Ollama Instance 1 (port 11435) → Qwen 3 Coder 80B (tensor parallel)
- **CPU:** Dual Xeon (24 cores / 48 threads)
- **RAM:** 256GB DDR4

## Standard Benchmark Task

**"Bookmark Manager REST API"** — Level 5 on our difficulty scale (REST API + database + validation).

> Build a bookmark manager REST API with Flask. Features: add/remove/update bookmarks with URL, title, tags. Search by tag. Pagination. Input validation. SQLite storage. Structure: models.py (Bookmark dataclass), database.py (CRUD operations), validators.py (input validation), app.py (Flask routes). Test files: test_models.py, test_database.py, test_validators.py, test_app.py.

4 source files, 4 test files, 8 total. `--max-iterations 3`.

---

## Results Summary

| Version | Date | DoD | Tests Passing | Source 1st-Cand | Iterations | Duration | Outcome |
|---------|------|-----|---------------|-----------------|------------|----------|---------|
| v0.9.9a | 2026-02-13 | 2/2 | 41/43 (95%) | 3/4 | 1 | ~45 min | ✅ Pass (reduced task) |
| v0.9.9b | 2026-02-13 | 2/8 | 46/70 (66%) | 3/4 | 3 | ~2h+ | ❌ Fail |
| v1.0 | 2026-02-14 | 6/8 | 66/79 (84%) | 4/4 | 1 (crashed iter 2) | ~1h 15m | ⚠️ Crash (missing field) |
| **v1.0c** | **2026-02-14** | **8/8** | **64/65 (98%)** | **4/4** | **2** | **1h 14m** | **✅ Pass** |

## Detailed Breakdown: v1.0c (Latest)

### Iteration 1 — Build Phase

| File | Type | Result | Candidate | Notes |
|------|------|--------|-----------|-------|
| models.py | source | ✅ 3/3 | 1st (t=0.0) | Clean first-candidate |
| database.py | source | ✅ 3/3 | 1st (t=0.0) | Clean first-candidate |
| validators.py | source | ✅ 3/3 | 1st (t=0.0) | Clean first-candidate |
| app.py | source | ✅ 3/3 | 1st (t=0.0) | Clean first-candidate |
| test_models.py | test | ✅ 5/5 | 1st (t=0.3) | All tests pass immediately |
| test_validators.py | test | ✅ 30/30 | 1st (t=0.3) | All tests pass immediately |
| test_database.py | test | ⚠️ 17/18 | best of 4 | Edit repair failed (0/3 rounds) |
| test_app.py | test | ⚠️ 11/27 | best of 4 | Edit repair failed (0/3 rounds) |

**Iteration 1 DoD:** 3/8 — failed on test_database.py and test_app.py

### RCA Analysis

Root cause identified: dual `BookmarkDB(':memory:')` instances — `create_app()` creates its own DB while routes reference module-level `db`. SQLite `:memory:` is per-connection, so test data isn't visible to the app.

**Action:** Refactor to `get_db()` pattern with dependency injection via `app.config['DATABASE']`.

### Iteration 2 — Targeted Rebuild

| File | Action | Result |
|------|--------|--------|
| models.py | ⏭️ Skipped (verified) | — |
| database.py | ⏭️ Skipped (verified) | — |
| validators.py | ⏭️ Skipped (verified) | — |
| app.py | 🔄 Rebuilt with get_db() | ✅ 1st candidate |
| test_models.py | ⏭️ Skipped (passing) | — |
| test_validators.py | ⏭️ Skipped (passing) | — |
| test_database.py | 🔄 Rebuilt | ✅ 4th candidate |
| test_app.py | 🔄 Rebuilt | ⚠️ 21/22 (best of 4) |

**Iteration 2 DoD:** 8/8 — all criteria passed

**Final:** 64/65 individual tests passing (98.5%)

## Version Progression

### Source File Quality (1st-candidate success rate)

```
v0.9.9a  ███████████████░  3/4  (75%)
v0.9.9b  ███████████████░  3/4  (75%)
v1.0     ████████████████  4/4  (100%)
v1.0c    ████████████████  4/4  (100%)
```

### test_app.py (hardest file — Flask integration tests)

```
v0.9.9a  (not tested — reduced task)
v0.9.9b  ░░░░░░░░░░░░░░░░  0/1   import_error (dead end)
v1.0     ████████████░░░░  20/29  (69%)
v1.0c    ███████████████░  21/22  (95%) → DoD pass via get_db() fix
```

### Overall Test Pass Rate

```
v0.9.9a  ████████████████  41/43  (95%)  — but reduced task scope
v0.9.9b  ██████████░░░░░░  46/70  (66%)
v1.0     █████████████░░░  66/79  (84%)
v1.0c    ████████████████  64/65  (98%)
```

## Key Improvements by Version

**v0.9.9a → v0.9.9b:** Added multi-candidate sampling, temperature sweep, edit repair. Increased task complexity but pass rate dropped — exposed test generation weakness.

**v0.9.9b → v1.0:** AST-based signature extraction (compact API contracts instead of full source). Source files went from 3/4 to 4/4 first-candidate. test_app.py went from import_error to 20/29 passing.

**v1.0 → v1.0c:** Fixed `rca_history` crash bug. RCA successfully identified and fixed the dual-DB pattern. Dependency-aware retry skipped verified files. 98.5% pass rate.

## Difficulty Scale

| Level | Description | Example | Status |
|-------|-------------|---------|--------|
| 1 | Single function | Fibonacci with tests | ✅ Trivial |
| 2 | Single class with tests | Linked list | ✅ Trivial |
| 3 | Multi-file, known patterns | Miniqueue | ✅ Passing |
| 4 | Inter-module state, file I/O | Task tracker CLI with JSON | 🔲 Not tested |
| 5 | REST API + database + validation | **Bookmark manager** | **✅ 8/8 DoD, 98.5%** |
| 6 | Complex business logic, auth, edge cases | Expense tracker with JWT | 🔲 Next target |
| 7 | Async, state machines, protocols | Chat server with WebSocket | 🔲 Future |
| 8+ | Concurrency, distributed systems | Job scheduler, CRDT editor | 🔲 Future |

## Running Benchmarks

```bash
# Standard benchmark (Level 5)
make benchmark

# Specific task
python3 standalone_main.py "your task" --max-iterations 3 --working-dir /tmp/test-run

# Full benchmark suite (5 tasks)
python3 benchmark.py --suite standard --output BENCHMARKS.md
```
