# Benchmark Results

All benchmarks run on the same hardware to track orchestrator improvements across versions.

## Hardware

- **GPU 0:** RTX 5060 Ti 16GB → Ollama Instance 2 (port 11436) → Qwen 2.5 Coder 7B
- **GPU 1:** RTX 3090 24GB → Ollama Instance 1 (port 11435) → Qwen3-Coder-Next 80B MoE (Q4_K_M)
- **GPU 2:** RTX 4070 Super 12GB → Ollama Instance 1 (tensor parallel)
- **GPU 3:** RTX 4070 Super 12GB → Ollama Instance 1 (tensor parallel)
- **CPU:** Dual Xeon (24 cores / 48 threads)
- **RAM:** 64GB DDR4 ECC RDIMM
- **Context:** 131K (primary), 32K (secondary)

---

## Benchmark Suite (v1.2)

Five tasks of increasing difficulty, from simple single-class to complex multi-file REST API with auth.

| # | Task | Level | Files | Tests | Description |
|---|------|-------|-------|-------|-------------|
| 1 | Calculator | 2 | 2 | 5 | Single class with operations + tests |
| 2 | Miniqueue | 3 | 3 | 20 | Multi-file, known patterns |
| 3 | Task Tracker CLI | 4 | 5 | 38 | Inter-module state, file I/O, argparse |
| 4 | Bookmark Manager API | 5 | 8 | 78 | REST API + database + validation + pagination |
| 5 | Expense Tracker + Auth | 6 | 10 | ~90 | JWT auth, budget limits, CSV export, edge cases |

---

## v1.2 Results (2026-02-15)

**Model:** Qwen3-Coder-Next 80B MoE (Q4_K_M) + Qwen 2.5 Coder 7B
**Features:** Grammar-constrained structured edits, thinking tokens, repeat_penalty=1.0, AST-RAG

| Task | Level | Outcome | Tests | Iterations | Duration |
|------|-------|---------|-------|------------|----------|
| Calculator | 2 | ✅ Pass | 5/5 (100%) | 1 | 21m |
| Miniqueue | 3 | ✅ Pass | 20/20 (100%) | 1 | 11m |
| Task Tracker CLI | 4 | ✅ Pass | 38/38 (100%) | 3 | 21m |
| Bookmark Manager API | 5 | ✅ Pass | 70/78 (90%) | 2 | 29m |
| Expense Tracker + Auth | 6 | ✅ Pass | 145/159 (91%) | 3 | 170m |

**Overall: 5/5 tasks passing. 278/300 tests (93%) across all tasks.**

### Variance Analysis (3 runs)

| Task | Run 1 | Run 2 | Run 3 | Notes |
|------|-------|-------|-------|-------|
| Calculator (L2) | ✅ 5/5, 1 iter | ✅ 5/5, 1 iter | ✅ 5/5, 1 iter | Rock solid |
| Miniqueue (L3) | ✅ 16/16, 2 iter | ✅ 16/16, 1 iter | ✅ 20/20, 1 iter | Consistent |
| Task Tracker (L4) | ✅ 38/38, 3 iter | ❌ 36/37, 3 iter | ✅ 38/38, 3 iter | Stubborn edge case |
| Bookmark API (L5) | ✅ 70/78, 2 iter | ❌ 44/58, 3 iter | ✅ 70/78, 2 iter | High variance |
| Expense Tracker (L6) | ✅ 145/159, 3 iter | ✅ 110/127, 3 iter | ✅ 145/159, 3 iter | Passes but margins vary |

**Key finding:** Tasks 4-5 have iteration-count bottlenecks. With `max_iterations=3`, Task 4 sometimes stalls at 36/37 and Task 5 drops to 76% tests. Bumping to `max_iterations=5` for Level 4+ tasks (applied in v1.2.1) provides headroom for the model to recover from high-variance runs.

### Key Observations

**First-iteration success:** Tasks 1-2 pass in a single iteration with 100% test rates. The 80B model with thinking tokens generates correct code on the first try for simpler tasks.

**Task 3 (Level 4):** Required 3 iterations — the CLI argparse patterns are tricky for models. RCA correctly identified missing flags and the cascade rebuild fixed dependent test files.

**Task 4 (Level 5):** 2 iterations, 70/78 tests. The remaining 8 tests are edge cases in `test_app.py` (Flask integration). RCA identified and fixed the dual-DB pattern (`:memory:` per-connection issue).

**Task 5 (Level 6):** The hardest task — 5 source files + 5 test files with JWT auth, bcrypt, budget limits, CSV export, and complex edge cases. Passed in 3 iterations (170 minutes) with 145/159 tests (91%). Required the full retry budget but RCA successfully identified and fixed issues across iterations. 6/8 source files passed on first candidate.

---

## v1.0c Results (2026-02-14) — Bookmark Manager Only

**Model:** Qwen3-Coder-Next 80B MoE + Qwen 2.5 Coder 7B
**Task:** Bookmark Manager API (Level 5) only

| Metric | Result |
|--------|--------|
| DoD Criteria | 8/8 |
| Tests Passing | 64/65 (98.5%) |
| Source 1st-Candidate | 4/4 (100%) |
| Iterations | 2 |
| Duration | 1h 14m |

### v1.0c Iteration 1 — Build Phase

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

### v1.0c RCA → Iteration 2

Root cause: dual `BookmarkDB(':memory:')` instances — `create_app()` creates its own DB while routes reference module-level `db`. SQLite `:memory:` is per-connection, so test data isn't visible to the app.

**Action:** Refactor to `get_db()` pattern with dependency injection.

| File | Action | Result |
|------|--------|--------|
| app.py | 🔄 Rebuilt with get_db() | ✅ 1st candidate |
| test_database.py | 🔄 Rebuilt | ✅ 4th candidate |
| test_app.py | 🔄 Rebuilt | ⚠️ 21/22 (best of 4) |

**Final:** 8/8 DoD, 64/65 tests (98.5%)

---

## Earlier Versions (Bookmark Manager Only)

| Version | Date | DoD | Tests | Source 1st-Cand | Iterations | Duration | Outcome |
|---------|------|-----|-------|-----------------|------------|----------|---------|
| v0.9.9a | 2026-02-13 | 2/2 | 41/43 (95%) | 3/4 | 1 | ~45m | ✅ Pass (reduced task) |
| v0.9.9b | 2026-02-13 | 2/8 | 46/70 (66%) | 3/4 | 3 | ~2h+ | ❌ Fail |
| v1.0 | 2026-02-14 | 6/8 | 66/79 (84%) | 4/4 | 1 (crash) | ~1h 15m | ⚠️ Crash |
| **v1.0c** | **2026-02-14** | **8/8** | **64/65 (98%)** | **4/4** | **2** | **1h 14m** | **✅ Pass** |

---

## Version Progression

### Source File Quality (1st-candidate success rate)

```
v0.9.9a  ███████████████░  3/4  (75%)
v0.9.9b  ███████████████░  3/4  (75%)
v1.0     ████████████████  4/4  (100%)
v1.0c    ████████████████  4/4  (100%)
v1.2     ████████████████  4/4  (100%)  + structured edits
```

### test_app.py (hardest file — Flask integration tests)

```
v0.9.9a  (not tested — reduced task)
v0.9.9b  ░░░░░░░░░░░░░░░░  0/1   import_error (dead end)
v1.0     ████████████░░░░  20/29  (69%)
v1.0c    ███████████████░  21/22  (95%) → DoD pass via get_db() fix
v1.2     ████████████████  70/78  (90%) → full 8-file benchmark
```

### Overall Test Pass Rate

```
v0.9.9a  ████████████████  41/43  (95%)  — reduced task scope
v0.9.9b  ██████████░░░░░░  46/70  (66%)
v1.0     █████████████░░░  66/79  (84%)
v1.0c    ████████████████  64/65  (98%)
v1.2     ████████████████  278/300 (93%) — 5/5 tasks (Levels 2-6)
```

---

## Difficulty Scale

| Level | Description | Example | Status |
|-------|-------------|---------|--------|
| 1 | Single function | Fibonacci with tests | ✅ Trivial |
| 2 | Single class with tests | Calculator | ✅ 5/5, 1 iter |
| 3 | Multi-file, known patterns | Miniqueue | ✅ 20/20, 1 iter |
| 4 | Inter-module state, file I/O | Task tracker CLI | ✅ 38/38, 3 iter |
| 5 | REST API + database + validation | Bookmark manager | ✅ 70/78, 2 iter |
| 6 | Complex business logic, auth, edge cases | Expense tracker with JWT | ✅ 145/159, 3 iter |
| 7 | Async, state machines, protocols | Chat server with WebSocket | 🔲 Future |
| 8+ | Concurrency, distributed systems | Job scheduler, CRDT editor | 🔲 Future |

---

## Running Benchmarks

```bash
# Full benchmark suite (5 tasks)
make benchmark

# Quick smoke test (Level 2)
make benchmark-quick

# Level 5 only
make benchmark-l5

# Specific task with custom timeout
python3 benchmark.py --task 5 --timeout 10800

# List available tasks
python3 benchmark.py --list
```

---

## What Changed in v1.2.1 (Bug Fixes + Iteration Tuning)

| Fix | Before | After |
|---|---|---|
| Dependency mismatch | requirements.txt listed `httpx`, code used `requests` | Fixed: requirements.txt now lists `requests` |
| Path traversal guard | ToolExecutor accepted any path from LLM | Validates all paths resolve within working_dir |
| Per-task max iterations | Flat `max_iterations=3` for all tasks | Level 2-3: 3, Level 4-6: 5 (CLI still overrides) |

## What Changed in v1.2 (Inference Optimizations)

| Optimization | Before | After |
|---|---|---|
| Edit repair format | Free-text SEARCH/REPLACE (15-20% malformed) | JSON schema structured output (100% well-formed) |
| Thinking tokens | Not used | /think for plan+build, /no_think for fast agents |
| Repetition penalty | Ollama default (1.1) | Disabled (1.0) — Qwen-Next sensitive |
| Sampling | Default | top_p=0.95 nucleus sampling |
| Code retrieval | Line-based chunking | AST-aware semantic chunks |
| Training data | Not collected | JSONL pairs on successful tasks |
| Context window | Was defaulting to 2K (bug) | Fixed: 131K primary, 32K secondary |
