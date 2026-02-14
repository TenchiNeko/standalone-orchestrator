# 🧠 The Unbroken Method — Standalone Orchestrator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![No Frameworks](https://img.shields.io/badge/frameworks-none-red.svg)](#design-philosophy)
[![Local LLMs](https://img.shields.io/badge/LLMs-100%25%20local-purple.svg)](#architecture)

**Fully autonomous multi-agent coding system running on local LLMs. Zero API costs.**

> Plan → Build → Test → Debug → Fix — iterating until all tests pass, with no human in the loop and no cloud APIs. Runs entirely on your own hardware with Ollama.

Give it a task description. It plans, builds, tests, debugs, and fixes — iterating until all tests pass or it escalates to you with a detailed handoff report. No human in the loop. No cloud APIs. Everything runs on your own GPUs.

### Why This Exists

| | Frameworks (LangChain, CrewAI, AutoGen) | This Project |
|---|---|---|
| **Control** | Framework owns the loop, you fill in callbacks | You own every line of the control flow |
| **Dependencies** | 50+ packages, breaking changes monthly | Python stdlib + httpx |
| **Debugging** | Stack traces through 12 layers of abstraction | Read the Python, read the Ollama logs |
| **Lock-in** | Married to the framework's abstractions | Swap Ollama for vLLM/llama.cpp by changing one URL |
| **Cost** | Usually wraps OpenAI/Anthropic APIs | 100% local, zero API costs |

```
Task: "Build a bookmark manager REST API with Flask..."

ITERATION 1
├─ EXPLORE    → Analyze requirements, identify patterns
├─ PLAN       → Generate DoD criteria with verification commands
├─ BUILD      → Sequential micro-builds with multi-candidate sampling
│  ├─ models.py      ✅ (1st candidate, temp=0.0)
│  ├─ database.py    ✅ (1st candidate, temp=0.0)
│  ├─ validators.py  ✅ (1st candidate, temp=0.0)
│  ├─ app.py         ✅ (2nd candidate, temp=0.4)
│  ├─ test_models.py ✅ (2nd candidate, 9/9 tests pass)
│  ├─ test_database.py ✅ (1st candidate, 15/15 tests pass)
│  └─ test_app.py    ⚠️  (best of 4: 20/22 pass → edit repair)
└─ TEST       → DoD verification: 6/8 criteria passed

ITERATION 2 (dependency-aware retry)
├─ RCA        → "database.py has incomplete tag filtering"
├─ BUILD      → Rebuild database.py + cascade dependents
└─ TEST       → ✅ ALL DoD CRITERIA PASSED

🎉 TASK COMPLETED SUCCESSFULLY
```

---

## Architecture

```
Cortana (Dell 7920) — 4x GPU, 64GB VRAM
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Instance 1 (port 11435)          Instance 2 (port 11436)   │
│  Qwen 3 Coder 80B                    Qwen 2.5 Coder 7B/14B    │
│  GPUs 1+2+3 (48GB VRAM)               GPU 0 (16GB)              │
│  ┌──────────┐ ┌──────────┐       ┌──────────┐              │
│  │ Planner  │ │ Builder  │       │ Explorer │              │
│  │ (reason) │ │ (code)   │       │ Init     │              │
│  └──────────┘ └──────────┘       │ Tester   │              │
│        │            │            │ Librarian│              │
│        ▼            ▼            └──────────┘              │
│  ┌──────────────────────────────────────────┐              │
│  │         standalone_orchestrator.py        │              │
│  │  Plan → Build → Test → RCA → Retry       │              │
│  │  AST signatures │ Import graphs │ DoD     │              │
│  └──────────────────────────────────────────┘              │
│        │                                                    │
│  ┌─────┴─────┐  ┌──────────┐  ┌──────────────┐            │
│  │ RAG KB    │  │ Librarian│  │ Trace        │            │
│  │ 81 pats   │  │ Journal  │  │ Collector    │            │
│  │ 15K docs  │  │ Snippets │  │ JSONL export │            │
│  └───────────┘  └──────────┘  └──────────────┘            │
└─────────────────────────┬───────────────────────────────────┘
                          │ rsync (auto-sync after each run)
                          ▼
PVE Node (homeserver) — Qwen 2.5 Coder 7B
┌─────────────────────────────────────────────────────────────┐
│  Subconscious Daemon (24/7)                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ Analyze  │→ │ Reflect  │→ │ Curate   │                  │
│  │ sessions │  │ patterns │  │ playbook │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
│        │                           │                        │
│        ▼                           ▼                        │
│  Training pairs              playbook.json ──→ Cortana      │
│  (LoRA fine-tuning)          (injected into agent prompts)  │
└─────────────────────────────────────────────────────────────┘
```

Both Ollama instances run simultaneously — **no model swapping**. The 80B handles planning and code generation while the 7B/14B handles exploration, testing, and curation in parallel.

---

## How It Works

### The Loop

Each task runs through up to N iterations (default 3). Each iteration:

1. **EXPLORE** — Scan the workspace, identify existing code and patterns
2. **PLAN** — Generate Definition of Done (DoD) criteria with concrete verification commands
3. **BUILD** — Sequential micro-builds with multi-candidate sampling per file
4. **TEST** — Run all tests, verify DoD criteria
5. **RCA** — If tests fail, perform 5-Whys Root Cause Analysis with concrete edit suggestions
6. **RETRY** — Rebuild only the broken files + their dependents (cascade-aware)

### Multi-Candidate Sampling

Each file gets multiple candidates at different temperatures. The orchestrator picks the best one by score:

| Source files | Test files |
|---|---|
| 3 candidates (temp 0.0, 0.4, 0.8) | 4 candidates (temp 0.3, 0.6, 0.8, 1.0) |
| Score = compiles + imports + exports match | Score = tests passing / total tests |

If no candidate is perfect, **Wave 2** kicks in: error-aware re-sampling that includes the specific errors from Wave 1 in the prompt.

### Edit Repair

For files where the best candidate has most tests passing but a few failures, the orchestrator uses **SEARCH/REPLACE** blocks to surgically fix the failing tests without regenerating the whole file. Up to 3 rounds of iterative repair.

Small files (<80 lines) skip edit repair and go straight to whole-file regeneration — it's faster and more reliable for short files.

### Dependency-Aware Cascade Rebuilds

The orchestrator builds an **AST-based import dependency graph** at retry time:

```
models.py ← database.py ← app.py
              ↑               ↑
         validators.py    test_app.py
```

When `database.py` is identified as the root cause, the orchestrator automatically rebuilds `app.py`, `test_database.py`, and `test_app.py` too — preventing stale dependency failures.

### AST-Based Signature Extraction

Instead of dumping full source files into the manifest (context rot for small models), the orchestrator extracts **compact API contracts** using Python's `ast` module:

```
models.py exports:
  Bookmark(id, url, title, tags=..., created_at=...)

database.py exports:
  BookmarkDB[__init__(db_path=...), .get_all(page=..., tag=...), .delete(bookmark_id)]
  init_db(), get_db()
```

This gives the model exact function signatures and constructor parameters in ~200 tokens instead of ~9,000 tokens of full source.

---

## File Reference

### Core Orchestrator (runs on Cortana)

| File | Lines | Purpose |
|------|-------|---------|
| `standalone_main.py` | 131 | CLI entry point. Parses args, loads config, starts orchestrator |
| `standalone_orchestrator.py` | 4,351 | Main loop: plan → build → test → RCA → retry. Multi-candidate sampling, edit repair, AST signatures, import graphs, cascade rebuilds |
| `standalone_agents.py` | 4,642 | Agent implementations for all roles. Ollama HTTP client, tool-use parsing, prompt construction, context injection |
| `standalone_config.py` | 272 | Dual-instance Ollama config. Model routing, GPU assignments, agent role mapping |
| `standalone_models.py` | 191 | Data models: `AgentResult`, `TaskState`, `BuildStep` |
| `standalone_session.py` | 135 | Session state persistence (JSON) |
| `standalone_memory.py` | 197 | Cross-session memory management |
| `standalone_trace_collector.py` | 468 | Records build/test/RCA failures as structured JSONL for analysis and LoRA training |
| `kb_client.py` | ~200 | RAG Knowledge Base client. Queries pattern matches and documentation chunks on port 8787 |
| `librarian.py` | 693 | Post-session curation using 7B model. Extracts journal entries, code snippets, and error patterns from completed sessions |
| `librarian_store.py` | 428 | SQLite storage for librarian data (journal entries, code snippets) |
| `playbook_reader.py` | ~150 | Reads the evolving playbook from the subconscious daemon. Injects top-scored bullets into agent prompts by role |

### Agent Prompts

| File | Role | Used By |
|------|------|---------|
| `prompts/initializer.txt` | Set up workspace, git repo, venv, install deps | Qwen 7B/14B |
| `prompts/explore.txt` | Analyze requirements, identify files and patterns | Qwen 7B/14B |
| `prompts/plan.txt` | Generate DoD criteria with verification commands | Qwen 80B |
| `prompts/build.txt` | Generate source code files | Qwen 80B |
| `prompts/build_markdown.txt` | Alternative build format for markdown output | Qwen 80B |
| `prompts/test_gen.txt` | Generate test files against source | Qwen 80B |
| `prompts/test.txt` | Run tests, report results | Qwen 7B/14B |
| `prompts/edit_repair.txt` | Surgical SEARCH/REPLACE fixes | Qwen 80B |

### Subconscious Daemon (runs on PVE node)

| File | Purpose |
|------|---------|
| `subconscious-daemon/daemon.py` | Main daemon loop with priority queue. Analyzes sessions, updates playbook, extracts training pairs. Never idles |
| `subconscious-daemon/config.py` | Daemon configuration: paths, timing, thresholds, safety limits |
| `subconscious-daemon/playbook.py` | ACE-style evolving playbook. Delta updates, deduplication, quality scoring, stale pruning |
| `subconscious-daemon/ollama_client.py` | Async Ollama HTTP client for the daemon's 7B model |
| `subconscious-daemon/session_scanner.py` | Watches `/shared/sessions` for completed orchestrator sessions, parses into structured traces |
| `subconscious-daemon/deploy.sh` | Installs daemon as systemd service on PVE node |
| `subconscious-daemon/seed_playbook.py` | Pre-loads playbook with known patterns from debugging sessions |

### Sync Scripts

| File | Where | Purpose |
|------|-------|---------|
| `sync-session.sh` | Cortana | Pushes completed sessions to PVE node + pulls latest playbook back |
| `backfill-sessions.sh` | Cortana | Syncs all existing `/tmp/bookmark-*` sessions to PVE |

---

## The Knowledge Stack

Three layers of accumulated knowledge, each operating at a different timescale:

### Layer 1: RAG Knowledge Base (static, curated)
- **81 error→solution patterns** extracted from past failures
- **15,500+ documentation chunks** (CPython, Flask, pytest, sqlite3, dataclasses)
- Queried per-file during builds: "KB provided proactive context for database.py"
- Runs as a systemd service on port 8787

### Layer 2: Librarian (per-session, 7B curator)
- Runs after each session completes (success or failure)
- Uses the 7B model to extract **journal entries** (lessons learned) and **code snippets** (reusable patterns)
- Stored in persistent SQLite — survives across benchmark runs
- Injected into future sessions as supplementary context

### Layer 3: Subconscious Daemon (continuous, cross-session)
- Runs 24/7 on a separate node with its own 7B model
- Implements the **ACE (Agentic Context Engineering)** framework from Stanford/SambaNova
- Maintains an evolving **playbook** of bullet-point heuristics
- Each bullet has helpful/harmful counters tracked by real test outcomes
- Priority queue ensures most valuable work happens first:

| Priority | Task | Trigger |
|----------|------|---------|
| P0 | Analyze new session | Session completed |
| P1 | Update playbook | After analysis |
| P2 | Extract training pairs | Successful sessions |
| P3 | Re-analyze old sessions | Every 24h (multi-epoch) |
| P7 | Self-evaluate playbook | Nightly (prune stale/harmful bullets) |

---

## Hardware Requirements

### Minimum (single GPU)
- 1x GPU with 24GB+ VRAM (RTX 3090, RTX 4090)
- Run a single Ollama instance with a 14B-34B model
- Modify `standalone_config.py` to point both roles at the same endpoint

### Recommended (multi-GPU, as built)

```
Cortana (Dell 7920 Workstation)
├─ GPU 0: RTX 5060 Ti 16GB  → Ollama Instance 2 (port 11436) → Qwen 2.5 Coder 7B
├─ GPU 1: RTX 3090 24GB     → Ollama Instance 1 (port 11435) → Qwen 3 Coder 80B
├─ GPU 2: RTX 4070 Super 12GB → Ollama Instance 1 (tensor parallel)
├─ GPU 3: RTX 4070 Super 12GB → Ollama Instance 1 (tensor parallel)
└─ Total: 64GB VRAM, no model swapping

PVE Node (homeserver)
└─ Any GPU with 8GB+ → Ollama → Qwen 2.5 Coder 7B → Subconscious daemon
```

---

## Quick Start

### 1. Set up Ollama instances

```bash
# Instance 1: Heavy reasoning (80B)
CUDA_VISIBLE_DEVICES=1,2,3 OLLAMA_HOST=0.0.0.0:11435 ollama serve

# Instance 2: Fast agents (7B/14B)
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11436 ollama serve

# Pull models
OLLAMA_HOST=localhost:11435 ollama pull qwen3-coder:80b
OLLAMA_HOST=localhost:11436 ollama pull qwen2.5-coder:7b
```

### 2. Start the RAG Knowledge Base

```bash
# If you have the KB server set up:
sudo systemctl start rag-kb
# Runs on port 8787
```

### 3. Run a task

```bash
cd ~/standalone-orchestrator

python3 standalone_main.py \
  "Build a bookmark manager REST API with Flask. Features: add/remove/update bookmarks
   with URL, title, tags. Search by tag. Pagination. Input validation. SQLite storage." \
  --max-iterations 3 \
  --working-dir /tmp/bookmark-test
```

### 4. (Optional) Set up the subconscious daemon

```bash
# On PVE node:
cd subconscious-daemon
bash deploy.sh
python3 seed_playbook.py       # Pre-load known patterns
sudo systemctl start subconscious

# On main node:
bash sync-session.sh /tmp/bookmark-test   # Push session for analysis
```

---

## Design Philosophy

This project follows the **library approach** — not the framework approach. There are no base classes to inherit from, no decorators to register agents, no YAML-driven pipelines. The orchestrator is a Python script that calls Ollama's HTTP API directly.

**Why no frameworks?**

- **Full control over execution flow.** When the model generates broken JSON, we handle it. When a file needs 5 retries at different temperatures, we control that loop. Frameworks abstract this away and break when edge cases hit.
- **No dependency rot.** The entire system depends on: Python stdlib, `httpx`, `ollama` HTTP API. That's it. No LangChain, no CrewAI, no AutoGen, no vendor lock-in.
- **Debuggable.** When something fails, you read the Python code and the Ollama logs. No framework magic, no middleware chains, no callback hell.
- **Portable.** Swap Ollama for vLLM, llama.cpp, or any OpenAI-compatible API by changing the HTTP endpoint in `standalone_config.py`. The orchestration logic doesn't care what serves the model.

---

## Key Techniques

| Technique | Source | Implementation |
|-----------|--------|----------------|
| Multi-candidate sampling | Aider, AlphaCode | Temperature sweep per file, best-of-N selection |
| AST-based repo maps | Aider (tree-sitter) | `_extract_signatures_ast()` in orchestrator |
| Import dependency graphs | Custom | `_build_import_graph()` + `_get_dependents()` |
| Edit-based repair | Aider (SEARCH/REPLACE) | 5-layer fuzzy matching for surgical fixes |
| Spec-anchored TDD | EvalPlus, TiCoder | Tests written against task spec, not source |
| Context engineering | Anthropic research | Compact API contracts to prevent context rot |
| Localization → Repair → Validation | Agentless (UIUC) | RCA identifies root cause, retry fixes targeted files |
| ACE playbook evolution | Stanford/SambaNova | Subconscious daemon with delta updates + quality tracking |
| Architect/Editor split | Aider | 80B reasons about the problem, 7B/14B handles fast execution |

---

## Project Status

**v1.0** — Active development. The orchestrator successfully completes multi-file Flask REST API tasks with 4-8 source+test files in 1-3 iterations. Typical benchmark: bookmark manager API with models, database, validators, routes, and comprehensive tests.

Known limitations:
- Edit repair SEARCH/REPLACE matching has ~50% apply rate with 80B models (fuzzy matching helps but isn't perfect)
- Test files for complex endpoints (app.py with 6+ routes) remain the hardest to generate correctly on first attempt
- The subconscious daemon's training pair extraction is implemented but LoRA fine-tuning pipeline is not yet automated end-to-end

---

## Star History

If this project is useful to you, a ⭐ helps others find it.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with patience, mass quantities of GPU hours, and zero framework dependencies by [@TenchiNeko](https://github.com/TenchiNeko).*

---

### Keywords

`autonomous-coding` `ai-agents` `local-llm` `ollama` `multi-agent` `code-generation` `self-improving` `test-driven` `no-framework` `multi-gpu` `qwen` `agentic` `orchestrator` `swe-bench` `ace-framework`
