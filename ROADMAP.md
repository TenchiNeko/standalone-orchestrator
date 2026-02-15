# Standalone Orchestrator — Roadmap

## Current State: v1.2 (February 2026)

Core autonomous coding loop fully operational: EXPLORE → PLAN → BUILD → TEST → RCA → RETRY

**Benchmark: 4/5 tasks passing (Levels 2-5), 133/141 tests (94%)**

### v1.2 Features
- Grammar-constrained structured output for edit repair (JSON schema, 100% well-formed)
- Thinking token injection (/think for plan+build, /no_think for fast agents)
- Repetition penalty disabled (1.0) — Qwen-Next sensitivity fix
- Nucleus sampling (top_p=0.95)
- AST-aware RAG chunking for code retrieval
- Self-play training data collection (JSONL pairs for QLoRA)
- Speculative decoding support (server-side config)
- All optimizations configurable via JSON overrides

### v1.1 Fixes
- num_ctx bug fix (128K→2K context truncation)
- AST-guarded stdlib imports (prevents injecting into try blocks)
- Hashline tag stripping from SEARCH blocks
- Edit repair fallback (structured → text graceful degradation)
- Fuzzy threshold tuning for SEARCH/REPLACE matching
- Branch audit: 6 fixes across all conditional paths

### v1.0 Foundation
- Multi-candidate sampling (temperature sweep, best-of-N)
- AST-based signature extraction (compact API contracts)
- Import dependency graphs with cascade rebuilds
- 5-Whys RCA with concrete edit suggestions
- Spec-anchored TDD (tests from spec, not source)
- Snapshot protection (rollback regressions)
- Error-aware Wave 2 re-sampling
- Trace collector for failure analysis

### Infrastructure
- Primary: Qwen3-Coder-Next 80B MoE (Q4_K_M) on RTX 3090 + 2x 4070 Super
- Secondary: Qwen 2.5 Coder 7B on RTX 5060 Ti
- RAG KB: 81 patterns, 15K+ doc chunks (port 8787)
- Subconscious daemon: ACE playbook evolution (PVE node)
- CI: GitHub Actions (ruff lint + mypy type checking)

---

## Immediate: v1.3 — Level 6 + Performance

### Beat the Level 6 benchmark
- [ ] **Task 5 completion** — Expense Tracker with JWT auth, budget limits, CSV export
  Currently timing out at 2h. Need to profile where time is spent and optimize.
- [ ] **Build phase profiling** — measure time per agent call, identify bottlenecks
- [ ] **Parallel candidate sampling** — sample multiple candidates concurrently (currently sequential)
- [ ] **Smarter retry targeting** — only rebuild files that actually failed, not all dependents

### Performance tuning
- [ ] **Speculative decoding deployment** — pair Qwen 7B as draft model for 80B inference
- [ ] **KV cache optimization** — tune num_keep for longer conversations
- [ ] **Prompt compression** — reduce prompt size for faster inference on large tasks

### Testing & reliability
- [ ] **End-to-end benchmark CI** — run Level 2 benchmark on every push (smoke test)
- [ ] **Structured output for all agents** — extend JSON schema approach beyond edit repair
- [ ] **Better timeout handling** — per-agent timeouts, not just per-task

---

## Near-term: v1.4 — Multi-Agent Coordination

### Parallel execution
- [ ] **Parallel agent execution** — run explore on secondary while plan runs on primary
- [ ] **Parallel micro-builds** — build independent files concurrently
- [ ] **Atomic state writes** — fcntl locks or atomic rename for parallel-safe state

### Smarter context
- [ ] **Dynamic context budgets** — scale per-section budgets based on actual context_window
- [ ] **Iteration summarization** — summarize old iterations instead of truncating
- [ ] **Cross-session learning** — inject librarian snippets from past similar tasks

### LoRA fine-tuning pipeline
- [ ] **Automated QLoRA training** — Unsloth pipeline from self-play JSONL pairs
- [ ] **Domain-specific adapters** — web_api.lora, cli.lora, database.lora
- [ ] **Hot-swap evaluation** — A/B test base model vs fine-tuned on benchmark

---

## Medium-term: v2.0 — Multi-Agent Teams

### tmux-based multi-agent spawning
- [ ] Each agent as a separate process in a tmux split
- [ ] Orchestrator manages lifecycle (spawn, monitor, kill)
- [ ] Visual: side-by-side panes showing agents working in parallel

### Structured inter-agent protocol
- [ ] Message types: task_assignment, progress_update, completion, error_report
- [ ] JSON schema for each message type
- [ ] Filesystem-based message passing (no HTTP, no database)

### Agent specialization per module
- [ ] For multi-module projects, assign different agents to different modules
- [ ] Each agent gets a subset of the plan
- [ ] Coordinator merges and runs integration tests

### Task dependency DAG
- [ ] Express "task B depends on task A" for multi-module builds
- [ ] Cycle detection, topological sort
- [ ] Enables parallel building of independent modules

---

## Long-term: v3.0 — External Integrations

### API-augmented planning
- [ ] **Claude/GPT-4 as planner** — Use cloud API for planning only, keep build/test local
- [ ] **LLM-as-Judge evaluation** — Score completed work beyond pass/fail
- [ ] **MCP integration** — Replace direct tool execution with Model Context Protocol

### Extended capabilities
- [ ] **Node.js / npm support** — Detect package.json, run npm install
- [ ] **Browser automation** — Playwright for end-to-end testing of web projects
- [ ] **Multi-language support** — TypeScript, Rust, Go (AST extraction per language)

### Advanced patterns
- [ ] **Intra-file PRM** — Build test files function-by-function with verification between each
- [ ] **GLM-4.7 Flash as build model** — test alternative code-focused models
- [ ] **Async/WebSocket project support** — Level 7+ benchmark tasks

---

## Completed Versions

<details>
<summary>v0.6.x — Sampling & Protection (2026-02-12)</summary>

- Snapshot protection (rollback regressions)
- Error-aware re-sampling (Wave 2 with error context)
- Second wave sampling (8 total candidates per test file)
- Trace collector (JSONL failure trajectories)
- Multi-patch sampling for test files (best-of-N with pytest validation)
</details>

<details>
<summary>v0.5.x — Verification & Context (2026-02-10)</summary>

- Post-build verification redesign (commands from workspace, not plan)
- Eliminated sanitizer pipeline (179 lines removed)
- RCA evidence enrichment (stderr, git diff, file contents)
- Context budget management (per-section caps)
- Retry fix mode (read_file → edit_file on retries)
- Testing pattern hints in build prompt
- Inner-loop lint guard (py_compile after every write)
</details>

<details>
<summary>v0.4.x — Structured Output & RCA (2026-02-08)</summary>

- Structured output for plan, explore, RCA agents
- LLM-based 5 Whys RCA
- Sequential micro-build architecture
- Direct RCA-to-Build injection (Spotify/Atla pattern)
- Auto-dependency installation
- .gitignore protection
</details>

<details>
<summary>v0.3.0 — Standalone System (2026-02-06)</summary>

- Direct Ollama API integration (no opencode dependency)
- Fallback tool-call parser for Qwen text-embedded JSON
- Safety rails on destructive commands
- Conversation memory with persistence
- Stuck-loop detection
</details>
