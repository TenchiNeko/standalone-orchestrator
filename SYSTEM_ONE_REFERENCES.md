# System-One reference review

These repositories were cloned read-only for source inspection. No cloud
integration, package, plugin, or model was installed from them.

| Project | inspected commit | useful pattern retained |
|---|---|---|
| [Canny](https://github.com/qkal/Canny) | `74bc3487370ae6d61cef69c5642cce504a8579cb` | append-only factual ledger; deterministic post-edit verification gate; model judgment can only relax a prose stop, never prove a check |
| [Winnow](https://github.com/GhalebDweikat/winnow) | `51d80b945c74c8384bc47fa817179f668289afd8` | conservative block-level relevance decisions; shadow mode; full-text cache and explicit recall; errors and uncertain scores stay visible |
| [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) | `1231850a0bf1a0c0341fe408ef1668dbbfdfac46` | separate browser perception, typed decision, bounded execution, and independent verification |
| [fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction) | `e3f262a7f4d42bd8dd32ced30d26176f7cb545b0` | pin recent/critical messages; fit state in stages; independently judge call/result retention; preserve recallable exact evidence |
| [OpenJev](https://github.com/razorback16/openjev) | `91d5005effcf8cc0ecccaa9538ceabbb130fef59` | Jev-compatible typed API, complete distributions, bounded batching/backpressure, deterministic request seeds, optional image questions |

## Applied to v2

### Completion evidence

`StateStore` now records an intake file manifest, source hashes, check
commands/results, and mutation outcomes. `Controller.completion_gate()` only
allows completion when a successful test/build-style evidence record has the
current permitted-file hash. A passing check from before a subsequent edit is
therefore stale. Contracts can set `visual_required`; a current
`visual_verify` evidence record is then mandatory. Unknown or missing mutation
outcomes route to review/blocking, and model prose never changes these facts.

### Context GC

`ContextGC` writes every block unchanged to a task-scoped artifact before any
decision. It supports `KEEP_VERBATIM`, `HIDE_BUT_RECALLABLE`,
`TRUNCATE_WITH_POINTER`, and `PIN`. Shadow mode is the default and returns the
original text while recording the proposed disposition. Active filtering is
opt-in only and requires a high provider probability; errors, recent
mutations, and pinned blocks always remain verbatim. `recall(ref)` validates
the local artifact path and returns the exact original bytes.

The local shadow fixture (`examples/context_gc_eval.py`) processed four blocks
(5,930 chars): model-facing size remained 5,930 chars, two error/recent-
mutation blocks were pinned, no blocks were hidden, and exact recall passed
for all four artifacts. Qwen prefill was intentionally not compared because
shadow mode does not rewrite the prompt.

### Browser routing

`BrowserTools.actionable_controls()` extracts a bounded local table of visible
buttons, links, textboxes, selects, and labels. `BrowserCandidateRouter` gives
the provider only those candidate IDs and permitted operations. It does not
accept selectors, JavaScript, shell commands, or coordinates from the model;
execution and post-action verification remain coordinator callbacks. The
router is shadow-only by default.

### Future model routing

`ModelRouter` defines `ROUTINE_QWEN`, `HARD_QWEN`, `ESCALATE_STRONG_MODEL`, and
`DETERMINISTIC_ONLY`. It records a typed provider suggestion but
`effective_route()` remains the deterministic default until a matched task
evaluation justifies promotion. Flash-Next is not started or changed.

## OpenJev feasibility

The inspected OpenJev implementation exposes `POST /v1/systemone` with
Noul/Choice/Score questions, probabilities/confidence, optional image inputs,
and a compatible `/v1/chat/completions` path. It uses DiffusionGemma
26B-A4B through vLLM, with seeded canvas reads, bounded concurrency, and
backpressure. The published local path expects roughly an 18 GB checkpoint
and at least 24 GB GPU memory, so it remains research-only on this host and
was not installed.
