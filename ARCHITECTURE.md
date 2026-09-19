# Standalone Orchestrator v2

The controller owns `INTAKE → INSPECT → PLAN → IMPLEMENT → TEST → REVIEW → FINAL_VERIFY → COMPLETE` and routes uncertainty to `BLOCKED`/`RECONCILE`. Only controller code can change phase or verify criteria.

`state.sqlite3` stores task payloads, append-only events, and evidence with source-tree hashes. Test/review evidence is stale after an edit. Artifacts remain ordinary files under the project state directory.

The Qwen adapter is a single authenticated OpenAI-compatible request path. The controller serializes calls; it does not launch a second backend. Native tool calls must pass `ToolRegistry` before reaching `WorkspaceTools`. Workspace paths are resolved beneath the disposable root, commands are allowlisted, child process groups are cancellable, and tests run in an available bubblewrap namespace with no host home/NAS mount. Secrets are not copied into the workspace. The implementation worker is bounded by the task call budget and stops on a response without a tool call.

OCR starts in delegation mode: deterministic `ocr delegate preview/rule` selection only. Findings are candidates and require independent checks. OCR-managed loops are deliberately not nested into v2.

Decision supervision is fully local by default. `--decision-provider off` is
the deterministic baseline; `--decision-provider nanojev --decision-mode
shadow` records NanoJev distributions without routing changes. NanoJev has no
authority over mutations, retries, tests, scope, or completion. The TypeSafe
Jev adapter remains optional reference code (`--jev` is off by default) and
does not require a credential for the normal install.

Vision is explicit: `BrowserTools` records URL, viewport, capture time/path, and SHA-256; its local-only sequence supports bounded navigate/click/type/inspect/screenshot steps; `QwenAdapter.chat_with_image` sends screenshot bytes as an image input. A filename, DOM text, or pixel-analysis result is not accepted as visual evidence.

`ContextGC` is a reversible shadow-first block selector. It stores exact
tool evidence locally and can emit recall pointers only when explicitly
enabled. `BrowserCandidateRouter` separates local control extraction,
typed candidate selection, deterministic Playwright execution, and fresh
verification. `ModelRouter` currently records only future route suggestions
(`ROUTINE_QWEN`, `HARD_QWEN`, `ESCALATE_STRONG_MODEL`,
`DETERMINISTIC_ONLY`); the effective route remains deterministic.
