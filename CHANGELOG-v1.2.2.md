# v1.2.2 Hotfix — Regression Fix (2026-02-17)

## Summary

Emergency hotfix reverting v1.2.1 regressions that broke all benchmark tasks.
Restores from pre-v1.2.1 backup with only safe v1.2.1 changes cherry-picked.

## Root Cause

The v1.2.1 patch introduced three critical regressions in `standalone_agents.py`:

### BUG 1 — f-string prompt interpolation destroyed (CRITICAL)

20+ f-string prompts had their `f` prefix stripped (likely a linter auto-fix for
F541 "f-string without placeholders" applied globally instead of selectively).
Template variables like `{goal}`, `{state.goal}`, `{filename}`, `{manifest}`,
`{kb_section}` became literal text. The model received `"{goal}"` instead of
the actual task description, causing it to hallucinate unrelated code.

**Observed**: Calculator task (L2) produced Bookmark Manager code in all 3 attempts
across 60+ minutes. The model literally could not see the task spec.

### BUG 2 — Ollama options dict gutted (CRITICAL)

The `options` dict in both `_call_ollama_chat` methods was reduced to only
`temperature` and `num_predict`. Removed:

| Parameter        | Default without it    | Impact                              |
|------------------|-----------------------|-------------------------------------|
| `num_ctx`        | 2048 tokens           | Prompts silently truncated to 2K    |
| `repeat_penalty` | 1.1 (Ollama default)  | Qwen-Next degrades with any penalty |
| `top_p`          | 1.0 (Ollama default)  | Sampling distribution changed       |

The `num_ctx` omission alone is fatal — Ollama defaults to 2048 tokens, so
the 128K context window configured in `standalone_config.py` was never applied.
This is the same bug documented on r/LocalLLaMA and warned about in the Qwen3
deployment guide.

### BUG 3 — Feature removal (MODERATE)

Several v1.2.0 features were removed without replacement:

- `/think` and `/no_think` Qwen3 thinking mode injection
- `FLASK_GOLDEN_SNIPPET` expert context for Flask tasks
- `EDIT_REPAIR_SCHEMA` structured output for edit repair
- Dependency API contract extraction for cross-file test generation
- Increased dep_content read limits (4000→2000, 6000→3000)

## Fix Applied

**Strategy**: Restore `standalone_agents.py` from `pre-v1.2.1` backup, then
cherry-pick only the three safe v1.2.1 changes:

### Cherry-picked from v1.2.1

1. **httpx migration** — `requests` → `httpx` (Session, exceptions)
2. **`_validate_path()`** — Path traversal guard on all 4 tool handlers
   (`_write_file`, `_read_file`, `_edit_file`, `_list_directory`)
3. **Port 11434 → 11435** — Runtime fix for Cortana's Ollama instance 1

### Reverted from v1.2.1

- All f-string prompt restorations (20+ prompts)
- Full Ollama options dict (`num_ctx`, `repeat_penalty`, `top_p`, `min_p`, `num_keep`)
- Qwen3 thinking mode injection (`/think` for plan/build, `/no_think` for fast agents)
- `FLASK_GOLDEN_SNIPPET` constant and injection logic
- `EDIT_REPAIR_SCHEMA` constant
- Dependency API contract extraction in test build prompts
- Increased dependency read limits

## Files Changed

- `standalone_agents.py` — Restored + cherry-picks (4935 → 4944 lines)
- `standalone_agents.py.broken-v1.2.1` — Backup of broken version
- `librarian.py` — No changes needed (v1.2.1 only had clean httpx migration)

## Verification

After applying, run:

```bash
# Quick smoke test — Calculator L2 should pass in <5 min
python3 benchmark.py --tasks 1 --max-iterations 1

# Full suite
python3 benchmark.py
```

## Prevention

The v1.2.1 regressions were caused by a batch linter auto-fix. To prevent:

1. **Never apply F541 fixes globally** — many f-strings have `f` prefix for
   template variables resolved at runtime, not at parse time
2. **Always diff before committing** — `diff standalone_agents.py.pre-v1.2.1 standalone_agents.py`
3. **Run benchmark smoke test** after any patch — at minimum task 1 (Calculator L2)
   and task 2 (Miniqueue L3) as sanity checks
