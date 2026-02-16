"""
librarian.py - The 7B Librarian: Post-Session Knowledge Curator

v0.9.0: Runs on the PVE node / Instance 2 (Qwen 14B or 7B) after each
orchestrator session completes. Acts as an intelligent filter that decides
what's worth remembering from a build session.

Three curation tasks:
  1. ERROR PATTERNS: Did the session fix an error? → auto-capture to patterns DB
  2. JOURNAL ENTRIES: Any strategic lessons or decisions? → write to journal DB
  3. CODE SNIPPETS: Any clean, reusable code worth saving? → write to snippets DB

The librarian uses structured output (Ollama format schema) to get deterministic
JSON responses from the 7B. No parsing regex needed.

Architecture:
    Build loop completes (success or failure)
        ↓
    Orchestrator calls librarian.curate_session(session_summary)
        ↓
    Librarian sends summary to 7B with curation prompts
        ↓
    7B returns structured JSON: {worth_storing: bool, entries: [...]}
        ↓
    Librarian writes approved entries to KB tables
        ↓
    Next session gets enriched context from past curated knowledge

Design principle: The 7B doesn't need to be smart. It just needs to be a
good FILTER. "Is this worth remembering? If yes, write a one-paragraph summary."
That's well within 7B/14B capability.
"""

import json
import logging
import time
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from standalone_config import ModelConfig
from librarian_store import (
    add_journal_entry,
    add_snippet,
    get_librarian_stats,
    init_librarian_tables,
    DEFAULT_DB_PATH,
)

logger = logging.getLogger(__name__)


# ── Session Summary (built by orchestrator, consumed by librarian) ──

@dataclass
class SessionSummary:
    """Everything the librarian needs to curate a completed session."""
    task_id: str
    goal: str
    outcome: str  # "success" or "failure"
    iterations: int
    total_duration_seconds: float = 0.0

    # What happened
    plan_summary: str = ""
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)

    # Error history (for pattern extraction)
    errors_encountered: List[Dict[str, str]] = field(default_factory=list)
    # Each: {"error": "NameError: ...", "fix": "import uuid", "iteration": 2}

    # Final code (for snippet extraction) — map of filename → content
    final_code: Dict[str, str] = field(default_factory=dict)

    # Failure history for lesson extraction
    failure_summaries: List[str] = field(default_factory=list)
    rca_summaries: List[str] = field(default_factory=list)


# ── Structured Output Schemas (Ollama format) ───────────────────

PATTERN_CURATION_SCHEMA = {
    "type": "object",
    "properties": {
        "patterns_to_store": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "error_pattern": {"type": "string"},
                    "solution": {"type": "string"},
                    "example": {"type": "string"},
                    "tags": {"type": "string"},
                },
                "required": ["error_pattern", "solution"],
            },
        },
    },
    "required": ["patterns_to_store"],
}

JOURNAL_CURATION_SCHEMA = {
    "type": "object",
    "properties": {
        "entries_to_store": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "lesson_type": {
                        "type": "string",
                        "enum": ["strategy", "approach", "pitfall", "architecture", "debugging"],
                    },
                    "domain": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
                "required": ["title", "content", "lesson_type", "domain"],
            },
        },
    },
    "required": ["entries_to_store"],
}

SNIPPET_CURATION_SCHEMA = {
    "type": "object",
    "properties": {
        "snippets_to_store": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "code": {"type": "string"},
                    "domain": {"type": "string"},
                    "tags": {"type": "string"},
                    "source_file": {"type": "string"},
                    "quality_score": {
                        "type": "string",
                        "enum": ["good", "excellent"],
                    },
                },
                "required": ["title", "description", "code", "domain"],
            },
        },
    },
    "required": ["snippets_to_store"],
}


# ── Curation Prompts ─────────────────────────────────────────────

PATTERN_CURATION_PROMPT = """You are a knowledge curator for an autonomous coding system.

A build session just completed. Below is a summary of errors that occurred and how they were fixed.

Your job: Decide which error→fix pairs are REUSABLE across future tasks.

STORE a pattern if:
- The error is a common Python/library mistake (import errors, type errors, API misuse)
- The fix is generalizable (not task-specific)
- A future task hitting the same error would benefit from this solution

SKIP a pattern if:
- The error was caused by task-specific logic (wrong variable name for this specific task)
- The fix only works in the exact context of this task
- The error is trivially obvious (syntax typo, missing colon)
- It duplicates a pattern that would already exist (very common errors)

Return an empty array if nothing is worth storing.

SESSION ERRORS AND FIXES:
{errors_json}

TASK CONTEXT: {goal}"""

JOURNAL_CURATION_PROMPT = """You are a knowledge curator for an autonomous coding system.

A build session just completed. Below is a summary of what happened.

Your job: Extract STRATEGIC LESSONS that would help future sessions work better.

STORE a lesson if:
- It's about an approach or architecture that worked well (or failed badly)
- It's about a common pitfall for a type of task (e.g., "Flask apps need app_context()")
- It would change how the planner approaches similar tasks in the future
- It took multiple iterations to figure out — so it's genuinely non-obvious

SKIP if:
- It's just a restatement of the task description
- It's about a one-off implementation detail
- It's common knowledge any developer would already know
- It doesn't generalize beyond this exact task

Keep entries SHORT — 1-3 sentences for the content. The title should be actionable.

For lesson_type use:
- "strategy": high-level approach decisions
- "approach": specific implementation techniques that worked
- "pitfall": things that caused failures / wasted iterations
- "architecture": structural decisions about file layout, module design
- "debugging": insights about how to diagnose specific types of failures

For domain, use the primary technology: "flask", "cli", "database", "testing", "api", "general"

Return an empty array if no strategic lessons are worth storing.

SESSION SUMMARY:
Task: {goal}
Outcome: {outcome} (after {iterations} iterations)
Plan: {plan_summary}
Files: {files}
Failures: {failures}
RCA insights: {rca}"""

SNIPPET_CURATION_PROMPT = """You are a knowledge curator for an autonomous coding system.

A build session just completed successfully. Below are the final source files.

Your job: Decide if any of these files (or parts of them) are CLEAN, REUSABLE code patterns worth saving as reference for future tasks.

STORE a snippet if:
- It demonstrates a clean implementation of a common pattern (REST API, CLI parser, DB model)
- The code is well-structured and could be adapted for similar tasks
- It shows proper error handling, testing patterns, or library usage
- A future task could use this as a starting point or reference

SKIP if:
- The code is task-specific logic with no reuse value
- It's boilerplate that any model would generate identically
- The code quality is poor (hacky fixes, no error handling)
- It's a test file (unless it shows an especially good testing pattern)

Extract only the RELEVANT portion — don't store entire files if only one function is noteworthy.
Keep descriptions concise — what it does, when to use it.

For domain, use: "flask", "cli", "database", "testing", "api", "general"
For tags, use comma-separated keywords relevant to searching.

Return an empty array if nothing is worth storing.

FINAL SOURCE FILES:
{code_files}

TASK CONTEXT: {goal}"""


# ── The Librarian ────────────────────────────────────────────────

class Librarian:
    """
    Post-session knowledge curator using the 7B/14B model.

    Runs on Instance 2 (Qwen 14B on GPU 0) or PVE node (7B).
    Uses structured output for deterministic JSON responses.
    All operations are fail-safe — librarian failures never affect the build loop.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        db_path: str = DEFAULT_DB_PATH,
        kb_server_url: str = "http://localhost:8787",
    ):
        self.model_config = model_config
        self.db_path = db_path
        self.kb_server_url = kb_server_url.rstrip("/")
        self.session = httpx.Client(timeout=120)
        self.session.headers.update({"Content-Type": "application/json"})

        # Ensure librarian tables exist
        init_librarian_tables(db_path)

    def curate_session(self, summary: SessionSummary) -> Dict[str, int]:
        """
        Main entry point. Run all three curation tasks on a completed session.

        Returns dict of counts: {"patterns": N, "journal": N, "snippets": N}

        This is designed to be called from the orchestrator's _finalize_success()
        or _escalate() methods. It runs the 7B synchronously but could be made
        async if latency matters.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📚 LIBRARIAN: Curating session {summary.task_id}")
        logger.info(f"   Task: {summary.goal[:80]}")
        logger.info(f"   Outcome: {summary.outcome} ({summary.iterations} iterations)")
        logger.info(f"{'='*60}")

        start = time.time()
        counts = {"patterns": 0, "journal": 0, "snippets": 0}

        # Task 1: Error→Fix patterns (always, success or failure)
        if summary.errors_encountered:
            counts["patterns"] = self._curate_patterns(summary)

        # Task 2: Journal entries (always, success or failure)
        counts["journal"] = self._curate_journal(summary)

        # Task 3: Code snippets (only on success — failed code isn't worth saving)
        if summary.outcome == "success" and summary.final_code:
            counts["snippets"] = self._curate_snippets(summary)

        elapsed = time.time() - start
        total = sum(counts.values())
        logger.info(f"\n📚 LIBRARIAN COMPLETE: {total} entries stored "
                    f"(patterns={counts['patterns']}, journal={counts['journal']}, "
                    f"snippets={counts['snippets']}) in {elapsed:.1f}s")

        return counts

    # ── Task 1: Pattern Curation ─────────────────────────────

    def _curate_patterns(self, summary: SessionSummary) -> int:
        """Ask 7B which error→fix pairs are worth storing."""
        errors_json = json.dumps(summary.errors_encountered, indent=2)

        prompt = PATTERN_CURATION_PROMPT.format(
            errors_json=errors_json,
            goal=summary.goal,
        )

        result = self._call_structured(
            prompt=prompt,
            schema=PATTERN_CURATION_SCHEMA,
            task_label="pattern curation",
        )

        if not result:
            return 0

        patterns = result.get("patterns_to_store", [])
        if not patterns:
            logger.info("  📚 Patterns: 7B found nothing worth storing")
            return 0

        stored = 0
        for p in patterns:
            error_pattern = p.get("error_pattern", "").strip()
            solution = p.get("solution", "").strip()
            if not error_pattern or not solution:
                continue

            # Store via KB server if available, direct DB otherwise
            success = self._store_pattern(
                error_pattern=error_pattern,
                solution=solution,
                example=p.get("example", ""),
                tags=p.get("tags", ""),
                source="librarian_auto",
            )
            if success:
                stored += 1

        logger.info(f"  📚 Patterns: stored {stored}/{len(patterns)} curated patterns")
        return stored

    # ── Task 2: Journal Curation ─────────────────────────────

    def _curate_journal(self, summary: SessionSummary) -> int:
        """Ask 7B what strategic lessons to extract."""
        prompt = JOURNAL_CURATION_PROMPT.format(
            goal=summary.goal,
            outcome=summary.outcome,
            iterations=summary.iterations,
            plan_summary=summary.plan_summary[:500],
            files=", ".join(summary.files_created + summary.files_modified),
            failures="\n".join(summary.failure_summaries[:5]),
            rca="\n".join(summary.rca_summaries[:3]),
        )

        result = self._call_structured(
            prompt=prompt,
            schema=JOURNAL_CURATION_SCHEMA,
            task_label="journal curation",
        )

        if not result:
            return 0

        entries = result.get("entries_to_store", [])
        if not entries:
            logger.info("  📓 Journal: 7B found no strategic lessons")
            return 0

        stored = 0
        for e in entries:
            title = e.get("title", "").strip()
            content = e.get("content", "").strip()
            if not title or not content:
                continue

            try:
                add_journal_entry(
                    title=title,
                    content=content,
                    lesson_type=e.get("lesson_type", "strategy"),
                    domain=e.get("domain", "general"),
                    session_id=summary.task_id,
                    task_description=summary.goal[:300],
                    iterations_taken=summary.iterations,
                    outcome=summary.outcome,
                    confidence=e.get("confidence", "medium"),
                    db_path=self.db_path,
                )
                stored += 1
            except Exception as ex:
                logger.warning(f"  Journal write failed: {ex}")

        logger.info(f"  📓 Journal: stored {stored}/{len(entries)} curated entries")
        return stored

    # ── Task 3: Snippet Curation ─────────────────────────────

    def _curate_snippets(self, summary: SessionSummary) -> int:
        """Ask 7B which code files/functions are worth saving."""
        # Build code preview — truncate large files
        code_sections = []
        for filename, content in summary.final_code.items():
            if filename.startswith("test_"):
                continue  # Skip test files by default
            # Cap at 2000 chars per file to stay within context
            preview = content[:2000]
            if len(content) > 2000:
                preview += f"\n# ... ({len(content) - 2000} chars truncated)"
            code_sections.append(f"### {filename}\n```python\n{preview}\n```")

        if not code_sections:
            return 0

        prompt = SNIPPET_CURATION_PROMPT.format(
            code_files="\n\n".join(code_sections),
            goal=summary.goal,
        )

        result = self._call_structured(
            prompt=prompt,
            schema=SNIPPET_CURATION_SCHEMA,
            task_label="snippet curation",
        )

        if not result:
            return 0

        snippets = result.get("snippets_to_store", [])
        if not snippets:
            logger.info("  📎 Snippets: 7B found no reusable code")
            return 0

        stored = 0
        for s in snippets:
            title = s.get("title", "").strip()
            code = s.get("code", "").strip()
            description = s.get("description", "").strip()
            if not title or not code or not description:
                continue

            try:
                add_snippet(
                    title=title,
                    description=description,
                    code=code,
                    domain=s.get("domain", "general"),
                    tags=s.get("tags", ""),
                    source_file=s.get("source_file", ""),
                    session_id=summary.task_id,
                    quality_score=s.get("quality_score", "good"),
                    db_path=self.db_path,
                )
                stored += 1
            except Exception as ex:
                logger.warning(f"  Snippet write failed: {ex}")

        logger.info(f"  📎 Snippets: stored {stored}/{len(snippets)} curated snippets")
        return stored

    # ── LLM Calls ────────────────────────────────────────────

    def _call_structured(
        self,
        prompt: str,
        schema: dict,
        task_label: str = "",
    ) -> Optional[Dict]:
        """
        Call the 7B/14B with structured output.

        Uses Ollama's format parameter for deterministic JSON output.
        Falls back gracefully on any failure.
        """
        endpoint = (self.model_config.endpoint or "http://127.0.0.1:11436").rstrip("/")
        url = f"{endpoint}/api/chat"

        payload = {
            "model": self.model_config.model_id,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": schema,
            "options": {
                "temperature": 0.1,  # Low temp for curation — we want consistency
                "num_predict": 4096,
            },
        }

        try:
            logger.debug(f"  Librarian calling 7B for {task_label}...")
            resp = self.session.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            content = data.get("message", {}).get("content", "")
            if not content:
                logger.warning(f"  Librarian: empty response for {task_label}")
                return None

            result = json.loads(content)
            eval_count = data.get("eval_count", 0)
            logger.debug(f"  Librarian {task_label}: {eval_count} tokens generated")
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"  Librarian: JSON parse error for {task_label}: {e}")
            return None
        except httpx.ConnectError:
            logger.warning(f"  Librarian: cannot connect to model at {endpoint}")
            return None
        except httpx.TimeoutException:
            logger.warning(f"  Librarian: timeout for {task_label}")
            return None
        except Exception as e:
            logger.warning(f"  Librarian: unexpected error for {task_label}: {e}")
            return None

    def _store_pattern(
        self,
        error_pattern: str,
        solution: str,
        example: str = "",
        tags: str = "",
        source: str = "librarian_auto",
    ) -> bool:
        """Store a pattern via the KB server (HTTP) or fall back to direct DB."""
        # Try KB server first
        try:
            payload = {
                "error": error_pattern[:500],
                "solution": solution[:500],
                "example": example[:1000],
                "tags": tags,
                "source": source,
            }
            data = json.dumps(payload).encode("utf-8")
            import urllib.request
            req = urllib.request.Request(
                f"{self.kb_server_url}/add_pattern",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode())
                return result.get("status") in ("created", "added")
        except Exception:
            # Fall back to direct DB write
            try:
                from kb_store import add_pattern
                add_pattern(
                    error_pattern=error_pattern,
                    solution=solution,
                    example=example,
                    tags=tags,
                    source=source,
                    db_path=self.db_path,
                )
                return True
            except Exception as e:
                logger.debug(f"  Pattern direct write also failed: {e}")
                return False


# ── Helper: Build SessionSummary from Orchestrator State ─────

def build_session_summary(
    task_state,
    working_dir: Path,
    memory_records: Optional[list] = None,
) -> SessionSummary:
    """
    Build a SessionSummary from the orchestrator's TaskState.

    Called by the orchestrator before invoking the librarian.
    Collects all the information the librarian needs to curate.
    """
    summary = SessionSummary(
        task_id=task_state.task_id,
        goal=task_state.goal,
        outcome="success" if task_state.phase.value == "complete" else "failure",
        iterations=task_state.iteration,
        plan_summary=task_state.current_plan or "",
    )

    # Collect files
    try:
        py_files = [f.name for f in working_dir.iterdir()
                    if f.is_file() and f.suffix == '.py'
                    and not f.name.startswith('.')]
        summary.files_created = py_files
    except Exception:
        pass

    # Collect error→fix pairs from failure history
    if task_state.failure_history:
        for i, failure in enumerate(task_state.failure_history):
            error = failure.get("error", "")
            rca = failure.get("rca", "")
            rca_data = failure.get("rca_data", {})

            if error:
                entry = {
                    "error": error[:300],
                    "iteration": failure.get("iteration", i + 1),
                }

                # If there's a subsequent iteration that succeeded,
                # the RCA from this failure likely contains the fix
                if rca:
                    entry["fix_hint"] = rca[:300]

                if rca_data:
                    what_to_change = rca_data.get("what_to_change", "")
                    if what_to_change:
                        entry["fix"] = what_to_change[:300]

                    edits = rca_data.get("concrete_edits", [])
                    if edits:
                        entry["concrete_edits"] = json.dumps(edits)[:500]

                summary.errors_encountered.append(entry)

            # Failure summaries for journal
            phase = failure.get("phase", "unknown")
            summary.failure_summaries.append(
                f"Iteration {failure.get('iteration', '?')} failed at {phase}: {error[:150]}"
            )

            if rca:
                summary.rca_summaries.append(rca[:300])

    # Collect final code (for snippet extraction) — only on success
    if summary.outcome == "success":
        for f in working_dir.iterdir():
            if (f.is_file() and f.suffix == '.py'
                    and not f.name.startswith('.')
                    and not f.name.startswith('test_')
                    and f.stat().st_size < 50000):  # skip huge files
                try:
                    summary.final_code[f.name] = f.read_text()
                except Exception:
                    pass

    # Extract error→fix pairs where we can match error to resolution
    # If the session succeeded and there were failures, the last RCA
    # likely describes what finally fixed it
    if (summary.outcome == "success"
            and summary.errors_encountered
            and not any(e.get("fix") for e in summary.errors_encountered)):
        # No explicit fixes captured — use memory records if available
        if memory_records:
            for rec in memory_records:
                if rec.success and rec.rca:
                    # This iteration succeeded after an RCA — the RCA was the fix
                    for err_entry in summary.errors_encountered:
                        if not err_entry.get("fix"):
                            err_entry["fix"] = rec.rca[:300]
                            break

    return summary


if __name__ == "__main__":
    """Quick test — show librarian stats."""
    stats = get_librarian_stats()
    print(json.dumps(stats, indent=2))
