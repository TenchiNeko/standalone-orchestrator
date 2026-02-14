"""
Knowledge Base Client for the Orchestrator.

v0.8.0: Provides two modes of access to the RAG Knowledge Base:
  - "remote" mode: HTTP calls to the retrieval_server.py (port 8787)
  - "direct" mode: SQLite queries via kb_store.py (same machine, no network)

The orchestrator uses this to:
  1. PROACTIVE lookup: Before building a file, fetch relevant patterns/docs
  2. REACTIVE lookup: When a build fails, query with the error text
  3. AUTO-CAPTURE: When a fix works, save the error→solution pattern

All methods are fail-safe — KB unavailability never blocks the build loop.
"""

import json
import logging
import urllib.request
import urllib.parse
import urllib.error
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class KBClient:
    """
    Knowledge Base client for orchestrator integration.

    Usage:
        kb = KBClient(server_url="http://localhost:8787")

        # Proactive: get patterns relevant to what we're building
        context = kb.get_build_context("Flask API with SQLite")

        # Reactive: get fix suggestions for a specific error
        fix = kb.get_fix_for_error("NameError: name 'uuid' is not defined")

        # Auto-capture: save a new error→solution pattern
        kb.capture_pattern(error_text, solution_text, example_code)
    """

    def __init__(self, server_url: str = "http://localhost:8787", timeout: int = 5):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._available: Optional[bool] = None  # lazy check

    def is_available(self) -> bool:
        """Check if KB server is reachable. Caches result for session."""
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(f"{self.server_url}/stats", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                self._available = resp.status == 200
        except Exception:
            self._available = False
            logger.debug("KB server not available — running without knowledge base")
        return self._available

    def get_build_context(self, query: str, max_results: int = 3) -> str:
        """
        PROACTIVE lookup: Get relevant patterns and docs before building.

        Args:
            query: Task description or filename context (e.g., "Flask REST API", "test_cli.py argparse")
            max_results: Max doc chunks to return

        Returns:
            Formatted context string for injection into the build prompt.
            Empty string if KB unavailable or no relevant results.
        """
        if not self.is_available():
            # v1.1: Fall back to local librarian store when KB server is down
            return self._local_fallback(query, max_results)

        try:
            data = self._lookup(query)
            if not data:
                return ""

            sections = []

            # Tier 1: Known patterns
            patterns = data.get("tier1_matches", [])
            if patterns:
                # Deduplicate by error_pattern
                seen = set()
                unique = []
                for p in patterns:
                    key = p.get("error_pattern", "")
                    if key not in seen:
                        seen.add(key)
                        unique.append(p)

                pattern_lines = []
                for p in unique[:5]:
                    pattern_lines.append(
                        f"- Error: `{p.get('error_pattern', '')}`\n"
                        f"  Fix: {p.get('solution', '')}\n"
                        f"  Example: `{p.get('example', '')[:150]}`"
                    )
                sections.append(
                    "### Known Error Patterns (avoid these mistakes)\n" +
                    "\n".join(pattern_lines)
                )

            # Tier 2: Relevant docs
            docs = data.get("tier2_matches", [])
            if docs:
                doc_lines = []
                for d in docs[:max_results]:
                    code_examples = d.get("code_examples", [])
                    code_block = ""
                    if code_examples:
                        code_block = f"\n  ```python\n  {code_examples[0][:300]}\n  ```"
                    doc_lines.append(
                        f"- {d.get('title', 'Doc')}: {d.get('content', '')[:200]}{code_block}"
                    )
                sections.append(
                    "### Reference Documentation\n" +
                    "\n".join(doc_lines)
                )

            if not sections:
                return ""

            return (
                "\n## 📚 Knowledge Base Context (auto-retrieved)\n" +
                "\n\n".join(sections) +
                "\n"
            )

        except Exception as e:
            logger.debug(f"KB proactive lookup failed: {e}")
            return ""

    def get_fix_for_error(self, error_text: str) -> str:
        """
        REACTIVE lookup: Get fix suggestions for a specific error.

        Args:
            error_text: The actual error output from a failed test/compile

        Returns:
            Formatted fix context string for injection into Wave 2 prompts.
            Empty string if no relevant fix found.
        """
        if not self.is_available():
            return ""

        try:
            # Extract the most relevant error line (first traceback line with Error)
            query = self._extract_error_query(error_text)
            if not query:
                return ""

            data = self._lookup(query)
            if not data:
                return ""

            # Use the recommended fix if available
            rec = data.get("recommended_fix")
            if rec and rec.get("solution"):
                example = rec.get("example", "")
                example_block = f"\n```python\n{example}\n```" if example else ""
                return (
                    f"\n## 📚 KB Fix Suggestion (from known patterns)\n"
                    f"**Solution:** {rec['solution']}\n"
                    f"{example_block}\n"
                    f"Source: {rec.get('source', 'knowledge_base')}\n"
                )

            # Fall back to agent_context if available
            agent_ctx = data.get("agent_context", "")
            if agent_ctx and len(agent_ctx) > 50:
                return f"\n## 📚 KB Reference\n{agent_ctx[:1500]}\n"

            return ""

        except Exception as e:
            logger.debug(f"KB reactive lookup failed: {e}")
            return ""

    def capture_pattern(
        self,
        error_pattern: str,
        solution: str,
        example: str = "",
        source: str = "auto_captured",
        confidence: str = "medium",
    ) -> bool:
        """
        AUTO-CAPTURE: Save a new error→solution pattern to the KB.

        Called when Wave 2 successfully fixes an error that Wave 1 couldn't.

        Returns True if pattern was saved successfully.
        """
        if not self.is_available():
            return False

        try:
            payload = {
                "error": error_pattern[:500],
                "solution": solution[:500],
                "example": example[:1000],
                "source": source,
                "confidence": confidence,
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.server_url}/add_pattern",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode())
                if result.get("status") == "added":
                    logger.info(f"  📚 KB: Auto-captured pattern: {error_pattern[:80]}")
                    return True
                return False

        except Exception as e:
            logger.debug(f"KB auto-capture failed: {e}")
            return False

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get KB statistics."""
        if not self.is_available():
            return None
        try:
            req = urllib.request.Request(f"{self.server_url}/stats", method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return None

    # --- Internal helpers ---

    def _local_fallback(self, query: str, max_results: int = 3) -> str:
        """v1.1: Fall back to local librarian SQLite store when KB server is unavailable.

        This ensures the build agent still gets past-session knowledge even
        when the KB server isn't running (e.g., during benchmarks).
        """
        try:
            from librarian_store import search_snippets, search_journal
            from pathlib import Path

            db_path = str(Path(__file__).parent / 'knowledge_base.db')
            parts = []

            # Get relevant journal entries
            journal_hits = search_journal(query, limit=2, db_path=db_path)
            if journal_hits:
                for j in journal_hits:
                    parts.append(f"- [{j['lesson_type'].upper()}] {j['content'][:200]}")

            # Get relevant code snippets
            snippet_hits = search_snippets(query, limit=max_results, db_path=db_path)
            if snippet_hits:
                for s in snippet_hits:
                    parts.append(f"- {s['title']}: {s['description'][:150]}\n  ```python\n  {s['code'][:300]}\n  ```")

            if parts:
                result = "## Knowledge Base (local cache)\n" + "\n".join(parts) + "\n"
                logger.debug(f"KB local fallback: {len(parts)} results for '{query[:50]}'")
                return result
        except Exception as e:
            logger.debug(f"KB local fallback failed: {e}")

        return ""


    def _lookup(self, query: str) -> Optional[Dict]:
        """Raw lookup against the KB server."""
        encoded_query = urllib.parse.quote(query[:200])
        url = f"{self.server_url}/lookup?error={encoded_query}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def _extract_error_query(self, error_text: str) -> str:
        """
        Extract the most searchable error string from raw test/compile output.

        Priorities:
        1. Python exception lines (NameError, ImportError, etc.)
        2. Assertion errors with context
        3. First non-empty stderr line
        """
        import re

        # Look for Python exception patterns
        exception_patterns = [
            r"((?:Name|Import|Module(?:NotFound)|Attribute|Type|Value|Key|Runtime|OSError|FileNotFound)Error:?\s*.+?)(?:\n|$)",
            r"(SyntaxError:.+?)(?:\n|$)",
            r"(AssertionError:?\s*.+?)(?:\n|$)",
        ]

        for pattern in exception_patterns:
            match = re.search(pattern, error_text)
            if match:
                return match.group(1).strip()[:200]

        # Fall back to last non-empty line (often the actual error)
        lines = [line.strip() for line in error_text.strip().split("\n") if line.strip()]
        if lines:
            # Skip "EXIT_CODE" lines
            for line in reversed(lines):
                if not line.startswith("EXIT_CODE") and len(line) > 10:
                    return line[:200]

        return error_text[:200]
