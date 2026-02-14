"""
librarian_store.py - Extended Knowledge Base Storage for the Librarian System

v0.9.0: Three-database knowledge architecture (all in one SQLite file):

  Table 1: patterns (EXISTING) — error→solution mappings (Tier 1)
  Table 2: docs (EXISTING) — chunked documentation (Tier 2)
  Table 3: journal — strategic lessons and decisions from completed sessions
  Table 4: snippets — clean, reusable code patterns from successful builds

The 7B librarian curates entries into tables 3 and 4 after each session.
Tables 1-2 remain managed by the existing kb_store.py system.
"""

import sqlite3
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "./knowledge_base.db"


def get_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get a database connection with FTS5 support."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_librarian_tables(db_path: str = DEFAULT_DB_PATH):
    """
    Create the journal and snippets tables + FTS indexes.

    Safe to call multiple times — uses IF NOT EXISTS.
    Run this once at startup or after upgrading from v0.8.x.

    v0.9.5: Split into individual statements with error handling.
    executescript() silently aborts on error, so if the journal FTS
    trigger had any issue, the snippets table was never created.
    """
    conn = get_db(db_path)

    # ── Table 3: Journal — Strategic Lessons ──────────────────
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                lesson_type TEXT NOT NULL,
                domain TEXT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                task_description TEXT,
                iterations_taken INTEGER,
                outcome TEXT,
                confidence TEXT DEFAULT 'medium',
                hit_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    except Exception as e:
        logger.warning(f"Journal table creation: {e}")

    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS journal_fts USING fts5(
                title, content, domain, lesson_type, task_description,
                content=journal, content_rowid=id,
                tokenize='porter unicode61'
            )
        """)
        conn.commit()
    except Exception as e:
        logger.warning(f"Journal FTS creation: {e}")

    for trigger_sql in [
        """CREATE TRIGGER IF NOT EXISTS journal_ai AFTER INSERT ON journal BEGIN
            INSERT INTO journal_fts(rowid, title, content, domain, lesson_type, task_description)
            VALUES (new.id, new.title, new.content, new.domain, new.lesson_type, new.task_description);
        END""",
        """CREATE TRIGGER IF NOT EXISTS journal_ad AFTER DELETE ON journal BEGIN
            INSERT INTO journal_fts(journal_fts, rowid, title, content, domain, lesson_type, task_description)
            VALUES ('delete', old.id, old.title, old.content, old.domain, old.lesson_type, old.task_description);
        END""",
        """CREATE TRIGGER IF NOT EXISTS journal_au AFTER UPDATE ON journal BEGIN
            INSERT INTO journal_fts(journal_fts, rowid, title, content, domain, lesson_type, task_description)
            VALUES ('delete', old.id, old.title, old.content, old.domain, old.lesson_type, old.task_description);
            INSERT INTO journal_fts(rowid, title, content, domain, lesson_type, task_description)
            VALUES (new.id, new.title, new.content, new.domain, new.lesson_type, new.task_description);
        END""",
    ]:
        try:
            conn.execute(trigger_sql)
            conn.commit()
        except Exception as e:
            logger.debug(f"Journal trigger (may already exist): {e}")

    # ── Table 4: Snippets — Reusable Code Patterns ───────────
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS snippets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                code TEXT NOT NULL,
                language TEXT DEFAULT 'python',
                domain TEXT,
                tags TEXT,
                source_file TEXT,
                quality_score TEXT DEFAULT 'good',
                hit_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    except Exception as e:
        logger.warning(f"Snippets table creation: {e}")

    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS snippets_fts USING fts5(
                title, description, domain, tags,
                content=snippets, content_rowid=id,
                tokenize='porter unicode61'
            )
        """)
        conn.commit()
    except Exception as e:
        logger.warning(f"Snippets FTS creation: {e}")

    for trigger_sql in [
        """CREATE TRIGGER IF NOT EXISTS snippets_ai AFTER INSERT ON snippets BEGIN
            INSERT INTO snippets_fts(rowid, title, description, domain, tags)
            VALUES (new.id, new.title, new.description, new.domain, new.tags);
        END""",
        """CREATE TRIGGER IF NOT EXISTS snippets_ad AFTER DELETE ON snippets BEGIN
            INSERT INTO snippets_fts(snippets_fts, rowid, title, description, domain, tags)
            VALUES ('delete', old.id, old.title, old.description, old.domain, old.tags);
        END""",
        """CREATE TRIGGER IF NOT EXISTS snippets_au AFTER UPDATE ON snippets BEGIN
            INSERT INTO snippets_fts(snippets_fts, rowid, title, description, domain, tags)
            VALUES ('delete', old.id, old.title, old.description, old.domain, old.tags);
            INSERT INTO snippets_fts(rowid, title, description, domain, tags)
            VALUES (new.id, new.title, new.description, new.domain, new.tags);
        END""",
    ]:
        try:
            conn.execute(trigger_sql)
            conn.commit()
        except Exception as e:
            logger.debug(f"Snippets trigger (may already exist): {e}")

    # ── Verify both tables exist ──────────────────────────────
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}

    missing = []
    if 'journal' not in tables:
        missing.append('journal')
    if 'snippets' not in tables:
        missing.append('snippets')

    if missing:
        logger.error(f"LIBRARIAN INIT FAILED: missing tables: {missing}")

    conn.close()
    logger.info(f"✓ Librarian tables initialized in {db_path}"
                f" (journal={'✅' if 'journal' in tables else '❌'}"
                f", snippets={'✅' if 'snippets' in tables else '❌'})")


# ── Journal Operations ────────────────────────────────────────

def add_journal_entry(
    title: str,
    content: str,
    lesson_type: str = "strategy",
    domain: str = "general",
    session_id: str = "",
    task_description: str = "",
    iterations_taken: int = 0,
    outcome: str = "success",
    confidence: str = "medium",
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[int]:
    """Add a strategic lesson to the journal."""
    conn = get_db(db_path)
    cursor = conn.execute("""
        INSERT INTO journal (session_id, lesson_type, domain, title, content,
                            task_description, iterations_taken, outcome, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (session_id, lesson_type, domain, title, content,
          task_description, iterations_taken, outcome, confidence))
    conn.commit()
    entry_id = cursor.lastrowid
    conn.close()
    logger.info(f"  📓 Journal entry #{entry_id}: {title}")
    return entry_id


def search_journal(
    query: str,
    domain: Optional[str] = None,
    limit: int = 5,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict]:
    """Search journal entries via FTS5."""
    conn = get_db(db_path)

    import re
    clean = re.sub(r'[^\w\s]', ' ', query)
    terms = [t for t in clean.split() if len(t) > 1]
    if not terms:
        conn.close()
        return []

    fts_query = " OR ".join(terms[:10])

    try:
        if domain:
            rows = conn.execute("""
                SELECT j.*, rank
                FROM journal_fts
                JOIN journal j ON j.id = journal_fts.rowid
                WHERE journal_fts MATCH ? AND j.domain = ?
                ORDER BY rank
                LIMIT ?
            """, (fts_query, domain, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT j.*, rank
                FROM journal_fts
                JOIN journal j ON j.id = journal_fts.rowid
                WHERE journal_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (fts_query, limit)).fetchall()
    except sqlite3.OperationalError:
        rows = []

    results = [dict(row) for row in rows]

    # Bump hit counts
    for r in results:
        conn.execute("UPDATE journal SET hit_count = hit_count + 1 WHERE id = ?", (r['id'],))
    conn.commit()
    conn.close()

    return results


# ── Snippet Operations ────────────────────────────────────────

def add_snippet(
    title: str,
    description: str,
    code: str,
    domain: str = "general",
    tags: str = "",
    source_file: str = "",
    session_id: str = "",
    quality_score: str = "good",
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[int]:
    """Add a reusable code snippet."""
    conn = get_db(db_path)
    cursor = conn.execute("""
        INSERT INTO snippets (session_id, title, description, code, domain, tags,
                             source_file, quality_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (session_id, title, description, code, domain, tags,
          source_file, quality_score))
    conn.commit()
    snippet_id = cursor.lastrowid
    conn.close()
    logger.info(f"  📎 Snippet #{snippet_id}: {title}")
    return snippet_id


def search_snippets(
    query: str,
    domain: Optional[str] = None,
    limit: int = 3,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict]:
    """Search snippets via FTS5."""
    conn = get_db(db_path)

    import re
    clean = re.sub(r'[^\w\s]', ' ', query)
    terms = [t for t in clean.split() if len(t) > 1]
    if not terms:
        conn.close()
        return []

    fts_query = " OR ".join(terms[:10])

    try:
        if domain:
            rows = conn.execute("""
                SELECT s.*, rank
                FROM snippets_fts
                JOIN snippets s ON s.id = snippets_fts.rowid
                WHERE snippets_fts MATCH ? AND s.domain = ?
                ORDER BY rank
                LIMIT ?
            """, (fts_query, domain, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT s.*, rank
                FROM snippets_fts
                JOIN snippets s ON s.id = snippets_fts.rowid
                WHERE snippets_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (fts_query, limit)).fetchall()
    except sqlite3.OperationalError:
        rows = []

    results = [dict(row) for row in rows]

    for r in results:
        conn.execute("UPDATE snippets SET hit_count = hit_count + 1 WHERE id = ?", (r['id'],))
    conn.commit()
    conn.close()

    return results


# ── Combined Retrieval (for orchestrator) ─────────────────────

def get_session_context(
    task_description: str,
    db_path: str = DEFAULT_DB_PATH,
    max_journal: int = 3,
    max_snippets: int = 2,
) -> str:
    """
    Main entry point for the orchestrator's planning phase.

    Queries journal + snippets and returns formatted context
    to inject into the 70B's planning prompt.
    """
    parts = []

    # Journal: strategic lessons
    journal_hits = search_journal(task_description, limit=max_journal, db_path=db_path)
    if journal_hits:
        journal_lines = []
        for j in journal_hits:
            journal_lines.append(
                f"- [{j['lesson_type'].upper()}] {j['title']}\n"
                f"  {j['content'][:300]}\n"
                f"  (from: {j['task_description'][:100]}, "
                f"outcome: {j['outcome']}, iters: {j['iterations_taken']})"
            )
        parts.append(
            "### Strategic Lessons (from past sessions)\n" +
            "\n".join(journal_lines)
        )

    # Snippets: reusable code
    snippet_hits = search_snippets(task_description, limit=max_snippets, db_path=db_path)
    if snippet_hits:
        snippet_lines = []
        for s in snippet_hits:
            code_preview = s['code'][:400]
            snippet_lines.append(
                f"- {s['title']} ({s['domain']})\n"
                f"  {s['description'][:200]}\n"
                f"  ```python\n  {code_preview}\n  ```"
            )
        parts.append(
            "### Reusable Code Patterns (from past sessions)\n" +
            "\n".join(snippet_lines)
        )

    if not parts:
        return ""

    return (
        "\n## 🧠 Librarian Context (curated from past sessions)\n" +
        "\n\n".join(parts) +
        "\n"
    )


# ── Stats ─────────────────────────────────────────────────────

def get_librarian_stats(db_path: str = DEFAULT_DB_PATH) -> Dict:
    """Get stats for the librarian tables."""
    conn = get_db(db_path)
    stats = {}
    try:
        stats["journal_entries"] = conn.execute("SELECT COUNT(*) FROM journal").fetchone()[0]
        stats["snippets"] = conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
        stats["journal_by_domain"] = dict(conn.execute(
            "SELECT domain, COUNT(*) FROM journal GROUP BY domain"
        ).fetchall())
        stats["journal_by_type"] = dict(conn.execute(
            "SELECT lesson_type, COUNT(*) FROM journal GROUP BY lesson_type"
        ).fetchall())
        stats["top_journal"] = [dict(r) for r in conn.execute(
            "SELECT title, hit_count, domain FROM journal ORDER BY hit_count DESC LIMIT 5"
        ).fetchall()]
        stats["top_snippets"] = [dict(r) for r in conn.execute(
            "SELECT title, hit_count, domain FROM snippets ORDER BY hit_count DESC LIMIT 5"
        ).fetchall()]
    except sqlite3.OperationalError:
        # Tables don't exist yet
        stats["journal_entries"] = 0
        stats["snippets"] = 0
    conn.close()
    return stats


if __name__ == "__main__":
    init_librarian_tables()
    print(json.dumps(get_librarian_stats(), indent=2))
