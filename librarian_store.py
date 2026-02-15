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
from typing import Any, Dict, List, Optional

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
) -> List[Dict[str, Any]]:
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
) -> List[Dict[str, Any]]:
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


# ── v1.2: AST-Aware Code Chunking ────────────────────────────
# Replaces naive text chunking with semantically coherent units
# based on function/class boundaries. cAST approach (CMU + Augment Code):
# improves Recall@5 by 4.3 points on RepoEval.

def chunk_python_ast(
    source: str,
    filename: str = "",
    max_chunk_lines: int = 80,
    min_chunk_lines: int = 5,
) -> List[Dict[str, Any]]:
    """
    Parse Python source into AST-aware chunks.

    Each chunk is a self-contained code unit (function, class, or module-level block)
    with metadata for better retrieval. Siblings under max_chunk_lines are merged.

    Returns:
        List of dicts with keys: name, type, signature, code, imports, docstring, line_start, line_end
    """
    import ast as _ast

    chunks: List[Dict[str, Any]] = []
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        # Fallback: return entire file as single chunk
        if len(source.strip()) > 0:
            chunks.append({
                "name": filename or "unknown",
                "type": "file",
                "signature": "",
                "code": source[:2000],
                "imports": [],
                "docstring": "",
                "line_start": 1,
                "line_end": source.count('\n') + 1,
            })
        return chunks

    lines = source.split('\n')

    # Collect module-level imports
    module_imports = []
    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                module_imports.append(f"import {alias.name}")
        elif isinstance(node, _ast.ImportFrom):
            names = ", ".join(a.name for a in node.names)
            module_imports.append(f"from {node.module or ''} import {names}")

    # Process top-level definitions
    pending_lines: list[str] = []  # Module-level code between definitions
    pending_start = 1

    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            # Flush pending module-level code
            if pending_lines:
                code = '\n'.join(pending_lines).strip()
                if len(code) > 20:
                    chunks.append({
                        "name": f"{filename}:module",
                        "type": "module_code",
                        "signature": "",
                        "code": code[:1500],
                        "imports": module_imports,
                        "docstring": "",
                        "line_start": pending_start,
                        "line_end": node.lineno - 1,
                    })
                pending_lines = []

            # Extract function signature
            args = []
            for arg in node.args.args:
                arg_name = arg.arg
                if arg.annotation:
                    try:
                        arg_name += f": {_ast.unparse(arg.annotation)}"
                    except Exception:
                        pass
                args.append(arg_name)

            returns = ""
            if node.returns:
                try:
                    returns = f" -> {_ast.unparse(node.returns)}"
                except Exception:
                    pass

            sig = f"def {node.name}({', '.join(args)}){returns}"

            # Extract docstring
            docstring = ""
            if (node.body and isinstance(node.body[0], _ast.Expr)
                    and isinstance(node.body[0].value, (_ast.Constant, _ast.Str))):
                docstring = str(getattr(node.body[0].value, 'value', getattr(node.body[0].value, 's', '')))

            end_line = node.end_lineno or node.lineno + len(node.body)
            code = '\n'.join(lines[node.lineno - 1:end_line])

            chunks.append({
                "name": node.name,
                "type": "async_function" if isinstance(node, _ast.AsyncFunctionDef) else "function",
                "signature": sig,
                "code": code[:2000],
                "imports": module_imports,
                "docstring": docstring[:300],
                "line_start": node.lineno,
                "line_end": end_line,
            })

            pending_start = end_line + 1

        elif isinstance(node, _ast.ClassDef):
            if pending_lines:
                code = '\n'.join(pending_lines).strip()
                if len(code) > 20:
                    chunks.append({
                        "name": f"{filename}:module",
                        "type": "module_code",
                        "signature": "",
                        "code": code[:1500],
                        "imports": module_imports,
                        "docstring": "",
                        "line_start": pending_start,
                        "line_end": node.lineno - 1,
                    })
                pending_lines = []

            # Class with all methods
            bases = []
            for base in node.bases:
                try:
                    bases.append(_ast.unparse(base))
                except Exception:
                    bases.append("?")
            sig = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"

            docstring = ""
            if (node.body and isinstance(node.body[0], _ast.Expr)
                    and isinstance(node.body[0].value, (_ast.Constant, _ast.Str))):
                docstring = str(getattr(node.body[0].value, 'value', getattr(node.body[0].value, 's', '')))

            # Method signatures for class summary
            methods = []
            for item in node.body:
                if isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    m_args = [a.arg for a in item.args.args if a.arg != 'self']
                    methods.append(f"{item.name}({', '.join(m_args)})")

            end_line = node.end_lineno or node.lineno + len(node.body)
            code = '\n'.join(lines[node.lineno - 1:end_line])

            # If class is very large, chunk it into methods
            if end_line - node.lineno > max_chunk_lines:
                # Add class header chunk
                header_end = min(node.lineno + 5, end_line)
                chunks.append({
                    "name": node.name,
                    "type": "class",
                    "signature": sig,
                    "code": '\n'.join(lines[node.lineno - 1:header_end]),
                    "imports": module_imports,
                    "docstring": docstring[:300],
                    "line_start": node.lineno,
                    "line_end": header_end,
                    "methods": methods,
                })
                # Add individual methods as separate chunks
                for item in node.body:
                    if isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                        m_end = item.end_lineno or item.lineno + len(item.body)
                        m_code = '\n'.join(lines[item.lineno - 1:m_end])
                        m_args = [a.arg for a in item.args.args if a.arg != 'self']
                        chunks.append({
                            "name": f"{node.name}.{item.name}",
                            "type": "method",
                            "signature": f"{node.name}.{item.name}({', '.join(m_args)})",
                            "code": m_code[:2000],
                            "imports": [],
                            "docstring": "",
                            "line_start": item.lineno,
                            "line_end": m_end,
                        })
            else:
                chunks.append({
                    "name": node.name,
                    "type": "class",
                    "signature": sig,
                    "code": code[:2000],
                    "imports": module_imports,
                    "docstring": docstring[:300],
                    "line_start": node.lineno,
                    "line_end": end_line,
                    "methods": methods,
                })

            pending_start = end_line + 1

        else:
            # Module-level code
            if hasattr(node, 'lineno'):
                end = getattr(node, 'end_lineno', node.lineno)
                pending_lines.extend(lines[node.lineno - 1:end])

    # Merge small sibling MODULE-LEVEL chunks only (cAST approach)
    # Functions and classes are kept separate for precise retrieval
    merged: List[Dict[str, Any]] = []
    buffer: Optional[Dict[str, Any]] = None
    for chunk in chunks:
        chunk_lines = chunk.get("line_end", 0) - chunk.get("line_start", 0) + 1
        # Only merge module_code chunks — keep functions/classes/methods separate
        if chunk["type"] == "module_code" and chunk_lines < min_chunk_lines:
            if buffer is not None and buffer["type"] == "module_code":
                buffer_lines = buffer.get("line_end", 0) - buffer.get("line_start", 0) + 1
                if buffer_lines + chunk_lines <= max_chunk_lines:
                    buffer["name"] += f" + {chunk['name']}"
                    buffer["code"] += f"\n\n{chunk['code']}"
                    buffer["line_end"] = chunk["line_end"]
                    continue
                else:
                    merged.append(buffer)
                    buffer = chunk
                    continue
            buffer = chunk
        else:
            if buffer:
                merged.append(buffer)
                buffer = None
            merged.append(chunk)

    if buffer:
        merged.append(buffer)

    return merged


def add_ast_chunks(
    source: str,
    filename: str,
    domain: str = "general",
    session_id: str = "",
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """
    Parse a Python file and store AST-aware chunks as snippets.

    Returns number of chunks stored.
    """
    chunks = chunk_python_ast(source, filename)
    count = 0
    for chunk in chunks:
        tags = f"{chunk['type']},{chunk.get('signature', '')}"
        add_snippet(
            title=f"{filename}:{chunk['name']}",
            description=f"{chunk['type']}: {chunk.get('signature', chunk['name'])}. {chunk.get('docstring', '')}",
            code=chunk['code'],
            domain=domain,
            tags=tags[:200],
            source_file=filename,
            session_id=session_id,
            db_path=db_path,
        )
        count += 1
    return count
