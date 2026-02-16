"""
Standalone Agent Runner — Direct LLM API + Tool Execution Loop.

This is the core of the standalone system. It:
1. Calls Ollama (or Anthropic) HTTP API directly — no CLI wrappers
2. Defines tools (run_command, write_file, read_file, list_directory)
3. Implements the agentic loop: send prompt → get response → if tool_call,
   execute it locally → feed result back → repeat until text response
4. Zero dependency on external CLI agent wrappers

The tool execution loop is what makes agents actually DO things instead of
just generating text about what they would do.
"""

import os
import json
import time
import subprocess
import re
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import httpx

from standalone_config import Config, AgentConfig, ModelConfig
from standalone_models import TaskState, AgentResult, DoD, IterationResult
from playbook_reader import PlaybookReader

logger = logging.getLogger(__name__)


# ============================================================
# Context Budget Utilities
# ============================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text. Uses ~4 chars/token heuristic
    (accurate within 20% for English code-heavy content).
    Good enough for budget management — we don't need exact counts.
    """
    return len(text) // 4


def truncate_to_budget(text: str, max_chars: int, label: str = "") -> str:
    """
    Smart truncation that preserves head and tail of content.

    For debugging context, the beginning (what the error is) and the end
    (the final state / exit code) are most valuable. The middle is
    usually repetitive stack frames or verbose output.

    Guarantees output length <= max_chars.

    Args:
        text: Content to truncate
        max_chars: Maximum character budget
        label: Optional label for the truncation marker
    """
    if len(text) <= max_chars:
        return text

    marker = f"\n\n... ({label + ': ' if label else ''}truncated {len(text) - max_chars} chars) ...\n\n"
    marker_len = len(marker)

    usable = max_chars - marker_len
    if usable < 20:
        # Budget too small for meaningful head+tail — just hard cut
        return text[:max_chars]

    # 60% head, 40% tail — errors are usually at the end
    head_budget = int(usable * 0.6)
    tail_budget = usable - head_budget

    return text[:head_budget] + marker + text[-tail_budget:]


def truncate_diff(diff_text: str, max_chars: int) -> str:
    """
    Truncate a git diff intelligently — keep file headers and the first
    chunk of each file's diff rather than cutting mid-hunk.
    """
    if len(diff_text) <= max_chars:
        return diff_text

    lines = diff_text.split("\n")
    result = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1  # +1 for newline
        if current_size + line_size > max_chars - 60:
            result.append(f"... (diff truncated, {len(diff_text) - current_size} chars remaining)")
            break
        result.append(line)
        current_size += line_size

    return "\n".join(result)


# ============================================================
# Context budget defaults (characters, not tokens)
# These are conservative — roughly 1/4 of these = tokens.
# A 128K token model can fit ~512K chars, but we want to leave
# plenty of room for tool calls, system prompts, and responses.
# ============================================================

# Per-section budgets for RCA evidence gathering
RCA_EVIDENCE_BUDGET = {
    "file_listing": 500,       # just filenames
    "diff_stat": 600,          # git diff --stat
    "diff_content": 3000,      # actual code changes
    "test_files": 3000,        # test file contents (total across all files)
    "source_files": 2000,      # source file contents for RCA
    "total": 8000,             # hard cap on all evidence combined
}

# Per-section budgets for the full RCA prompt
RCA_PROMPT_BUDGET = {
    "failure_context": 3000,   # DoD results + error details
    "evidence": 8000,          # from _gather_rca_evidence
    "plan_context": 2000,      # abbreviated plan
    "failure_history": 1500,   # previous iteration failures
    "total": 16000,            # total user message budget
}

# Budget for memory context injected into build agent
MEMORY_CONTEXT_BUDGET = {
    "per_iteration": 2000,     # chars per iteration record
    "total": 6000,             # total for last N iterations
}

# Budget for build agent prompt sections
BUILD_PROMPT_BUDGET = {
    "memory_context": 6000,    # previous iteration history
    "plan_context": 8000,      # the plan to follow
    "dod_context": 2000,       # definition of done
    "total": 20000,            # total user message (excluding system prompt + tools)
}

# Budget for plan agent prompt sections
PLAN_PROMPT_BUDGET = {
    "failure_context": 4000,   # previous failures + RCA
    "exploration_context": 5000,  # explore findings + librarian context (v0.9.1)
    "total": 12000,            # total user message (v0.9.1: bumped for librarian)
}


# ============================================================
# Tool Definitions (sent to the LLM so it knows what it can call)
# ============================================================

# JSON Schema for structured plan output — forces the model to produce
# deterministic, parseable plans with verification commands for every criterion.
PLAN_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "plan_summary": {
            "type": "string",
            "description": "Brief 1-2 sentence summary of the implementation approach"
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_number": {"type": "integer"},
                    "description": {"type": "string"},
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files to create or modify in this step"
                    }
                },
                "required": ["step_number", "description"]
            },
            "description": "Ordered list of implementation steps"
        },
        "definition_of_done": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Human-readable criterion description. Be specific and testable."
                    },
                    "verification_type": {
                        "type": "string",
                        "description": "Type of verification: 'test' (unit tests cover this), 'file_exists' (file must exist), 'syntax' (valid Python syntax), 'import' (module importable). Default: 'test'."
                    },
                    "target_file": {
                        "type": "string",
                        "description": "Optional: specific file this criterion relates to (e.g., 'app.py', 'test_app.py')"
                    }
                },
                "required": ["description"]
            },
            "description": "Specific, testable criteria. Verification commands are generated automatically AFTER build — you only need descriptions."
        }
    },
    "required": ["plan_summary", "steps", "definition_of_done"]
}

# JSON Schema for structured exploration output — forces the explore agent
# to produce deterministic, parseable reports instead of free-text summaries.
# This gives the plan agent cleaner input with real file paths and patterns.
EXPLORE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "project_type": {
            "type": "string",
            "description": "Type of project (e.g., 'python_cli', 'flask_api', 'new_project', 'library')"
        },
        "summary": {
            "type": "string",
            "description": "Brief 1-2 sentence summary of what was found"
        },
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"},
                    "role": {"type": "string", "description": "What this file does (e.g., 'entry point', 'test file', 'config')"},
                    "key_contents": {"type": "string", "description": "Key classes, functions, or exports in this file"}
                },
                "required": ["path", "role"]
            },
            "description": "Files discovered in the project"
        },
        "dependencies": {
            "type": "array",
            "items": {"type": "string"},
            "description": "External dependencies found (from requirements.txt, imports, package.json, etc.)"
        },
        "patterns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Code patterns observed (e.g., 'uses unittest', 'Flask blueprints', 'dataclass models')"
        },
        "existing_tests": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "framework": {"type": "string", "description": "Test framework (pytest, unittest, etc.)"},
                    "count": {"type": "integer", "description": "Approximate number of test cases"}
                },
                "required": ["path"]
            },
            "description": "Test files found"
        },
        "relevant_to_task": {
            "type": "string",
            "description": "What in the codebase is most relevant to the current task and why"
        }
    },
    "required": ["project_type", "summary", "files"]
}

# JSON Schema for structured test/verification reports
TEST_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_passed": {
            "type": "boolean",
            "description": "Whether all DoD criteria passed"
        },
        "criteria_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion_id": {"type": "string", "description": "e.g., criterion-0"},
                    "passed": {"type": "boolean"},
                    "command_run": {"type": "string", "description": "The actual command that was executed"},
                    "exit_code": {"type": "integer"},
                    "stdout_snippet": {"type": "string", "description": "First 200 chars of stdout"},
                    "stderr_snippet": {"type": "string", "description": "First 200 chars of stderr"},
                    "failure_reason": {"type": "string", "description": "If failed, why (e.g., 'ImportError', 'assertion failed', 'file not found')"}
                },
                "required": ["criterion_id", "passed"]
            }
        },
        "passed_count": {"type": "integer"},
        "total_count": {"type": "integer"}
    },
    "required": ["overall_passed", "criteria_results", "passed_count", "total_count"]
}

# JSON Schema for structured RCA output — replaces heuristic string matching
# with actual LLM analysis of what went wrong and why.
RCA_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause": {
            "type": "string",
            "description": "The actual root cause of the failure in one sentence"
        },
        "why_chain": {
            "type": "array",
            "items": {"type": "string"},
            "description": "5 Whys chain — each item answers 'why did the previous thing happen?'"
        },
        "failed_phase": {
            "type": "string",
            "description": "Which phase failed (explore, plan, build, test)"
        },
        "failed_criteria": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Which specific DoD criteria failed"
        },
        "what_to_change": {
            "type": "string",
            "description": "Specific, actionable change for the next iteration"
        },
        "concrete_edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "The file to edit (e.g. 'cli.py', 'test_models.py')"
                    },
                    "action": {
                        "type": "string",
                        "description": "What to do: 'add_import', 'replace_line', 'add_code', 'fix_syntax', 'rewrite_function'"
                    },
                    "details": {
                        "type": "string",
                        "description": "The EXACT code change. For add_import: the import line. For replace_line: old → new. For add_code: the code to add and where."
                    }
                },
                "required": ["file", "action", "details"]
            },
            "description": "List of CONCRETE edit commands the build agent must execute. Each must specify the exact file, action, and code. Example: {file: 'cli.py', action: 'add_import', details: 'Add `from models import Task` after line 2'}"
        },
        "needs_re_exploration": {
            "type": "boolean",
            "description": "Whether the project needs re-exploration before replanning"
        },
        "severity": {
            "type": "string",
            "description": "low (minor fix), medium (plan adjustment needed), high (fundamental approach change needed)"
        }
    },
    "required": ["root_cause", "why_chain", "what_to_change", "concrete_edits", "needs_re_exploration", "severity"]
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command in the working directory. Returns stdout, stderr, and exit code. Use for: running tests, git commands, installing packages, compiling, checking file existence, grep, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute (run via bash -c)"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 60)",
                        "default": 60
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories if needed. Use for creating new files or overwriting existing ones completely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to working directory"
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file content to write"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns the file content as a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to working directory"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at the given path. Returns a directory listing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to working directory (default: '.')",
                        "default": "."
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list recursively (default: false)",
                        "default": False
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace a specific string in a file with new content. The old_str must appear exactly once in the file. Use for surgical edits to existing files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to working directory"
                    },
                    "old_str": {
                        "type": "string",
                        "description": "The exact string to find (must be unique in the file)"
                    },
                    "new_str": {
                        "type": "string",
                        "description": "The replacement string"
                    }
                },
                "required": ["path", "old_str", "new_str"]
            }
        }
    }
]


# ============================================================
# Tool Executor — actually runs the tools on the local system
# ============================================================

class ToolExecutor:
    """Executes tool calls from the LLM on the local filesystem."""

    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
        self._current_build_target = ""  # Set by orchestrator before build calls

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return the result as a string."""
        try:
            if tool_name == "run_command":
                return self._run_command(arguments)
            elif tool_name == "write_file":
                return self._write_file(arguments)
            elif tool_name == "read_file":
                return self._read_file(arguments)
            elif tool_name == "list_directory":
                return self._list_directory(arguments)
            elif tool_name == "edit_file":
                return self._edit_file(arguments)
            else:
                return f"ERROR: Unknown tool '{tool_name}'"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def _run_command(self, args: dict) -> str:
        command = args.get("command", "")
        timeout = args.get("timeout", 60)

        if not command:
            return "ERROR: No command provided"

        # Safety rails: block commands that would destroy orchestrator files
        PROTECTED_PATTERNS = [
            "rm -rf .",
            "rm -rf /",
            "rm -rf *",
            "git rm -rf .",
            "git rm -rf *",
            "git rm -rf .agents",
            "git rm -rf prompts",
            "git clean -fdx",
            "sudo ",  # No sudo access
        ]
        # Block deletion of orchestrator files
        PROTECTED_FILES = [
            "standalone_main.py", "standalone_agents.py", "standalone_config.py",
            "standalone_models.py", "standalone_orchestrator.py", "standalone_session.py",
            "test_standalone_integration.py",
        ]
        PROTECTED_DIRS = ["prompts", ".agents"]
        cmd_lower = command.lower().strip()
        for pattern in PROTECTED_PATTERNS:
            if pattern in cmd_lower:
                return f"ERROR: Blocked dangerous command: '{command[:80]}'. This command could damage the orchestrator."
        if "rm " in cmd_lower or "git rm" in cmd_lower:
            for pf in PROTECTED_FILES:
                if pf in command:
                    return f"ERROR: Cannot delete protected orchestrator file: {pf}"
            for pd in PROTECTED_DIRS:
                if pd in command and ("rm " in cmd_lower or "git rm" in cmd_lower):
                    return f"ERROR: Cannot delete protected directory: {pd}"

        logger.debug(f"  TOOL run_command: {command[:100]}")

        try:
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "TERM": "dumb", "NO_COLOR": "1"}
            )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += ("\n" if output else "") + f"STDERR: {result.stderr}"
            output += f"\nEXIT_CODE: {result.returncode}"

            # Truncate very long output
            if len(output) > 10000:
                output = output[:5000] + "\n\n... (truncated) ...\n\n" + output[-3000:]

            # Enrich unhelpful git messages so the LLM understands the situation
            if "nothing to commit" in output and "working tree clean" in output:
                output += "\n\nNOTE: All changes are already committed. No further git action needed. Proceed to the next task or finish."

            return output

        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {timeout}s"

    # Files that agents should not overwrite or delete
    PROTECTED_FILES = {".gitignore"}

    # Required .gitignore entries that must always be present
    REQUIRED_GITIGNORE_ENTRIES = [
        "venv/",
        "__pycache__/",
        "*.pyc",
        ".agents/backups/",
        ".pytest_cache/",
        "node_modules/",
    ]

    def _validate_path(self, path: str) -> Path:
        """Resolve path and ensure it stays within working_dir. Raises ValueError on traversal."""
        full_path = (self.working_dir / path).resolve()
        working_resolved = self.working_dir.resolve()
        if not str(full_path).startswith(str(working_resolved)):
            raise ValueError(f"Path traversal blocked: '{path}' resolves outside working directory")
        return full_path

    def _write_file(self, args: dict) -> str:
        path = args.get("path", "")
        content = args.get("content", "")

        if not path:
            # Try fallback: if we have a build target filename set, use it
            if self._current_build_target:
                path = self._current_build_target
                logger.info(f"  TOOL write_file: using build target fallback: {path}")
            else:
                return "ERROR: No path provided"

        # Protect .gitignore — agents can add entries but not remove required ones
        if path == ".gitignore" or path.endswith("/.gitignore"):
            content = self._protect_gitignore(content)

        full_path = self._validate_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

        logger.debug(f"  TOOL write_file: {path} ({len(content)} bytes)")

        # Lint guard: syntax-check Python files immediately after writing
        lint_result = self._lint_python_file(path, full_path)
        if lint_result:
            return lint_result

        return f"OK: Wrote {len(content)} bytes to {path}"

    def _protect_gitignore(self, new_content: str) -> str:
        """
        Ensure critical .gitignore entries are preserved when agents overwrite the file.

        Agents often write a minimal .gitignore that forgets venv/, which causes
        git commits with 3000+ venv files. This method merges the agent's entries
        with the required entries.
        """
        existing_lines = set(line.strip() for line in new_content.splitlines() if line.strip())

        # Add any required entries that are missing
        missing = []
        for entry in self.REQUIRED_GITIGNORE_ENTRIES:
            if entry not in existing_lines:
                missing.append(entry)

        if missing:
            logger.debug(f"  Protected .gitignore: added missing entries: {missing}")
            if not new_content.endswith("\n"):
                new_content += "\n"
            new_content += "# Required by orchestrator\n"
            new_content += "\n".join(missing) + "\n"

        return new_content

    def _read_file(self, args: dict) -> str:
        path = args.get("path", "")
        if not path:
            return "ERROR: No path provided"

        full_path = self._validate_path(path)
        if not full_path.exists():
            return f"ERROR: File not found: {path}"
        if not full_path.is_file():
            return f"ERROR: Not a file: {path}"

        content = full_path.read_text()

        # Truncate very large files
        if len(content) > 15000:
            content = content[:7000] + "\n\n... (truncated) ...\n\n" + content[-5000:]

        logger.debug(f"  TOOL read_file: {path} ({len(content)} bytes)")
        return content

    def _list_directory(self, args: dict) -> str:
        path = args.get("path", ".")
        recursive = args.get("recursive", False)

        full_path = self._validate_path(path)
        if not full_path.exists():
            return f"ERROR: Directory not found: {path}"

        lines = []
        if recursive:
            for p in sorted(full_path.rglob("*")):
                if any(part.startswith(".") for part in p.relative_to(full_path).parts):
                    if not str(p.relative_to(full_path)).startswith(".agents"):
                        continue
                rel = p.relative_to(full_path)
                prefix = "📁 " if p.is_dir() else "📄 "
                lines.append(f"{prefix}{rel}")
        else:
            for p in sorted(full_path.iterdir()):
                prefix = "📁 " if p.is_dir() else "📄 "
                lines.append(f"{prefix}{p.name}")

        if len(lines) > 200:
            lines = lines[:200] + [f"... and {len(lines) - 200} more"]

        return "\n".join(lines) if lines else "(empty directory)"

    def _edit_file(self, args: dict) -> str:
        path = args.get("path", "")
        old_str = args.get("old_str", "")
        new_str = args.get("new_str", "")

        if not path:
            return "ERROR: No path provided"

        full_path = self._validate_path(path)
        if not full_path.exists():
            return f"ERROR: File not found: {path}"

        content = full_path.read_text()
        count = content.count(old_str)

        if count == 0:
            return f"ERROR: old_str not found in {path}"
        if count > 1:
            # v0.9.2: If old_str appears many times AND roughly matches file size,
            # the model is trying to do a whole-file replacement. Handle gracefully.
            # Research insight (Aider): weaker models ignore search/replace semantics.
            # Aider's --edit-format whole bypasses uniqueness constraints entirely.
            if len(old_str) > len(content) * 0.8:
                # Model passed ~entire file as old_str — treat as whole-file write
                logger.info(f"  🔧 edit_file: old_str covers {len(old_str)}/{len(content)} bytes — "
                           f"treating as whole-file replacement for {path}")
                full_path.write_text(new_str)
                lint_result = self._lint_python_file(path, full_path)
                if lint_result:
                    return lint_result
                return f"OK: Replaced entire content of {path} (whole-file fallback)"
            return f"ERROR: old_str appears {count} times in {path} (must be unique)"

        new_content = content.replace(old_str, new_str, 1)
        full_path.write_text(new_content)

        logger.debug(f"  TOOL edit_file: {path}")

        # Lint guard: syntax-check Python files immediately after editing
        lint_result = self._lint_python_file(path, full_path)
        if lint_result:
            return lint_result

        return f"OK: Replaced text in {path}"

    def _auto_fix_syntax(self, content: str) -> str:
        """v0.8.2: Deterministic auto-fix for known LLM syntax patterns.

        v0.9.2: Added multi-line string triple-quote fix.
        Research insight (Khati et al. 2026, arXiv:2601.19106): Deterministic
        AST-based repair achieves 100% precision and 77% auto-correction
        without involving the LLM. Key principle: intercept and fix KNOWN
        bad patterns before they reach the file system.

        Patterns fixed:
        1. cursor.execute ''' ... ''' → cursor.execute(\"\"\"...\"\"\" )
        2. "CREATE TABLE...(newline)..." → triple-quoted strings for SQL
        """
        import re

        # === Pattern 1: cursor.execute triple-single-quote (original v0.8.2) ===
        lines = content.split("\n")
        result = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Detect: <indent>self.cursor.execute '''( or cursor.execute '''
            m = re.match(r"^(\s+)((?:self\.)?cursor\.execute)\s+'''", line)
            if m:
                indent = m.group(1)
                obj = m.group(2)
                # Grab everything after the triple-quote opener
                after_open = re.sub(r"^.*?'''", "", line, count=1)
                sql_parts = [after_open]
                i += 1
                params = ""
                found_close = False
                while i < len(lines):
                    if "'''" in lines[i]:
                        before_close = lines[i].split("'''")[0]
                        sql_parts.append(before_close)
                        after_close = lines[i].split("'''", 1)[1].strip()
                        # Check for params like , (val1, val2)
                        pm = re.search(r',\s*(\(.*?\))', after_close)
                        if pm:
                            params = pm.group(1)
                        found_close = True
                        i += 1
                        break
                    sql_parts.append(lines[i])
                    i += 1
                if found_close:
                    sql = "\n".join(sql_parts).strip().lstrip("(").rstrip(")")
                    result.append(f'{indent}_query = """{sql}"""')
                    if params:
                        result.append(f"{indent}{obj}(_query, {params})")
                    else:
                        result.append(f"{indent}{obj}(_query)")
                else:
                    # Couldn't find close, leave original
                    result.append(line)
            else:
                result.append(line)
                i += 1
        content = "\n".join(result)

        # === Pattern 2: Multi-line double-quoted strings (v0.9.2) ===
        # Llama 3.3 70B consistently generates:
        #   cursor.execute("CREATE TABLE IF NOT EXISTS projects (
        #       id INTEGER PRIMARY KEY,
        #       ...
        #   )")
        # This is a SyntaxError because " strings can't span lines.
        # Fix: convert to triple-quoted strings.
        #
        # Strategy: find opening " on a line that contains SQL keywords,
        # where the closing " is on a DIFFERENT line. Replace " with """.
        SQL_KEYWORDS = r'(?:CREATE\s+TABLE|INSERT\s+INTO|SELECT\s+|UPDATE\s+|DELETE\s+FROM|ALTER\s+TABLE|DROP\s+TABLE|CREATE\s+INDEX|PRAGMA\s+)'

        lines = content.split("\n")
        result = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Look for opening pattern: something("SQL_KEYWORD or = "SQL_KEYWORD
            # where the line does NOT have a matching closing quote
            open_match = re.search(
                r'(\()?\s*"(' + SQL_KEYWORDS + r')',
                line
            )
            if open_match:
                # Count unescaped quotes on this line AFTER the match
                after_match = line[open_match.start():]
                # Simple quote counting: if odd number of ", the string isn't closed
                quote_count = after_match.count('"') - after_match.count('\\"')
                if quote_count % 2 == 1:  # Odd = unclosed string
                    # Found multi-line string! Scan forward for closing "
                    close_line_idx = None
                    for j in range(i + 1, min(i + 50, len(lines))):
                        # Look for a line with a closing " that's not inside the SQL
                        stripped = lines[j].rstrip()
                        if stripped.endswith('")') or stripped.endswith('"') or stripped.endswith('",'):
                            close_line_idx = j
                            break
                        # Also check for ") or ", (params)
                        close_m = re.search(r'"\s*[,)]', lines[j])
                        if close_m:
                            close_line_idx = j
                            break
                    if close_line_idx is not None:
                        # Replace opening " with """ and closing " with """
                        # Opening line: replace first " before SQL keyword with """
                        fixed_open = line[:open_match.start()] + re.sub(
                            r'"(' + SQL_KEYWORDS + r')',
                            r'"""\1',
                            line[open_match.start():],
                            count=1
                        )
                        result.append(fixed_open)
                        # Middle lines: unchanged
                        for k in range(i + 1, close_line_idx):
                            result.append(lines[k])
                        # Closing line: replace the closing " with """
                        close_line = lines[close_line_idx]
                        # Find the last " that closes the SQL string
                        # Replace it with """
                        close_fixed = re.sub(r'"(\s*[,)])', r'"""\1', close_line, count=1)
                        if close_fixed == close_line:
                            # Try: line ends with just "
                            close_fixed = re.sub(r'"\s*$', '"""', close_line, count=1)
                        result.append(close_fixed)
                        i = close_line_idx + 1
                        logger.info(f"  🔧 AUTO-FIX: converted multi-line SQL string to triple-quotes (lines {i-close_line_idx+i}–{close_line_idx+1})")
                        continue
            result.append(line)
            i += 1

        return "\n".join(result)

    def _lint_python_file(self, path: str, full_path: Path) -> Optional[str]:
        """
        Syntax-check a Python file immediately after write/edit.

        Inspired by Aider and SWE-agent: both lint after every edit to catch
        syntax errors before they cascade into import failures across modules.

        Returns None if file is OK, or an error message string if syntax error found.
        The file is kept on disk (so the agent can read_file and fix it) but
        the error is returned inline so the agent can fix it in the SAME turn.
        """
        if not path.endswith(".py"):
            return None

        # v0.8.2: Auto-fix known syntax patterns BEFORE lint check
        try:
            content = full_path.read_text()
            fixed_content = self._auto_fix_syntax(content)
            if fixed_content != content:
                full_path.write_text(fixed_content)
                logger.info(f"  🔧 AUTO-FIX: repaired known syntax pattern in {path}")
        except Exception:
            pass

        try:
            import py_compile
            py_compile.compile(str(full_path), doraise=True)
            return None
        except py_compile.PyCompileError as e:
            # Extract the useful part of the error
            error_msg = str(e)

            # Show the broken line with context if possible
            context_lines = ""
            try:
                lines = full_path.read_text().splitlines()
                # py_compile errors usually include line number
                import re
                line_match = re.search(r'line (\d+)', error_msg)
                if line_match:
                    error_line = int(line_match.group(1))
                    start = max(0, error_line - 3)
                    end = min(len(lines), error_line + 2)
                    numbered = []
                    for i in range(start, end):
                        marker = ">>>" if i == error_line - 1 else "   "
                        numbered.append(f"  {marker} {i+1:3d} | {lines[i]}")
                    context_lines = "\n" + "\n".join(numbered)
            except Exception:
                pass

            logger.warning(f"  LINT GUARD: SyntaxError in {path}: {error_msg[:100]}")

            return (
                f"SYNTAX ERROR in {path} — file was written but has invalid Python syntax.\n"
                f"Error: {error_msg}\n"
                f"{context_lines}\n\n"
                "⚠️ You MUST fix this syntax error before proceeding.\n"
                "Use read_file to see the current content, then use edit_file or "
                "write_file to fix the error.\n"
                "Common fixes:\n"
                "  - @dataclass decorator: must be '@dataclass' on its own line, "
                "then 'class ClassName:' on the next line\n"
                "  - Missing colons, parentheses, or indentation errors\n"
                "  - Unclosed strings or brackets"
            )


# ============================================================
# LLM Client — talks to Ollama or Anthropic HTTP API directly
# ============================================================

class LLMClient:
    """
    Direct HTTP client for LLM APIs.

    Supports:
    - Ollama: POST /api/chat with tool support
    - Anthropic: POST /v1/messages (fallback for bootstrapping)
    """

    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.session = httpx.Client(timeout=900)
        self.session.headers.update({"Content-Type": "application/json"})

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat request and return the raw response.

        Returns dict with:
          - "content": str (text response)
          - "tool_calls": list of tool call objects (if any)
          - "done": bool
        """
        if self.config.provider == "ollama":
            return self._chat_ollama(messages, tools, temperature)
        elif self.config.provider == "anthropic":
            return self._chat_anthropic(messages, tools, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def chat_structured(
        self,
        messages: List[Dict[str, Any]],
        schema: dict,
        temperature: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Call Ollama with structured output (JSON schema constraint).

        The model is forced to output valid JSON matching the provided schema.
        No regex parsing needed — the output is deterministic.

        Args:
            messages: Chat messages
            schema: JSON schema dict that the output must conform to
            temperature: Override temperature

        Returns:
            Parsed JSON dict, or None on failure
        """
        if self.config.provider != "ollama":
            logger.warning("Structured output only supported for Ollama provider")
            return None

        endpoint = (self.config.endpoint or "http://127.0.0.1:11434").rstrip("/")
        url = f"{endpoint}/api/chat"

        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }

        try:
            resp = self.session.post(url, json=payload, timeout=900)  # v0.9.9c: 600→900 (edit repair on large files)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Structured output request failed: {e}")
            return None

        content = data.get("message", {}).get("content", "")
        if not content:
            logger.warning("Structured output returned empty content")
            return None

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured output as JSON: {e}")
            logger.debug(f"Raw content: {content[:500]}")
            return None

    def _chat_ollama(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Call Ollama /api/chat endpoint."""
        endpoint = (self.config.endpoint or "http://127.0.0.1:11434").rstrip("/")
        url = f"{endpoint}/api/chat"

        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }

        if tools and self.config.supports_tools:
            payload["tools"] = tools

        try:
            resp = self.session.post(url, json=payload, timeout=900)  # v0.9.9c: 600→900 (edit repair on large files)
            resp.raise_for_status()
            data = resp.json()
        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to Ollama at {endpoint}: {e}")
        except httpx.TimeoutException:
            raise TimeoutError("Ollama request timed out after 900s")
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

        # Parse Ollama response format
        message = data.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])

        return {
            "content": content,
            "tool_calls": tool_calls,
            "done": data.get("done", True),
            "total_duration": data.get("total_duration"),
            "eval_count": data.get("eval_count"),
        }

    def _chat_anthropic(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Call Anthropic /v1/messages endpoint."""
        api_key = os.environ.get(self.config.api_key_env or "ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(f"API key not found in env var: {self.config.api_key_env}")

        url = "https://api.anthropic.com/v1/messages"

        # Convert messages to Anthropic format
        # Anthropic uses system as a top-level param, not in messages
        system_text = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text += msg["content"] + "\n"
            else:
                api_messages.append(msg)

        payload = {
            "model": self.config.model_id,
            "max_tokens": self.config.max_tokens,
            "messages": api_messages,
        }
        if system_text:
            payload["system"] = system_text.strip()
        if temperature is not None:
            payload["temperature"] = temperature

        # Convert tools to Anthropic format
        if tools:
            anthropic_tools = []
            for t in tools:
                if t.get("type") == "function":
                    func = t["function"]
                    anthropic_tools.append({
                        "name": func["name"],
                        "description": func["description"],
                        "input_schema": func["parameters"]
                    })
            if anthropic_tools:
                payload["tools"] = anthropic_tools

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        try:
            resp = self.session.post(url, json=payload, headers=headers, timeout=300)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")

        # Parse Anthropic response
        content_text = ""
        tool_calls = []
        for block in data.get("content", []):
            if block["type"] == "text":
                content_text += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append({
                    "function": {
                        "name": block["name"],
                        "arguments": block["input"],  # already a dict
                    },
                    "id": block["id"],
                })

        return {
            "content": content_text,
            "tool_calls": tool_calls,
            "done": data.get("stop_reason") != "tool_use",
        }


# ============================================================
# AgentRunner — the agentic tool-use loop
# ============================================================

class AgentRunner:
    """
    Runs agents with real tool execution.

    The core loop:
    1. Send prompt + tools to LLM
    2. If response contains tool_calls → execute each tool locally
    3. Append tool results to conversation
    4. Send back to LLM for next step
    5. Repeat until LLM responds with just text (no more tool calls)
    """

    def __init__(self, config: Config, working_dir: Path):
        self.config = config
        self.working_dir = working_dir
        self.tool_executor = ToolExecutor(working_dir)
        self.prompts_dir = Path(__file__).parent
        self._llm_clients: Dict[str, LLMClient] = {}

        # ACE playbook integration — inject learned patterns into agent prompts
        _pb_path = "/home/brandon/standalone-orchestrator/playbook.json"
        if not Path(_pb_path).exists():
            _pb_path = str(working_dir.parent / "playbook.json")
        self._playbook = PlaybookReader(playbook_path=_pb_path)
        self._role_map = {
            "initializer": "initializer",
            "explore": "explorer",
            "plan": "planner",
            "build": "builder",
            "test": "test_gen",
            "test_gen": "test_gen",
        }

    def _get_llm_client(self, model_config: ModelConfig) -> LLMClient:
        """Get or create an LLM client for the given model config."""
        key = f"{model_config.provider}:{model_config.endpoint}:{model_config.model_id}"
        if key not in self._llm_clients:
            self._llm_clients[key] = LLMClient(model_config)
        return self._llm_clients[key]

    def _load_prompt(self, prompt_file: str) -> str:
        prompt_path = self.prompts_dir / prompt_file
        if prompt_path.exists():
            return prompt_path.read_text()
        # Also check relative to working dir
        alt_path = self.working_dir / prompt_file
        if alt_path.exists():
            return alt_path.read_text()
        logger.warning(f"Prompt file not found: {prompt_path}")
        return ""

    def _run_agent(
        self,
        agent_config: AgentConfig,
        system_prompt: str,
        user_prompt: str,
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> AgentResult:
        """
        THE CORE AGENTIC LOOP.

        Sends the prompt to the LLM with tool definitions, then loops:
        - If the LLM returns tool calls → execute them → feed results back
        - If the LLM returns text only → done, return the result

        Includes stuck-loop detection: if the agent repeats the same tool calls
        3 times in a row, it's stuck and we bail out early.
        """
        client = self._get_llm_client(agent_config.model)
        # If temperature override provided, temporarily adjust client config
        original_temp = None
        if temperature is not None:
            original_temp = client.config.temperature
            client.config.temperature = temperature
        max_rounds = agent_config.max_tool_rounds
        start_time = time.time()

        # Build initial messages
        messages = []
        if system_prompt:
            # ACE playbook injection — add learned patterns to agent context
            playbook_role = self._role_map.get(agent_config.role, "general")
            playbook_context = self._playbook.get_context_for_agent(
                role=playbook_role,
                task_goal=user_prompt[:500],
            )
            if playbook_context:
                system_prompt = system_prompt + "\n\n" + playbook_context
                logger.debug(f"Playbook injected for {agent_config.role}: {len(playbook_context)} chars")
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        all_output = []  # Collect all text output across rounds
        round_count = 0
        recent_commands = []  # Track last N rounds for stuck-loop detection
        STUCK_THRESHOLD = 3  # Bail if same commands repeat this many times

        logger.info(f"Running {agent_config.role} agent...")
        logger.debug(f"  Model: {agent_config.model.name} ({agent_config.model.model_id})")
        logger.debug(f"  Endpoint: {agent_config.model.endpoint}")

        try:
            while round_count < max_rounds:
                round_count += 1

                # Call the LLM
                response = client.chat(messages, tools=tools)

                content = response.get("content", "")
                tool_calls = response.get("tool_calls", [])

                # Tool call resolution strategy:
                # 1. If model returned native tool_calls (Qwen3, Llama 3.3 with native support),
                #    use them directly — no parsing needed
                # 2. If no native tool_calls but content has JSON tool calls embedded,
                #    parse them from text (fallback for Qwen 2.5, older models)
                if not tool_calls and content and tools:
                    parsed_calls = self._extract_tool_calls_from_text(content)
                    if parsed_calls:
                        tool_calls = parsed_calls
                        # Strip the tool-call JSON from the text output
                        content = self._strip_tool_json_from_text(content)
                        logger.debug(f"  Extracted {len(tool_calls)} tool call(s) from text content")
                elif tool_calls:
                    # Native tool calls — strip any <think> blocks from content
                    if content:
                        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                    logger.debug(f"  Native tool call(s): {len(tool_calls)}")

                if content:
                    all_output.append(content)

                # If no tool calls, we're done
                if not tool_calls:
                    logger.debug(f"  Agent finished after {round_count} rounds")
                    break

                # Process tool calls
                logger.debug(f"  Round {round_count}: {len(tool_calls)} tool call(s)")

                # Add assistant message with tool calls to conversation
                assistant_msg = {"role": "assistant", "content": content or ""}
                if agent_config.model.provider == "ollama":
                    assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)

                # Execute each tool call and add results
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    tool_args = func.get("arguments", {})

                    # Ollama sometimes returns arguments as a string
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {"command": tool_args}

                    # Execute the tool
                    tool_result = self.tool_executor.execute(tool_name, tool_args)

                    logger.debug(f"  Tool {tool_name}: {tool_result[:100]}...")

                    # Add tool result to conversation
                    # Format depends on provider
                    if agent_config.model.provider == "anthropic":
                        messages.append({
                            "role": "user",
                            "content": [{  # type: ignore[dict-item]
                                "type": "tool_result",
                                "tool_use_id": tc.get("id", ""),
                                "content": tool_result
                            }]
                        })
                    else:
                        # Ollama format — include tool_name for Qwen3 native tool calling
                        tool_msg = {
                            "role": "tool",
                            "content": tool_result,
                        }
                        # Qwen3 expects tool_name in the tool result message
                        if agent_config.model.native_tool_calling:
                            tool_msg["tool_name"] = tool_name
                        messages.append(tool_msg)

                # --- Stuck-loop detection ---
                # Build a fingerprint of this round's tool calls
                round_fingerprint = "|".join(
                    f"{tc.get('function', {}).get('name', '')}:{json.dumps(tc.get('function', {}).get('arguments', {}), sort_keys=True)}"
                    for tc in tool_calls
                )
                recent_commands.append(round_fingerprint)

                # Check if the last STUCK_THRESHOLD rounds are identical
                if len(recent_commands) >= STUCK_THRESHOLD:
                    last_n = recent_commands[-STUCK_THRESHOLD:]
                    if len(set(last_n)) == 1:
                        elapsed = time.time() - start_time
                        logger.warning(
                            f"  🔄 STUCK LOOP DETECTED: same commands repeated {STUCK_THRESHOLD}x "
                            f"(round {round_count}). Bailing out."
                        )
                        logger.debug(f"  Repeated commands: {last_n[0][:200]}")
                        return AgentResult(
                            success=False,
                            output="\n".join(all_output),
                            error=f"Stuck loop detected at round {round_count}: same tool calls repeated {STUCK_THRESHOLD} times. "
                                  "The agent cannot solve this problem with its current approach.",
                            duration_seconds=elapsed,
                        )

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > agent_config.timeout_seconds:
                    logger.warning(f"  Agent timed out after {elapsed:.0f}s")
                    return AgentResult(
                        success=False,
                        output="\n".join(all_output),
                        error=f"Timeout after {elapsed:.0f}s ({round_count} rounds)",
                        duration_seconds=elapsed
                    )

            duration = time.time() - start_time

            if round_count >= max_rounds:
                logger.warning(f"  Agent hit max rounds ({max_rounds})")
                return AgentResult(
                    success=False,
                    output="\n".join(all_output),
                    error=f"Hit max tool rounds ({max_rounds})",
                    duration_seconds=duration
                )

            return AgentResult(
                success=True,
                output="\n".join(all_output),
                duration_seconds=duration
            )

        except (ConnectionError, TimeoutError) as e:
            duration = time.time() - start_time
            logger.error(f"  Connection error: {e}")
            return AgentResult(
                success=False, output="\n".join(all_output),
                error=str(e), duration_seconds=duration
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"  Agent execution failed: {e}")
            return AgentResult(
                success=False, output="\n".join(all_output),
                error=str(e), duration_seconds=duration
            )

    # ============================================================
    # Fallback: extract tool calls from text content
    # ============================================================

    def _extract_tool_calls_from_text(self, text: str) -> List[dict]:
        """
        Extract tool calls when the model puts them as JSON in text content
        instead of in the structured tool_calls field.

        v0.9.2: Backtrack fix for Llama 3.3 70B which outputs
        {"type":"function","name":"write_file",...} where "name" isn't the first key.
        Old regex required { immediately before "name", missing these calls entirely.
        New approach: find "name":"tool_name" anywhere, backtrack to find enclosing {.

        Handles formats like:
          {"name": "write_file", "arguments": {"path": "hello.py", "content": "..."}}
          {"type": "function", "name": "write_file", "arguments": {...}}
        Also handles Qwen artifacts like <|im_start|> tokens.
        """
        tool_calls = []
        known_tools = {"run_command", "write_file", "read_file", "list_directory", "edit_file"}

        # Clean up common model artifacts
        cleaned = text.replace("<|im_start|>", "\n").replace("<|im_end|>", "\n")

        # Strategy: find "name": "tool_name" anywhere, then backtrack to find
        # the enclosing { and parse the full JSON object.
        decoder = json.JSONDecoder()

        for match in re.finditer(r'"name"\s*:\s*"(\w+)"', cleaned):
            tool_name = match.group(1)
            if tool_name not in known_tools:
                continue

            # Backtrack to find the opening { (up to 200 chars back)
            start = match.start()
            brace_pos = cleaned.rfind('{', max(0, start - 200), start)
            if brace_pos == -1:
                # No opening brace found, try parsing from match start (old behavior)
                brace_pos = start
                # But only if it starts with {
                if brace_pos < len(cleaned) and cleaned[brace_pos] != '{':
                    continue

            try:
                obj, end = decoder.raw_decode(cleaned, brace_pos)
                if isinstance(obj, dict) and "name" in obj and ("arguments" in obj or "parameters" in obj):
                    args = obj.get("arguments") or obj.get("parameters", {})
                    tool_calls.append({
                        "function": {
                            "name": obj["name"],
                            "arguments": args
                        }
                    })
            except (json.JSONDecodeError, ValueError):
                # Backtrack didn't work, try from the match position directly
                # (handles case where { is right before "name")
                if brace_pos != start:
                    try:
                        # Look for { right at or before the "name" key
                        direct_brace = cleaned.rfind('{', max(0, start - 5), start + 1)
                        if direct_brace >= 0:
                            obj, end = decoder.raw_decode(cleaned, direct_brace)
                            if isinstance(obj, dict) and "name" in obj and ("arguments" in obj or "parameters" in obj):
                                args = obj.get("arguments") or obj.get("parameters", {})
                                tool_calls.append({
                                    "function": {
                                        "name": obj["name"],
                                        "arguments": args
                                    }
                                })
                    except (json.JSONDecodeError, ValueError):
                        continue

        # Deduplicate — model sometimes repeats the same call
        seen = set()
        unique_calls = []
        for tc in tool_calls:
            key = json.dumps(tc, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_calls.append(tc)

        return unique_calls

    def _strip_tool_json_from_text(self, text: str) -> str:
        """Remove tool-call JSON from text content, keeping any surrounding prose.

        v0.9.2: Backtrack fix matching _extract_tool_calls_from_text.
        """
        cleaned = text.replace("<|im_start|>", " ").replace("<|im_end|>", " ")

        # Remove JSON tool call blocks by finding and removing them
        decoder = json.JSONDecoder()
        known_tools = {"run_command", "write_file", "read_file", "list_directory", "edit_file"}

        # Find all tool call JSON positions and remove them
        for match in re.finditer(r'"name"\s*:\s*"(\w+)"', cleaned):
            if match.group(1) not in known_tools:
                continue
            # Backtrack to find opening {
            start = match.start()
            brace_pos = cleaned.rfind('{', max(0, start - 200), start)
            if brace_pos == -1:
                brace_pos = start
                if brace_pos < len(cleaned) and cleaned[brace_pos] != '{':
                    continue
            try:
                _, end = decoder.raw_decode(cleaned, brace_pos)
                # Mark this region for removal
                cleaned = cleaned[:brace_pos] + " " + cleaned[end:]
            except (json.JSONDecodeError, ValueError):
                continue

        # Clean up whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        return cleaned

    # ============================================================
    # v0.9.3: Plain-text file content parser
    # ============================================================

    @staticmethod
    def parse_plain_file_content(output: str) -> Optional[str]:
        """
        v0.9.3: Extract file content from plain-text markers.

        The plain-text build mode asks the model to output code between
        markers instead of wrapping it in JSON tool calls. This eliminates
        JSON escaping failures (triple-quote SQL, multi-line strings, etc.)
        that local models consistently produce.

        Extraction priority:
        1. <<<CONTENT>>> ... <<<END>>> markers (preferred)
        2. ```python ... ``` code blocks (fallback)
        3. Raw Python if output looks like pure code (last resort)
        4. None if nothing found
        """
        # Strategy 1: <<<CONTENT>>> ... <<<END>>> markers
        content_pattern = r'<<<CONTENT>>>[ \t]*\n?(.*?)<<<END>>>'
        match = re.search(content_pattern, output, re.DOTALL)
        if match:
            content = match.group(1).rstrip()
            if len(content) > 20:
                return content

        # Strategy 2: ```python ... ``` code blocks — take longest
        backticks = chr(96) * 3
        pattern = backticks + r'python\n(.*?)' + backticks
        blocks = re.findall(pattern, output, re.DOTALL)
        if blocks:
            best = max(blocks, key=len)
            if len(best.strip()) > 20:
                return best.strip()

        # Strategy 3: If the entire output looks like pure Python
        stripped = output.strip()
        # Remove any <think>...</think> blocks first
        stripped = re.sub(r'<think>.*?</think>', '', stripped, flags=re.DOTALL).strip()
        if stripped and len(stripped) > 100:
            first_line = stripped.split('\n')[0].strip()
            python_starts = ('import ', 'from ', 'class ', 'def ', '#!', '#')
            if first_line.startswith(('"""', "'''")):
                return stripped
            if any(first_line.startswith(s) for s in python_starts):
                return stripped

        return None

    # ============================================================
    # Role-specific agent methods
    # ============================================================

    def run_initializer(self, goal: str, task_id: str) -> AgentResult:
        """
        Initialize the project environment.

        Before delegating to the LLM, we do deterministic setup that the
        build agent shouldn't waste rounds on: git init and venv creation.
        """
        # --- Deterministic setup (no LLM needed) ---
        self._setup_project_environment(task_id)

        # --- LLM-driven initialization (feature breakdown, etc.) ---
        agent_config = self.config.get_agent("initializer")

        system_prompt = self._load_prompt(agent_config.system_prompt_file or "prompts/initializer.txt")
        if not system_prompt:
            system_prompt = "You are the INITIALIZER agent. Set up the project environment for the given task."

        user_prompt = """## Task
{goal}

## Task ID
{task_id}

## Instructions

The project environment is already set up (git repo initialized, Python venv created).

1. Run `ls -la` to see the current directory
2. Analyze the goal and break it into specific, testable features
3. Create feature_list.json with the feature breakdown
4. Commit the feature list: `git add -A && git commit -m "chore: initialize task {task_id}"`

IMPORTANT:
- Git is already initialized — do NOT run `git init`
- A Python venv exists at `./venv` — use `source venv/bin/activate` before pip commands
- Use `python3 -m pytest` to run tests (not bare `pytest`)
"""
        return self._run_agent(agent_config, system_prompt, user_prompt, tools=TOOL_DEFINITIONS)

    def _setup_project_environment(self, task_id: str):
        """
        Deterministic project setup — git repo, venv, .gitignore.

        This runs before any LLM agent touches the project, so the build
        agent never wastes rounds on 'git init' or 'pip install pytest'.
        """
        wd = self.working_dir

        # Git init (if not already a repo)
        git_dir = wd / ".git"
        if not git_dir.exists():
            logger.info("Setting up git repository...")
            result = subprocess.run(
                ["git", "init"], cwd=wd, capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.debug("  Git repo initialized")
            else:
                logger.warning(f"  Git init failed: {result.stderr}")

            # Configure git user if not set
            subprocess.run(
                ["git", "config", "user.email", "agent@orchestrator.local"],
                cwd=wd, capture_output=True
            )
            subprocess.run(
                ["git", "config", "user.name", "Orchestrator Agent"],
                cwd=wd, capture_output=True
            )

        # .gitignore
        gitignore = wd / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(
                "venv/\n__pycache__/\n*.pyc\n.agents/backups/\n"
                "*.egg-info/\ndist/\nbuild/\n.pytest_cache/\n"
            )
            logger.debug("  Created .gitignore")

        # Python venv (if not already created)
        venv_dir = wd / "venv"
        if not venv_dir.exists():
            logger.info("Creating Python virtual environment...")
            result = subprocess.run(
                ["python3", "-m", "venv", str(venv_dir)],
                cwd=wd, capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.debug("  Venv created at ./venv")
                # Install pytest in the venv
                pip_path = venv_dir / "bin" / "pip"
                if pip_path.exists():
                    result = subprocess.run(
                        [str(pip_path), "install", "pytest"],
                        cwd=wd, capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        logger.debug("  Installed pytest in venv")
                    else:
                        logger.warning(f"  Failed to install pytest: {result.stderr[:200]}")
            else:
                logger.warning(f"  Venv creation failed: {result.stderr}")

        # .agents directory structure
        for subdir in ["plans", "reports", "logs", "backups"]:
            (wd / ".agents" / subdir).mkdir(parents=True, exist_ok=True)

        # Initial commit (if repo is empty)
        result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=wd, capture_output=True, text=True
        )
        if result.returncode != 0:
            # No commits yet — make initial commit
            subprocess.run(["git", "add", "-A"], cwd=wd, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", f"chore: initialize project for task {task_id}"],
                cwd=wd, capture_output=True
            )
            logger.debug("  Created initial git commit")

    def run_explore(self, state: TaskState) -> AgentResult:
        agent_config = self.config.get_agent("explore")

        system_prompt = self._load_prompt(agent_config.system_prompt_file or "prompts/explore.txt")
        if not system_prompt:
            system_prompt = "You are the EXPLORE agent. Gather context about the codebase. READ files, don't modify them."

        user_prompt = """## Task
{state.goal}

## Task ID
{state.task_id}

## Iteration
{state.iteration}

## Instructions
1. Run `ls -la` and examine the project structure
2. Read key files to understand the codebase
3. Identify patterns, dependencies, and relevant code
4. Summarize your findings for the planning agent

Focus on what's relevant to the task. Be selective.
"""
        # Phase 1: Run the agentic exploration (read files, list dirs, etc.)
        result = self._run_agent(agent_config, system_prompt, user_prompt, tools=TOOL_DEFINITIONS)

        if not result.success:
            return result

        # Phase 2: Summarize findings as structured output
        structured_report = self._summarize_exploration_structured(
            agent_config, state, result.output
        )

        if structured_report:
            # Format the structured report for the plan agent
            formatted = self._format_exploration_report(structured_report)
            result.output = formatted
            result.exploration_summary = formatted
            logger.info(f"  Explore agent produced structured report: {len(structured_report.get('files', []))} files, "
                       f"{len(structured_report.get('patterns', []))} patterns")
        else:
            # Fallback: use raw output
            result.exploration_summary = result.output
            logger.debug("  Explore agent: structured summary unavailable, using raw output")

        return result

    def _summarize_exploration_structured(
        self,
        agent_config: AgentConfig,
        state: TaskState,
        raw_exploration: str,
    ) -> Optional[dict]:
        """
        Take the raw exploration output and produce a structured JSON report.

        Uses the same model as the explore agent with structured output mode.
        This is a second LLM call but it's cheap (summarization, not tool use).
        """
        model_config = agent_config.model
        if model_config.provider != "ollama":
            return None

        client = LLMClient(model_config)
        messages = [
            {"role": "system", "content": (
                "You are a code analysis agent. Summarize the exploration findings "
                "into a structured report. Be precise about file paths and patterns. "
                "Only include files and patterns that actually exist — do not hallucinate."
            )},
            {"role": "user", "content": """## Task
{state.goal}

## Raw Exploration Output
{raw_exploration[:6000]}

Summarize the above exploration into a structured report.
Include only real file paths and patterns you can see in the output.
"""},
        ]

        try:
            return client.chat_structured(messages, schema=EXPLORE_OUTPUT_SCHEMA, temperature=0.0)
        except Exception as e:
            logger.warning(f"Structured exploration summary failed: {e}")
            return None

    def _format_exploration_report(self, report: dict) -> str:
        """Format a structured exploration report as readable text for the plan agent."""
        lines = ["## Exploration Report (Structured)", ""]

        lines.append(f"**Project Type:** {report.get('project_type', 'unknown')}")
        lines.append(f"**Summary:** {report.get('summary', 'No summary')}")
        lines.append("")

        files = report.get("files", [])
        if files:
            lines.append("### Files")
            for f in files:
                key = f.get("key_contents", "")
                role = f.get("role", "")
                line = f"- `{f['path']}` — {role}"
                if key:
                    line += f" ({key})"
                lines.append(line)
            lines.append("")

        deps = report.get("dependencies", [])
        if deps:
            lines.append(f"### Dependencies: {', '.join(deps)}")
            lines.append("")

        patterns = report.get("patterns", [])
        if patterns:
            lines.append("### Patterns")
            for p in patterns:
                lines.append(f"- {p}")
            lines.append("")

        tests = report.get("existing_tests", [])
        if tests:
            lines.append("### Existing Tests")
            for t in tests:
                fw = t.get("framework", "unknown")
                count = t.get("count", "?")
                lines.append(f"- `{t['path']}` ({fw}, ~{count} tests)")
            lines.append("")

        relevant = report.get("relevant_to_task", "")
        if relevant:
            lines.append(f"### Relevance to Task\n{relevant}")
            lines.append("")

        return "\n".join(lines)

    def run_plan(self, state: TaskState) -> AgentResult:
        agent_config = self.config.get_agent("plan")

        system_prompt = self._load_prompt(agent_config.system_prompt_file or "prompts/plan.txt")
        if not system_prompt:
            system_prompt = "You are the PLAN agent. Create detailed implementation plans with measurable Definition of Done criteria."

        failure_context = ""
        if state.failure_history:
            failure_context = "\n## Previous Failures (MUST ADDRESS)\n"
            for f in state.failure_history[-3:]:
                failure_context += f"- Iteration {f.get('iteration')}: [{f.get('phase')}] {f.get('error')}\n"
                if f.get('rca'):
                    failure_context += f"  RCA: {f.get('rca')}\n"

            # Extract the most recent RCA action directive if available
            last_rca = state.failure_history[-1].get("rca", "")
            if "ACTION:" in last_rca:
                action_line = last_rca.split("ACTION:")[1].split("\n")[0].strip()
                failure_context += f"\n## ⚠️ MANDATORY FIX FOR THIS ITERATION\n{action_line}\n"
                failure_context += "Your plan MUST directly address this action item.\n"

            # Include which specific criteria failed AND which passed
            last_dod_results = state.failure_history[-1].get("dod_results", {})
            if last_dod_results:
                failed_criteria = []
                passed_criteria = []
                if isinstance(last_dod_results, dict) and "criteria_results" in last_dod_results:
                    for cr in last_dod_results["criteria_results"]:
                        if cr.get("passed"):
                            passed_criteria.append(f"  - ✅ {cr['criterion_id']}: {cr.get('description', '?')}")
                        else:
                            reason = cr.get("failure_reason", "unknown")
                            failed_criteria.append(f"  - ❌ {cr['criterion_id']}: {cr.get('description', '?')} — {reason}")
                elif isinstance(last_dod_results, dict):
                    for cid, res in last_dod_results.items():
                        if isinstance(res, dict):
                            if res.get("passed"):
                                passed_criteria.append(f"  - ✅ {cid}: passed")
                            else:
                                evidence = res.get("evidence", "unknown")[:100]
                                failed_criteria.append(f"  - ❌ {cid}: {evidence}")

                if passed_criteria:
                    failure_context += "\n## Passing Criteria (DO NOT BREAK THESE)\n"
                    failure_context += "\n".join(passed_criteria) + "\n"
                    failure_context += "\nThese criteria PASSED. Your plan MUST preserve them.\n"

                if failed_criteria:
                    failure_context += "\n## Failed Criteria (MUST FIX)\n"
                    failure_context += "\n".join(failed_criteria) + "\n"
                    failure_context += "\nYour plan MUST address these specific failures.\n"
                    failure_context += "Focus your steps on FIXING failures, not rewriting everything.\n"

            # Budget cap failure context
            failure_context = truncate_to_budget(
                failure_context, PLAN_PROMPT_BUDGET["failure_context"], "plan failure context"
            )

        exploration_context = ""
        if state.exploration_context:
            exploration_context = truncate_to_budget(
                state.exploration_context,
                PLAN_PROMPT_BUDGET["exploration_context"],
                "exploration"
            )
            exploration_context = f"\n## Exploration Findings\n{exploration_context}\n"

        user_prompt = """## Task
{state.goal}

## Task ID
{state.task_id}

## Iteration
{state.iteration}
{failure_context}
{exploration_context}

## Instructions

Create a detailed implementation plan as structured JSON with:

1. **plan_summary** — brief description of the approach
2. **steps** — ordered implementation steps, each with files to create/modify
3. **definition_of_done** — specific, testable criteria describing WHAT must be true when the task is complete

## Rules for definition_of_done
- Each criterion needs only a **description** — verification commands are generated AUTOMATICALLY after the build
- Include a **verification_type** when helpful: 'test' (default), 'file_exists', 'syntax', 'import'
- Include a **target_file** when the criterion relates to a specific file
- Be specific — "works correctly" is not measurable
- Good examples: "app.py has /health endpoint returning {{status: ok}}", "divide(1,0) raises ValueError", "test_app.py passes all tests"
- The build agent MUST write unit tests — verification primarily runs those tests
- If this is a retry: keep the SAME DoD criteria that were passing, and focus your plan steps on fixing the failures
- Do NOT inflate the number of criteria on retries — fix the failing ones, keep the passing ones
- If this is a retry, your plan MUST address previous failures
"""

        # Try structured output first (Ollama JSON schema mode)
        result = self._run_plan_structured(agent_config, system_prompt, user_prompt, state)
        if result is not None:
            return result

        # Fallback: unstructured with regex parsing (for non-Ollama or old Ollama versions)
        logger.info("Structured output unavailable, falling back to regex parsing")
        return self._run_plan_unstructured(agent_config, system_prompt, user_prompt, state)

    def _run_plan_structured(
        self,
        agent_config: AgentConfig,
        system_prompt: str,
        user_prompt: str,
        state: TaskState,
    ) -> Optional[AgentResult]:
        """
        Run plan agent with Ollama structured output (JSON schema mode).

        Returns AgentResult on success, None if structured output is unavailable.
        """
        model_config = agent_config.model
        if model_config.provider != "ollama":
            return None

        client = LLMClient(model_config)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("Running plan agent (structured output)...")
        logger.debug(f"  Model: {model_config.name} ({model_config.model_id})")
        logger.debug(f"  Endpoint: {model_config.endpoint}")

        plan_data = client.chat_structured(messages, schema=PLAN_OUTPUT_SCHEMA)

        if plan_data is None:
            return None  # Signal to caller to use fallback

        # Validate we got what we need
        dod_items = plan_data.get("definition_of_done", [])
        steps = plan_data.get("steps", [])
        summary = plan_data.get("plan_summary", "")

        if not dod_items:
            logger.warning("Structured plan had no DoD criteria, falling back")
            return None

        # Build DoD from structured data
        dod = DoD()
        for item in dod_items:
            desc = item.get("description", "")
            if desc:
                # verification_type and target_file are hints for post-build verification
                v_type = item.get("verification_type", "test")
                target = item.get("target_file", "")
                # Store verification hints in the criterion for post-build use
                dod.add(desc, method=v_type, command=None)
                # Stash target_file in the criterion if provided
                if target and dod.criteria:
                    dod.criteria[-1].target_file = target

        # Build readable plan text for the build agent
        plan_text = f"## Plan Summary\n{summary}\n\n"
        plan_text += "## Implementation Steps\n"
        for step in steps:
            files_str = ", ".join(step.get("files", []))
            plan_text += f"{step.get('step_number', '?')}. {step.get('description', '')}"
            if files_str:
                plan_text += f" (files: {files_str})"
            plan_text += "\n"

        plan_text += "\n## Definition of Done\n"
        for c in dod.criteria:
            v_type = c.verification_method or "test"
            target = getattr(c, 'target_file', '')
            target_str = f" (file: {target})" if target else ""
            plan_text += f"- [ ] {c.description} [verify: {v_type}]{target_str}\n"
        plan_text += "\nNote: Verification commands are generated AFTER build based on actual code.\n"

        logger.debug(f"  Structured plan: {len(dod.criteria)} DoD criteria, {len(steps)} steps")
        logger.info(f"  Plan agent produced {len(dod.criteria)} DoD criteria (all with verification commands)")

        return AgentResult(
            success=True,
            output=plan_text,
            plan=plan_text,
            dod=dod,
        )

    def _run_plan_unstructured(
        self,
        agent_config: AgentConfig,
        system_prompt: str,
        user_prompt: str,
        state: TaskState,
    ) -> AgentResult:
        """Fallback: run plan agent with free-form output and regex parsing."""
        # Adjust prompt for free-form output
        user_prompt += """

Output your plan in this EXACT format:

```dod
- [ ] Criterion 1: description (verify: `bash_command_here`)
- [ ] Criterion 2: description (verify: `bash_command_here`)
```
"""
        result = self._run_agent(agent_config, system_prompt, user_prompt, tools=TOOL_DEFINITIONS)

        if result.success:
            result.plan = result.output
            result.dod = self._parse_dod_from_output(result.output)

        return result

    def run_build(self, state: TaskState, memory_context: str = "") -> AgentResult:
        agent_config = self.config.get_agent("build")

        system_prompt = self._load_prompt(agent_config.system_prompt_file or "prompts/build.txt")
        if not system_prompt:
            system_prompt = """You are the BUILD agent. You implement the plan by writing actual code and running commands.

You have tools available: run_command, write_file, read_file, list_directory, edit_file.
USE THEM. Do not just describe what you would do — actually do it using the tools.

## PYTHON PATTERNS (ALWAYS FOLLOW)
- dataclass: Always use `@dataclass` decorator syntax. NEVER write `dataclass(ClassName):`.
- CLI main(): Accept optional `argv` and `storage` params for testability:
  `def main(argv=None, storage=None) -> int:` with `parser.parse_args(argv)`
- CLI/argparse testing: Use `unittest.mock.patch('sys.argv', ['prog', ...])` — never raw main().
- Test isolation: Always use `tempfile.TemporaryDirectory()` for file-based tests.
- Flask API testing: NEVER call route functions directly. ALWAYS use the test client:
  ```python
  from app import app
  class TestApp(unittest.TestCase):
      def setUp(self):
          app.config['TESTING'] = True
          self.client = app.test_client()
      def test_post(self):
          response = self.client.post('/endpoint', json={'key': 'value'})
          self.assertEqual(response.status_code, 201)
          data = response.get_json()
      def test_get(self):
          response = self.client.get('/endpoint')
          self.assertEqual(response.status_code, 200)
  ```
  NEVER do `from app import my_route; my_route()` — this causes RuntimeError (outside request context).
- IDs: Use `str(uuid.uuid4())[:8]` — always strings, never uuid.int.
- Import-safe: argparse/input()/sys.exit() must be inside `if __name__ == '__main__':` or a function.
- After writing a file, ALWAYS verify it compiles before moving on.
- datetime + JSON: datetime is NOT JSON-serializable. Use `default=str`:
  `json.dump(data, f, default=str)` — converts datetime (and anything else) to strings automatically.
  For loading back: `datetime.fromisoformat(d["created_at"])` to reconstruct.
  NEVER call json.dump without `default=str` when data may contain datetime objects.
- API contracts: Test files MUST call methods with the EXACT signature the source defines.
  Read the source file FIRST, check method signatures, then write tests that match.
  If source has `save_tasks(self)` (no args), tests must call `store.save_tasks()` not `store.save_tasks([task])`.
"""

        plan_context = state.current_plan or "No plan available — implement based on the goal."
        # Budget cap plan context
        plan_context = truncate_to_budget(
            plan_context, BUILD_PROMPT_BUDGET["plan_context"], "plan"
        )

        # Build the iteration history section
        history_section = ""
        if memory_context:
            # Budget cap memory context
            capped_memory = truncate_to_budget(
                memory_context, BUILD_PROMPT_BUDGET["memory_context"], "memory"
            )
            history_section = """
## ⚠️ PREVIOUS ITERATION HISTORY — READ THIS CAREFULLY
{capped_memory}
You MUST review the failures above and NOT repeat them.
If edit_file failed before, use read_file first to get exact content.
If a file was destroyed, restore it from the plan.
"""

        # Budget cap DoD
        dod_text = state.dod.to_markdown() if state.dod else "No DoD defined."
        dod_text = truncate_to_budget(
            dod_text, BUILD_PROMPT_BUDGET["dod_context"], "DoD"
        )

        # On retry iterations, switch to fix mode:
        # - Read existing files FIRST
        # - Make targeted edits instead of full rewrites
        # - Only touch files related to failures
        is_retry = state.iteration > 1
        if is_retry:
            fix_instructions = """
## 🔧 FIX MODE (Iteration {iteration} — this is a RETRY)

You are fixing code from a PREVIOUS iteration. Files already exist on disk.

**MANDATORY WORKFLOW FOR RETRIES:**
1. FIRST: Use `list_directory` to see what files exist
2. THEN: Use `read_file` on EACH file to see the current code
3. ONLY THEN: Fix the specific problems identified in the failure history
4. Use `edit_file` for targeted fixes when possible
5. Only use `write_file` if the entire file needs to be rewritten

**CRITICAL: Do NOT blindly rewrite files that were working.**
If a file was passing tests before, READ it first and only change what's broken.
If the error is a syntax issue (e.g., missing decorator, wrong indentation),
use `read_file` to see the exact current content, then use `edit_file` to fix
just the broken part.
""".format(iteration=state.iteration)

            # Inject concrete RCA edits directly into build prompt
            # This is the key insight from Spotify/Atla research:
            # precise, concrete edit commands bypass the "telephone game"
            # of RCA → Plan → Build where the action gets diluted
            rca_edits_section = self._build_rca_edits_section(state)
            if rca_edits_section:
                fix_instructions += rca_edits_section
        else:
            fix_instructions = ""

        user_prompt = """## Task
{state.goal}

## Task ID
{state.task_id}

## Iteration
{state.iteration}
{history_section}{fix_instructions}
## The Plan (FOLLOW THIS)
{plan_context}

## Definition of Done
{dod_text}

## CRITICAL INSTRUCTIONS

You MUST use the tools to actually implement the plan:
- Use `write_file` to create files
- Use `run_command` to execute shell commands
- Use `read_file` to check existing files
- Use `edit_file` to make surgical changes (ALWAYS read_file FIRST to get exact content)

## ENVIRONMENT RULES
- Use `python3 -m pytest` or `python3 -m unittest` to run tests (NOT bare `pytest`)
- A Python venv may already exist at `./venv` — if so, activate it: `source venv/bin/activate && pip install ...`
- If `pip install` fails with "externally-managed-environment", create a venv first:
  run_command: "python3 -m venv venv && source venv/bin/activate && pip install pytest"
- If git fails with "not a git repository", run: "git init" first
- If you try the same command 2+ times and it fails the same way, STOP and try a different approach
- On iteration 1: prefer `write_file` to create new files from scratch
- On retries: prefer `read_file` then `edit_file` to fix specific issues without destroying working code

## SELF-VERIFY (v0.7.2 — MANDATORY)
After writing or editing EACH Python file, verify it immediately:
1. run_command: "python3 -c \\"import py_compile; py_compile.compile('<filename>', doraise=True)\\""
2. If it fails, read the error and fix with edit_file. Repeat until it compiles.
3. For source modules, also run: "python3 -c \\"from <module> import *; print('OK')\\""
Do NOT move to the next file until the current one compiles and imports cleanly.

After implementing ALL files, commit your work:
- run_command: "git add -A && git commit -m 'feat: implement task {state.task_id}'"
- If git says "nothing to commit" — that is SUCCESS, not failure. Do NOT retry.

DO NOT just describe what you would do. ACTUALLY DO IT using the tools.
"""
        return self._run_agent(agent_config, system_prompt, user_prompt, tools=TOOL_DEFINITIONS)

    def run_test(self, state: TaskState) -> AgentResult:
        """
        Run verification of DoD criteria.

        v0.5.0 Strategy:
        1. Post-build verification (primary) — generates commands from actual
           code on disk, runs them deterministically. No LLM needed.
        2. If post-build returns None (no DoD criteria), fall back to LLM test agent.
        """
        agent_config = self.config.get_agent("test")

        # Primary: post-build verification — generates commands from actual workspace
        if state.dod and state.dod.criteria:
            direct_results = self._run_direct_verification(state)
            if direct_results is not None:
                return direct_results

        # Fallback: LLM-based test agent
        system_prompt = self._load_prompt(agent_config.system_prompt_file or "prompts/test.txt")
        if not system_prompt:
            system_prompt = "You are the TEST agent. Verify each DoD criterion by running actual commands."

        dod_criteria_list = ""
        if state.dod and state.dod.criteria:
            for idx, criterion in enumerate(state.dod.criteria):
                cmd = criterion.verification_command or "Manual inspection required"
                dod_criteria_list += f"\n### criterion-{idx}: {criterion.description}\nVerification command: `{cmd}`\n"

        user_prompt = """## VERIFY EACH CRITERION BY RUNNING COMMANDS

For EACH criterion below, use run_command to execute the verification command.

## DoD Criteria
{dod_criteria_list}

Output results as:
```test-results
{{
    "criterion-0": {{"passed": true, "evidence": "output"}},
    "criterion-1": {{"passed": false, "evidence": "error"}}
}}
```
"""
        result = self._run_agent(agent_config, system_prompt, user_prompt, tools=TOOL_DEFINITIONS)

        if result.success:
            result.test_report = self._parse_test_results(result.output)
            if state.dod and state.dod.criteria:
                test_results = result.test_report
                for idx, criterion in enumerate(state.dod.criteria):
                    cid = f"criterion-{idx}"
                    if cid in test_results and test_results[cid].get("passed"):
                        criterion.passed = True
                        criterion.evidence = test_results[cid].get("evidence", "")
                        logger.info(f"DoD {cid} PASSED: {criterion.description[:60]}")
                    else:
                        criterion.passed = False
                        logger.warning(f"DoD {cid} FAILED: {criterion.description[:60]}")

                passed_count = sum(1 for c in state.dod.criteria if c.passed)
                total_count = len(state.dod.criteria)
                logger.info(f"DoD final count: {passed_count}/{total_count} criteria passed")
                if passed_count < total_count:
                    result.success = False
                    result.error = f"{passed_count}/{total_count} DoD criteria passed"

        return result

    def run_build_single_file(
        self,
        state: TaskState,
        filename: str,
        step_info: dict,
        manifest: str,
        step_number: int,
        total_steps: int,
        temperature: Optional[float] = None,
        error_context: Optional[str] = None,
        kb_context: Optional[str] = None,
        agent_name: str = "build",
    ) -> AgentResult:
        """
        Build a single file with focused context.

        v0.6: Instead of giving the model all files at once, we give it ONE file
        to write plus a manifest of what already exists. This keeps the context
        small and focused — exactly what 70B models handle well.

        v0.6.2: error_context injects actual failure output from previous
        sampling attempts so the model can avoid the same mistakes.
        """
        agent_config = self.config.get_agent(agent_name)

        system_prompt = self._load_prompt(agent_config.system_prompt_file or "prompts/build.txt")
        if not system_prompt:
            system_prompt = """You are the BUILD agent. You implement code by writing files and running commands.

You have tools available: run_command, write_file, read_file, list_directory, edit_file.
USE THEM. Do not just describe what you would do — actually do it using the tools.

## PYTHON PATTERNS (ALWAYS FOLLOW)
- dataclass: Always use `@dataclass` decorator syntax. NEVER write `dataclass(ClassName):`.
- CLI main(): Accept optional `argv` and `storage` params for testability:
  `def main(argv=None, storage=None) -> int:` with `parser.parse_args(argv)`
- CLI/argparse testing: Use `unittest.mock.patch('sys.argv', ['prog', ...])` — never raw main().
- Test isolation: Always use `tempfile.TemporaryDirectory()` for file-based tests.
- Flask API testing: NEVER call route functions directly. ALWAYS use the test client:
  ```python
  from app import app
  class TestApp(unittest.TestCase):
      def setUp(self):
          app.config['TESTING'] = True
          self.client = app.test_client()
      def test_post(self):
          response = self.client.post('/endpoint', json={'key': 'value'})
          self.assertEqual(response.status_code, 201)
          data = response.get_json()
      def test_get(self):
          response = self.client.get('/endpoint')
          self.assertEqual(response.status_code, 200)
  ```
  NEVER do `from app import my_route; my_route()` — this causes RuntimeError (outside request context).
- IDs: Use `str(uuid.uuid4())[:8]` — always strings, never uuid.int.
- Imports between modules: If A imports B, B must have valid syntax or A fails too.
- After writing a file, ALWAYS verify it compiles before moving on.
- datetime + JSON: datetime is NOT JSON-serializable. Use `default=str`:
  `json.dump(data, f, default=str)` — converts datetime (and anything else) to strings automatically.
  For loading back: `datetime.fromisoformat(d["created_at"])` to reconstruct.
  NEVER call json.dump without `default=str` when data may contain datetime objects.
- API contracts: Test files MUST call methods with the EXACT signature the source defines.
  Read the source file FIRST, check method signatures, then write tests that match.
- Flask request.get_json() CONTRACT: request.get_json() returns PLAIN DICTS with STRING values.
  WRONG: task.priority.value  (AttributeError — it's already a string, NOT an enum instance!)
  WRONG: task["priority"].value  (same error — dict values from JSON are strings)
  RIGHT: task["priority"]  (use the string directly)
  RIGHT: task.get("priority", "medium")  (use dict .get() with default)
  NEVER use .value on anything from request.get_json() — JSON only has strings, numbers, bools, and null.
- Flask Application Factory: ALWAYS use the create_app() factory pattern:
  ```python
  def create_app(config=None):
      app = Flask(__name__)
      if config:
          app.config.update(config)
      # register routes...
      return app
  ```
  This ensures tests can create fresh app instances with `app = create_app({'TESTING': True})`.
"""

        is_test = step_info.get("is_test", False)
        tests_for = step_info.get("tests_for", "")
        depends_on = step_info.get("depends_on", [])

        # If this file tests a source module, read that module first for context
        source_context = ""
        test_template_section = ""
        contract_section = ""
        if is_test and tests_for:
            source_path = self.working_dir / tests_for
            # v0.9.7: tests_for may be "task_queue" (no .py) from per-source splitting
            if not source_path.exists() and not tests_for.endswith('.py'):
                source_path = self.working_dir / (tests_for + '.py')
            if source_path.exists():
                try:
                    source_content = source_path.read_text()[:3000]
                    source_context = """
## Source Module Being Tested: `{tests_for}`
```python
{source_content}
```
Read this carefully. Your tests must import from and test the actual classes/functions defined above.
"""
                    # v0.7.1: Generate a test template based on what's in the source
                    test_template_section = self._generate_test_template(
                        filename, tests_for, source_content
                    )

                    # v0.7.4: Extract method signatures as a compact contract reference.
                    # This is the #1 cause of test failures: model writes tests calling
                    # methods with wrong signatures (e.g., save_tasks([list]) when source
                    # defines save_tasks(self) with no args).
                    contract_section = self._extract_api_contract(tests_for, source_content)
                except Exception:
                    pass

        # v0.9.7: Multi-source injection for unified test files (e.g., test_miniqueue.py)
        # When tests_for is empty but the test file covers multiple modules,
        # inject ALL source files so the model sees every class/function signature.
        if is_test and not tests_for and not source_context:
            all_source_parts = []
            all_contract_parts = []
            for src_file in sorted(self.working_dir.glob("*.py")):
                if src_file.name.startswith("test_") or src_file.name.startswith("__"):
                    continue
                try:
                    src_content = src_file.read_text()[:2000]
                    all_source_parts.append(
                        f"### `{src_file.name}`\n```python\n{src_content}\n```"
                    )
                    contract = self._extract_api_contract(src_file.name, src_content)
                    if contract:
                        all_contract_parts.append(contract)
                except Exception:
                    continue
            if all_source_parts:
                source_context = (
                    "## Source Modules Being Tested (ALL files)\n\n"
                    + "\n\n".join(all_source_parts)
                    + "\n\nRead these carefully. Your tests MUST use the EXACT class names, "
                    "method names, and parameter names defined above."
                )
                contract_section = "\n".join(all_contract_parts)

        # v0.8.2: TDD Contract - inject locked test as contract for source files
        tdd_contract = ""
        if not is_test:
            test_filename = f"test_{filename}"
            test_path = self.working_dir / test_filename
            if test_path.exists():
                try:
                    test_content = test_path.read_text()[:3000]
                    tdd_contract = (
                        "\n## TDD CONTRACT - Your code MUST pass these tests\n"
                        "The following test file is LOCKED. You cannot change it.\n"
                        "Your implementation MUST satisfy these tests exactly.\n"
                        "Read the method calls carefully - match EXACT signatures.\n\n"
                        f"### `{test_filename}`\n"
                        f"```python\n{test_content}\n```\n\n"
                        "CRITICAL: Match the exact method signatures the tests use.\n"
                    )
                except Exception:
                    pass

        # If this module depends on others, show their interfaces
        deps_context = ""
        if depends_on and not is_test:
            deps_lines = []
            for dep in depends_on:
                dep_path = self.working_dir / dep
                if dep_path.exists():
                    try:
                        dep_content = dep_path.read_text()[:2000]
                        deps_lines.append(f"### `{dep}`\n```python\n{dep_content}\n```")
                    except Exception:
                        pass
            if deps_lines:
                deps_context = """
## Dependency Modules (you MUST import from these correctly)
{chr(10).join(deps_lines)}

Import from these modules as needed. Use the exact class/function names shown above.
"""

        # v0.6.2: Error context from previous failed sampling attempts
        error_section = ""
        if error_context:
            error_section = """
## ⚠️ PREVIOUS ATTEMPTS FAILED — AVOID THESE MISTAKES
Multiple previous attempts to write this test file ALL failed with errors.
Study these errors carefully and write DIFFERENT code that avoids them:

{error_context[:1500]}

Common fixes:
- If tests error on import: make sure you import from the correct module names
- If tests error on datetime: use `from datetime import datetime` not `import datetime`
- If argparse tests error: use `unittest.mock.patch('sys.argv', [...])` to mock CLI args
- If tests error on missing attribute: read the source module carefully for exact names
- If tests error with "Working outside of request context" or "Working outside of application context":
  you are calling Flask route functions directly. NEVER do that. Use `app.test_client()`:
  `client = app.test_client(); response = client.get('/endpoint'); data = response.get_json()`
"""

        # v0.8.0: Knowledge Base context (proactive patterns + docs)
        kb_section = ""
        if kb_context:
            kb_section = kb_context

        user_prompt = """{kb_section}
## Task
{state.goal}

## Step {step_number} of {total_steps}: Create `{filename}`

You are building ONE file: **{filename}**

{manifest}
{tdd_contract}
{deps_context}
{source_context}
{contract_section}
{test_template_section}
{error_section}

## Instructions for `{filename}`

{step_info.get('description', f'Create {filename}')}

**RULES:**
1. Create ONLY the file `{filename}` using `write_file`
2. Do NOT modify or recreate any other files
3. Make sure all imports reference modules that already exist (see manifest above)
4. If this is a test file, import from the actual source module and test real behavior
5. Do NOT write empty or placeholder tests — each test must assert something meaningful

## SELF-VERIFY (v0.7.2 — MANDATORY)
After writing `{filename}`, you MUST verify it before finishing:

1. run_command: "python3 -c \\"import py_compile; py_compile.compile('{filename}', doraise=True)\\""
2. run_command: "python3 -c \\"from {filename.replace('.py', '')} import *; print('OK')\\""
3. If EITHER fails, read the error, fix with `edit_file`, and verify again.
4. Do NOT declare done until both checks pass.

If this is a test file, also run:
5. run_command: "python3 -m pytest {filename} -v --tb=short 2>&1 | head -40"
6. If tests fail, read the error output and fix with `edit_file`.

Write the file now using `write_file`, then verify.
"""

        return self._run_agent(agent_config, system_prompt, user_prompt, tools=TOOL_DEFINITIONS, temperature=temperature)

    def run_build_single_file_plain(
        self,
        state: TaskState,
        filename: str,
        step_info: dict,
        manifest: str,
        step_number: int,
        total_steps: int,
        temperature: Optional[float] = None,
        error_context: Optional[str] = None,
        kb_context: Optional[str] = None,
        agent_name: str = "build",
    ) -> AgentResult:
        """
        v0.9.3: Plain-text build mode — eliminates JSON escaping failures.

        Instead of asking the model to use write_file tool calls (which wrap
        file content in JSON string values requiring perfect escaping of
        newlines, quotes, and triple-quotes), this method asks the model to
        output raw file content between <<<CONTENT>>> and <<<END>>> markers.

        Inspired by Aider's --edit-format whole, which found that local models
        produce far fewer syntax errors when outputting code directly vs.
        wrapping it in JSON. The model gets NO tools — single-shot generation.
        Verification is handled by the orchestrator after parsing the markers.
        """
        agent_config = self.config.get_agent(agent_name)

        is_test = step_info.get("is_test", False)
        tests_for = step_info.get("tests_for", "")
        depends_on = step_info.get("depends_on", [])

        # --- Build context sections (same as run_build_single_file) ---

        # Source context for test files
        source_context = ""
        contract_section = ""
        if is_test and tests_for:
            source_path = self.working_dir / tests_for
            # v0.9.7: tests_for may be "task_queue" (no .py) from per-source splitting
            if not source_path.exists() and not tests_for.endswith('.py'):
                source_path = self.working_dir / (tests_for + '.py')
            if source_path.exists():
                try:
                    source_content = source_path.read_text()[:3000]
                    source_context = """
## Source Module Being Tested: `{tests_for}`
```python
{source_content}
```
Read this carefully. Your tests must import from and test the actual classes/functions defined above.
"""
                    contract_section = self._extract_api_contract(tests_for, source_content)
                except Exception:
                    pass

        # v0.9.7: Multi-source injection fallback (same as above)
        if is_test and not tests_for and not source_context:
            all_source_parts = []
            all_contract_parts = []
            for src_file in sorted(self.working_dir.glob("*.py")):
                if src_file.name.startswith("test_") or src_file.name.startswith("__"):
                    continue
                try:
                    src_content = src_file.read_text()[:2000]
                    all_source_parts.append(
                        f"### `{src_file.name}`\n```python\n{src_content}\n```"
                    )
                    contract = self._extract_api_contract(src_file.name, src_content)
                    if contract:
                        all_contract_parts.append(contract)
                except Exception:
                    continue
            if all_source_parts:
                source_context = (
                    "## Source Modules Being Tested (ALL files)\n\n"
                    + "\n\n".join(all_source_parts)
                    + "\n\nRead these carefully. Your tests MUST use the EXACT class names, "
                    "method names, and parameter names defined above."
                )
                contract_section = "\n".join(all_contract_parts)

        # TDD contract for source files
        tdd_contract = ""
        if not is_test:
            test_filename = f"test_{filename}"
            test_path = self.working_dir / test_filename
            if test_path.exists():
                try:
                    test_content = test_path.read_text()[:3000]
                    tdd_contract = (
                        "\n## TDD CONTRACT - Your code MUST pass these tests\n"
                        "The following test file is LOCKED. You cannot change it.\n"
                        "Your implementation MUST satisfy these tests exactly.\n"
                        "Read the method calls carefully - match EXACT signatures.\n\n"
                        f"### `{test_filename}`\n"
                        f"```python\n{test_content}\n```\n\n"
                        "CRITICAL: Match the exact method signatures the tests use.\n"
                    )
                except Exception:
                    pass

        # Dependency context
        deps_context = ""
        if depends_on and not is_test:
            deps_lines = []
            for dep in depends_on:
                dep_path = self.working_dir / dep
                if dep_path.exists():
                    try:
                        dep_content = dep_path.read_text()[:2000]
                        deps_lines.append(f"### `{dep}`\n```python\n{dep_content}\n```")
                    except Exception:
                        pass
            if deps_lines:
                deps_context = """
## Dependency Modules (you MUST import from these correctly)
{chr(10).join(deps_lines)}

Import from these modules as needed. Use the exact class/function names shown above.
"""

        # Error context from previous attempts
        error_section = ""
        if error_context:
            error_section = """
## PREVIOUS ATTEMPTS FAILED — AVOID THESE MISTAKES
{error_context[:1500]}
"""

        # KB context
        kb_section = ""
        if kb_context:
            kb_section = kb_context

        # --- System prompt: raw output, no tools ---
        system_prompt = (
            "You are the BUILD agent. You write complete Python files.\n\n"
            "## OUTPUT FORMAT\n"
            "Output ONLY the complete file content between these exact markers:\n\n"
            "<<<CONTENT>>>\n"
            "(your complete Python file here)\n"
            "<<<END>>>\n\n"
            "Do NOT use any tool calls or JSON. Do NOT add explanation outside the markers.\n"
            "Just output the markers with the complete, working Python file between them.\n\n"
            "## PYTHON PATTERNS (ALWAYS FOLLOW)\n"
            "- dataclass: Always use `@dataclass` decorator syntax.\n"
            "- Multi-line SQL: ALWAYS use triple-quoted strings for SQL:\n"
            '  cursor.execute("""\n'
            "      CREATE TABLE IF NOT EXISTS projects (\n"
            "          id TEXT PRIMARY KEY\n"
            "      )\n"
            '  """)\n'
            '  NEVER use double-quoted strings with \\n for SQL.\n'
            "- IDs: Use `str(uuid.uuid4())[:8]` — always strings.\n"
            "- datetime + JSON: Use `default=str` in json.dump.\n"
            "- Flask: Use create_app() factory pattern.\n"
            "- Flask testing: Use app.test_client(), never call routes directly.\n"
            "- Flask request.get_json() returns plain dicts — never use .value on JSON data.\n"
            "- Imports between modules: If A imports B, B must have valid syntax or A fails too.\n"
        )

        user_prompt = """{kb_section}
## Task
{state.goal}

## Step {step_number} of {total_steps}: Create `{filename}`

You are writing ONE file: **{filename}**

{manifest}
{tdd_contract}
{deps_context}
{source_context}
{contract_section}
{error_section}

## Instructions for `{filename}`

{step_info.get('description', f'Create {filename}')}

Now write the complete `{filename}` between the markers:

<<<CONTENT>>>
"""

        # Call WITHOUT tools — single-shot generation, no agentic loop
        return self._run_agent(
            agent_config, system_prompt, user_prompt,
            tools=None,
            temperature=temperature
        )


    # ──────────────────────────────────────────────────────────────
    # v0.9.8: Edit-based repair — surgical fixes instead of regen
    # ──────────────────────────────────────────────────────────────

    def run_edit_repair(
        self,
        state: TaskState,
        filename: str,
        current_content: str,
        test_output: str,
        source_context: str = "",
        contract_section: str = "",
        temperature: float = 0.2,
    ) -> AgentResult:
        """
        v0.9.8: Ask the model to produce SEARCH/REPLACE edits to fix test failures.

        Instead of regenerating the entire file (which throws away passing code),
        this gives the model the current file + test output and asks for targeted
        edits. Research shows this has 2x higher fix rate than full regeneration.
        """
        agent_config = self.config.get_agent("build")

        system_prompt = self._load_prompt("prompts/edit_repair.txt")
        if not system_prompt:
            system_prompt = (
                "You are the REPAIR agent. Fix specific bugs using SEARCH/REPLACE blocks.\n"
                "Output format: <<<SEARCH>>>\nold code\n<<<REPLACE>>>\nnew code\n<<<END>>>\n"
                "Fix ONLY failing tests. Do NOT modify passing code."
            )

        # v0.9.9: Tag content with hashlines for stable line referencing
        tagged_content = self.hashline_tag(current_content)

        # v0.9.9: Include failure feedback from previous edit attempts if available
        feedback_section = ""
        if state.edit_feedback:
            feedback_section = "## Previous Edit Failures\n"
            for fb in state.edit_feedback[-3:]:  # Last 3 failures
                feedback_section += f"\n{fb}\n"
            feedback_section += "\nFix these matching issues in your new SEARCH blocks.\n"

        user_prompt = """## Task
{state.goal}

## Current File: `{filename}` (with line:hash tags)
Each line is tagged as `LINE:HASH| content` for reference.
Your SEARCH blocks should match the CONTENT part (after the `| `), not the tags.

```
{tagged_content}
```

{source_context}
{contract_section}

{feedback_section}

## Test Failures to Fix
```
{test_output[-2000:]}
```

## Instructions
Produce SEARCH/REPLACE edits to fix the failing tests.
- SEARCH blocks must match the actual file content (without line:hash tags)
- You can reference line numbers/hashes to identify locations, but the SEARCH
  text itself must be the raw code as it appears in the file
- Fix ONLY the specific failures shown above
- Do NOT rewrite the entire file
- Keep changes minimal and surgical
"""

        return self._run_agent(
            agent_config, system_prompt, user_prompt,
            tools=None, temperature=temperature
        )

    # Alias for orchestrator compatibility
    run_edit_repair_structured = run_edit_repair

    @staticmethod
    def parse_search_replace(model_output: str) -> list:
        """
        v0.9.8: Parse SEARCH/REPLACE blocks from model output.

        Returns list of (search_text, replace_text) tuples.
        """
        blocks = []
        parts = model_output.split("<<<SEARCH>>>")

        for part in parts[1:]:  # Skip everything before first SEARCH
            if "<<<REPLACE>>>" not in part:
                continue

            search_part, rest = part.split("<<<REPLACE>>>", 1)

            if "<<<END>>>" in rest:
                replace_part = rest.split("<<<END>>>")[0]
            else:
                # No END marker — take everything up to next SEARCH or end
                replace_part = rest.split("<<<SEARCH>>>")[0] if "<<<SEARCH>>>" in rest else rest

            search_text = search_part.strip('\n\r')
            replace_text = replace_part.strip('\n\r')

            if search_text:  # Don't add empty searches
                blocks.append((search_text, replace_text))

        return blocks

    @staticmethod
    def apply_search_replace(filepath, edits: list) -> tuple:
        """
        v0.9.9: Apply SEARCH/REPLACE edits with 5-layer fuzzy matching.

        Research: Aider's layered matching reduces edit failures by 9X.
        RooCode uses middle-out Levenshtein distance for robust matching.

        Matching layers (tried in order):
        1. Exact match
        2. Trailing-whitespace-stripped match
        3. difflib.SequenceMatcher fuzzy match (≥0.85 similarity)
        4. Content-stripped match (ignore all leading whitespace)
        5. Partial-line match (for single-line edits)

        Returns (num_applied, failure_feedback) where failure_feedback is
        a list of strings describing why failed edits didn't match (for
        sending back to the model on retry).
        """
        import difflib
        from pathlib import Path
        filepath = Path(filepath)

        try:
            content = filepath.read_text()
            original_content = content  # v0.9.9a: save for rollback on compile failure
        except Exception:
            return 0, ["Could not read file"]

        applied = 0
        feedback = []

        def _indent_replace(replace_text, orig_first_line):
            """Apply replacement preserving RELATIVE indentation.

            v0.9.9b fix: old code applied a flat indent from the first matched
            line to every replacement line, destroying nested indentation
            (if/else bodies, inner functions, etc.). Now we compute the indent
            delta between the original location and the replacement's own base
            indent, then shift every line by that delta.
            """
            orig_indent = len(orig_first_line) - len(orig_first_line.lstrip())
            rep_lines = replace_text.split('\n')
            # Find base indent of replacement (first non-empty line)
            rep_base = 0
            for rl in rep_lines:
                if rl.strip():
                    rep_base = len(rl) - len(rl.lstrip())
                    break
            delta = orig_indent - rep_base
            adjusted = []
            for rl in rep_lines:
                if not rl.strip():
                    adjusted.append('')
                else:
                    cur = len(rl) - len(rl.lstrip())
                    new_indent = max(0, cur + delta)
                    adjusted.append(' ' * new_indent + rl.lstrip())
            return adjusted

        for search_text, replace_text in edits:
            matched = False
            best_ratio = 0.0  # v0.9.9b: must init here — Layer 3 may be skipped

            # === Layer 1: Exact match ===
            if search_text in content:
                content = content.replace(search_text, replace_text, 1)
                applied += 1
                continue

            # === Layer 2: Strip trailing whitespace per line ===
            # v0.9.9b: Match in stripped space but replace in ORIGINAL content
            # by line position. Old code replaced in rstripped content, destroying
            # all trailing whitespace in the entire file.
            search_lines_rs = [l.rstrip() for l in search_text.split('\n')]
            content_lines_rs = [l.rstrip() for l in content.split('\n')]
            search_rs_joined = '\n'.join(search_lines_rs)
            content_rs_joined = '\n'.join(content_lines_rs)
            if search_rs_joined in content_rs_joined:
                # Find start line by counting newlines before match
                match_pos = content_rs_joined.index(search_rs_joined)
                start_line = content_rs_joined[:match_pos].count('\n')
                num_lines = len(search_lines_rs)
                # Replace those lines in original content (preserving other lines)
                orig_lines = content.split('\n')
                replace_lines = replace_text.split('\n')
                orig_lines[start_line:start_line + num_lines] = replace_lines
                content = '\n'.join(orig_lines)
                applied += 1
                continue

            # === Layer 3: difflib fuzzy match (Aider/RooCode approach) ===
            search_lines = search_text.split('\n')
            content_lines = content.split('\n')
            best_ratio = 0.0
            best_start = -1
            window_size = len(search_lines)

            if window_size >= 2 and len(content_lines) >= window_size:
                for i in range(len(content_lines) - window_size + 1):
                    candidate_block = '\n'.join(content_lines[i:i + window_size])
                    ratio = difflib.SequenceMatcher(
                        None, search_text, candidate_block
                    ).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_start = i

                # Accept if similarity ≥ 0.85 (v0.9.9a: raised from 0.7 — too aggressive)
                if best_ratio >= 0.85 and best_start >= 0:
                    # v0.9.9b: Preserve relative indentation (not flat)
                    indented_replace = _indent_replace(replace_text, content_lines[best_start])

                    content_lines[best_start:best_start + window_size] = indented_replace
                    content = '\n'.join(content_lines)
                    applied += 1
                    continue

            # === Layer 4: Content-stripped match (ignore ALL whitespace) ===
            # v0.9.9a: require 3+ non-blank lines — too easy to false-match short blocks
            # v0.9.9b: keep blank lines for positional matching (old code filtered them
            # from search but not content, so a search with blank lines could never match)
            search_all_stripped = [l.strip() for l in search_text.split('\n')]
            non_blank_count = sum(1 for l in search_all_stripped if l)
            if non_blank_count >= 3:
                content_lines = content.split('\n')
                for i in range(len(content_lines) - len(search_all_stripped) + 1):
                    candidate = [content_lines[i + j].strip()
                                 for j in range(len(search_all_stripped))]
                    if candidate == search_all_stripped:
                        # v0.9.9b: relative indent preservation
                        indented_replace = _indent_replace(replace_text, content_lines[i])

                        content_lines[i:i + len(search_all_stripped)] = indented_replace
                        content = '\n'.join(content_lines)
                        applied += 1
                        matched = True
                        break

                if matched:
                    continue

            # === Layer 5: Partial-line / single-line match ===
            search_clean = search_text.strip()
            if '\n' not in search_clean and len(search_clean) >= 15:
                # Single line edit — find closest match
                content_lines = content.split('\n')
                best_line_ratio = 0.0
                best_line_idx = -1
                for i, cl in enumerate(content_lines):
                    ratio = difflib.SequenceMatcher(
                        None, search_clean, cl.strip()
                    ).ratio()
                    if ratio > best_line_ratio:
                        best_line_ratio = ratio
                        best_line_idx = i

                if best_line_ratio >= 0.85 and best_line_idx >= 0:
                    # v0.9.9b: relative indent for single-line replacement
                    indented_lines = _indent_replace(replace_text, content_lines[best_line_idx])
                    content_lines[best_line_idx:best_line_idx + 1] = indented_lines
                    content = '\n'.join(content_lines)
                    applied += 1
                    continue

            # === All layers failed — generate feedback for model ===
            # (Like Aider: show what the file actually contains near the search)
            preview_lines = search_text.split('\n')[:3]
            preview_stripped = [l.strip() for l in preview_lines if l.strip()]

            if preview_stripped:
                content_lines = content.split('\n')
                # Find most similar region using first line
                best_sim: float = 0.0
                best_idx: int = 0
                for i, cl in enumerate(content_lines):
                    r = difflib.SequenceMatcher(None, preview_stripped[0], cl.strip()).ratio()
                    if r > best_sim:
                        best_sim = r
                        best_idx = i

                # Show nearby lines
                start = max(0, best_idx - 2)
                end = min(len(content_lines), best_idx + len(preview_lines) + 2)
                nearby = '\n'.join(f"  {j+1}: {content_lines[j]}" for j in range(start, end))
                fb = (
                    f"SEARCH block failed to match (best similarity: {best_ratio:.0%}).\n"
                    "Your SEARCH started with:\n"
                    f"  {preview_stripped[0]}\n"
                    f"Nearest actual lines in file (lines {start+1}-{end}):\n{nearby}\n"
                    "The SEARCH must match the file EXACTLY including whitespace."
                )
                feedback.append(fb)

        if applied > 0:
            # v0.9.9a: Compile check before writing — don't break working files
            filepath.write_text(content)
            if str(filepath).endswith('.py'):
                import subprocess
                try:
                    result = subprocess.run(
                        ['python3', '-c', f'import py_compile; py_compile.compile("{filepath}", doraise=True)'],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode != 0:
                        # Edits broke syntax — rollback
                        filepath.write_text(original_content)
                        feedback.append(
                            f"ROLLBACK: {applied} edits were applied but broke Python syntax. "
                            f"Error: {result.stderr.strip()[:200]}. "
                            "All edits reverted. Try smaller, more targeted changes."
                        )
                        return 0, feedback
                except Exception:
                    pass  # If compile check itself fails, keep the edits

        return applied, feedback

    @staticmethod
    def hashline_tag(content: str) -> str:
        """
        v0.9.9: Tag each line with a short content hash for stable referencing.

        Research: hashline approach (can1357, 2026) improved weakest models by
        10X by giving them stable line identifiers instead of requiring perfect
        text recall. When the model edits, it can reference line tags rather
        than reproducing exact content.

        Format: "  7:a3| def hello():"
                ^^^^  ^^^^^^^^^^^^^^^^
                line:hash  original content

        Hash is first 2 chars of hex digest of stripped line content.
        """
        import hashlib
        lines = content.split('\n')
        tagged = []
        for i, line in enumerate(lines, 1):
            h = hashlib.md5(line.strip().encode()).hexdigest()[:2]
            tagged.append(f"{i:3d}:{h}| {line}")
        return '\n'.join(tagged)

    @staticmethod
    def strip_hashline_tags(tagged_content: str) -> str:
        """
        v0.9.9: Remove hashline tags from content, returning plain code.

        Strips the "  7:a3| " prefix from each line.
        """
        import re
        lines = tagged_content.split('\n')
        stripped = []
        for line in lines:
            # Match pattern: optional spaces, digits, colon, 2 hex chars, pipe, space
            m = re.match(r'^\s*\d+:[0-9a-f]{2}\| (.*)$', line)
            if m:
                stripped.append(m.group(1))
            else:
                stripped.append(line)
        return '\n'.join(stripped)


    def _extract_api_contract(self, source_filename: str, source_content: str) -> str:
        """
        v0.7.4: Extract method signatures from source as a compact API contract.
        v0.9.7: Also extracts @dataclass field names as constructor parameters.

        The #1 cause of test file failures is API mismatch: the model writes
        tests calling methods with wrong arguments (e.g., `save_tasks([list])`
        when the source defines `save_tasks(self)` with no extra args).

        This creates a compact, impossible-to-miss contract block showing
        every class and its method signatures, so the model knows exactly
        what it can call and with what arguments.
        """
        import re

        source_module = source_filename.replace('.py', '')
        lines = []

        # Extract class definitions with their methods
        class_pattern = re.compile(r'^class\s+(\w+)(?:\(.*?\))?:', re.MULTILINE)
        method_pattern = re.compile(r'^\s+def\s+(\w+)\((.*?)\)', re.MULTILINE)

        # v0.9.7: Detect @dataclass classes for constructor extraction
        # @dataclass auto-generates __init__ from field annotations, so there's
        # no explicit def __init__ to regex-match. We must extract field names
        # as constructor parameters directly from class body annotations.
        dataclass_classes = set()
        for m in re.finditer(r'@dataclass(?:\(.*?\))?\s*\nclass\s+(\w+)', source_content):
            dataclass_classes.add(m.group(1))

        classes = list(class_pattern.finditer(source_content))

        for i, class_match in enumerate(classes):
            class_name = class_match.group(1)
            # Find the end of this class (start of next class or end of file)
            class_start = class_match.end()
            class_end = classes[i + 1].start() if i + 1 < len(classes) else len(source_content)
            class_body = source_content[class_start:class_end]

            # v0.9.7: For @dataclass classes, extract field names as constructor params
            if class_name in dataclass_classes:
                dc_fields = []
                for field_m in re.finditer(
                    r'^\s+(\w+)\s*:\s*([^=\n]+?)(?:\s*=\s*(.+))?$',
                    class_body, re.MULTILINE
                ):
                    field_name = field_m.group(1)
                    field_type = field_m.group(2).strip()
                    has_default = field_m.group(3) is not None
                    if field_name.startswith('_'):
                        continue
                    dc_fields.append((field_name, field_type, has_default))
                if dc_fields:
                    param_list = ', '.join(
                        f"{name}=..." if has_default else name
                        for name, ftype, has_default in dc_fields
                    )
                    lines.append(f"**{class_name} (@dataclass — constructor uses THESE field names):**")
                    lines.append(f"  CONSTRUCTOR: `{class_name}({param_list})`")
                    for name, ftype, has_default in dc_fields:
                        req = "optional" if has_default else "REQUIRED"
                        lines.append(f"  - {name}: {ftype}  [{req}]")
                    field_names = ', '.join(name for name, _, _ in dc_fields)
                    lines.append("  ⚠️ Do NOT use 'id', 'payload', 'name', or any other param names.")
                    lines.append(f"     Use EXACTLY: {field_names}")

            methods = []
            for m in method_pattern.finditer(class_body):
                method_name = m.group(1)
                params = m.group(2).strip()
                # Clean up params: remove type annotations for compact display
                clean_params = []
                for p in params.split(','):
                    p = p.strip()
                    if not p:
                        continue
                    # Strip type annotations and defaults for signature display
                    name = p.split(':')[0].split('=')[0].strip()
                    has_default = '=' in p
                    if name == 'self':
                        continue
                    clean_params.append(f"{name}{'=...' if has_default else ''}")

                sig = ', '.join(clean_params) if clean_params else ''
                methods.append(f"  - {method_name}({sig})")

            if methods:
                if class_name not in dataclass_classes:
                    lines.append(f"**{class_name}:**")
                lines.extend(methods)

        # Extract standalone functions
        standalone_funcs = []
        for m in re.finditer(r'^def\s+(\w+)\((.*?)\)', source_content, re.MULTILINE):
            func_name = m.group(1)
            params = m.group(2).strip()
            clean_params = []
            for p in params.split(','):
                p = p.strip()
                if not p:
                    continue
                name = p.split(':')[0].split('=')[0].strip()
                has_default = '=' in p
                clean_params.append(f"{name}{'=...' if has_default else ''}")
            sig = ', '.join(clean_params) if clean_params else ''
            standalone_funcs.append(f"  - {func_name}({sig})")

        if standalone_funcs:
            lines.append("**Standalone functions:**")
            lines.extend(standalone_funcs)

        if not lines:
            return ""

        contract = '\n'.join(lines)
        return """
## ⚠️ API CONTRACT (from `{source_filename}` — MUST MATCH EXACTLY)
Your tests MUST call these methods with EXACTLY these signatures.
Do NOT invent extra parameters or change the calling convention.

{contract}

If a method takes no args (besides self), call it with NO args: `obj.method()`
If a method takes args, call it with those args: `obj.method(arg1, arg2)`
"""

    def _generate_test_template(self, test_filename: str, source_filename: str, source_content: str) -> str:
        """
        v0.7.1: Generate a deterministic test template from the actual source module.

        Instead of asking the LLM to invent tests from scratch (high entropy,
        frequent NameError/ImportError), we analyze the source code and generate
        a known-good scaffold with correct imports and class names. The LLM only
        needs to fill in the assertion logic.

        Returns a prompt section with the template, or empty string if we can't
        generate one.
        """
        import re

        source_module = source_filename.replace('.py', '')

        # Extract classes and their __init__ signatures
        classes = []
        for match in re.finditer(
            r'^class\s+(\w+)(?:\(.*?\))?:\s*\n((?:\s+.+\n)*)',
            source_content, re.MULTILINE
        ):
            class_name = match.group(1)
            class_body = match.group(2)

            # Check if it's a dataclass (look backwards for @dataclass decorator)
            class_pos = match.start()
            preceding = source_content[:class_pos].rstrip()
            is_dataclass = preceding.endswith('@dataclass')

            # Extract __init__ params or dataclass fields
            init_params = []
            if is_dataclass:
                for field_match in re.finditer(r'^\s+(\w+):\s*(\w+)', class_body, re.MULTILINE):
                    field_name = field_match.group(1)
                    field_type = field_match.group(2)
                    init_params.append((field_name, field_type))
            else:
                init_match = re.search(r'def __init__\(self,?\s*(.*?)\):', class_body)
                if init_match:
                    params_str = init_match.group(1)
                    for param in params_str.split(','):
                        param = param.strip()
                        if param and '=' not in param:
                            parts = param.split(':')
                            name = parts[0].strip()
                            ptype = parts[1].strip() if len(parts) > 1 else 'str'
                            init_params.append((name, ptype))

            classes.append({
                'name': class_name,
                'is_dataclass': is_dataclass,
                'init_params': init_params,
            })

        # Extract standalone functions
        functions = []
        for match in re.finditer(r'^def\s+(\w+)\((.*?)\)', source_content, re.MULTILINE):
            func_name = match.group(1)
            if not func_name.startswith('_'):
                functions.append(func_name)

        if not classes and not functions:
            return ""

        # Detect module type
        is_storage = any(kw in source_module.lower() for kw in ['storage', 'store', 'repo', 'db'])
        is_flask = 'Flask(' in source_content or 'from flask import' in source_content
        is_cli = not is_flask and any(kw in source_module.lower() for kw in ['cli', 'main', 'app'])

        # Build import line
        importable_names = [c['name'] for c in classes] + functions
        import_line = f"from {source_module} import {', '.join(importable_names)}"  # type: ignore[arg-type]

        # For Flask apps, import the app object — NOT route functions
        if is_flask:
            import_line = f"from {source_module} import app"

        # Build the template
        template_lines = [
            "import unittest",
            "import tempfile",
            "from pathlib import Path",
            "from datetime import datetime",
            "",
            import_line,
        ]

        # Add storage-specific imports
        if is_storage:
            template_lines.insert(2, "import json")

        # Add Flask-specific imports and setup
        if is_flask:
            template_lines.insert(2, "import json")

        # Add CLI-specific imports
        if is_cli:
            template_lines.insert(2, "import io")
            template_lines.insert(3, "from contextlib import redirect_stdout")
            template_lines.insert(4, "from unittest.mock import patch")
            # Also import storage/models if they exist
            for dep_module in ['storage', 'models']:
                dep_path = self.working_dir / f"{dep_module}.py"
                if dep_path.exists():
                    dep_content = dep_path.read_text()
                    dep_classes = re.findall(r'^class\s+(\w+)', dep_content, re.MULTILINE)
                    if dep_classes:
                        template_lines.append(f"from {dep_module} import {', '.join(dep_classes)}")

        template_lines.append("")
        template_lines.append("")

        # Generate test class
        class_name = f"Test{source_module.title().replace('_', '')}"
        template_lines.append(f"class {class_name}(unittest.TestCase):")

        if is_storage and classes:
            # v0.7.4: Storage test template — API-aware using actual method signatures
            storage_class = str(classes[0]['name'])

            # Extract method signatures to generate correct test calls
            import re as re_mod
            class_body_match = re_mod.search(
                rf'^class\s+{re_mod.escape(storage_class)}.*?(?=\nclass\s|\Z)',  # type: ignore[type-var]
                source_content, re_mod.MULTILINE | re_mod.DOTALL
            )
            class_body = class_body_match.group(0) if class_body_match else ""

            # Detect add_task/save_tasks signatures
            has_add_task = bool(re_mod.search(r'def\s+add_task\s*\(\s*self\s*,\s*\w+', class_body))
            save_takes_args = bool(re_mod.search(r'def\s+save_tasks\s*\(\s*self\s*,\s*\w+', class_body))

            # Find import for the model class
            model_import = ""
            for dep_module in ['models', 'model', 'task']:
                dep_path = self.working_dir / f"{dep_module}.py"
                if dep_path.exists():
                    dep_content = dep_path.read_text()
                    dep_classes = re_mod.findall(r'^class\s+(\w+)', dep_content, re_mod.MULTILINE)
                    if dep_classes:
                        model_import = f"from {dep_module} import {', '.join(dep_classes)}"
                        break

            if model_import:
                template_lines.append(model_import)
                template_lines.append("")

            template_lines.extend([
                "",
                "    def test_save_and_load_roundtrip(self):",
                "        with tempfile.TemporaryDirectory() as d:",
                "            path = Path(d) / 'data.json'",
                f"            store = {storage_class}(path)",
            ])

            if has_add_task:
                # Use add_task method (manages internal state)
                template_lines.extend([
                    "            # Use the add_task method to add items (it calls save internally)",
                    "            # store.add_task(Task(id='1', title='Test', status='pending', created_at=datetime.now().isoformat()))",
                    "            # Then reload and verify:",
                    f"            # store2 = {storage_class}(path)",
                    "            # self.assertEqual(len(store2.tasks), 1)",
                    "            pass  # Replace with actual test logic",
                ])
            elif save_takes_args:
                # save_tasks takes a list argument
                template_lines.extend([
                    "            # save_tasks takes a list argument:",
                    "            # store.save_tasks([task1, task2])",
                    "            pass  # Replace with actual test logic",
                ])
            else:
                # save_tasks uses self.tasks (no args)
                template_lines.extend([
                    "            # save_tasks() uses internal self.tasks — NO arguments:",
                    "            # store.tasks.append(task)",
                    "            # store.save_tasks()  # <-- NO args!",
                    "            pass  # Replace with actual test logic",
                ])
        elif is_flask:
            # v0.7.4: Flask test template — uses test_client, NEVER calls routes directly
            # Extract routes from the source to generate targeted tests
            import re as re_mod
            # Handle both single and double quotes in route decorators
            routes = re_mod.findall(
                r"""@app\.route\(['"](.*?)['"],\s*methods=\[([^\]]+)\]\)""",
                source_content
            )

            template_lines.extend([
                "",
                "    def setUp(self):",
                "        app.config['TESTING'] = True",
                "        self.client = app.test_client()",
                "        # Set up test database if needed:",
                "        # with tempfile.TemporaryDirectory() as d:",
                "        #     app.config['DATABASE_PATH'] = Path(d) / 'test.db'",
            ])

            if routes:
                for route_path, methods_str in routes[:4]:  # Cap at 4 routes
                    methods = [m.strip().strip("'\"") for m in methods_str.split(',')]
                    for method in methods:
                        method_lower = method.strip().lower()
                        # Generate test name from route
                        test_name = route_path.strip('/').replace('/', '_').replace('<', '').replace('>', '').replace('int:', '')
                        if not test_name:
                            test_name = 'root'

                        if method_lower == 'post':
                            template_lines.extend([
                                "",
                                f"    def test_{method_lower}_{test_name}(self):",
                                f"        response = self.client.post('{route_path}', json={{",
                                "            # Add required fields here",
                                "        }})",
                                "        self.assertIn(response.status_code, [200, 201])",
                                "        data = response.get_json()",
                                "        self.assertIsNotNone(data)",
                            ])
                        elif method_lower == 'get':
                            template_lines.extend([
                                "",
                                f"    def test_{method_lower}_{test_name}(self):",
                                f"        response = self.client.get('{route_path}')",
                                "        self.assertEqual(response.status_code, 200)",
                                "        data = response.get_json()",
                                "        self.assertIsNotNone(data)",
                            ])
                        elif method_lower == 'delete':
                            template_lines.extend([
                                "",
                                f"    def test_{method_lower}_{test_name}(self):",
                                "        # First create an item, then delete it",
                                f"        response = self.client.delete('{route_path}')",
                                "        self.assertIn(response.status_code, [200, 204])",
                            ])
            else:
                # No routes found, give generic Flask test scaffold
                template_lines.extend([
                    "",
                    "    def test_health(self):",
                    "        response = self.client.get('/')",
                    "        self.assertEqual(response.status_code, 200)",
                ])

        elif is_cli and 'main' in functions:
            # CLI test template with injectable main()
            template_lines.extend([
                "",
                "    def test_cli_add(self):",
                "        with tempfile.TemporaryDirectory() as d:",
                "            path = Path(d) / 'data.json'",
                "            # TODO: Call main() with injectable argv and storage",
                "            # main(argv=['add', 'Test Item'], storage=TempStorage(path))",
                "            pass  # Replace with actual test call",
            ])
        else:
            # Generic test template
            for cls in classes[:2]:
                cls_name: str = str(cls['name'])
                init_params: list[tuple[str, str]] = cls.get('init_params', []) or []  # type: ignore[assignment]
                template_lines.extend([
                    "",
                    f"    def test_{cls_name.lower()}_creation(self):",
                ])
                if init_params:
                    param_strs = []
                    for name, ptype in init_params[:4]:
                        if ptype in ('str', 'String'):
                            param_strs.append(f"{name}='test'")
                        elif ptype in ('int', 'Integer', 'float'):
                            param_strs.append(f"{name}=1")
                        elif ptype == 'datetime':
                            param_strs.append(f"{name}=datetime.now()")
                        else:
                            param_strs.append(f"{name}='test'")
                    params = ', '.join(param_strs)
                    template_lines.append(f"        obj = {cls_name}({params})")
                    template_lines.append("        self.assertIsNotNone(obj)")
                else:
                    template_lines.append(f"        obj = {cls_name}()")
                    template_lines.append("        self.assertIsNotNone(obj)")

            for func in functions[:3]:
                template_lines.extend([
                    "",
                    f"    def test_{func}(self):",
                    f"        # TODO: Call {func}() with appropriate args",
                    "        pass  # Replace with actual test",
                ])

        template_lines.extend([
            "",
            "",
            "if __name__ == '__main__':",
            "    unittest.main()",
        ])

        template_code = '\n'.join(template_lines)

        return """
## 📋 TEST TEMPLATE (START FROM THIS — DO NOT IGNORE)
The following template has been generated from the actual source module `{source_filename}`.
It has correct imports and class names. **Start from this template** and flesh out the
test methods with real assertions. Do NOT rewrite the imports or class instantiation
from scratch — they are already correct.

```python
{template_code}
```

**RULES FOR USING THIS TEMPLATE:**
1. Keep ALL the import lines exactly as shown (they match the actual source)
2. Replace `pass` / TODO comments with real test logic
3. You may add more test methods but do NOT remove the existing imports
4. Use the exact class and function names from the template (they come from the real source code)
"""

    def _build_rca_edits_section(self, state: TaskState) -> str:
        """
        Build a mandatory edits section from RCA concrete_edits.

        Inspired by Spotify's verification loop architecture and Atla's
        actor-critic research: concrete, precise critiques at failure points
        boost agent performance by ~30% vs. vague natural language instructions.

        Instead of: RCA → Plan Agent → "fix imports" → Build Agent (ignores it)
        We do:      RCA → Build Agent receives: "MANDATORY: edit cli.py, add import X"

        This bypasses the "telephone game" where actionable fixes get diluted
        through multiple agent hops.
        """
        if not state.failure_history:
            return ""

        last_failure = state.failure_history[-1]
        rca_data = last_failure.get("rca_data")
        if not rca_data:
            return ""

        concrete_edits = rca_data.get("concrete_edits", [])
        if not concrete_edits:
            return ""

        section = "\n## 🎯 MANDATORY FIRST ACTIONS (from Root Cause Analysis)\n"
        section += "The following edits were identified by automated analysis.\n"
        section += "You MUST execute these BEFORE doing anything else:\n\n"

        for i, edit in enumerate(concrete_edits, 1):
            file = edit.get("file", "unknown")
            action = edit.get("action", "unknown")
            details = edit.get("details", "unknown")
            section += f"**Edit {i}:** `{file}` — {action}\n"
            section += f"  → {details}\n\n"

        section += (
            "**WORKFLOW:** For each edit above:\n"
            "1. `read_file` the target file to see current content\n"
            "2. `edit_file` to make the exact change specified\n"
            "3. Verify the file still has valid syntax\n"
            "4. THEN proceed with any remaining plan steps\n\n"
            "Do NOT skip these edits. Do NOT rewrite entire files. "
            "Make the targeted changes specified above.\n"
        )

        logger.info(f"  Injected {len(concrete_edits)} concrete RCA edits into build prompt")
        return section

    def run_rca(self, state: TaskState, result: IterationResult) -> Optional[dict]:
        """
        Run LLM-based Root Cause Analysis using 5 Whys methodology.

        v0.5.0: Now includes actual evidence (stderr, file listing, git diff)
        so the RCA can reason about code, not just about vague error summaries.

        v0.5.1: All prompt sections are budget-capped to prevent context window
        bloat on multi-file tasks. Total prompt is hard-capped.

        Uses the plan agent's model (70B) for better reasoning quality.
        Returns structured RCA dict, or None if analysis fails.
        """
        agent_config = self.config.get_agent("plan")  # Use the heavy model for RCA
        model_config = agent_config.model

        if model_config.provider != "ollama":
            return None

        budget = RCA_PROMPT_BUDGET

        # Build failure context with ACTUAL evidence (budget-capped)
        failure_context = ""
        if state.failure_history:
            last = state.failure_history[-1]
            failure_context = f"Phase: {last.get('phase', 'unknown')}\n"
            failure_context += f"Error: {last.get('error', 'unknown')}\n"

            # Include DoD results with evidence — cap per-criterion and total
            dod_results = last.get("dod_results") or (result.dod_results if result else None)
            if dod_results:
                failure_context += "\nDoD Results:\n"
                for cid, res in dod_results.items():
                    if isinstance(res, dict):
                        status = "PASSED" if res.get("passed") else "FAILED"
                        cmd = res.get("command", "unknown")
                        failure_context += f"  {cid}: {status}\n"
                        failure_context += f"    Command: {cmd}\n"
                        if not res.get("passed"):
                            # v0.8.0: Extract error TAIL not head — pytest puts actual error at bottom
                            evidence = self._extract_test_error_tail(res.get("evidence", ""), 500)
                            failure_context += f"    Output: {evidence}\n"

            # Hard cap on failure context section
            failure_context = truncate_to_budget(
                failure_context, budget["failure_context"], "failure context"
            )

        # Gather real evidence from workspace (already budget-capped internally)
        evidence_context = self._gather_rca_evidence(state)
        evidence_context = truncate_to_budget(
            evidence_context, budget["evidence"], "evidence"
        )

        # Include plan summary for context
        plan_context = truncate_to_budget(
            state.current_plan or "No plan available",
            budget["plan_context"], "plan"
        )

        # Format failure history (budget-capped)
        failure_history = truncate_to_budget(
            self._format_failure_history(state),
            budget["failure_history"], "failure history"
        )

        user_content = """## Task
{state.goal}

## Iteration {state.iteration} FAILED

## Failure Details
{failure_context}

## Actual Evidence from Workspace
{evidence_context}

## Plan That Was Followed
{plan_context}

## Previous Failures
{failure_history}

Perform a 5 Whys analysis. What is the SPECIFIC root cause? What SPECIFIC change should be made for the next iteration?
"""

        # Hard cap on total user message
        user_content = truncate_to_budget(
            user_content, budget["total"], "RCA prompt"
        )

        messages = [
            {"role": "system", "content": (
                "You are a Root Cause Analysis agent. Analyze build failures using the "
                "5 Whys methodology. Be specific and actionable.\n\n"
                "IMPORTANT: You are given actual stderr output, file listings, and git diffs. "
                "Use this CONCRETE evidence to identify the root cause. Do NOT give vague "
                "answers like 'insufficient test coverage' or 'implementation incomplete'. "
                "Instead, cite specific errors, missing imports, wrong function signatures, "
                "or structural problems you can see in the evidence.\n\n"
                "CRITICAL — concrete_edits field:\n"
                "You MUST provide concrete_edits — an array of EXACT edit commands that the "
                "build agent can execute mechanically. Each edit must specify:\n"
                "  - file: exact filename (e.g., 'cli.py')\n"
                "  - action: what to do (add_import, replace_line, add_code, fix_syntax, rewrite_function)\n"
                "  - details: the EXACT code. Not 'add an import', but 'Add `from models import Task` after `from storage import TaskStorage`'\n\n"
                "Example concrete_edits:\n"
                "[\n"
                '  {"file": "cli.py", "action": "add_import", "details": "Add `from models import Task` after the existing `from storage import TaskStorage` line"},\n'
                '  {"file": "test_cli.py", "action": "replace_line", "details": "Replace empty assertEqual with assertIn to check for expected output"}\n'
                "]\n\n"
                "These edits will be injected DIRECTLY into the build agent's prompt as mandatory first actions. "
                "The more precise you are, the more likely the fix will succeed.\n\n"
                "GUARDRAIL (v0.7.0):\n"
                "- concrete_edits MUST only target files that appear in the traceback.\n"
                "- If the error is NameError/ImportError/SyntaxError in test_*.py, "
                "  your concrete_edits MUST ONLY target test files, NOT source modules.\n"
                "- DO NOT propose adding __eq__ to models unless the test explicitly "
                "  compares objects with == and AssertionError shows different instances.\n"
                "- Quote the exact exception type and file:line from evidence before proposing edits."
            )},
            {"role": "user", "content": user_content},
        ]

        client = LLMClient(model_config)

        try:
            logger.info("Running LLM-based RCA (5 Whys) with enriched evidence...")
            prompt_tokens = estimate_tokens(user_content)
            logger.debug(f"  RCA prompt: ~{prompt_tokens} tokens "
                        f"(budget: ~{budget['total'] // 4} tokens, "
                        f"model context: {model_config.context_window})")
            rca_data = client.chat_structured(messages, schema=RCA_OUTPUT_SCHEMA, temperature=0.0)

            if rca_data and rca_data.get("root_cause"):
                logger.debug(f"  RCA root cause: {rca_data['root_cause']}")
                logger.debug(f"  RCA action: {rca_data.get('what_to_change', 'none')}")
                concrete_edits = rca_data.get("concrete_edits", [])
                if concrete_edits:
                    logger.info(f"  RCA produced {len(concrete_edits)} concrete edits:")
                    for edit in concrete_edits:
                        logger.info(f"    → {edit.get('file')}: {edit.get('action')} — {edit.get('details', '')[:80]}")
                return rca_data
            else:
                logger.warning("  RCA returned empty or invalid data")
                return None

        except Exception as e:
            logger.warning(f"  LLM-based RCA failed: {e}")
            return None

    def _gather_rca_evidence(self, state: TaskState) -> str:
        """
        Gather concrete evidence from the workspace for RCA.

        Collects (within budget):
        1. File listing (what files exist)
        2. Git diff (what changed in this iteration)
        3. Key file contents (test files, main source files) — abbreviated

        v0.5.1: All sections are budget-capped to prevent context window bloat
        on multi-file tasks. Total evidence is hard-capped at RCA_EVIDENCE_BUDGET["total"].
        """
        budget = RCA_EVIDENCE_BUDGET
        evidence_parts = []
        total_chars = 0

        # 1. File listing (cheap — just filenames)
        try:
            py_files = sorted([
                f.name for f in self.working_dir.iterdir()
                if f.name.endswith(".py") and not f.name.startswith("standalone_")
            ])
            if py_files:
                listing = f"### Python files in workspace\n{', '.join(py_files)}"
                listing = listing[:budget["file_listing"]]
                evidence_parts.append(listing)
                total_chars += len(listing)
        except Exception:
            pass

        # 2. Git diff (what the build agent changed)
        try:
            diff_result = subprocess.run(
                ["git", "dif", "HEAD~1", "--stat"],
                cwd=self.working_dir, capture_output=True, text=True, timeout=10
            )
            if diff_result.returncode == 0 and diff_result.stdout.strip():
                stat_text = truncate_to_budget(
                    diff_result.stdout, budget["diff_stat"], "diff stat"
                )
                evidence_parts.append(f"### Git diff (files changed)\n{stat_text}")
                total_chars += len(stat_text) + 30

            # Actual diff content — most valuable for RCA but also biggest
            if total_chars < budget["total"] - 500:
                diff_content = subprocess.run(
                    ["git", "dif", "HEAD~1", "--", "*.py"],
                    cwd=self.working_dir, capture_output=True, text=True, timeout=10
                )
                if diff_content.returncode == 0 and diff_content.stdout.strip():
                    remaining_for_diff = min(
                        budget["diff_content"],
                        budget["total"] - total_chars - 500  # reserve for file contents
                    )
                    diff_text = truncate_diff(diff_content.stdout, remaining_for_diff)
                    evidence_parts.append(f"### Code changes\n{diff_text}")
                    total_chars += len(diff_text) + 20
        except Exception:
            pass

        # 3. Key file contents — test files first (most useful for RCA), then source
        try:
            remaining_file_budget = min(
                budget["test_files"],
                budget["total"] - total_chars
            )

            if remaining_file_budget > 200:
                files_added = 0
                for f in sorted(self.working_dir.iterdir()):
                    if f.name.startswith("test_") and f.name.endswith(".py") and f.name not in {
                        "test_memory.py", "test_standalone_integration.py",
                        "test_orchestrator.py", "test_tool_executor.py",
                    }:
                        content = f.read_text()
                        if len(content) <= remaining_file_budget - 50:
                            block = f"### {f.name}\n```python\n{content}\n```"
                        else:
                            # Show head + tail of file — errors often at bottom
                            truncated = truncate_to_budget(
                                content, remaining_file_budget - 50, f.name
                            )
                            block = f"### {f.name} (truncated)\n```python\n{truncated}\n```"

                        evidence_parts.append(block)
                        total_chars += len(block)
                        remaining_file_budget -= len(block)
                        files_added += 1

                        # Max 2 test files to save budget for source
                        if files_added >= 2 or remaining_file_budget < 200:
                            break
        except Exception:
            pass

        # 4. Source file snippets — only if we have budget remaining
        try:
            remaining_source_budget = min(
                budget["source_files"],
                budget["total"] - total_chars
            )

            if remaining_source_budget > 300:
                for f in sorted(self.working_dir.iterdir()):
                    if (f.name.endswith(".py")
                        and not f.name.startswith("test_")
                        and not f.name.startswith("standalone_")
                        and f.stat().st_size < 5000):
                        content = f.read_text()
                        truncated = truncate_to_budget(
                            content, remaining_source_budget - 50, f.name
                        )
                        block = f"### {f.name}\n```python\n{truncated}\n```"
                        evidence_parts.append(block)
                        total_chars += len(block)
                        break  # Only first source file
        except Exception:
            pass

        if not evidence_parts:
            return "No additional evidence available."

        result = "\n\n".join(evidence_parts)

        # Hard cap — should rarely hit this given per-section budgets
        if len(result) > budget["total"]:
            result = truncate_to_budget(result, budget["total"], "total evidence")

        return result

    def _format_failure_history(self, state: TaskState) -> str:
        """Format failure history for RCA context."""
        if not state.failure_history:
            return "No previous failures."

        lines = []
        for f in state.failure_history[-3:]:
            lines.append(f"- Iteration {f.get('iteration', '?')}: [{f.get('phase', '?')}] {(f.get('error') or '?')[:150]}")
            if f.get('rca'):
                lines.append(f"  Previous RCA: {f['rca'][:200]}")
        return "\n".join(lines)

    def _run_direct_verification(self, state: TaskState) -> Optional[AgentResult]:
        """
        Post-build verification — generates and runs verification commands based on
        what actually exists on disk, NOT what the plan agent predicted.

        v0.5.0 Architecture (inspired by SWE-bench, Refact.ai, Verdent, Claude Code):
        1. Scan the workspace for test files, source files, and project structure
        2. Generate verification commands deterministically based on actual code
        3. Primary verification: run all unit tests (unittest discover / pytest)
        4. Secondary: syntax checks, file existence, import validation
        5. Match results back to DoD criteria
        """
        if not state.dod or not state.dod.criteria:
            return None

        logger.info("Running post-build verification (v0.5.0)...")

        # Detect venv (for dependency installation only)
        venv_path = self.working_dir / "venv" / "bin" / "activate"

        # v0.7.4 CRITICAL: Do NOT use venv for test execution.
        # The micro-build (_run_test_file) uses system Python without venv.
        # DoD MUST use the same Python environment, otherwise tests that pass
        # in micro-build will fail in DoD due to missing packages (pytest),
        # different sys.path, or other environment differences.
        # The venv prefix is ONLY used for auto-installing dependencies.
        venv_prefix = ""

        # Auto-install dependencies
        if venv_path.exists():
            self._auto_install_dependencies(venv_path)

        # -- Step 1: Scan workspace --
        ORCHESTRATOR_TEST_FILES = {
            "test_standalone_integration.py", "test_memory.py",
            "test_orchestrator.py", "test_tool_executor.py",
        }
        ORCHESTRATOR_SRC_FILES = {
            "standalone_agents.py", "standalone_orchestrator.py",
            "standalone_config.py", "standalone_memory.py",
            "standalone_models.py", "standalone_session.py",
        }

        source_files = [
            f for f in self.working_dir.iterdir()
            if f.name.endswith(".py")
            and not f.name.startswith("test_")
            and f.name not in ORCHESTRATOR_SRC_FILES
        ]

        test_files = [
            f for f in self.working_dir.iterdir()
            if f.name.startswith("test_") and f.name.endswith(".py")
            and f.name not in ORCHESTRATOR_TEST_FILES
        ]

        # Validate test files can actually import (exclude stale ones)
        valid_test_files = []
        for tf in test_files:
            check_cmd = f'{venv_prefix}python3 -c "import {tf.stem}"'
            check_result = self.tool_executor.execute(
                "run_command", {"command": check_cmd, "timeout": 10}
            )
            if "EXIT_CODE: 0" in check_result:
                valid_test_files.append(tf)
            else:
                logger.debug(f"  Excluding stale test file {tf.name} (import failed)")

        logger.info(f"  Workspace: {len(source_files)} source files, "
                    f"{len(valid_test_files)} valid test files "
                    f"(excluded {len(test_files) - len(valid_test_files)} stale)")

        # -- Step 1.5: Syntax pre-check on source files --
        # If a source file has a SyntaxError, ALL tests will fail on import.
        # Catch this early and report the exact error + line number.
        syntax_errors = {}
        for sf in source_files:
            check_cmd = f'{venv_prefix}python3 -m py_compile {sf.name}'
            check_result = self.tool_executor.execute(
                "run_command", {"command": check_cmd, "timeout": 10}
            )
            if "EXIT_CODE: 0" not in check_result:
                syntax_errors[sf.name] = check_result
                logger.warning(f"  ⚠️ SYNTAX ERROR in {sf.name}: {check_result[:150]}")

        if syntax_errors:
            # Fast-fail: source files have syntax errors, no point running tests
            logger.warning(f"  {len(syntax_errors)} source file(s) have syntax errors — "
                          "tests will fail on import. Reporting syntax errors directly.")

            test_report: Dict[str, Dict[str, Any]] = {}
            all_evidence: list[str] = []
            error_summary = "\n".join(
                f"  {name}: {err[:200]}" for name, err in syntax_errors.items()
            )

            for idx, criterion in enumerate(state.dod.criteria):
                cid = f"criterion-{idx}"
                criterion.passed = False
                criterion.evidence = f"BLOCKED by syntax error(s):\n{error_summary}"
                test_report[cid] = {
                    "passed": False,
                    "evidence": criterion.evidence[:500],
                    "command": "py_compile (pre-check)",
                }
                all_evidence.append(f"{cid}: BLOCKED — syntax error in {', '.join(syntax_errors.keys())}")

            output = f"SYNTAX ERRORS FOUND — all criteria blocked:\n{error_summary}"
            output += "\n\nDOD VERIFICATION FAILED\nFailed: all (source files have syntax errors)"

            return AgentResult(
                success=False,
                output=output,
                error=f"0/{len(state.dod.criteria)} DoD criteria passed (syntax errors in source files)",
                test_report={
                    "overall_passed": False,
                    "passed_count": 0,
                    "total_count": len(state.dod.criteria),
                    "verification_method": "syntax_precheck_v0.5.3",
                    "syntax_errors": {name: err[:300] for name, err in syntax_errors.items()},
                    "criteria_results": [
                        {
                            "criterion_id": f"criterion-{i}",
                            "description": c.description,
                            "passed": False,
                            "command_run": "py_compile (pre-check)",
                            "evidence": f"BLOCKED: syntax error in {', '.join(syntax_errors.keys())}",
                            "failure_reason": f"SyntaxError in {', '.join(syntax_errors.keys())}",
                        }
                        for i, c in enumerate(state.dod.criteria)
                    ],
                },
                dod_results=test_report,  # type: ignore[arg-type]
            )

        # -- Step 2: v0.7.4 — Run ALL tests with proper isolation --
        #
        # ROOT CAUSE OF 24-HOUR FAILURE (finally identified):
        #
        # The micro-build phase runs each test file right after creating it.
        # These tests create artifacts (tasks.json, etc.) that persist on disk.
        # When DoD re-runs the SAME tests, it finds stale data from micro-build:
        #
        #   Micro-build: test_storage.py runs → creates tasks.json → PASSES (clean slate)
        #   Micro-build: test_cli.py runs → modifies tasks.json → PASSES
        #   DoD: test_cli.py runs → finds stale tasks.json → wrong task count → FAILS
        #   DoD: test_storage.py runs → finds stale tasks.json → FAILS
        #
        # The 70B doesn't generate setUp()/tearDown() with temp directories.
        # Fix: clean test artifacts before each test file to ensure clean slate.

        if valid_test_files:
            per_file_results: Dict[str, Dict[str, Any]] = {}
            all_tests_pass = True
            combined_output = []

            for tf in valid_test_files:
                # CRITICAL: Clean test artifacts before EACH test file.
                # This ensures each test starts with a clean filesystem,
                # matching the conditions when micro-build first ran the test.
                # Clean: JSON data files, temp files, __pycache__ (stale bytecode)
                cleanup_cmd = "rm -f *.json *.json.bak *.db *.sqlite3 && rm -rf __pycache__"
                self.tool_executor.execute("run_command", {"command": cleanup_cmd, "timeout": 5})

                # Run test with SAME command as micro-build's _run_test_file:
                # - System Python (no venv) — micro-build doesn't use venv
                # - pytest first, fall back to unittest
                # - --tb=short for readable error output
                file_cmd = f"python3 -m pytest {tf.name} -v --tb=short 2>&1 || python3 -m unittest {tf.stem} -v 2>&1"
                file_result = self.tool_executor.execute(
                    "run_command", {"command": file_cmd, "timeout": 60}
                )
                file_passed = "EXIT_CODE: 0" in file_result
                per_file_results[tf.name] = {"passed": file_passed, "output": file_result}
                combined_output.append(f"--- {tf.name}: {'PASS' if file_passed else 'FAIL'} ---\n{file_result[:300]}")
                if not file_passed:
                    all_tests_pass = False
                    logger.warning(f"  ❌ {tf.name}: FAILED")
                else:
                    logger.info(f"  ✅ {tf.name}: PASSED")

            # Final cleanup after all tests
            self.tool_executor.execute(
                "run_command", {"command": "rm -f *.json *.json.bak *.db *.sqlite3", "timeout": 5}
            )

            tests_passed = all_tests_pass
            test_result_output = "\n".join(combined_output)
        else:
            test_command = "python3 -m unittest discover -s . -p 'test_*.py' -v"
            test_result_output = self.tool_executor.execute(
                "run_command", {"command": test_command, "timeout": 120}
            )
            tests_passed = "EXIT_CODE: 0" in test_result_output
            per_file_results: Dict[str, Dict[str, Any]] = {}

        # Parse aggregate counts
        import re as re_mod
        n_passed = sum(1 for r in per_file_results.values() if r["passed"])
        n_total = len(per_file_results) if per_file_results else 0
        test_summary = f"{n_passed}/{n_total} test files pass"
        logger.info(f"  Test suite result: {test_summary} — {'ALL PASS ✅' if tests_passed else 'FAILURES ❌'}")

        # v0.8.4: Parse individual test function results from pytest -v output
        # This enables granular criteria mapping instead of binary all-or-nothing.
        # Inspired by SWE-bench: "tests pass" IS the verification — no separate DoD layer.
        individual_tests: Dict[str, bool] = {}  # {"test_create_project": True, "test_delete_task": False}
        for tf_name, tf_result in per_file_results.items():
            output: str = str(tf_result.get("output", ""))
            # Parse pytest -v output: "test_app.py::test_create_project PASSED"
            for match in re_mod.finditer(r'(\w+::|^)(test_\w+)\s+(PASSED|FAILED|ERROR)', output, re_mod.MULTILINE):
                func_name = match.group(2)
                status = match.group(3)
                individual_tests[func_name] = (status == "PASSED")
            # Also parse unittest output: "test_create_project (test_app.TestApp) ... ok"
            for match in re_mod.finditer(r'(test_\w+)\s+\([^)]+\)\s+\.\.\.\s+(ok|FAIL|ERROR)', output, re_mod.MULTILINE):
                func_name = match.group(1)
                status = match.group(2)
                individual_tests[func_name] = (status == "ok")

        if individual_tests:
            ind_passed = sum(1 for v in individual_tests.values() if v)
            ind_total = len(individual_tests)
            logger.info(f"  Individual test functions: {ind_passed}/{ind_total} pass")

        # -- Step 3: Map test results to DoD criteria --
        # v0.8.4: Three-tier matching:
        #   1. Independent checks (file existence, import validation)
        #   2. Test file name matching (criterion mentions "test_app")
        #   3. Test FUNCTION matching (criterion text overlaps with test function names)
        #   4. Default: proportional to individual test results (NOT binary all-or-nothing)
        test_report: Dict[str, Dict[str, Any]] = {}
        all_evidence: list[str] = []
        unmapped_criteria_indices: list[int] = []  # Track criteria that don't match anything specific

        for idx, criterion in enumerate(state.dod.criteria):
            cid = f"criterion-{idx}"
            desc_lower = criterion.description.lower()

            # Check if this criterion can be verified independently of tests
            independent_check = None
            was_mapped = False

            # File existence checks
            if self._desc_is_file_check(desc_lower):
                filename = self._extract_filename_from_desc(criterion.description)
                if filename:
                    independent_check = f"test -f {filename}"

            # Import checks
            elif "import" in desc_lower and "can be imported" in desc_lower:
                filename = self._extract_filename_from_desc(criterion.description)
                if filename:
                    module = filename.replace('.py', '')
                    independent_check = f'{venv_prefix}python3 -c "import {module}"'

            # Criteria that reference a specific test file — use that file's result
            elif per_file_results:
                matched_file = None
                for tf in valid_test_files:
                    if tf.stem in desc_lower or tf.name in desc_lower:
                        matched_file = tf.name
                        break
                if matched_file and matched_file in per_file_results:
                    file_result = per_file_results[matched_file]
                    criterion.passed = bool(file_result["passed"])
                    criterion.evidence = str(file_result.get("output", ""))[:500]
                    test_report[cid] = {"passed": criterion.passed, "evidence": criterion.evidence, "command": f"pytest {matched_file}"}
                    if criterion.passed:
                        logger.info(f"DoD {cid} PASSED: {criterion.description[:60]}")
                        all_evidence.append(f"{cid}: PASSED -- {matched_file} tests pass")
                    else:
                        logger.warning(f"DoD {cid} FAILED: {criterion.description[:60]}")
                        fail_tail = self._extract_test_error_tail(str(file_result.get('output', '')))
                        all_evidence.append(f"{cid}: FAILED -- {matched_file}: {fail_tail}")
                    was_mapped = True

            if independent_check:
                result = self.tool_executor.execute("run_command", {"command": independent_check, "timeout": 30})
                passed = "EXIT_CODE: 0" in result
                criterion.passed = passed
                criterion.evidence = result[:500]
                test_report[cid] = {"passed": passed, "evidence": result[:500], "command": independent_check}
                if passed:
                    logger.info(f"DoD {cid} PASSED: {criterion.description[:60]}")
                    all_evidence.append(f"{cid}: PASSED -- {independent_check}")
                else:
                    logger.warning(f"DoD {cid} FAILED: {criterion.description[:60]}")
                    all_evidence.append(f"{cid}: FAILED -- {result[:200]}")
                was_mapped = True

            if was_mapped:
                continue

            # v0.8.4 Tier 3: Match criterion text to individual test function names
            # Extract keywords from criterion description and find matching test functions
            if individual_tests:
                # Extract meaningful words from criterion (skip common words)
                skip_words = {'the', 'a', 'an', 'is', 'are', 'has', 'have', 'can', 'should',
                              'must', 'will', 'be', 'to', 'o', 'in', 'for', 'with', 'and',
                              'or', 'not', 'all', 'each', 'that', 'this', 'it', 'on', 'at',
                              'by', 'from', 'as', 'when', 'i', 'no', 'any', 'only', 'test',
                              'correctly', 'properly', 'successfully', 'valid', 'returns',
                              'return', 'field', 'fields', 'using', 'via', 'endpoint'}
                desc_words = set(re_mod.findall(r'[a-z_]+', desc_lower)) - skip_words

                best_match = None
                best_overlap = 0
                for func_name, func_passed in individual_tests.items():
                    func_words = set(func_name.lower().replace('test_', '').split('_'))
                    overlap = len(desc_words & func_words)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = (func_name, func_passed)

                if best_match and best_overlap >= 2:
                    func_name, func_passed = best_match
                    criterion.passed = func_passed
                    criterion.evidence = f"Matched to test function: {func_name} ({'PASSED' if func_passed else 'FAILED'})"
                    test_report[cid] = {"passed": func_passed, "evidence": criterion.evidence, "command": f"pytest -k {func_name}"}
                    if func_passed:
                        logger.info(f"DoD {cid} PASSED: {criterion.description[:60]} (matched: {func_name})")
                        all_evidence.append(f"{cid}: PASSED -- matched test function {func_name}")
                    else:
                        logger.warning(f"DoD {cid} FAILED: {criterion.description[:60]} (matched: {func_name})")
                        all_evidence.append(f"{cid}: FAILED -- test function {func_name} failed")
                    continue

            # v0.8.4 Tier 4: Proportional mapping based on test pass rate
            # OLD (BROKEN): criterion.passed = tests_passed  (binary: 7/8 pass → ALL 16 criteria FAIL)
            # NEW: Map proportionally to individual test results.
            # If 7/8 tests pass, this criterion is LIKELY passing.
            # SWE-bench principle: "as long as tests pass, task is scored as completed"
            if individual_tests:
                ind_passed = sum(1 for v in individual_tests.values() if v)
                ind_total = len(individual_tests)
                pass_rate = ind_passed / ind_total if ind_total > 0 else 0
                # If >80% of individual tests pass, map unmapped criteria as passing
                # This prevents 7/8 → 0/16 while still failing on catastrophic breakage
                criterion.passed = pass_rate >= 0.8
                criterion.evidence = f"Test pass rate: {ind_passed}/{ind_total} ({pass_rate:.0%}). Unmapped criterion — using pass rate threshold (≥80%)."
            else:
                criterion.passed = tests_passed
                criterion.evidence = f"Test suite: {test_summary}" if tests_passed else test_result_output[:500]

            test_report[cid] = {"passed": criterion.passed, "evidence": criterion.evidence[:500], "command": "pytest (proportional)"}

            if criterion.passed:
                logger.info(f"DoD {cid} PASSED: {criterion.description[:60]} (proportional)")
                all_evidence.append(f"{cid}: PASSED -- test pass rate ≥80%")
            else:
                logger.warning(f"DoD {cid} FAILED: {criterion.description[:60]} (proportional)")
                all_evidence.append(f"{cid}: FAILED -- test pass rate below 80%")

        passed_count = sum(1 for c in state.dod.criteria if c.passed)
        total_count = len(state.dod.criteria)
        logger.info(f"DoD final count: {passed_count}/{total_count} criteria passed")

        # v0.8.4: PYTEST AUTHORITY OVERRIDE
        # SWE-bench, Aider, Agentless all agree: "tests pass" = success.
        # If ALL test files pass, the task IS done — regardless of how well
        # abstract DoD criteria descriptions mapped to test function names.
        # The criteria were generated by the PLAN agent (which can hallucinate).
        # The TESTS were generated from the SPEC. Tests are authoritative.
        success = passed_count == total_count
        if not success and tests_passed and n_total > 0:
            logger.info(f"  🔑 PYTEST AUTHORITY: All {n_total} test files pass — "
                       f"overriding DoD ({passed_count}/{total_count}) → SUCCESS")
            success = True
            # Mark remaining failed criteria as passed (test-overridden)
            for c in state.dod.criteria:
                if not c.passed:
                    c.passed = True
                    c.evidence = f"PYTEST OVERRIDE: All {n_total} test files pass"
            passed_count = total_count

        # -- Step 4: Build structured test report with workspace info --
        structured_report: Dict[str, Any] = {
            "overall_passed": success,
            "passed_count": passed_count,
            "total_count": total_count,
            "verification_method": "post_build_v0.5",
            "workspace_info": {
                "source_files": [f.name for f in source_files],
                "valid_test_files": [f.name for f in valid_test_files],
                "stale_test_files": [f.name for f in test_files if f not in valid_test_files],
            },
            "criteria_results": [],
        }
        for idx, criterion in enumerate(state.dod.criteria):
            cid = f"criterion-{idx}"
            report_entry = test_report.get(cid, {})
            evidence_str = str(report_entry.get("evidence", ""))
            structured_report["criteria_results"].append({
                "criterion_id": cid,
                "description": criterion.description,
                "passed": criterion.passed,
                "command_run": report_entry.get("command", "none"),
                "evidence": evidence_str[:300],
                "failure_reason": "" if criterion.passed else self._extract_failure_reason(evidence_str),
            })

        output = "\n".join(all_evidence)
        if success:
            output += "\n\nALL DOD CRITERIA PASSED"
        else:
            failed_ids = [f"criterion-{i}" for i, c in enumerate(state.dod.criteria) if not c.passed]
            output += f"\n\nDOD VERIFICATION FAILED\nFailed: {', '.join(failed_ids)}"

        return AgentResult(
            success=success,
            output=output,
            error=None if success else f"{passed_count}/{total_count} DoD criteria passed",
            test_report=structured_report,
            dod_results=test_report,  # type: ignore[arg-type]
        )

    def _generate_post_build_commands(
        self,
        criteria: list,
        source_files: list,
        test_files: list,
        venv_prefix: str,
    ) -> Dict[str, str]:
        """
        Generate verification commands AFTER build, based on what actually exists on disk.

        This is the core fix -- verification commands are generated from actual code structure,
        not from plan agent predictions about code that doesn't exist yet.

        Strategy per verification_type:
        - 'test':        Run unittest discover (primary) -- covers most functional criteria
        - 'file_exists': Check file exists with test -f
        - 'syntax':      Run py_compile on target file
        - 'import':      Try importing the target module

        For 'test' criteria, we run ALL tests once and share the result across all
        test-type criteria. This is more efficient and more reliable than trying to
        map individual criteria to individual test cases.
        """
        commands = {}

        # Build the canonical test command once
        if test_files:
            test_modules = " ".join(f.stem for f in test_files)
            test_command = f"{venv_prefix}python3 -m unittest {test_modules} -v"
        else:
            # No test files found -- try discover as last resort
            test_command = f"{venv_prefix}python3 -m unittest discover -s . -p 'test_*.py' -v"

        for idx, criterion in enumerate(criteria):
            cid = f"criterion-{idx}"
            desc_lower = criterion.description.lower()
            v_type = criterion.verification_method or "test"
            target = getattr(criterion, 'target_file', '') or ''

            if v_type == "file_exists" or self._desc_is_file_check(desc_lower):
                # File existence check
                filename = target or self._extract_filename_from_desc(criterion.description)
                if filename:
                    commands[cid] = f"test -f {filename}"
                else:
                    commands[cid] = test_command

            elif v_type == "syntax" or ("syntax" in desc_lower and "valid" in desc_lower):
                # Syntax check
                filename = target or self._extract_filename_from_desc(criterion.description)
                if filename:
                    commands[cid] = f"{venv_prefix}python3 -m py_compile {filename}"
                else:
                    # Syntax check all source files
                    if source_files:
                        files_str = " ".join(f.name for f in source_files[:5])
                        commands[cid] = f"{venv_prefix}python3 -m py_compile {files_str}"
                    else:
                        commands[cid] = test_command

            elif v_type == "import" or ("import" in desc_lower and "can be imported" in desc_lower):
                # Import check
                filename = target or self._extract_filename_from_desc(criterion.description)
                if filename:
                    module = filename.replace('.py', '')
                    commands[cid] = f'{venv_prefix}python3 -c "import {module}"'
                else:
                    commands[cid] = test_command

            else:
                # Default: 'test' type -- run unit tests
                commands[cid] = test_command

        return commands

    def _desc_is_file_check(self, desc_lower: str) -> bool:
        """Check if a description is about file existence."""
        return any(p in desc_lower for p in [
            "file exists", "file is created", "file present",
            "creates file", "generates file", "has file",
        ])

    def _extract_filename_from_desc(self, description: str) -> Optional[str]:
        """Extract a filename (something.py, something.json, etc.) from a description."""
        # Match quoted or backticked filenames
        match = re.search(r'[`"\']?([\w/.-]+\.\w{1,4})[`"\']?', description)
        if match:
            filename = match.group(1)
            # Sanity: must look like a real file extension
            if '.' in filename and not filename.startswith('.'):
                return filename
        return None


    def _auto_install_dependencies(self, venv_activate_path: Path) -> None:
        """
        Auto-detect and install project dependencies into the venv before verification.

        Scans for common dependency manifests and installs them:
          - requirements.txt → pip install -r requirements.txt
          - setup.py         → pip install -e .
          - pyproject.toml   → pip install -e .
          - package.json     → npm install (if node project)

        This is an infrastructure concern, not an LLM concern. The build agent
        should focus on writing code, not managing pip installs in the right env.
        """
        wd = self.working_dir
        pip_path = wd / "venv" / "bin" / "pip"
        if not pip_path.exists():
            return

        # Track what we install so we don't repeat
        installed_something = False

        # requirements.txt — most common
        req_txt = wd / "requirements.txt"
        if req_txt.exists() and req_txt.stat().st_size > 0:
            logger.info("  Auto-installing dependencies from requirements.txt...")
            result = subprocess.run(
                [str(pip_path), "install", "-r", str(req_txt), "-q"],
                cwd=wd, capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                installed_something = True
                logger.debug("  ✅ requirements.txt installed successfully")
            else:
                logger.warning(f"  ⚠️ requirements.txt install failed: {result.stderr[:200]}")

        # setup.py or pyproject.toml — editable install
        setup_py = wd / "setup.py"
        pyproject = wd / "pyproject.toml"
        if (setup_py.exists() or pyproject.exists()) and not installed_something:
            manifest = "setup.py" if setup_py.exists() else "pyproject.toml"
            logger.info(f"  Auto-installing project from {manifest}...")
            result = subprocess.run(
                [str(pip_path), "install", "-e", ".", "-q"],
                cwd=wd, capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                installed_something = True
                logger.debug(f"  ✅ Editable install from {manifest} succeeded")
            else:
                logger.warning(f"  ⚠️ Editable install failed: {result.stderr[:200]}")

        # Also check for any .txt files that look like requirements (e.g., requirements-dev.txt)
        for req_file in wd.glob("requirements*.txt"):
            if req_file.name == "requirements.txt":
                continue  # Already handled above
            if req_file.stat().st_size > 0:
                logger.debug(f"  Auto-installing from {req_file.name}...")
                result = subprocess.run(
                    [str(pip_path), "install", "-r", str(req_file), "-q"],
                    cwd=wd, capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    logger.debug(f"  ✅ {req_file.name} installed")
                else:
                    logger.debug(f"  ⚠️ {req_file.name} install failed: {result.stderr[:100]}")

    def _extract_failure_reason(self, evidence: str) -> str:
        """Extract a concise failure reason from verification output."""
        if not evidence:
            return "no output"

        evidence_lower = evidence.lower()

        # Common failure patterns
        if "importerror" in evidence_lower or "modulenotfounderror" in evidence_lower:
            # Try to extract the module name
            import re
            match = re.search(r"(?:ImportError|ModuleNotFoundError):\s*(.+?)(?:\n|$)", evidence)
            return f"ImportError: {match.group(1).strip()}" if match else "ImportError"

        if "syntaxerror" in evidence_lower:
            return "SyntaxError in source code"

        if "assertionerror" in evidence_lower:
            return "Assertion failed"

        if "filenotfounderror" in evidence_lower or "no such file" in evidence_lower:
            return "File not found"

        if "nameerror" in evidence_lower:
            return "NameError — undefined variable or function"

        if "typeerror" in evidence_lower:
            return "TypeError"

        if "attributeerror" in evidence_lower:
            return "AttributeError"

        if "exit_code: 1" in evidence_lower:
            # Generic failure — grab first line of stderr if present
            for line in evidence.split("\n"):
                if line.startswith("STDERR:"):
                    return line[7:].strip()[:100]
            return "Command exited with error"

        if "exit_code: 2" in evidence_lower:
            return "Command not found or usage error"

        return "Unknown failure"

    def _extract_test_error_tail(self, output: str, max_chars: int = 400) -> str:
        """
        v0.8.0: Extract the TAIL of pytest output where the actual error lives.

        Pytest puts the error summary (e.g. sqlite3.ProgrammingError) at the
        bottom. Taking output[:400] gives you headers and traceback frames
        but misses the actual error line. This grabs the useful part.
        """
        if not output or len(output) <= max_chars:
            return output or ""

        lines = output.strip().split(chr(10))

        # Strategy 1: Find the short test summary line and take everything after
        for i, line in enumerate(lines):
            if "short test summary" in line.lower():
                tail = chr(10).join(lines[i:])
                return tail[:max_chars]

        # Strategy 2: Find the last Exception/Error line and grab context around it
        for i in range(len(lines) - 1, -1, -1):
            if any(err in lines[i] for err in ["Error:", "Error(", "FAILED", "Exception"]):
                start = max(0, i - 3)
                tail = chr(10).join(lines[start:])
                return tail[:max_chars]

        # Fallback: last N chars
        return output[-max_chars:]

    def _parse_dod_from_output(self, output: str) -> Optional[DoD]:
        """Parse Definition of Done from agent output."""
        dod = DoD()

        # Look for ```dod block
        dod_match = re.search(r'```dod\n(.*?)\n```', output, re.DOTALL)
        if not dod_match:
            dod_match = re.search(r'### Definition of Done\n(.*?)(?=###|\Z)', output, re.DOTALL)
        if not dod_match:
            # Also try ## Definition of Done
            dod_match = re.search(r'## Definition of Done\n(.*?)(?=##|\Z)', output, re.DOTALL)

        if not dod_match:
            return None

        dod_text = dod_match.group(1)

        # Also look for verification commands section
        verify_section = ""
        verify_match = re.search(r'(?:### |## )?Verification Commands?\n(.*?)(?=###|##|\Z)', output, re.DOTALL)
        if verify_match:
            verify_section = verify_match.group(1)

        # Extract individual commands from verification section
        verify_commands = re.findall(r'`([^`]+)`', verify_section)

        for line in dod_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            match = re.match(r'-\s*\[([ xX])\]\s*(.+)', line)
            if match:
                checked = match.group(1).lower() == 'x'
                full_text = match.group(2).strip()

                # Try to extract inline verification command: (verify: `command`)
                verify_cmd = None
                cmd_match = re.search(r'\(verify:\s*`([^`]+)`\)', full_text)
                if cmd_match:
                    verify_cmd = cmd_match.group(1)
                    # Remove the verify part from description
                    description = re.sub(r'\s*\(verify:\s*`[^`]+`\)', '', full_text).strip()
                else:
                    description = full_text.rstrip(':').strip()

                cid = dod.add(description, command=verify_cmd)
                if checked:
                    dod.mark_passed(cid)

        # If we didn't find inline commands, try to match from verify section by index
        if verify_commands:
            for idx, criterion in enumerate(dod.criteria):
                if not criterion.verification_command and idx < len(verify_commands):
                    criterion.verification_command = verify_commands[idx]

        return dod if dod.criteria else None

    def _parse_test_results(self, output: str) -> dict:
        """Parse test results JSON from agent output."""
        json_match = re.search(r'```(?:test-results|json)\n(\{.*?\})\n```', output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Fallback: check for keywords
        results = {}
        if "ALL DOD CRITERIA PASSED" in output:
            results["overall"] = {"passed": True}
        elif "VERIFICATION FAILED" in output:
            results["overall"] = {"passed": False}

        return results
