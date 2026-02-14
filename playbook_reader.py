"""
Playbook Reader — Orchestrator-side integration.

This module runs on the MAIN NODE and reads the playbook
maintained by the subconscious daemon on the PVE node.
It injects the most relevant bullets into agent system prompts.

Usage in standalone_orchestrator.py:
    from playbook_reader import PlaybookReader

    reader = PlaybookReader("/shared/playbook.json")

    # In agent call:
    extra_context = reader.get_context_for_agent("builder", task_goal)
    system_prompt = base_system_prompt + "\n\n" + extra_context
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


# Role → section relevance mapping
ROLE_SECTIONS = {
    "planner": ["architecture", "build_ordering", "general"],
    "builder": [
        "import_resolution", "flask_patterns", "dataclass_patterns",
        "sqlite_patterns", "stdlib_usage", "error_recovery", "general",
    ],
    "test_gen": ["test_generation", "import_resolution", "general"],
    "initializer": ["architecture", "general"],
    "explorer": ["architecture", "general"],
}


class PlaybookReader:
    """
    Reads the subconscious daemon's playbook and provides
    context injection for orchestrator agents.
    """

    def __init__(self, playbook_path: str = "/shared/playbook.json"):
        self.path = Path(playbook_path)
        self._cache = None
        self._cache_mtime: float = 0

    def _load(self) -> Optional[dict]:
        """Load playbook, with simple mtime-based caching."""
        if not self.path.exists():
            return None

        try:
            mtime = self.path.stat().st_mtime
            if self._cache and mtime == self._cache_mtime:
                return self._cache

            data = json.loads(self.path.read_text())
            self._cache = data
            self._cache_mtime = mtime
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"Could not read playbook: {e}")
            return None

    def get_context_for_agent(self, role: str, task_goal: str = "",
                              max_bullets: int = 25, max_chars: int = 3000) -> str:
        """
        Get playbook context formatted for injection into an agent's system prompt.

        Args:
            role: Agent role (planner, builder, test_gen, etc.)
            task_goal: Current task description (for future relevance scoring)
            max_bullets: Maximum number of bullets to include
            max_chars: Maximum character count for the context block

        Returns:
            Formatted text block to append to system prompt, or empty string.
        """
        data = self._load()
        if not data:
            return ""

        sections = data.get("sections", {})
        if not sections:
            return ""

        # Get relevant sections for this role
        relevant = ROLE_SECTIONS.get(role, list(sections.keys()))

        # Collect and score bullets
        scored_bullets = []
        for section_name in relevant:
            if section_name not in sections:
                continue
            for bullet in sections[section_name]:
                helpful = bullet.get("helpful_count", 0)
                harmful = bullet.get("harmful_count", 0)
                total = helpful + harmful
                quality = helpful / max(total, 1)

                # Score: quality * log(1 + references)
                score = quality * math.log1p(total + 1)
                scored_bullets.append((score, bullet))

        if not scored_bullets:
            return ""

        # Sort by score descending
        scored_bullets.sort(key=lambda x: x[0], reverse=True)

        # Build context block
        lines = [
            "## Coding Playbook (learned patterns — follow these)",
            ""
        ]
        char_count = sum(len(l) for l in lines)
        count = 0

        for score, bullet in scored_bullets:
            if count >= max_bullets:
                break
            bid = bullet.get("id", "?")
            content = bullet.get("content", "")
            line = f"- [{bid}] {content}"

            if char_count + len(line) + 1 > max_chars:
                break

            lines.append(line)
            char_count += len(line) + 1
            count += 1

        if count == 0:
            return ""

        result = "\n".join(lines) + "\n"
        logger.debug(f"Playbook context for {role}: {count} bullets, {char_count} chars")
        return result

    def report_bullet_usage(self, bullet_ids: List[str], was_successful: bool):
        """
        Report which bullets were in context during a session.
        Writes to a feedback file that the daemon picks up.

        Called by the orchestrator after a session completes.
        """
        feedback_dir = self.path.parent / "daemon" / "feedback"
        feedback_dir.mkdir(parents=True, exist_ok=True)

        feedback = {
            "bullet_ids": bullet_ids,
            "was_successful": was_successful,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }

        feedback_file = feedback_dir / "bullet_feedback.jsonl"
        try:
            with open(feedback_file, "a") as f:
                f.write(json.dumps(feedback) + "\n")
        except OSError as e:
            logger.debug(f"Could not write bullet feedback: {e}")

    @property
    def stats(self) -> dict:
        """Get quick stats about the current playbook."""
        data = self._load()
        if not data:
            return {"available": False}

        sections = data.get("sections", {})
        total = sum(len(bullets) for bullets in sections.values())
        return {
            "available": True,
            "total_bullets": total,
            "sections": {name: len(bullets) for name, bullets in sections.items()},
            "last_updated": data.get("last_updated", "unknown"),
        }
