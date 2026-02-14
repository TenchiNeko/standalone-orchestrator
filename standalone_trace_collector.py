"""
Standalone Trace Collector — v0.6.2

Captures failure trajectories from orchestrator runs for the
ML + LLM co-training loop:

  1. LLM attempts solution → orchestrator runs
  2. This module captures: prompt, generated code, error output, RCA
  3. User pastes traces to Claude/GPT-4 → gets reasoning traces back
  4. Reasoning traces → LoRA fine-tune the 15B/9B models
  5. Repeat — models get smarter at the patterns they fail on

Trace format is designed to be:
  - Machine-readable (JSONL for training pipelines)
  - Human-readable (markdown summaries for pasting to Claude)
  - Compact (only what matters for learning)

Usage:
    collector = TraceCollector(working_dir)
    collector.record_build_failure(filename, prompt, code, error, ...)
    collector.record_test_failure(filename, prompt, code, test_output, ...)
    collector.export_for_training()  # → JSONL
    collector.export_for_claude()    # → markdown for pasting
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TraceCollector:
    """
    Collects failure trajectories from orchestrator runs.

    Each trace captures the full context needed to generate
    a training example: what was asked, what was produced,
    why it failed, and what the correct fix looks like.
    """

    def __init__(self, working_dir: Path):
        self.working_dir = Path(working_dir)
        self.traces_dir = self.working_dir / ".agents" / "traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)

        # Current session traces
        self.traces: list[dict] = []

        # Load existing traces if any
        self.traces_file = self.traces_dir / "failure_traces.jsonl"
        self.summary_file = self.traces_dir / "failure_summary.md"

    def record_build_failure(
        self,
        filename: str,
        prompt_excerpt: str,
        generated_code: str,
        error_output: str,
        error_category: str = "unknown",
        model_used: str = "unknown",
        temperature: float = 0.0,
        iteration: int = 0,
        task_goal: str = "",
        depends_on: Optional[list] = None,
        manifest_context: str = "",
    ):
        """
        Record a build failure (file didn't compile, wrong structure, etc).

        This captures cases where the generated code itself is wrong —
        syntax errors, missing imports, wrong class structure.
        """
        trace = {
            "type": "build_failure",
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "task_goal": task_goal[:500],
            "model_used": model_used,
            "temperature": temperature,
            "iteration": iteration,
            "depends_on": depends_on or [],
            "error_category": error_category,
            "prompt_excerpt": prompt_excerpt[:2000],
            "generated_code": generated_code[:3000],
            "error_output": error_output[:1000],
            "manifest_context": manifest_context[:500],
            "correct_code": None,  # Filled in during distillation
            "reasoning_trace": None,  # Filled in during distillation
        }

        self.traces.append(trace)
        self._append_to_file(trace)

        logger.info(f"  📝 Trace recorded: build_failure for {filename} ({error_category})")

    def record_test_failure(
        self,
        test_filename: str,
        source_filename: str,
        prompt_excerpt: str,
        generated_test_code: str,
        source_code: str,
        test_output: str,
        passed: int = 0,
        failed: int = 0,
        errors: int = 0,
        error_category: str = "unknown",
        model_used: str = "unknown",
        temperature: float = 0.0,
        iteration: int = 0,
        task_goal: str = "",
        candidate_number: int = 0,
        total_candidates: int = 0,
    ):
        """
        Record a test failure (test file runs but tests fail/error).

        This is the highest-value trace type — captures cases where
        the model writes tests with logic errors (datetime.now equality,
        wrong argparse mocking, etc). These are the patterns we most
        need to teach via LoRA.
        """
        trace = {
            "type": "test_failure",
            "timestamp": datetime.now().isoformat(),
            "test_filename": test_filename,
            "source_filename": source_filename,
            "task_goal": task_goal[:500],
            "model_used": model_used,
            "temperature": temperature,
            "iteration": iteration,
            "candidate_number": candidate_number,
            "total_candidates": total_candidates,
            "error_category": error_category,
            "prompt_excerpt": prompt_excerpt[:2000],
            "generated_test_code": generated_test_code[:3000],
            "source_code": source_code[:2000],
            "test_output": test_output[:1500],
            "test_stats": {
                "passed": passed,
                "failed": failed,
                "errors": errors,
            },
            "correct_code": None,  # Filled in during distillation
            "reasoning_trace": None,  # Filled in during distillation
        }

        self.traces.append(trace)
        self._append_to_file(trace)

        logger.info(
            f"  📝 Trace recorded: test_failure for {test_filename} "
            f"({passed}/{passed+failed+errors} pass, {error_category})"
        )

    def record_rca_failure(
        self,
        rca_diagnosis: str,
        actual_root_cause: str,
        iteration: int = 0,
        task_goal: str = "",
        was_fix_applied: bool = False,
        did_fix_work: bool = False,
    ):
        """
        Record when RCA misdiagnoses the problem.

        E.g., RCA says "add __eq__ method" when real fix is
        "use same timestamp in test". These traces teach the
        pattern classifier to avoid known misdiagnosis patterns.
        """
        trace = {
            "type": "rca_failure",
            "timestamp": datetime.now().isoformat(),
            "task_goal": task_goal[:500],
            "iteration": iteration,
            "rca_diagnosis": rca_diagnosis[:1000],
            "actual_root_cause": actual_root_cause[:1000],
            "was_fix_applied": was_fix_applied,
            "did_fix_work": did_fix_work,
        }

        self.traces.append(trace)
        self._append_to_file(trace)

        logger.info(f"  📝 Trace recorded: rca_failure (iter {iteration})")

    def record_sampling_result(
        self,
        filename: str,
        total_candidates: int,
        wave1_results: list,
        wave2_results: Optional[list] = None,
        winner: Optional[int] = None,
        task_goal: str = "",
    ):
        """
        Record the full sampling trajectory for a test file.

        Captures all candidates and their results, which wave
        produced the winner (if any), and all error outputs.
        This data drives the failure pattern classifier.
        """
        trace = {
            "type": "sampling_result",
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "task_goal": task_goal[:500],
            "total_candidates": total_candidates,
            "wave1_results": wave1_results,
            "wave2_results": wave2_results,
            "winner": winner,
            "all_failed": winner is None,
        }

        self.traces.append(trace)
        self._append_to_file(trace)

    def _append_to_file(self, trace: dict):
        """Append a trace to the JSONL file."""
        try:
            with open(self.traces_file, "a") as f:
                f.write(json.dumps(trace, default=str) + "\n")
        except Exception as e:
            logger.debug(f"Failed to write trace: {e}")

    def get_session_stats(self) -> dict:
        """Get statistics for the current session's traces."""
        stats = {
            "total": len(self.traces),
            "build_failures": sum(1 for t in self.traces if t["type"] == "build_failure"),
            "test_failures": sum(1 for t in self.traces if t["type"] == "test_failure"),
            "rca_failures": sum(1 for t in self.traces if t["type"] == "rca_failure"),
            "sampling_results": sum(1 for t in self.traces if t["type"] == "sampling_result"),
        }
        return stats

    def export_for_training(self, output_path: Optional[str] = None) -> str:
        """
        Export all traces as JSONL for training pipeline.

        Format: one JSON object per line, ready for LoRA fine-tuning
        data processing scripts.
        """
        output = output_path or str(self.traces_dir / "training_traces.jsonl")

        # Load all traces from the file
        all_traces = []
        if self.traces_file.exists():
            with open(self.traces_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            all_traces.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        with open(output, "w") as f:
            for trace in all_traces:
                f.write(json.dumps(trace, default=str) + "\n")

        logger.info(f"📦 Exported {len(all_traces)} traces to {output}")
        return output

    def export_for_claude(self, output_path: Optional[str] = None, max_traces: int = 20) -> str:
        """
        Export traces as a markdown document optimized for pasting
        to Claude/GPT-4 to generate reasoning traces.

        This is the key step in the co-training loop:
        1. Run orchestrator → failures collected here
        2. Export this markdown
        3. Paste to Claude: "Generate correct code with reasoning for each"
        4. Claude returns reasoning traces
        5. Format as training data → LoRA fine-tune

        Output format is designed to be copy-pasteable into a chat.
        """
        output = output_path or str(self.traces_dir / "for_claude.md")

        # Load traces
        all_traces = []
        if self.traces_file.exists():
            with open(self.traces_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            all_traces.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        # Focus on test failures — highest value for LoRA training
        test_failures = [t for t in all_traces if t["type"] == "test_failure"]
        build_failures = [t for t in all_traces if t["type"] == "build_failure"]

        lines = [
            "# Failure Traces for Reasoning Distillation",
            "",
            f"Generated: {datetime.now().isoformat()}",
            f"Total traces: {len(all_traces)}",
            f"Test failures: {len(test_failures)}",
            f"Build failures: {len(build_failures)}",
            "",
            "---",
            "",
            "## Instructions for Claude/GPT-4",
            "",
            "For each failure below, generate:",
            "1. The CORRECT code that would pass all tests",
            "2. A step-by-step REASONING TRACE explaining your thought process",
            "3. What the model got WRONG and WHY",
            "",
            "Format each response as:",
            "```",
            "### Trace N: [filename]",
            "",
            "**What went wrong:** [1-2 sentences]",
            "",
            "**Reasoning trace:**",
            "1. First, I read the source module to understand the API...",
            "2. For argparse testing, I need to mock sys.argv...",
            "3. I need to capture stdout because the CLI prints results...",
            "...",
            "",
            "**Correct code:**",
            "```python",
            "# ... correct implementation",
            "```",
            "```",
            "",
            "---",
            "",
        ]

        # Emit test failures (highest priority)
        trace_num = 0
        for trace in test_failures[:max_traces]:
            trace_num += 1
            lines.extend([
                f"## Failure {trace_num}: {trace.get('test_filename', 'unknown')}",
                "",
                f"**Task:** {trace.get('task_goal', 'unknown')[:200]}",
                f"**Source module:** {trace.get('source_filename', 'unknown')}",
                f"**Model:** {trace.get('model_used', 'unknown')} (temp={trace.get('temperature', '?')})",
                f"**Result:** {trace.get('test_stats', {})}",
                f"**Candidate:** {trace.get('candidate_number', '?')}/{trace.get('total_candidates', '?')}",
                "",
                "**Source code being tested:**",
                "```python",
                trace.get("source_code", "# not captured")[:1500],
                "```",
                "",
                "**Generated test code (WRONG):**",
                "```python",
                trace.get("generated_test_code", "# not captured")[:2000],
                "```",
                "",
                "**Test output / error:**",
                "```",
                trace.get("test_output", "# not captured")[:800],
                "```",
                "",
                "---",
                "",
            ])

        # Emit build failures
        for trace in build_failures[:max(0, max_traces - trace_num)]:
            trace_num += 1
            lines.extend([
                f"## Failure {trace_num}: {trace.get('filename', 'unknown')} (build)",
                "",
                f"**Task:** {trace.get('task_goal', 'unknown')[:200]}",
                f"**Error category:** {trace.get('error_category', 'unknown')}",
                f"**Model:** {trace.get('model_used', 'unknown')}",
                "",
                "**Generated code (WRONG):**",
                "```python",
                trace.get("generated_code", "# not captured")[:2000],
                "```",
                "",
                "**Error output:**",
                "```",
                trace.get("error_output", "# not captured")[:500],
                "```",
                "",
                "---",
                "",
            ])

        content = "\n".join(lines)
        with open(output, "w") as f:
            f.write(content)

        logger.info(f"📋 Exported {trace_num} traces for Claude to {output}")
        return output

    def export_training_pairs(self, output_path: Optional[str] = None) -> str:
        """
        Export traces that have been enriched with correct_code and
        reasoning_trace as training pairs for LoRA fine-tuning.

        Format: {"prompt": "...", "response": "...", "system": "..."}

        Only exports traces where correct_code has been filled in
        (i.e., after the Claude distillation step).
        """
        output = output_path or str(self.traces_dir / "lora_training_pairs.jsonl")

        all_traces = []
        if self.traces_file.exists():
            with open(self.traces_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trace = json.loads(line)
                            if trace.get("correct_code") and trace.get("reasoning_trace"):
                                all_traces.append(trace)
                        except json.JSONDecodeError:
                            pass

        pairs = []
        for trace in all_traces:
            if trace["type"] == "test_failure":
                pair = {
                    "system": (
                        "You are a Python test writer. You write tests that actually pass. "
                        "Think step by step before writing code. Consider imports, mocking, "
                        "cleanup, and edge cases."
                    ),
                    "prompt": (
                        f"Write a test file for the following module.\n\n"
                        f"Source code:\n```python\n{trace.get('source_code', '')[:2000]}\n```\n\n"
                        f"Requirements: {trace.get('task_goal', '')[:300]}"
                    ),
                    "response": (
                        f"{trace['reasoning_trace']}\n\n"
                        f"```python\n{trace['correct_code']}\n```"
                    ),
                }
                pairs.append(pair)

            elif trace["type"] == "build_failure":
                pair = {
                    "system": (
                        "You are a Python developer. Write clean, correct code. "
                        "Think step by step about imports, dependencies, and structure."
                    ),
                    "prompt": trace.get("prompt_excerpt", "")[:2000],
                    "response": (
                        f"{trace['reasoning_trace']}\n\n"
                        f"```python\n{trace['correct_code']}\n```"
                    ),
                }
                pairs.append(pair)

        with open(output, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        logger.info(f"🎓 Exported {len(pairs)} training pairs to {output}")
        return output
