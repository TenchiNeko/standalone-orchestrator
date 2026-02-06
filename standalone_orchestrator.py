"""
Standalone Orchestrator — Multi-Agent Execution Loop.

Implements: EXPLORE → PLAN → BUILD → TEST → DECISION GATE
with recursive self-correction on failures.

Zero dependency on opencode or any external CLI agent tools.
"""

import subprocess
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from typing import Optional
from dataclasses import asdict

from standalone_config import Config
from standalone_session import SessionManager
from standalone_models import TaskState, ExecutionPhase, DoD, IterationResult
from standalone_agents import AgentRunner

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Central orchestrator implementing the AGENTS.md execution loop.
    """

    def __init__(self, config: Config, working_dir: Path):
        self.config = config
        self.working_dir = working_dir
        self.session = SessionManager(working_dir)
        self.agent_runner = AgentRunner(config, working_dir)
        self.max_iterations = config.max_iterations

    def run(self, goal: str, resume: bool = False) -> bool:
        logger.info(f"{'='*60}")
        logger.info(f"ORCHESTRATOR STARTING")
        logger.info(f"Task: {goal}")
        logger.info(f"Working directory: {self.working_dir}")
        logger.info(f"Max iterations: {self.max_iterations}")
        logger.info(f"{'='*60}")

        if resume and self.session.has_existing_session():
            task_state = self.session.load_state()
            logger.info(f"Resumed session: iteration {task_state.iteration}")
        else:
            task_state = self._initialize_session(goal)
            logger.info(f"New session initialized: {task_state.task_id}")

        while task_state.iteration <= self.max_iterations:
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {task_state.iteration}")
            logger.info(f"{'='*60}")

            result = self._execute_iteration(task_state)

            if result.success:
                logger.info("✅ ALL DoD CRITERIA PASSED")
                self._finalize_success(task_state)
                return True

            if result.unrecoverable:
                logger.error(f"❌ UNRECOVERABLE ERROR: {result.error}")
                self._escalate(task_state, result.error)
                return False

            logger.warning(f"⚠️ DoD FAILED: {result.error}")
            task_state.add_failure(result)
            task_state.iteration += 1
            self.session.save_state(task_state)

            if task_state.iteration > self.max_iterations:
                logger.error(f"❌ MAX ITERATIONS ({self.max_iterations}) REACHED")
                self._escalate(task_state, "Max iterations exceeded")
                return False

            logger.info("Performing Root Cause Analysis...")
            self._perform_root_cause_analysis(task_state, result)

        return False

    def _initialize_session(self, goal: str) -> TaskState:
        task_id = str(uuid.uuid4())[:8]

        task_state = TaskState(
            task_id=task_id,
            goal=goal,
            iteration=1,
            phase=ExecutionPhase.EXPLORE,
            started_at=datetime.now().isoformat(),
        )

        logger.info("Running INITIALIZER phase...")
        init_result = self.agent_runner.run_initializer(goal, task_id)

        if not init_result.success:
            logger.warning(f"Initializer had issues: {init_result.error}")

        self.session.save_state(task_state)
        self.session.update_progress(task_state, "Session initialized")

        return task_state

    def _execute_iteration(self, task_state: TaskState) -> IterationResult:
        """Execute one full EXPLORE → PLAN → BUILD → TEST iteration."""

        # PHASE 1: EXPLORE (first iteration or after re-exploration flag)
        if task_state.iteration == 1 or task_state.needs_re_exploration:
            logger.info("\n📍 PHASE 1: EXPLORE")
            explore_result = self.agent_runner.run_explore(task_state)

            if not explore_result.success:
                return IterationResult(
                    success=False,
                    error=f"Exploration failed: {explore_result.error}",
                    phase=ExecutionPhase.EXPLORE
                )

            task_state.exploration_context = explore_result.output
            task_state.needs_re_exploration = False
            task_state.phase = ExecutionPhase.PLAN
            self.session.save_state(task_state)

        # PHASE 2: PLAN
        logger.info("\n📍 PHASE 2: PLAN")
        plan_result = self.agent_runner.run_plan(task_state)

        if not plan_result.success:
            return IterationResult(
                success=False,
                error=f"Planning failed: {plan_result.error}",
                phase=ExecutionPhase.PLAN
            )

        task_state.current_plan = plan_result.output
        task_state.dod = plan_result.dod
        task_state.phase = ExecutionPhase.BUILD
        self.session.save_state(task_state)

        # PHASE 3: BUILD
        logger.info("\n📍 PHASE 3: BUILD")
        backup_path = self._create_backup(task_state)
        build_result = self.agent_runner.run_build(task_state)

        if not build_result.success:
            return IterationResult(
                success=False,
                error=f"Build failed: {build_result.error}",
                phase=ExecutionPhase.BUILD
            )

        task_state.phase = ExecutionPhase.TEST
        self.session.save_state(task_state)

        # PHASE 4: TEST
        logger.info("\n📍 PHASE 4: TEST (VERIFY)")
        test_result = self.agent_runner.run_test(task_state)

        if not test_result.success:
            return IterationResult(
                success=False,
                error=f"Verification failed: {test_result.error}",
                phase=ExecutionPhase.TEST,
                dod_results=test_result.dod_results
            )

        return IterationResult(
            success=True,
            phase=ExecutionPhase.COMPLETE,
            dod_results=test_result.dod_results
        )

    def _perform_root_cause_analysis(self, task_state: TaskState, result: IterationResult):
        """Analyze why the last iteration failed and annotate state."""
        last_failure = task_state.failure_history[-1] if task_state.failure_history else {}
        phase = last_failure.get("phase", "unknown")
        error = last_failure.get("error", "unknown")

        rca = f"Phase '{phase}' failed: {error}"

        # Simple heuristics for common failure patterns
        if "timeout" in error.lower():
            rca += " | RCA: Agent took too long. Consider simplifying the plan or increasing timeout."
            task_state.needs_re_exploration = True
        elif "connection" in error.lower():
            rca += " | RCA: Cannot reach LLM endpoint. Check Ollama is running."
        elif "0/" in error:
            rca += " | RCA: No DoD criteria passed. Build agent may not have executed tools. Check model tool-use capability."
        elif "dod criteria passed" in error.lower():
            rca += " | RCA: Partial DoD pass. Review failed criteria and focus plan on those."

        # Store RCA in last failure entry
        if task_state.failure_history:
            task_state.failure_history[-1]["rca"] = rca

        logger.info(f"RCA: {rca}")

    def _finalize_success(self, task_state: TaskState):
        task_state.phase = ExecutionPhase.COMPLETE
        task_state.completed_at = datetime.now().isoformat()

        self.session.save_state(task_state)
        self.session.update_progress(
            task_state,
            f"✅ TASK COMPLETED SUCCESSFULLY after {task_state.iteration} iteration(s)"
        )

        self.session.mark_feature_complete(task_state.task_id)
        self._git_commit(f"feat: complete task {task_state.task_id} - {task_state.goal[:50]}")

        logger.info(f"\n{'='*60}")
        logger.info("🎉 TASK COMPLETED SUCCESSFULLY")
        logger.info(f"Task ID: {task_state.task_id}")
        logger.info(f"Iterations: {task_state.iteration}")
        logger.info(f"Duration: {task_state.started_at} → {task_state.completed_at}")
        logger.info(f"{'='*60}")

    def _escalate(self, task_state: TaskState, reason: str):
        task_state.phase = ExecutionPhase.ESCALATED
        task_state.escalation_reason = reason

        self.session.save_state(task_state)

        handoff_path = self.working_dir / f".agents/reports/handoff-{task_state.task_id}.md"
        handoff_path.parent.mkdir(parents=True, exist_ok=True)
        handoff_path.write_text(self._generate_handoff_report(task_state, reason))

        self.session.update_progress(
            task_state,
            f"❌ ESCALATED TO HUMAN: {reason}\nHandoff report: {handoff_path}"
        )

        logger.error(f"\n{'='*60}")
        logger.error("🚨 TASK ESCALATED TO HUMAN")
        logger.error(f"Reason: {reason}")
        logger.error(f"Handoff report: {handoff_path}")
        logger.error(f"{'='*60}")

    def _generate_handoff_report(self, task_state: TaskState, reason: str) -> str:
        failures_table = ""
        if task_state.failure_history:
            failures_table = "| Iteration | Phase | Error | RCA |\n|---|---|---|---|\n"
            for f in task_state.failure_history:
                failures_table += f"| {f.get('iteration', '?')} | {f.get('phase', '?')} | {(f.get('error') or '?')[:80]} | {(f.get('rca') or 'N/A')[:80]} |\n"

        return f"""# Handoff Report — {task_state.task_id}

**Generated:** {datetime.now().isoformat()}
**Escalation Reason:** {reason}

## Task Summary
**Goal:** {task_state.goal}
**Iterations:** {task_state.iteration}
**Final Phase:** {task_state.phase.value}

## Failure History
{failures_table or "No recorded failures."}

## Current Plan
```
{(task_state.current_plan or 'No plan')[:1000]}
```

## DoD Status
{task_state.dod.to_markdown() if task_state.dod else 'No DoD recorded'}

## Resume Instructions
```bash
cd {self.working_dir}
python3 standalone_main.py --resume
```
"""

    def _create_backup(self, task_state: TaskState) -> Optional[Path]:
        """
        Create a timestamped backup of the working directory before build phase.

        Backups are stored in .agents/backups/ and can be restored if the
        build agent damages the project.
        """
        backup_dir = self.working_dir / ".agents" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"iter{task_state.iteration}_{timestamp}"
        backup_path = backup_dir / backup_name

        try:
            import shutil

            # Copy everything except .agents/backups to avoid recursive backup
            def ignore_backups(directory, contents):
                if Path(directory) == backup_dir.parent:
                    return ["backups"]
                # Also skip .git internals (large, and we have git history anyway)
                if Path(directory).name == ".git":
                    return contents
                return []

            shutil.copytree(
                self.working_dir,
                backup_path,
                ignore=ignore_backups,
                dirs_exist_ok=False,
            )

            logger.info(f"📦 Backup created: .agents/backups/{backup_name}")

            # Prune old backups — keep only last 5
            backups = sorted(backup_dir.iterdir(), key=lambda p: p.name)
            while len(backups) > 5:
                old = backups.pop(0)
                shutil.rmtree(old)
                logger.debug(f"  Pruned old backup: {old.name}")

            return backup_path

        except Exception as e:
            logger.warning(f"⚠️ Backup failed (continuing anyway): {e}")
            return None

    def restore_backup(self, backup_name: str = None):
        """
        Restore from the most recent backup (or a specific one).

        Usage from CLI:
            python3 -c "
            from standalone_orchestrator import Orchestrator
            from standalone_config import load_config
            from pathlib import Path
            o = Orchestrator(load_config(), Path('.'))
            o.restore_backup()
            "
        """
        import shutil

        backup_dir = self.working_dir / ".agents" / "backups"
        if not backup_dir.exists():
            logger.error("No backups directory found")
            return False

        if backup_name:
            backup_path = backup_dir / backup_name
        else:
            # Use most recent
            backups = sorted(backup_dir.iterdir(), key=lambda p: p.name)
            if not backups:
                logger.error("No backups found")
                return False
            backup_path = backups[-1]

        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            return False

        logger.info(f"🔄 Restoring from backup: {backup_path.name}")

        # Restore files from backup (skip .agents dir itself)
        for item in backup_path.iterdir():
            if item.name == ".agents":
                continue
            dest = self.working_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        logger.info("✅ Restore complete")
        return True

    def _git_commit(self, message: str):
        """Run git add -A && git commit with the given message."""
        try:
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.working_dir, capture_output=True, timeout=30
            )
            subprocess.run(
                ["git", "commit", "-m", message, "--allow-empty"],
                cwd=self.working_dir, capture_output=True, timeout=30
            )
        except Exception as e:
            logger.warning(f"Git commit failed: {e}")
