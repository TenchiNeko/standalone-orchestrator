"""
Standalone Orchestrator — Multi-Agent Execution Loop.

Implements: EXPLORE → PLAN → BUILD → TEST → DECISION GATE
with recursive self-correction on failures.

Zero dependency on opencode or any external CLI agent tools.
"""

import subprocess
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from standalone_config import Config
from standalone_session import SessionManager
from standalone_models import TaskState, ExecutionPhase, IterationResult, AgentResult
from standalone_agents import AgentRunner
from standalone_memory import ConversationMemory
from standalone_trace_collector import TraceCollector

# v1.0: Subconscious playbook integration
try:
    from playbook_reader import PlaybookReader
    _HAS_PLAYBOOK = True
except ImportError:
    _HAS_PLAYBOOK = False
from kb_client import KBClient
from librarian import Librarian, build_session_summary
from librarian_store import init_librarian_tables, get_librarian_stats, get_session_context

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
        self.memory = ConversationMemory(
            memory_file=working_dir / ".agents" / "memory.json"
        )
        self.trace_collector = TraceCollector(working_dir)
        # v0.8.0: RAG Knowledge Base client
        self.kb = KBClient(server_url=getattr(config, 'kb_url', 'http://localhost:8787'))

        # v0.9.0: Librarian — 7B knowledge curator (post-session learning)
        self.librarian = None
        self.librarian_db_path = None
        if hasattr(config, 'librarian_model') and config.librarian_model:
            try:
                # v0.9.9c: Use persistent path so journal entries survive across
                # benchmark runs. Previous behavior put the DB in working_dir,
                # meaning every fresh --working-dir started with empty knowledge.
                _default_db = str(Path(__file__).parent / 'knowledge_base.db')
                self.librarian_db_path = getattr(config, '_kb_db_path', _default_db)
                self.librarian = Librarian(
                    model_config=config.librarian_model,
                    db_path=self.librarian_db_path,
                    kb_server_url=getattr(config, 'kb_url', 'http://localhost:8787'),
                )
                logger.info("Librarian: ✅ initialized (post-session curation enabled)")
            except Exception as e:
                logger.warning(f"Librarian: ⚠️ init failed: {e} (running without librarian)")

        # v1.0: Subconscious playbook reader
        self.playbook_reader = None
        if _HAS_PLAYBOOK:
            try:
                _playbook_path = str(Path(__file__).parent / 'playbook.json')
                self.playbook_reader = PlaybookReader(_playbook_path)
                _stats = self.playbook_reader.stats
                if _stats.get('available'):
                    logger.info(f"Playbook: ✅ {_stats['total_bullets']} bullets loaded")
                else:
                    logger.info("Playbook: 📝 no playbook.json yet (will be created by subconscious daemon)")
            except Exception as e:
                logger.warning(f"Playbook: ⚠️ init failed: {e}")

    def _get_playbook_context(self, role: str = "builder", task_goal: str = "") -> str:
        """Get playbook context for injection into agent prompts."""
        if not self.playbook_reader:
            return ""
        try:
            ctx = self.playbook_reader.get_context_for_agent(role, task_goal)
            if ctx:
                logger.debug(f"Playbook: injecting {len(ctx)} chars for {role}")
            return ctx
        except Exception as e:
            logger.debug(f"Playbook context error: {e}")
            return ""

    def _report_playbook_feedback(self, was_successful: bool):
        """Report session outcome to playbook for quality tracking."""
        if not self.playbook_reader:
            return
        try:
            # Report all bullet IDs that were in context
            playbook_data = self.playbook_reader._load()
            if playbook_data:
                all_ids = []
                for bullets in playbook_data.get("sections", {}).values():
                    for b in bullets:
                        all_ids.append(b.get("id", ""))
                if all_ids:
                    self.playbook_reader.report_bullet_usage(all_ids, was_successful)
                    logger.info(f"📊 Playbook feedback: {'✅' if was_successful else '❌'} for {len(all_ids)} bullets")
        except Exception as e:
            logger.debug(f"Playbook feedback error: {e}")

    def _sync_session_to_pve(self):
        """Auto-sync completed session to PVE node for subconscious analysis."""
        sync_script = Path(__file__).parent / 'sync-session.sh'
        if not sync_script.exists():
            return
        try:
            import subprocess
            result = subprocess.run(
                ['bash', str(sync_script), str(self.working_dir)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                logger.info("📤 Session synced to PVE node for subconscious analysis")
            else:
                logger.debug(f"Session sync failed: {result.stderr[:200]}")
        except Exception as e:
            logger.debug(f"Session sync error: {e}")

    @staticmethod
    def _safe_run(cmd, **kwargs):
        """subprocess.run wrapper that catches TimeoutExpired instead of crashing."""
        try:
            return subprocess.run(cmd, **kwargs)
        except subprocess.TimeoutExpired:
            # Return a fake CompletedProcess with failure
            return subprocess.CompletedProcess(
                args=cmd, returncode=1,
                stdout="", stderr="TIMEOUT: process killed after timeout"
            )

    def run(self, goal: str, resume: bool = False) -> bool:
        logger.info(f"{'='*60}")
        logger.info("ORCHESTRATOR STARTING")
        logger.info(f"Task: {goal}")
        logger.info(f"Working directory: {self.working_dir}")
        logger.info(f"Max iterations: {self.max_iterations}")
        # v0.8.0: KB status
        if self.kb.is_available():
            kb_stats = self.kb.get_stats()
            if kb_stats:
                logger.info(f"Knowledge Base: ✅ {kb_stats.get('patterns', 0)} patterns, "
                           f"{kb_stats.get('docs', 0)} doc chunks")
        else:
            logger.info("Knowledge Base: ⚠️ not available (running without KB)")
        # v0.9.0: Librarian status
        if self.librarian:
            try:
                lib_stats = get_librarian_stats(db_path=self.librarian_db_path or "")
                logger.info(f"Librarian: ✅ {lib_stats.get('journal_entries', 0)} journal entries, "
                           f"{lib_stats.get('snippets', 0)} snippets")
            except Exception:
                logger.info("Librarian: ✅ ready (no stats available)")
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
                # Record success in memory
                self.memory.add_iteration(
                    iteration=task_state.iteration,
                    phase_reached="complete",
                    success=True,
                    dod_results=result.dod_results or {},
                )
                logger.info("✅ ALL DoD CRITERIA PASSED")
                self._export_traces()
                self._finalize_success(task_state)
                return True

            if result.unrecoverable:
                logger.error(f"❌ UNRECOVERABLE ERROR: {result.error}")
                self._export_traces()
                self._escalate(task_state, result.error or "Unknown error")
                return False

            logger.warning(f"⚠️ DoD FAILED: {result.error}")
            task_state.add_failure(result)
            task_state.iteration += 1
            self.session.save_state(task_state)

            if task_state.iteration > self.max_iterations:
                logger.error(f"❌ MAX ITERATIONS ({self.max_iterations}) REACHED")
                self._export_traces()
                self._escalate(task_state, "Max iterations exceeded")
                return False

            logger.info("Performing Root Cause Analysis...")
            self._perform_root_cause_analysis(task_state, result)

            # Record failure in memory (after RCA so we capture it)
            rca_text = ""
            if task_state.failure_history:
                rca_text = task_state.failure_history[-1].get("rca", "")

            # Use structured test report if available, fall back to legacy dod_results
            dod_data = {}
            if result.dod_results:
                if isinstance(result.dod_results, dict) and "criteria_results" in result.dod_results:
                    # New structured format — pass it through
                    dod_data = result.dod_results
                else:
                    dod_data = result.dod_results

            self.memory.add_iteration(
                iteration=task_state.iteration - 1,  # The iteration that just failed
                phase_reached=result.phase.value if result.phase else "unknown",
                success=False,
                errors=[result.error or "unknown"],
                dod_results=dod_data,
                rca=rca_text,
                plan_summary=(task_state.current_plan or "")[:300],
            )

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
        """Execute one full EXPLORE → PLAN → BUILD → TEST iteration.

        v0.6: For multi-file tasks, BUILD phase decomposes into sequential
        micro-builds — one file at a time with verification between each.
        Single-file tasks use the original monolithic build.
        """

        # PHASE 1: EXPLORE (first iteration or after re-exploration flag)
        if task_state.iteration == 1 or task_state.needs_re_exploration:
            logger.info("\n📍 PHASE 1: EXPLORE")
            explore_result = self.agent_runner.run_explore(task_state)

            if not explore_result.success:
                # v0.8.0: Greenfield projects have nothing to explore.
                # Instead of failing the iteration, provide a default context
                # and proceed to PLAN. The task description IS the context.
                existing_files = [f.name for f in self.working_dir.iterdir()
                                  if f.is_file() and f.suffix == '.py']
                if not existing_files:
                    logger.info("  📋 Greenfield project — no source files to explore, proceeding to PLAN")
                    task_state.exploration_context = (
                        f"Greenfield project. No existing source files. "
                        f"Task: {task_state.goal} "
                        f"Working directory: {self.working_dir} "
                        f"This is a new project — all files need to be created from scratch."
                    )
                else:
                    return IterationResult(
                        success=False,
                        error=f"Exploration failed: {explore_result.error}",
                        phase=ExecutionPhase.EXPLORE
                    )
            else:
                task_state.exploration_context = explore_result.output
            task_state.needs_re_exploration = False
            task_state.phase = ExecutionPhase.PLAN
            self.session.save_state(task_state)

        # PHASE 2: PLAN
        logger.info("\n📍 PHASE 2: PLAN")

        # v0.9.1: Inject librarian context (journal lessons + code snippets)
        # PREPEND so truncate_to_budget preserves it (head-biased). Exploration report
        # gets trimmed (it's regenerable) while librarian lessons (curated) survive.
        try:
            lib_ctx = get_session_context(task_state.goal, db_path=self.librarian_db_path or str(Path(__file__).parent / 'knowledge_base.db'))
            if lib_ctx:
                task_state.exploration_context = lib_ctx + "\n" + (task_state.exploration_context or "")
                logger.info(f"  🧠 Librarian injected {len(lib_ctx)} chars of strategic context")
        except Exception as e:
            logger.debug(f"  Librarian context retrieval failed (non-fatal): {e}")

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

        # PHASE 3: BUILD — choose strategy based on task complexity
        logger.info("\n📍 PHASE 3: BUILD")
        backup_path = self._create_backup(task_state)

        # Decompose plan into file-level build sequence
        build_sequence = self._decompose_build_sequence(task_state)

        if len(build_sequence) > 1:
            # v0.7.3: ALWAYS use micro-builds for multi-file tasks (was iteration==1 only).
            # Monolithic build on retries was accomplishing nothing — the model got a
            # giant prompt, ran for 18 seconds, and the snapshot rolled back its changes.
            # Micro-builds are focused, one file at a time, with sampling and verification.
            is_retry = task_state.iteration > 1
            if is_retry:
                logger.info(f"  📦 Retry micro-build (iter {task_state.iteration}): {len(build_sequence)} files")
                # v0.7.3: On retry, inject RCA context and memory into micro-builds
                memory_context = self.memory.get_context(last_n=3, total_budget=6000)
                rca_edits_context = self._get_rca_edits_for_micro_build(task_state)
                build_result = self._run_micro_builds(
                    task_state, build_sequence,
                    memory_context=memory_context,
                    rca_edits_context=rca_edits_context,
                )
            else:
                logger.info(f"  📦 Sequential micro-build: {len(build_sequence)} files")
                build_result = self._run_micro_builds(task_state, build_sequence)
        else:
            # Retry iteration: snapshot passing files, build, then rollback regressions
            # v0.6.2 Fix 1: Snapshot Protection (Augment/Verdent pattern)
            passing_snapshots = self._snapshot_passing_files(task_state)

            memory_context = self.memory.get_context(last_n=3, total_budget=6000)
            build_result = self.agent_runner.run_build(task_state, memory_context=memory_context)

            # After build, check if any previously passing files now fail
            if passing_snapshots:
                self._rollback_regressions(passing_snapshots)

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

    def _decompose_build_sequence(self, task_state: TaskState) -> list:
        """
        Decompose the plan into an ordered sequence of single-file builds.

        Parses the plan text to extract files and their dependencies,
        then orders them so that dependencies are built first.

        v0.9.9c: On retry iterations, uses the FULL workspace file list instead
        of only what the plan mentions. This prevents cascade failures where
        rebuilding a source file (e.g., database.py) doesn't trigger dependent
        source files (e.g., app.py) or dependent tests (e.g., test_database.py)
        to be re-evaluated. The skip logic in _run_micro_builds efficiently
        skips files that don't need rebuilding by checking test results,
        source rebuild status, and RCA targets.

        Returns list of dicts: [{file, description, depends_on}, ...]
        """
        plan = task_state.current_plan or ""
        goal = task_state.goal or ""
        is_retry = task_state.iteration > 1

        import re

        # v0.9.9c: On retry, use full workspace file list.
        # The skip logic handles what actually needs rebuilding — but ALL files
        # must be IN the sequence to be evaluated. Without this, rebuilding
        # database.py leaves app.py (which imports from database.py) stale,
        # and test_database.py never gets re-run against the new API.
        if is_retry:
            files_mentioned = self._scan_workspace_project_files()
            if files_mentioned:
                # v1.0: Build import graph for precise cascade tracking
                import_graph = self._build_import_graph()
                rca_targets = set()
                for rca in (task_state.rca_history or []):
                    if isinstance(rca, dict):
                        for tf in rca.get("target_files", []):
                            rca_targets.add(tf)
                # Log which files will cascade from RCA targets
                cascade_files = set()
                for target in rca_targets:
                    deps = self._get_dependents(target, import_graph)
                    cascade_files.update(deps)
                if cascade_files:
                    logger.info(f"  🔄 Retry: {len(files_mentioned)} files, cascade from RCA: {sorted(cascade_files)}")
                else:
                    logger.info(f"  🔄 Retry: full workspace sequence ({len(files_mentioned)} files)")

        if not is_retry or not files_mentioned:
            # Extract files from plan steps (format: "1. Description (files: a.py, b.py)")
            files_mentioned = []
            for match in re.finditer(r'\(files?:\s*([^)]+)\)', plan):
                for f in match.group(1).split(','):
                    f = f.strip()
                    if f and f.endswith('.py'):
                        if f not in files_mentioned:
                            files_mentioned.append(f)

            # If plan doesn't have file annotations, try to extract from goal
            if not files_mentioned:
                # Look for .py filenames in the goal
                for match in re.finditer(r'(\w+\.py)', goal):
                    f = match.group(1)
                    if f not in files_mentioned:
                        files_mentioned.append(f)

        if len(files_mentioned) <= 1:
            # Single file or can't decompose — return single monolithic build
            return [{"file": "all", "description": "Full build"}]

        # Separate source files and test files
        source_files = [f for f in files_mentioned if not f.startswith('test_')]
        test_files = [f for f in files_mentioned if f.startswith('test_')]

        # Build the sequence: source files first (in dependency order), then test files
        # Simple heuristic: models/data files first, then logic, then CLI/main, then tests
        priority_keywords = {
            'model': 0, 'models': 0, 'data': 0, 'schema': 0, 'base': 0,
            'storage': 1, 'store': 1, 'db': 1, 'database': 1, 'repo': 1,
            'service': 2, 'logic': 2, 'utils': 2, 'helper': 2,
            'cli': 3, 'main': 3, 'app': 3, 'api': 3, 'server': 3,
        }

        def file_priority(filename):
            name = filename.replace('.py', '').lower()
            for keyword, priority in priority_keywords.items():
                if keyword in name:
                    return priority
            return 2  # default middle priority

        source_files.sort(key=file_priority)

        # Build the sequence with dependency tracking
        sequence = []
        built_so_far: List[str] = []

        for f in source_files:
            sequence.append({
                "file": f,
                "description": f"Create source module: {f}",
                "depends_on": list(built_so_far),
                "is_test": False,
            })
            built_so_far.append(f)

        # v0.9.6: Per-source test splitting (ClassEval/HITS research).
        # One test file covering N source files → N small test files.
        # Each ~40 lines instead of 200+, dramatically better for local models.
        if len(test_files) == 1 and len(source_files) > 1:
            monolithic_test = test_files[0]
            logger.info(f"  \U0001F4D0 Splitting {monolithic_test} into {len(source_files)} per-source test files")
            for sf in source_files:
                source_base = sf.replace('.py', '')
                per_test = f"test_{source_base}.py"
                sequence.append({
                    "file": per_test,
                    "description": f"Create focused tests for {sf} (split from {monolithic_test})",
                    "depends_on": list(built_so_far),
                    "is_test": True,
                    "tests_for": source_base,
                })
        else:
            for f in test_files:
                source_name = f.replace('test_', '')
                sequence.append({
                    "file": f,
                    "description": f"Create test module: {f}",
                    "depends_on": list(built_so_far),
                    "is_test": True,
                    "tests_for": source_name if source_name in source_files else None,
                })

        logger.info(f"  Build sequence: {' → '.join(s['file'] for s in sequence)}")
        return sequence

    def _scan_workspace_project_files(self) -> list:
        """
        v0.9.9c: Scan workspace for all project .py files.

        Returns a flat list of filenames (no paths) excluding __init__.py,
        venv, __pycache__, and .agents directories. Used on retry iterations
        to ensure the full dependency graph is available for the skip logic.
        """
        files = []
        for py_file in sorted(self.working_dir.glob("*.py")):
            name = py_file.name
            if name == "__init__.py" or name.startswith('.'):
                continue
            files.append(name)

        # Also check one level of subdirectories (but not venv/.agents)
        for subdir in sorted(self.working_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if subdir.name in ('venv', '__pycache__', '.agents', 'node_modules', '.git'):
                continue
            for py_file in sorted(subdir.glob("*.py")):
                if py_file.name == "__init__.py":
                    continue
                rel = f"{subdir.name}/{py_file.name}"
                files.append(rel)

        return files

    def _build_import_graph(self) -> dict:
        """
        v1.0: Build a dependency graph from actual import statements.

        Parses all project .py files using AST and extracts import relationships.
        Returns dict mapping filename -> set of module names it imports from.
        Example: {'app.py': {'database', 'models', 'validators'}}
        """
        import ast
        graph = {}
        project_modules = set()

        # Pass 1: identify all project module names
        for py_file in self.working_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            project_modules.add(py_file.stem)

        # Pass 2: parse imports
        for py_file in self.working_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            deps = set()
            try:
                content = py_file.read_text()
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            mod = alias.name.split('.')[0]
                            if mod in project_modules:
                                deps.add(mod)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            mod = node.module.split('.')[0]
                            if mod in project_modules:
                                deps.add(mod)
            except Exception:
                pass
            graph[py_file.name] = deps

        return graph

    def _get_dependents(self, changed_file: str, import_graph: dict) -> set:
        """
        v1.0: Find all files that directly or transitively depend on changed_file.
        Walks the import graph to find cascade dependencies.
        """
        module_name = changed_file.replace('.py', '')
        dependents = set()
        to_check = {module_name}
        checked = set()

        while to_check:
            current = to_check.pop()
            if current in checked:
                continue
            checked.add(current)

            for fname, deps in import_graph.items():
                if current in deps and fname not in dependents:
                    dependents.add(fname)
                    to_check.add(fname.replace('.py', ''))

        return dependents

    def _run_micro_builds(self, task_state: TaskState, sequence: list,
                          memory_context: str = "", rca_edits_context: str = "") -> AgentResult:
        """
        Execute sequential micro-builds: one file at a time with verification.

        v0.6.0: One brick at a time — focused single-file builds.
        v0.6.1: Multi-patch sampling for test files — generate N candidates,
                 run tests, pick the first one that passes.
        v0.7.2: Best-of-N sampling for source files too.
        v0.7.3: Also used on retry iterations with RCA context injection.
        """
        from standalone_models import AgentResult

        manifest = {}  # {filename: {exists: bool, exports: str, status: str}}
        errors = []
        is_retry = task_state.iteration > 1

        # v0.7.3: On retry, identify which files the RCA says need fixing
        rca_target_files = set()
        if is_retry and task_state.failure_history:
            last_rca = task_state.failure_history[-1].get("rca_data", {})
            for edit in last_rca.get("concrete_edits", []):
                rca_target_files.add(edit.get("file", ""))

        # v0.7.4: Track which source files were actually rebuilt this iteration.
        # When a source file changes, its dependent test file MUST be re-run
        # even if the test file itself wasn't in the RCA target list.
        rebuilt_source_files = set()

        # v0.9.5: Source-first test generation (Agentless pattern).
        # Research finding: every frontier coding agent that scores well generates
        # tests AFTER source, not before. Pre-generated tests embed wrong assumptions
        # about imports/structure before source exists → 100% stale rate.
        # Tests are now built in natural sequence order (source files first, then tests)
        # so test gen agents can see actual source code, imports, and function signatures.

        for i, step in enumerate(sequence):
            filename = step["file"]
            is_test = step.get("is_test", False)
            tests_for = step.get("tests_for", "")
            filepath = self.working_dir / filename

            logger.info(f"\n  🧱 Micro-build {i+1}/{len(sequence)}: {filename}")

            # v0.7.4: Dependent test invalidation — if the source file this test
            # depends on was rebuilt, we MUST re-run the test even if it previously
            # passed. The API contract may have changed (e.g., save_tasks() vs
            # save_tasks(tasks_list)), making the old passing test invalid.
            source_was_rebuilt = False
            if is_test and tests_for:
                source_name = tests_for if tests_for.endswith('.py') else tests_for + '.py'
                if source_name in rebuilt_source_files:
                    source_was_rebuilt = True
                    logger.info(f"  🔄 {filename} — source {source_name} was rebuilt, must re-test")

            # v0.7.3: On retry, skip files that already pass verification
            # v0.7.4: BUT never skip if the source dependency was rebuilt
            # v0.9.2: Source files ALSO check dependent test results before skipping.
            #   Research insight (Agentless, SWE-agent, Aider): compile/import status
            #   is a pre-filter, NOT a completion signal. Test execution is the sole
            #   "done" signal. Files that compile but have missing methods must be rebuilt.
            blamed_files = getattr(task_state, 'blamed_source_files', []) or []
            if is_retry and filepath.exists() and filename not in rca_target_files and not source_was_rebuilt and filename not in blamed_files:
                existing_verification = self._verify_single_file(filename, is_test)
                if existing_verification["status"] == "OK":
                    if is_test:
                        # Test files: also check if tests actually pass
                        test_result = self._run_test_file(filename, step.get("tests_for", ""))
                        if test_result.get("passed", 0) > 0 and test_result.get("errors", 0) == 0 and test_result.get("failed", 0) == 0:
                            logger.info(f"  ⏭️ {filename} — already passes, skipping")
                            manifest[filename] = existing_verification
                            continue
                    else:
                        # v0.9.2: Source files — check dependent tests before skipping.
                        # A file that compiles but has missing/broken methods will cause
                        # downstream test failures. Only skip if dependent tests pass.
                        source_base = filename.replace('.py', '')
                        dependent_tests_pass = True
                        dependent_test_checked = False
                        for other_step in sequence:
                            if other_step.get("tests_for") == source_base:
                                dep_test = other_step["file"]
                                dep_path = self.working_dir / dep_test
                                if dep_path.exists():
                                    dep_result = self._run_test_file(dep_test, source_base)
                                    dependent_test_checked = True
                                    if dep_result.get("failed", 0) > 0 or dep_result.get("errors", 0) > 0:
                                        dependent_tests_pass = False
                                        logger.info(f"  🔄 {filename} — compiles OK but dependent test {dep_test} failing, must rebuild")
                                        break
                        if dependent_tests_pass:
                            logger.info(f"  ⏭️ {filename} — verified OK{' (dependent tests pass)' if dependent_test_checked else ''}, skipping")
                            manifest[filename] = existing_verification
                            continue

            # Snapshot content before build for change detection
            old_content = filepath.read_text() if filepath.exists() else None

            # Build manifest context for this step
            manifest_text = self._format_manifest(manifest)

            # v0.7.3: On retry, inject RCA context as error_context
            retry_error_context = ""
            if is_retry and rca_edits_context and filename in rca_target_files:
                retry_error_context = rca_edits_context

            if is_test:
                # Multi-patch sampling for test files
                result, verification = self._sample_test_file(
                    task_state=task_state,
                    step=step,
                    manifest_text=manifest_text,
                    step_number=i + 1,
                    total_steps=len(sequence),
                )
                manifest[filename] = verification

                if verification["status"] == "OK":
                    logger.info(f"  ✅ {filename} — verified OK (tests pass)")
                else:
                    # v0.9.4: Apply stdlib import fix to test files too
                    if filepath.exists():
                        auto_imported = self._auto_fix_stdlib_imports(filepath)
                        if auto_imported:
                            re_verify = self._verify_single_file(filename, True)
                            if re_verify["status"] == "OK":
                                manifest[filename] = re_verify
                                verification = re_verify
                                logger.info(f"  ✅ {filename} — NOW verified OK after auto-import fix")
                            else:
                                logger.warning(f"  ⚠️ {filename} — {verification['status']}")
                                errors.append(f"{filename}: {verification['status']}")
                        else:
                            logger.warning(f"  ⚠️ {filename} — {verification['status']}")
                            errors.append(f"{filename}: {verification['status']}")
                    else:
                        logger.warning(f"  ⚠️ {filename} — {verification['status']}")
                        errors.append(f"{filename}: {verification['status']}")
            else:
                # v0.7.2: Source file sampling — best-of-N like test files
                result, verification = self._sample_source_file(
                    task_state=task_state,
                    step=step,
                    manifest_text=manifest_text,
                    step_number=i + 1,
                    total_steps=len(sequence),
                )
                manifest[filename] = verification

                if verification["status"] == "OK":
                    logger.info(f"  ✅ {filename} — verified OK")
                else:
                    logger.warning(f"  ⚠️ {filename} — {verification['status']}")
                    errors.append(f"{filename}: {verification['status']}")

                # v0.8.4: Deterministic stdlib auto-import BEFORE change detection
                # Catches missing `import datetime`, `import json`, etc. that cause
                # runtime NameError. Zero LLM cost, 100% fix rate for this bug class.
                if filepath.exists():
                    auto_imported = self._auto_fix_stdlib_imports(filepath)
                    if auto_imported:
                        # Re-verify after auto-fix — the file may now pass
                        if verification["status"] != "OK":
                            re_verify = self._verify_single_file(filename, False)
                            if re_verify["status"] == "OK":
                                manifest[filename] = re_verify
                                verification = re_verify
                                logger.info(f"  ✅ {filename} — NOW verified OK after auto-import fix")
                                # Remove from errors if it was added
                                errors = [e for e in errors if not e.startswith(f"{filename}:")]

                # v0.7.4: Track if this source file actually changed
                new_content = filepath.read_text() if filepath.exists() else None
                if new_content != old_content:
                    rebuilt_source_files.add(filename)
                    logger.info(f"  📝 {filename} — content changed, dependent tests will be re-run")

        # v0.9.4: Auto-create __init__.py in subdirectories
        # Without __init__.py, Python can't treat subdirs as packages,
        # so `from routes.books import books_blueprint` fails at runtime.
        try:
            created_inits = []
            for py_file in self.working_dir.rglob("*.py"):
                rel = py_file.relative_to(self.working_dir)
                if len(rel.parts) > 1:  # File is in a subdirectory
                    subdir = py_file.parent
                    init_file = subdir / "__init__.py"
                    if not init_file.exists():
                        init_file.write_text("")
                        created_inits.append(str(subdir.relative_to(self.working_dir)))
            if created_inits:
                unique_dirs = sorted(set(created_inits))
                logger.info(f"  📁 AUTO-INIT: created __init__.py in {', '.join(unique_dirs)}")
        except Exception as e:
            logger.debug(f"  __init__.py creation failed (non-fatal): {e}")

        # v0.9.4: Fix common datetime module/class confusion
        # Models write `import datetime` then call `datetime.strptime()` which fails
        # because datetime is the MODULE, not the CLASS. Fix: datetime.datetime.strptime()
        all_files = [s["file"] for s in sequence]
        try:
            dt_fixes = self._fix_datetime_confusion(all_files)
            if dt_fixes > 0:
                logger.info(f"  🔧 DATETIME FIX: corrected {dt_fixes} datetime module/class confusions")
        except Exception as e:
            logger.debug(f"  Datetime fix failed (non-fatal): {e}")

        # v0.9.3: Post-build import resolution — fix ALL import errors deterministically
        # Runs after all files are written so we have the complete picture of
        # what's defined where. Inspired by pyflyby's AST-based approach.
        try:
            fixes_applied = self._resolve_project_imports(all_files)
            if fixes_applied > 0:
                logger.info(f"  🔧 POST-BUILD: resolved {fixes_applied} import issues across project")

                # v0.9.4: RE-VERIFY files that had errors — import fix may have fixed them
                if errors:
                    re_verified = []
                    for err_str in errors[:]:  # Copy to allow mutation
                        err_filename = err_str.split(":")[0].strip()
                        err_step = None
                        for step in sequence:
                            if step["file"] == err_filename:
                                err_step = step
                                break
                        if err_step:
                            is_test_file = err_step.get("type") == "test" or err_filename.startswith("test_")
                            re_verify = self._verify_single_file(err_filename, is_test_file)
                            if re_verify["status"] == "OK":
                                manifest[err_filename] = re_verify
                                errors.remove(err_str)
                                re_verified.append(err_filename)
                                logger.info(f"  ✅ {err_filename} — NOW verified OK after import resolution")

                    if re_verified:
                        logger.info(f"  🎯 POST-BUILD RE-VERIFY: {len(re_verified)} files recovered: {', '.join(re_verified)}")
        except Exception as e:
            logger.debug(f"  Post-build import resolution failed (non-fatal): {e}")

        # Git commit all micro-builds together
        try:
            self._safe_run(
                ["git", "add", "-A"],
                cwd=str(self.working_dir),
                capture_output=True, timeout=10
            )
            self._safe_run(
                ["git", "commit", "-m", f"feat: implement task {task_state.task_id} (micro-build)"],
                cwd=str(self.working_dir),
                capture_output=True, timeout=10
            )
        except Exception as e:
            logger.debug(f"  Git commit after micro-builds: {e}")

        # Count successes
        ok_count = sum(1 for v in manifest.values() if v["status"] == "OK")
        total = len(sequence)
        logger.info(f"\n  📊 Micro-build complete: {ok_count}/{total} files verified OK")

        if errors:
            return AgentResult(
                success=True,  # Build phase succeeded (files were created)
                output=f"Micro-build: {ok_count}/{total} files OK. Issues: {'; '.join(errors[:3])}",
                error=None,
            )

        return AgentResult(
            success=True,
            output=f"Micro-build: all {total} files created and verified OK",
        )

    def _extract_blamed_source_files(self, test_output: str, test_filename: str) -> set:
        """
        v0.8.0: Parse pytest traceback to find source files that caused the failure.

        When test_app.py fails with 'database.py:19: ProgrammingError', the bug
        is in database.py, not test_app.py. Returns set of source filenames.
        """
        import re
        blamed = set()
        # Match traceback lines like: database.py:19: ProgrammingError
        # or: File "/tmp/.../database.py", line 19
        for match in re.finditer(r'(?:File "[^"]*[/\\])?(\w+\.py)(?:")?(?:,\s*line\s+|\:)\d+', test_output):
            source_file = match.group(1)
            # Don't blame the test file itself, or stdlib/site-packages
            if source_file != test_filename and not source_file.startswith('test_'):
                if (self.working_dir / source_file).exists():
                    blamed.add(source_file)
        return blamed

    def _rescue_uncreated_file(self, filename: str, output: str) -> bool:
        """Rescue a file when the model called write_file with wrong format."""
        import json as json_mod
        import re as re_mod

        filepath = self.working_dir / filename

        # Strategy 1: Find write_file JSON with content (valid JSON)
        decoder = json_mod.JSONDecoder()
        for m in re_mod.finditer(r'"name"\s*:\s*"write_file"', output):
            brace_pos = output.rfind('{', max(0, m.start() - 200), m.start())
            if brace_pos == -1:
                continue
            try:
                obj, _ = decoder.raw_decode(output, brace_pos)
                args = obj.get("arguments") or obj.get("parameters", {})
                file_content = args.get("content", "")
                if file_content and len(file_content) > 50:
                    filepath.write_text(file_content)
                    logger.info(f"    Rescued {filename}: {len(file_content)} bytes from tool call JSON")
                    return True
            except (json_mod.JSONDecodeError, ValueError):
                continue

        # Strategy 2: Extract content from malformed JSON (e.g. unescaped triple quotes)
        # Find "content": " and read until the closing pattern
        content_match = re_mod.search(r'"content"\s*:\s*"', output)
        if content_match:
            start = content_match.end()
            # Find the end: look for "}} or "} patterns that close the JSON
            # Walk forward tracking escape sequences
            result_chars = []
            i = start
            while i < len(output):
                ch = output[i]
                if ch == '\\' and i + 1 < len(output):
                    next_ch = output[i + 1]
                    if next_ch == 'n':
                        result_chars.append('\n')
                    elif next_ch == 't':
                        result_chars.append('\t')
                    elif next_ch == '"':
                        result_chars.append('"')
                    elif next_ch == '\\':
                        result_chars.append('\\')
                    else:
                        result_chars.append(ch + next_ch)
                    i += 2
                elif ch == '"':
                    break  # End of the JSON string value
                else:
                    result_chars.append(ch)
                    i += 1
            file_content = ''.join(result_chars)
            if len(file_content) > 50:
                filepath.write_text(file_content)
                logger.info(f"    Rescued {filename}: {len(file_content)} bytes from malformed JSON")
                return True

        # Strategy 3: Find python code blocks in markdown
        backticks = chr(96) * 3
        pattern = backticks + r"python\n(.*?)" + backticks
        code_blocks = re_mod.findall(pattern, output, re_mod.DOTALL)
        for block in code_blocks:
            if len(block.strip()) > 100:
                filepath.write_text(block.strip())
                logger.info(f"    Rescued {filename}: {len(block.strip())} bytes from markdown")
                return True

        return False

    def _format_manifest(self, manifest: dict) -> str:
        """Format the file manifest as context for the build agent."""
        if not manifest:
            return "No files built yet."

        lines = ["## Files Built So Far (use these EXACT imports and APIs)"]
        for filename, info in manifest.items():
            status = info.get("status", "unknown")
            exports = info.get("exports", "unknown")
            if status == "OK":
                lines.append(f"- ✅ `{filename}` — {exports}")
            else:
                lines.append(f"- ⚠️ `{filename}` — {status}")
        return "\n".join(lines)

    def _verify_single_file(self, filename: str, is_test: bool) -> dict:
        """
        Verify a single file after micro-build: syntax check + extract exports.

        v0.7.0 Fix 3: Import Hygiene — if syntax/import check fails with a known
        missing import pattern, auto-fix it before declaring failure.

        Returns dict with status, exists, exports.
        """
        filepath = self.working_dir / filename

        if not filepath.exists():
            return {"exists": False, "status": "FILE NOT CREATED", "exports": "none"}

        # Syntax check
        if filename.endswith('.py'):
            result = self._safe_run(
                ["python3", "-c", f"import py_compile; py_compile.compile('{filepath}', doraise=True)"],
                capture_output=True, text=True, timeout=10,
                cwd=str(self.working_dir)
            )
            if result.returncode != 0:
                error = result.stderr.strip()[-200:] if result.stderr else "unknown syntax error"
                return {"exists": True, "status": f"SYNTAX ERROR: {error}", "exports": "none (syntax error)"}

        # Extract exports (classes, functions, constants) for manifest
        exports = self._extract_exports(filepath)

        # For test files, try to import — and auto-fix if import fails
        if is_test and filename.endswith('.py'):
            module_name = filename.replace('.py', '')
            result = self._safe_run(
                ["python3", "-c", f"import {module_name}"],
                capture_output=True, text=True, timeout=10,
                cwd=str(self.working_dir)
            )
            if result.returncode != 0:
                error_output = result.stderr.strip()
                # v0.7.0: Try auto-fix before giving up
                if self._auto_fix_imports(filepath, error_output):
                    # Re-check after fix
                    result2 = self._safe_run(
                        ["python3", "-c", f"import {module_name}"],
                        capture_output=True, text=True, timeout=10,
                        cwd=str(self.working_dir)
                    )
                    if result2.returncode == 0:
                        logger.info(f"    🔧 Import hygiene: auto-fixed {filename}")
                        return {"exists": True, "status": "OK", "exports": exports}
                    # v0.7.3: stdlib fix wasn't enough, try project imports
                    error_output2 = result2.stderr.strip()
                    if self._auto_fix_project_imports(filepath, error_output2):
                        result3 = self._safe_run(
                            ["python3", "-c", f"import {module_name}"],
                            capture_output=True, text=True, timeout=10,
                            cwd=str(self.working_dir)
                        )
                        if result3.returncode == 0:
                            logger.info(f"    🔧 Project import hygiene: auto-fixed {filename}")
                            return {"exists": True, "status": "OK", "exports": exports}
                elif self._auto_fix_project_imports(filepath, error_output):
                    # v0.7.3: Try project imports directly
                    result2 = self._safe_run(
                        ["python3", "-c", f"import {module_name}"],
                        capture_output=True, text=True, timeout=10,
                        cwd=str(self.working_dir)
                    )
                    if result2.returncode == 0:
                        logger.info(f"    🔧 Project import hygiene: auto-fixed {filename}")
                        return {"exists": True, "status": "OK", "exports": exports}

                error = error_output[-200:] if error_output else "import failed"
                return {"exists": True, "status": f"IMPORT ERROR: {error}", "exports": exports}

        return {"exists": True, "status": "OK", "exports": exports}

    def _auto_fix_imports(self, filepath: Path, error_output: str) -> bool:
        """
        v0.7.0 Import Hygiene: Auto-fix common missing imports in test files.

        Handles the most frequent NameError patterns that cause 80% of test failures:
        - Missing `from datetime import datetime`
        - Missing `from pathlib import Path`
        - Missing `import tempfile` / `import io` / `import json`
        - Missing `from contextlib import redirect_stdout`
        - Missing `from unittest.mock import patch, MagicMock`

        Returns True if a fix was applied.
        """
        # Common auto-fix map: error pattern → import line to add
        IMPORT_FIXES = {
            "NameError: name 'datetime' is not defined": "from datetime import datetime",
            "NameError: name 'Path' is not defined": "from pathlib import Path",
            "NameError: name 'tempfile' is not defined": "import tempfile",
            "NameError: name 'TemporaryDirectory' is not defined": "from tempfile import TemporaryDirectory",
            "NameError: name 'io' is not defined": "import io",
            "NameError: name 'StringIO' is not defined": "from io import StringIO",
            "NameError: name 'json' is not defined": "import json",
            "NameError: name 'redirect_stdout' is not defined": "from contextlib import redirect_stdout",
            "NameError: name 'patch' is not defined": "from unittest.mock import patch",
            "NameError: name 'MagicMock' is not defined": "from unittest.mock import MagicMock",
            "NameError: name 'mock' is not defined": "from unittest import mock",
            "NameError: name 'uuid' is not defined": "import uuid",
            "NameError: name 'os' is not defined": "import os",
            "NameError: name 'sys' is not defined": "import sys",
            "NameError: name 'pytest' is not defined": "import pytest",
            "NameError: name 'dataclass' is not defined": "from dataclasses import dataclass",
        }

        try:
            content = filepath.read_text()
            fixed = False

            for error_pattern, import_line in IMPORT_FIXES.items():
                if error_pattern in error_output:
                    # Check if import already exists (might be wrong form)
                    module_name = import_line.split()[-1]
                    if import_line not in content:
                        # Add import at the top (after any existing imports)
                        lines = content.split('\n')
                        insert_idx = 0
                        for i, line in enumerate(lines):
                            if line.startswith(('import ', 'from ')):
                                insert_idx = i + 1
                            elif line.strip() and not line.startswith('#') and insert_idx > 0:
                                break

                        lines.insert(insert_idx, import_line)
                        content = '\n'.join(lines)
                        fixed = True
                        logger.info(f"    🔧 Auto-added: {import_line}")

            if fixed:
                filepath.write_text(content)
                return True

        except Exception as e:
            logger.debug(f"    Import auto-fix failed: {e}")

        return False

    def _auto_fix_project_imports(self, filepath: Path, error_output: str) -> bool:
        """
        v0.7.3: Auto-fix missing imports from PROJECT modules (not just stdlib).

        This is the #1 killer of test files: the model writes `TaskStorage(...)`
        without importing it, causing NameError. The stdlib auto-fix can't help
        because TaskStorage isn't in the IMPORT_FIXES map.

        This method scans all .py files in the workspace for the missing name
        and adds the correct import line.

        Returns True if a fix was applied.
        """
        import re

        # Parse all NameErrors from the error output
        missing_names = re.findall(r"NameError: name '(\w+)' is not defined", error_output)
        if not missing_names:
            return False

        try:
            content = filepath.read_text()
            fixed = False

            for missing_name in missing_names:
                # Skip if already imported (might be imported wrong way)
                if f"import {missing_name}" in content:
                    continue

                # Search all project .py files for this name
                found_in = None
                for py_file in self.working_dir.rglob("*.py"):
                    if py_file == filepath or py_file.name.startswith('test_'):
                        continue
                    # Skip venv, __pycache__, .agents
                    rel_str = str(py_file.relative_to(self.working_dir))
                    if any(skip in rel_str for skip in ['venv/', '__pycache__/', '.agents/', 'node_modules/']):
                        continue
                    try:
                        src = py_file.read_text()
                        # Check if this name is defined as a class, function, or assignment
                        if re.search(rf'^(class|def)\s+{re.escape(missing_name)}\b', src, re.MULTILINE):
                            found_in = py_file.stem
                            break
                        # Also check for top-level assignments (constants)
                        if re.search(rf'^{re.escape(missing_name)}\s*=', src, re.MULTILINE):
                            found_in = py_file.stem
                            break
                    except Exception:
                        continue

                if found_in:
                    import_line = f"from {found_in} import {missing_name}"
                    if import_line not in content:
                        # Insert after existing imports
                        lines = content.split('\n')
                        insert_idx = 0
                        for i, line in enumerate(lines):
                            if line.startswith(('import ', 'from ')):
                                insert_idx = i + 1
                            elif line.strip() and not line.startswith('#') and insert_idx > 0:
                                break

                        lines.insert(insert_idx, import_line)
                        content = '\n'.join(lines)
                        fixed = True
                        logger.info(f"    🔧 Auto-added project import: {import_line}")

            if fixed:
                filepath.write_text(content)
                return True

        except Exception as e:
            logger.debug(f"    Project import auto-fix failed: {e}")

        return False

    def _auto_fix_imports_precheck(self, filepath: Path) -> bool:
        """
        v0.7.1: Proactive import hygiene — run py_compile + import check on a test
        file BEFORE it enters the verification/test pipeline.

        v0.7.3: Also auto-fixes project-local imports (e.g., TaskStorage, Task)
        by scanning workspace files for the missing name.

        Unlike _auto_fix_imports (which requires error_output to be passed in),
        this method generates its own error output by attempting to compile and
        import the file. This allows it to run inside the sampling loop where
        we don't yet have error output.

        Runs up to 3 passes to handle cascading NameErrors (e.g., fixing
        `import tempfile` reveals a missing `from pathlib import Path` reveals
        a missing `from storage import TaskStorage`).

        Returns True if any fixes were applied.
        """
        if not filepath.exists() or not filepath.name.endswith('.py'):
            return False

        any_fixed = False

        for pass_num in range(3):  # v0.7.3: 3 passes (was 2) for project imports
            # Try py_compile first
            result = self._safe_run(
                ["python3", "-c", f"import py_compile; py_compile.compile('{filepath}', doraise=True)"],
                capture_output=True, text=True, timeout=10,
                cwd=str(self.working_dir)
            )

            if result.returncode == 0:
                # Compilation OK — also try importing to catch runtime NameErrors
                module_name = filepath.stem
                result = self._safe_run(
                    ["python3", "-c", f"import {module_name}"],
                    capture_output=True, text=True, timeout=10,
                    cwd=str(self.working_dir)
                )
                if result.returncode == 0:
                    break  # All good

            error_output = (result.stderr or "") + (result.stdout or "")
            if not error_output.strip():
                break

            # Try stdlib fix first, then project-local fix
            fixed = self._auto_fix_imports(filepath, error_output)
            if not fixed:
                # v0.7.3: Try project-local imports
                fixed = self._auto_fix_project_imports(filepath, error_output)
            if fixed:
                any_fixed = True
            else:
                break  # No fixable patterns found

        return any_fixed

    def _export_traces(self):
        """Export collected traces at the end of a run."""
        stats = self.trace_collector.get_session_stats()
        if stats["total"] > 0:
            logger.info("\n📝 Trace Collection Summary:")
            logger.info(f"  Test failures: {stats['test_failures']}")
            logger.info(f"  Build failures: {stats['build_failures']}")
            logger.info(f"  RCA failures: {stats['rca_failures']}")
            logger.info(f"  Total: {stats['total']}")

            # Export for Claude distillation
            claude_path = self.trace_collector.export_for_claude()
            training_path = self.trace_collector.export_for_training()
            logger.info(f"  📋 For Claude: {claude_path}")
            logger.info(f"  📦 For training: {training_path}")

    def _classify_test_error(self, error_output: str) -> str:
        """
        Classify test error output into a category for the pattern classifier.

        These categories become features for the ML failure pattern model.
        """
        error_lower = error_output.lower()

        if "importerror" in error_lower or "modulenotfounderror" in error_lower:
            return "import_error"
        elif "nameerror" in error_lower:
            return "name_error"
        elif "attributeerror" in error_lower:
            return "attribute_error"
        elif "typeerror" in error_lower:
            return "type_error"
        elif "assertionerror" in error_lower:
            return "assertion_error"
        elif "syntaxerror" in error_lower:
            return "syntax_error"
        elif "filenotfounderror" in error_lower:
            return "file_not_found"
        elif "json" in error_lower and ("decode" in error_lower or "serial" in error_lower):
            return "json_serialization"
        elif "argparse" in error_lower or "sys.argv" in error_lower:
            return "argparse_error"
        elif "working outside of" in error_lower and ("request context" in error_lower or "application context" in error_lower):
            return "flask_context_error"
        elif "datetime" in error_lower:
            return "datetime_error"
        elif "permission" in error_lower:
            return "permission_error"
        elif "timeout" in error_lower:
            return "timeout"
        elif "programmingerror" in error_lower or "operationalerror" in error_lower or "integrityerror" in error_lower:
            return "sqlite_error"
        elif "connectionerror" in error_lower or "connectionrefused" in error_lower:
            return "connection_error"
        elif "valueerror" in error_lower:
            return "value_error"
        elif "keyerror" in error_lower:
            return "key_error"
        elif "runtimeerror" in error_lower:
            return "runtime_error"
        elif "oserror" in error_lower or "ioerror" in error_lower:
            return "os_error"
        else:
            return "unknown"

    def _extract_exports(self, filepath: Path) -> str:
        """Extract public names from a Python file for the manifest.
        v1.0: Uses AST for accurate class/function signature extraction.
        Inspired by Aider's tree-sitter repo map — gives the model exact
        API contracts so it writes correct imports and function calls."""
        try:
            content = filepath.read_text()
            return self._extract_signatures_ast(content)
        except Exception:
            return "unknown"

    def _extract_signatures_ast(self, source_content: str) -> str:
        """
        v1.0: AST-based signature extraction — lightweight repo map.

        Returns compact function/class signatures including parameter names
        and defaults. Example output:
          classes: Bookmark(id, url, title, tags=..., created_at=...);
                   BookmarkDB[__init__(db_path=...), .get_all(page=..., tag=...)]
          functions: init_db(db_path=...), validate_url(url)
        """
        import ast

        try:
            tree = ast.parse(source_content)
        except SyntaxError:
            # Fallback to regex if AST fails (broken file)
            import re
            classes = re.findall(r'^class\s+(\w+)', source_content, re.MULTILINE)
            functions = re.findall(r'^def\s+(\w+)', source_content, re.MULTILINE)
            classes = [c for c in classes if not c.startswith('_')]
            functions = [f for f in functions if not f.startswith('_')]
            parts = []
            if classes:
                parts.append(f"classes: {', '.join(classes)}")
            if functions:
                parts.append(f"functions: {', '.join(functions)}")
            return "; ".join(parts) if parts else "no public exports"

        class_sigs = []
        func_sigs = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
                is_dataclass = any(
                    (isinstance(d, ast.Name) and d.id == 'dataclass') or
                    (isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                     and d.func.id == 'dataclass') or
                    (isinstance(d, ast.Attribute) and d.attr == 'dataclass')
                    for d in node.decorator_list
                )

                if is_dataclass:
                    fields = []
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            nm = item.target.id
                            if not nm.startswith('_'):
                                has_default = item.value is not None
                                fields.append(f"{nm}=..." if has_default else nm)
                    sig = f"{node.name}({', '.join(fields)})" if fields else node.name
                    class_sigs.append(sig)
                else:
                    methods = []
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if item.name.startswith('_') and item.name != '__init__':
                                continue
                            params = self._fmt_ast_params(item.args, skip_self=True)
                            if item.name == '__init__':
                                methods.append(f"__init__({params})")
                            else:
                                methods.append(f".{item.name}({params})")
                    if methods:
                        class_sigs.append(f"{node.name}[{', '.join(methods)}]")
                    else:
                        class_sigs.append(node.name)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith('_'):
                    params = self._fmt_ast_params(node.args, skip_self=False)
                    func_sigs.append(f"{node.name}({params})")

        parts = []
        if class_sigs:
            parts.append(f"classes: {'; '.join(class_sigs)}")
        if func_sigs:
            parts.append(f"functions: {', '.join(func_sigs)}")
        return "; ".join(parts) if parts else "no public exports"

    @staticmethod
    def _fmt_ast_params(args, skip_self=False) -> str:
        """Format AST function args into a readable signature string."""
        params = []
        all_args = args.args
        num_defaults = len(args.defaults)
        default_start = len(all_args) - num_defaults

        for i, arg in enumerate(all_args):
            name = arg.arg
            if skip_self and name == 'self':
                continue
            if i >= default_start:
                params.append(f"{name}=...")
            else:
                params.append(name)

        if args.vararg:
            params.append(f"*{args.vararg.arg}")
        for kw in args.kwonlyargs:
            params.append(f"{kw.arg}=...")
        if args.kwarg:
            params.append(f"**{args.kwarg.arg}")
        return ", ".join(params)

    def _sample_source_file(
        self,
        task_state: TaskState,
        step: dict,
        manifest_text: str,
        step_number: int,
        total_steps: int,
        num_candidates: int = 3,
    ) -> tuple:
        """
        v0.7.2: Best-of-N sampling for SOURCE files.

        Previously source files got exactly one attempt at temp=0.0. Now we
        generate N candidates at varied temperatures and pick the one that
        compiles, imports cleanly, and (if possible) passes a smoke test.

        Verifier chain (lightweight — no tests needed yet):
          1. py_compile — syntax check
          2. import <module> — catches runtime NameErrors at module level
          3. from <module> import * — catches broken exports

        Returns (AgentResult, verification_dict)
        """
        from standalone_models import AgentResult

        filename = step["file"]
        filepath = self.working_dir / filename
        module_name = filename.replace('.py', '')

        # v0.8.0: Proactive KB lookup — get relevant patterns before building
        kb_build_context = self.kb.get_build_context(
            f"{task_state.goal} {filename} {step.get('description', '')}"
        )
        if kb_build_context:
            logger.info(f"  📚 KB provided proactive context for {filename}")

        temperatures = [0.0, 0.4, 0.7][:num_candidates]
        logger.info(f"  🎲 Sampling {num_candidates} source candidates for {filename}")

        best_score = -1  # 0=created, 1=compiles, 2=imports, 3=exports
        best_content = None
        best_verification = {"exists": False, "status": "NO CANDIDATES", "exports": "none"}
        collected_errors = []

        for idx, temp in enumerate(temperatures):
            logger.info(f"    Source candidate {idx+1}/{num_candidates} (temp={temp})...")

            # v0.9.3: Plain-text build mode — no JSON tool calls
            result = self.agent_runner.run_build_single_file_plain(
                state=task_state,
                filename=filename,
                step_info=step,
                manifest=manifest_text,
                step_number=step_number,
                total_steps=total_steps,
                temperature=temp,
                kb_context=kb_build_context,
            )

            if not result.success:
                logger.info(f"    ❌ Source candidate {idx+1}: build failed")
                collected_errors.append(f"Candidate {idx+1}: build agent failed — {result.error}")
                continue

            # v0.9.3: Parse file content from plain-text markers
            parsed_content = self.agent_runner.parse_plain_file_content(result.output or "")
            if parsed_content:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(parsed_content)
                logger.debug(f"    Parsed {len(parsed_content)} bytes from plain-text output")
            elif not filepath.exists():
                # Fallback: try old rescue method in case model used tool calls anyway
                rescued = self._rescue_uncreated_file(filename, result.output or "")
                if rescued:
                    logger.info(f"    🔧 Source candidate {idx+1}: rescued from tool call output")
                else:
                    output_preview = (result.output or "")[:300]
                    logger.warning(f"    ❌ Source candidate {idx+1}: file not created")
                    logger.debug(f"    DEBUG model output: {output_preview}")
                    collected_errors.append(f"Candidate {idx+1}: file was not created")
                    continue

            candidate_content = filepath.read_text()
            score = 0  # File exists

            # v0.7.1: Import hygiene — auto-fix missing imports
            self._auto_fix_imports_precheck(filepath)
            candidate_content = filepath.read_text()

            # Score 1: Syntax check
            syntax_result = self._safe_run(
                ["python3", "-c", f"import py_compile; py_compile.compile('{filepath}', doraise=True)"],
                capture_output=True, text=True, timeout=10,
                cwd=str(self.working_dir)
            )
            if syntax_result.returncode != 0:
                error = syntax_result.stderr.strip()[-200:]
                logger.info(f"    ⚠️ Source candidate {idx+1}: syntax error")
                collected_errors.append(f"Candidate {idx+1}: syntax error — {error}")
                if score > best_score:
                    best_score = score
                    best_content = candidate_content
                continue
            score = 1  # Compiles

            # Score 2: Import check
            import_result = self._safe_run(
                ["python3", "-c", f"import {module_name}"],
                capture_output=True, text=True, timeout=10,
                cwd=str(self.working_dir)
            )
            if import_result.returncode != 0:
                error = import_result.stderr.strip()[-200:]
                logger.info(f"    ⚠️ Source candidate {idx+1}: import error")
                collected_errors.append(f"Candidate {idx+1}: import error — {error}")
                if score > best_score:
                    best_score = score
                    best_content = candidate_content
                continue
            score = 2  # Imports cleanly

            # Score 3: Export check (catches broken class/function definitions)
            export_result = self._safe_run(
                ["python3", "-c", f"from {module_name} import *; print('OK')"],
                capture_output=True, text=True, timeout=10,
                cwd=str(self.working_dir)
            )
            if export_result.returncode == 0:
                score = 3  # Exports cleanly

            # Score 3b: Runtime smoke test — catch name-shadowing & deadlocks
            if score == 3:
                smoke_code = (
                    f"import {module_name}, inspect\n"
                    f"errors = []\n"
                    f"for name, obj in inspect.getmembers({module_name}):\n"
                    f"    if inspect.isclass(obj) and not name.startswith('_'):\n"
                    f"        try:\n"
                    f"            sig = inspect.signature(obj.__init__)\n"
                    f"            required = [p for p in sig.parameters.values() if p.name != 'self' and p.default is inspect.Parameter.empty]\n"
                    f"            if not required:\n"
                    f"                inst = obj()\n"
                    f"                for mname in dir(inst):\n"
                    f"                    if mname.startswith('_'): continue\n"
                    f"                    attr = getattr(inst, mname)\n"
                    f"                    cls_attr = getattr(type(inst), mname, None)\n"
                    f"                    if callable(cls_attr) and not callable(attr):\n"
                    f"                        errors.append(f'{{name}}.{{mname}}: method shadowed by {{type(attr).__name__}}')\n"
                    f"        except Exception:\n"
                    f"            pass\n"
                    f"if errors:\n"
                    f"    raise RuntimeError('Smoke test: ' + '; '.join(errors))\n"
                    f"print('SMOKE_OK')"
                )
                try:
                    smoke_result = self._safe_run(
                        ["python3", "-c", smoke_code],
                        capture_output=True, text=True, timeout=10,
                        cwd=str(self.working_dir)
                    )
                    if smoke_result.returncode != 0:
                        smoke_err = smoke_result.stderr.strip()[-200:]
                        logger.info(f"    \u26a0\ufe0f Source candidate {idx+1}: smoke test FAILED — {smoke_err}")
                        collected_errors.append(f"Candidate {idx+1}: smoke test failed — {smoke_err}")
                        score = 2  # Downgrade: imports OK but runtime broken
                except subprocess.TimeoutExpired:
                    logger.info(f"    \u26a0\ufe0f Source candidate {idx+1}: smoke test TIMEOUT (threads may hang)")
                    collected_errors.append(f"Candidate {idx+1}: smoke test timed out — class instantiation hangs")
                    score = 2  # Downgrade: runtime hangs

            verification = self._verify_single_file(filename, False)

            if score >= 2:
                # Good enough — compiles and imports
                logger.info(f"    ✅ Source candidate {idx+1}: score {score}/3 — {'exports OK' if score == 3 else 'imports OK'}")
                if score == 3:
                    # Perfect candidate — use immediately
                    verification["status"] = "OK"
                    return result, verification

            if score > best_score:
                best_score = score
                best_content = candidate_content
                best_verification = verification

        # === Wave 2: Edit-based repair for source files (v0.9.8) ===
        if best_score >= 1 and best_score < 3 and best_content and collected_errors:
            # Source compiles but has issues — try surgical edit
            logger.info(f"  🔧 Source edit repair for {filename} (score {best_score}/3)")
            filepath.write_text(best_content)

            for edit_round in range(2):
                current_content = filepath.read_text()
                # Identify what's wrong
                error_summary = "\n".join(collected_errors[:2])

                edit_result = self.agent_runner.run_edit_repair(
                    state=task_state,
                    filename=filename,
                    current_content=current_content,
                    test_output=error_summary,
                    temperature=0.2,
                )

                if edit_result.success and edit_result.output:
                    edits = self.agent_runner.parse_search_replace(edit_result.output)
                    if edits:
                        apply_result = self.agent_runner.apply_search_replace(filepath, edits)
                        if isinstance(apply_result, tuple):
                            num_applied, edit_feedback = apply_result
                        else:
                            num_applied, edit_feedback = apply_result, []
                        logger.info(f"    📝 Source edit {edit_round + 1}: applied {num_applied}/{len(edits)} edits")

                        if num_applied == 0:
                            # v0.9.9b: store feedback for next round (was discarded)
                            if edit_feedback:
                                task_state.edit_feedback.extend(edit_feedback)
                                logger.info(f"    🔍 Source edit {edit_round + 1}: {len(edit_feedback)} match failures logged")
                                continue
                            break

                        # Clear feedback on success
                        task_state.edit_feedback = []

                        if num_applied > 0:
                            self._auto_fix_imports_precheck(filepath)
                            # Re-verify
                            import_ok = self._safe_run(
                                ["python3", "-c", f"import {module_name}"],
                                capture_output=True, text=True, timeout=10,
                                cwd=str(self.working_dir)
                            ).returncode == 0

                            export_ok = False
                            if import_ok:
                                export_ok = self._safe_run(
                                    ["python3", "-c", f"from {module_name} import *; print('OK')"],
                                    capture_output=True, text=True, timeout=10,
                                    cwd=str(self.working_dir)
                                ).returncode == 0

                            # v0.9.9b: actually check compile (was hardcoded True)
                            compile_ok = self._safe_run(
                                ["python3", "-c", f"import py_compile; py_compile.compile('{filepath}', doraise=True)"],
                                capture_output=True, text=True, timeout=10,
                                cwd=str(self.working_dir)
                            ).returncode == 0

                            new_score = (1 if compile_ok else 0) + (1 if import_ok else 0) + (1 if export_ok else 0)
                            if new_score > best_score:
                                best_score = new_score
                                best_content = filepath.read_text()
                                best_verification = self._verify_single_file(filename, False)
                                logger.info(f"    ✅ Source edit improved: score {new_score}/3")

                            if new_score >= 2:
                                best_verification["status"] = "OK"
                                return AgentResult(success=True, output=f"Source edit repair: score {new_score}/3"), best_verification

            filepath.write_text(best_content)  # Restore best version

        # === Wave 2 FALLBACK: Full regeneration (only if edit repair failed or score < 1) ===
        if best_score < 2 and collected_errors:
            logger.info(f"  🔬 Source Wave 2: error-aware re-sampling for {filename}")
            error_context = "\n".join(collected_errors[:3])

            # v0.8.0: Query KB for known fix patterns
            kb_context = ""
            for err_text in collected_errors[:2]:
                kb_fix = self.kb.get_fix_for_error(err_text)
                if kb_fix:
                    kb_context += kb_fix
                    break  # One good fix is enough
            if kb_context:
                error_context += "\n" + kb_context
                logger.info(f"  📚 KB provided fix context for {filename}")

            # Build code context from best attempt if available
            code_context = ""
            if best_content:
                code_context = f"""
## Best attempt so far (score {best_score}/3 — {'compiles' if best_score >= 1 else 'created but broken'}):
```python
{best_content[:2500]}
```
Fix the specific errors above. Keep the overall structure if it compiles.
"""

            temperatures_wave2 = [0.3, 0.6]
            for idx, temp in enumerate(temperatures_wave2):
                logger.info(f"    Source Wave2 {idx+1}/2 (temp={temp})...")

                # v0.9.3: Plain-text build mode
                result = self.agent_runner.run_build_single_file_plain(
                    state=task_state,
                    filename=filename,
                    step_info=step,
                    manifest=manifest_text,
                    step_number=step_number,
                    total_steps=total_steps,
                    temperature=temp,
                    error_context=error_context + "\n" + code_context,
                    kb_context=kb_build_context,
                )

                if not result.success:
                    continue

                # v0.9.3: Parse from plain-text markers
                parsed_content = self.agent_runner.parse_plain_file_content(result.output or "")
                if parsed_content:
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    filepath.write_text(parsed_content)
                elif not filepath.exists():
                    continue

                candidate_content = filepath.read_text()
                self._auto_fix_imports_precheck(filepath)
                candidate_content = filepath.read_text()

                # Quick verification
                syntax_ok = self._safe_run(
                    ["python3", "-c", f"import py_compile; py_compile.compile('{filepath}', doraise=True)"],
                    capture_output=True, text=True, timeout=10, cwd=str(self.working_dir)
                ).returncode == 0

                import_ok = False
                if syntax_ok:
                    import_ok = self._safe_run(
                        ["python3", "-c", f"import {module_name}"],
                        capture_output=True, text=True, timeout=10, cwd=str(self.working_dir)
                    ).returncode == 0

                score = 0 + (1 if syntax_ok else 0) + (1 if import_ok else 0)
                if import_ok:
                    export_ok = self._safe_run(
                        ["python3", "-c", f"from {module_name} import *; print('OK')"],
                        capture_output=True, text=True, timeout=10, cwd=str(self.working_dir)
                    ).returncode == 0
                    if export_ok:
                        score = 3

                if score > best_score:
                    best_score = score
                    best_content = candidate_content
                    best_verification = self._verify_single_file(filename, False)

                if score >= 2:
                    logger.info(f"    ✅ Source Wave2 {idx+1}: score {score}/3")
                    best_verification["status"] = "OK"
                    # v0.8.0: Auto-capture — Wave 2 fixed what Wave 1 couldn't
                    if collected_errors:
                        self.kb.capture_pattern(
                            error_pattern=collected_errors[0][:200],
                            solution=f"Fixed {filename} via error-aware re-sampling (score {score}/3)",
                            example=candidate_content[:500] if candidate_content else "",
                            source="auto_source_wave2",
                        )
                    return result, best_verification

        # Use best candidate
        total_attempts = num_candidates + (2 if collected_errors and best_score < 2 else 0)
        if best_content is not None:
            filepath.write_text(best_content)
            if best_score >= 2:
                best_verification["status"] = "OK"
            else:
                status_label = {0: "created but broken", 1: "compiles but import fails", 2: "imports OK", 3: "exports OK"}.get(best_score, "unknown")
                best_verification["status"] = f"BEST OF {total_attempts}: {status_label}"
            best_verification["exports"] = self._extract_exports(filepath)
            logger.info(f"  📊 Source sampling: best score {best_score}/3 from {total_attempts} candidates")
            return AgentResult(success=True, output=f"Source sampling: score {best_score}/3"), best_verification

        logger.warning(f"  ❌ All {total_attempts} source candidates failed for {filename}")
        return (
            AgentResult(success=False, output="All source candidates failed", error="source_sampling_exhausted"),
            {"exists": False, "status": "ALL CANDIDATES FAILED", "exports": "none"}
        )

    def _generate_spec_test(self, task_state: TaskState, step: dict, total_steps: int, source_files: Optional[list] = None) -> bool:
        """
        v0.8.2: Generate a test file from the SPEC ONLY — no source code visible.

        This enforces TDD: tests define the contract, source must satisfy them.
        The test file is generated with strong assertions derived from the task
        description, NOT from any implementation. This prevents the test-source
        collusion where the 70B writes weak assertions that match its own bugs.

        Returns True if a valid test file was created, False otherwise.
        """
        filename = step["file"]
        tests_for = step.get("tests_for", "")
        filepath = self.working_dir / filename

        system_prompt = self.agent_runner._load_prompt("prompts/test_gen.txt")
        if not system_prompt:
            logger.warning("  No test_gen.txt prompt found, falling back to normal sampling")
            return False

        # KB context for test patterns
        kb_context = ""
        if self.kb:
            query = f"{task_state.goal} {filename} {tests_for} pytest"
            kb_context = self.kb.get_build_context(query)

        user_prompt = f"""{kb_context}
## Task Specification
{task_state.goal}

## Generate Test File: {filename}
This test file verifies: {tests_for or filename.replace('test_', '').replace('.py', '') + '.py'}

Write the COMPLETE test file. Do NOT reference any existing implementation — write tests from the SPEC ONLY.

## Source files that WILL be created (use these EXACT module names for imports):
{', '.join(source_files or [])}
Import from these modules ONLY. Do not invent module names.

Use write_file to create {filename}.
"""
        # Sample 3 candidates, pick best by syntax + import validity
        best_content = None
        for temp in [0.3, 0.6, 0.9]:
            # v0.9.3: Plain-text build mode for test generation too
            result = self.agent_runner.run_build_single_file_plain(
                state=task_state,
                step_info=step,
                filename=filename,
                manifest="No source files built yet — write tests from spec only.",
                step_number=0,
                total_steps=total_steps,
                temperature=temp,
                agent_name="test_gen",
            )

            # v0.9.3: Parse from plain-text markers
            parsed_content = self.agent_runner.parse_plain_file_content(result.output or "")
            if parsed_content:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(parsed_content)

            if filepath.exists():
                content = filepath.read_text()
                # Verify syntax
                check = self._safe_run(
                    ["python3", "-m", "py_compile", filename],
                    cwd=str(self.working_dir),
                    capture_output=True, text=True, timeout=10
                )
                if check.returncode == 0 and len(content.strip()) > 50:
                    best_content = content
                    logger.info(f"    ✅ Spec test candidate (temp={temp}): syntax OK")
                    break
                else:
                    logger.info(f"    ⚠️ Spec test candidate (temp={temp}): syntax error")
            else:
                logger.info(f"    ⚠️ Spec test candidate (temp={temp}): file not created")

        if best_content:
            filepath.write_text(best_content)
            # Auto-fix imports: replace hallucinated module names with actual source files
            if source_files:
                self._fix_test_imports(filepath, source_files)
            # Auto-inject missing Flask test client fixture
            self._inject_flask_fixture(filepath)
            return True
        return False

    def _inject_flask_fixture(self, test_path):
        """Auto-inject Flask test client fixture if tests use 'client' but no fixture exists."""
        content = test_path.read_text()
        # Check if tests use client param but no fixture is defined
        if 'def test_' in content and '(client)' in content and '@pytest.fixture' not in content:
            # Find the app import to know the module name
            import re as re_mod
            app_import = re_mod.search(r'from\s+(\w+)\s+import\s+app', content)
            if app_import:
                module = app_import.group(1)
                fixture_code = (
                    "\n@pytest.fixture\n"
                    "def client():\n"
                    "    app.config['TESTING'] = True\n"
                    "    with app.test_client() as client:\n"
                    "        yield client\n\n"
                )
                # Insert after the import lines
                lines = content.split('\n')
                insert_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        insert_idx = i + 1
                    elif line.strip() and insert_idx > 0:
                        break
                lines.insert(insert_idx, fixture_code)
                test_path.write_text('\n'.join(lines))
                logger.info(f"    Injected Flask client fixture into {test_path.name}")

    def _fix_test_imports(self, test_path, source_files):
        """Fix hallucinated imports in generated test files.

        The 14B LoRA often invents module names like 'tracker' when the actual
        file is 'app.py'. This rewrites bad imports to use the real filenames.
        """
        import re as re_mod
        content = test_path.read_text()
        source_modules = {f.replace('.py', '') for f in source_files}
        changed = False

        lines = content.split('\n')
        new_lines = []
        for line in lines:
            # Match: from <module> import ... or import <module>
            m = re_mod.match(r'^(from\s+)([\w.]+)(\s+import\s+.+)$', line)
            if m:
                prefix, module, suffix = m.groups()
                root_module = module.split('.')[0]
                if root_module not in source_modules and root_module not in {'pytest', 'datetime', 'json', 'os', 'sys', 'tempfile', 'unittest', 'io', 'pathlib', 'contextlib', 'uuid', 'sqlite3', 'flask'}:
                    imported_names = [n.strip() for n in suffix.replace('import', '').split(',')]
                    best_match = None
                    # Priority 1: an imported name IS a source module
                    # e.g. "from tracker import app" -> "app" is in source_modules -> use "app"
                    for name in imported_names:
                        if name in source_modules:
                            best_match = name
                            break
                    # Priority 2: fuzzy substring match on root module name
                    if not best_match:
                        for src in source_modules:
                            if root_module.lower() in src.lower() or src.lower() in root_module.lower():
                                best_match = src
                                break
                    # Priority 3: match imported names to source modules
                    if not best_match:
                        for name in imported_names:
                            for src in source_modules:
                                if name.lower() == src.lower() or name.lower() in src.lower():
                                    best_match = src
                                    break
                            if best_match:
                                break
                    if best_match:
                        logger.info(f"    Import fix: {module} -> {best_match} in {test_path.name}")
                        line = f"{prefix}{best_match}{suffix}"
                        changed = True
            new_lines.append(line)

        if changed:
            test_path.write_text('\n'.join(new_lines))

    # ──────────────────────────────────────────────────────────────
    # v0.8.4 Fix 1: Deterministic stdlib auto-import (zero LLM cost)
    #
    # LlmFix paper (2024): NameError from missing stdlib imports is
    # the #1 auto-fixable LLM code generation error.  100% fix rate.
    # AST Hallucination Detection (2025): deterministic validation
    # beats LLM-in-the-loop for known error patterns.
    # Aider uses flake8 --select=F821 for the same purpose.
    #
    # Walk the AST, find names used but never imported/defined,
    # auto-inject `import <module>` for known stdlib modules.
    # ──────────────────────────────────────────────────────────────

    # Mapping: bare name → import statement
    STDLIB_IMPORT_MAP = {
        # Module-level names (import X)
        "datetime": "import datetime",
        "json": "import json",
        "os": "import os",
        "sys": "import sys",
        "re": "import re",
        "uuid": "import uuid",
        "sqlite3": "import sqlite3",
        "pathlib": "import pathlib",
        "tempfile": "import tempfile",
        "shutil": "import shutil",
        "hashlib": "import hashlib",
        "logging": "import logging",
        "collections": "import collections",
        "functools": "import functools",
        "itertools": "import itertools",
        "math": "import math",
        "time": "import time",
        "copy": "import copy",
        "io": "import io",
        "csv": "import csv",
        "traceback": "import traceback",
        "subprocess": "import subprocess",
        "threading": "import threading",
        "contextlib": "import contextlib",
        "typing": "import typing",
        "enum": "import enum",
        "secrets": "import secrets",
        "base64": "import base64",
        "http": "import http",
        # Class-level names (from X import Y)
        "Path": "from pathlib import Path",
        "Optional": "from typing import Optional",
        "List": "from typing import List",
        "Dict": "from typing import Dict",
        "Tuple": "from typing import Tuple",
        "Any": "from typing import Any",
        "Union": "from typing import Union",
        "Enum": "from enum import Enum",
        "IntEnum": "from enum import IntEnum",
        "dataclass": "from dataclasses import dataclass",
        "field": "from dataclasses import field",
        "asdict": "from dataclasses import asdict",
        "defaultdict": "from collections import defaultdict",
        "Counter": "from collections import Counter",
        "OrderedDict": "from collections import OrderedDict",
        "namedtuple": "from collections import namedtuple",
        "wraps": "from functools import wraps",
        "lru_cache": "from functools import lru_cache",
        "ABC": "from abc import ABC",
        "abstractmethod": "from abc import abstractmethod",
        "contextmanager": "from contextlib import contextmanager",
        # v0.9.4: Additional stdlib entries
        "dataclasses": "import dataclasses",
        "abc": "import abc",
        "warnings": "import warnings",
        "textwrap": "import textwrap",
        "struct": "import struct",
        "random": "import random",
        "string": "import string",
        "socket": "import socket",
        "signal": "import signal",
        "inspect": "import inspect",
        "unittest": "import unittest",
        "argparse": "import argparse",
        "glob": "import glob",
        "fnmatch": "import fnmatch",
        "decimal": "import decimal",
        "fractions": "import fractions",
        "pprint": "import pprint",
        "pickle": "import pickle",
        "gzip": "import gzip",
        "zipfile": "import zipfile",
        "tarfile": "import tarfile",
        "xml": "import xml",
        "html": "import html",
        "urllib": "import urllib",
        # Additional class-level stdlib imports
        "Decimal": "from decimal import Decimal",
        "NamedTuple": "from typing import NamedTuple",
        "Set": "from typing import Set",
        "Callable": "from typing import Callable",
        "ClassVar": "from typing import ClassVar",
        "Protocol": "from typing import Protocol",
        "TypeVar": "from typing import TypeVar",
        "Generic": "from typing import Generic",
        "deque": "from collections import deque",
        "ChainMap": "from collections import ChainMap",
        "partial": "from functools import partial",
        "reduce": "from functools import reduce",
        "total_ordering": "from functools import total_ordering",
        "suppress": "from contextlib import suppress",
        "redirect_stdout": "from contextlib import redirect_stdout",
        "redirect_stderr": "from contextlib import redirect_stderr",
        "patch": "from unittest.mock import patch",
        "MagicMock": "from unittest.mock import MagicMock",
        "Mock": "from unittest.mock import Mock",
        "PropertyMock": "from unittest.mock import PropertyMock",
        "TestCase": "from unittest import TestCase",
        "TemporaryDirectory": "from tempfile import TemporaryDirectory",
        "NamedTemporaryFile": "from tempfile import NamedTemporaryFile",
        "StringIO": "from io import StringIO",
        "BytesIO": "from io import BytesIO",
    }




    # ──────────────────────────────────────────────────────────────
    # v0.9.7: Deterministic test TypeError auto-fix
    #
    # When model writes Task(id="1", payload={}) but source defines
    # Task(task_id="1", data={}), pytest reports:
    #   TypeError: Task.__init__() got an unexpected keyword argument 'id'
    #
    # This method parses the error, looks up actual constructor params
    # from source files, finds the best match, and rewrites the test.
    # Zero LLM cost, handles the #1 test failure pattern with Qwen3.
    # ──────────────────────────────────────────────────────────────

    # Common semantic renames: wrong_kwarg → [possible_correct_kwargs]
    KWARG_RENAME_MAP = {
        "id": ["task_id", "item_id", "record_id", "entry_id", "uid", "identifier", "job_id"],
        "payload": ["data", "content", "body", "message", "task_data", "args"],
        "name": ["task_name", "title", "label", "display_name", "worker_name"],
        "callback": ["handler", "func", "function", "on_complete", "processor"],
        "timeout": ["max_delay", "delay", "wait_time", "deadline"],
        "worker_count": ["num_workers", "max_workers", "pool_size", "size", "concurrency"],
        "size": ["maxsize", "max_size", "capacity", "limit"],
        "maxsize": ["max_size", "size", "capacity"],
        "max_size": ["maxsize", "size", "capacity"],
        "retries": ["max_retries", "retry_count", "attempts", "max_attempts"],
        "max_retries": ["retries", "retry_count", "attempts", "max_attempts"],
        "delay": ["base_delay", "wait", "interval", "backoff"],
        "error": ["exception", "exc", "error_type", "errors"],
        "result": ["output", "return_value", "response", "value"],
        "status": ["state", "phase"],
        "created": ["created_at", "timestamp", "time", "created_time"],
        "updated": ["updated_at", "modified_at", "last_modified"],
        "description": ["desc", "details", "info", "summary"],
        "type": ["task_type", "kind", "category"],
    }

    def _auto_fix_test_type_errors(self, test_filepath, error_output: str) -> bool:
        """
        v0.9.7: Deterministic fix for TypeError from wrong constructor kwargs.

        Parses pytest error output for:
          TypeError: Task.__init__() got an unexpected keyword argument 'id'

        Looks up actual constructor params from source files (both @dataclass
        fields and explicit __init__ args), finds the best match, and rewrites.

        Returns True if any fixes were applied.
        """
        import ast
        import re as re_mod

        test_filepath = Path(test_filepath)

        type_errors = re_mod.findall(
            r"TypeError: (\w+)\.__init__\(\) got an unexpected keyword argument '(\w+)'",
            error_output
        )
        if not type_errors:
            return False

        # Build param map: {ClassName: [param_names]}
        class_params = {}
        for src_file in self.working_dir.glob("*.py"):
            if src_file.name.startswith("test_") or src_file.name == "__init__.py":
                continue
            try:
                src_content = src_file.read_text()
                src_tree = ast.parse(src_content)
            except Exception:
                continue
            for node in ast.walk(src_tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                is_dataclass = any(
                    (isinstance(d, ast.Name) and d.id == "dataclass") or
                    (isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                     and d.func.id == "dataclass")
                    for d in node.decorator_list
                )
                if is_dataclass:
                    params = []
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            params.append(item.target.id)
                    class_params[node.name] = params
                else:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                            params = [a.arg for a in item.args.args if a.arg != "self"]
                            class_params[node.name] = params

        if not class_params:
            return False

        try:
            test_content = test_filepath.read_text()
        except Exception:
            return False

        fixed = False
        for class_name, wrong_kwarg in set(type_errors):
            if class_name not in class_params:
                continue
            actual_params = class_params[class_name]
            if wrong_kwarg in actual_params:
                continue

            best_match = None
            # Strategy 1: Substring match
            for p in actual_params:
                if wrong_kwarg in p or p in wrong_kwarg:
                    best_match = p
                    break
            # Strategy 2: Semantic rename map
            if not best_match and wrong_kwarg in self.KWARG_RENAME_MAP:
                for candidate in self.KWARG_RENAME_MAP[wrong_kwarg]:
                    if candidate in actual_params:
                        best_match = candidate
                        break
            # Strategy 3: Reverse map
            if not best_match:
                for actual_p in actual_params:
                    if actual_p in self.KWARG_RENAME_MAP:
                        if wrong_kwarg in self.KWARG_RENAME_MAP[actual_p]:
                            best_match = actual_p
                            break

            if best_match:
                new_lines = []
                for line in test_content.split('\n'):
                    if class_name in line and f"{wrong_kwarg}=" in line:
                        line = line.replace(f"{wrong_kwarg}=", f"{best_match}=")
                        fixed = True
                    new_lines.append(line)
                test_content = '\n'.join(new_lines)
                logger.info(f"    🔧 API FIX: {class_name}.{wrong_kwarg} → {best_match}")

        # Also fix attribute access (.id → .task_id)
        for class_name, wrong_kwarg in set(type_errors):
            if class_name not in class_params:
                continue
            actual_params = class_params[class_name]
            best_match = None
            for p in actual_params:
                if wrong_kwarg in p or p in wrong_kwarg:
                    best_match = p
                    break
            if not best_match and wrong_kwarg in self.KWARG_RENAME_MAP:
                for candidate in self.KWARG_RENAME_MAP[wrong_kwarg]:
                    if candidate in actual_params:
                        best_match = candidate
                        break
            if best_match and best_match != wrong_kwarg:
                pattern = rf'\.{re_mod.escape(wrong_kwarg)}\b'
                new_content = re_mod.sub(pattern, f'.{best_match}', test_content)
                if new_content != test_content:
                    test_content = new_content
                    fixed = True

        if fixed:
            test_filepath.write_text(test_content)
            return True
        return False

    # v0.9.4: Third-party import map for commonly used packages
    # These aren't stdlib but are always pip-installable and commonly
    # used in the generated projects. The resolver tries these AFTER
    # stdlib and project imports.
    THIRD_PARTY_IMPORT_MAP = {
        # Flask ecosystem
        "Flask": "from flask import Flask",
        "request": "from flask import request",
        "jsonify": "from flask import jsonify",
        "Blueprint": "from flask import Blueprint",
        "render_template": "from flask import render_template",
        "redirect": "from flask import redirect",
        "url_for": "from flask import url_for",
        "abort": "from flask import abort",
        "make_response": "from flask import make_response",
        "Response": "from flask import Response",
        "g": "from flask import g",
        "session": "from flask import session",
        "current_app": "from flask import current_app",
        "flash": "from flask import flash",
        "send_file": "from flask import send_file",
        "send_from_directory": "from flask import send_from_directory",
        # Flask extensions
        "CORS": "from flask_cors import CORS",
        "SQLAlchemy": "from flask_sqlalchemy import SQLAlchemy",
        "Migrate": "from flask_migrate import Migrate",
        "LoginManager": "from flask_login import LoginManager",
        "login_required": "from flask_login import login_required",
        "current_user": "from flask_login import current_user",
        # Testing
        "pytest": "import pytest",
        "fixture": "import pytest",  # @pytest.fixture — just need pytest imported
        # Requests library
        "requests": "import requests",
        # FastAPI
        "FastAPI": "from fastapi import FastAPI",
        "HTTPException": "from fastapi import HTTPException",
        "Depends": "from fastapi import Depends",
        "APIRouter": "from fastapi import APIRouter",
        "Query": "from fastapi import Query",
        "Body": "from fastapi import Body",
        "BaseModel": "from pydantic import BaseModel",
        "Field": "from pydantic import Field",
        "validator": "from pydantic import validator",
        # SQLAlchemy standalone
        "create_engine": "from sqlalchemy import create_engine",
        "Column": "from sqlalchemy import Column",
        "Integer": "from sqlalchemy import Integer",
        "String": "from sqlalchemy import String",
        "ForeignKey": "from sqlalchemy import ForeignKey",
        "relationship": "from sqlalchemy.orm import relationship",
        "sessionmaker": "from sqlalchemy.orm import sessionmaker",
        "declarative_base": "from sqlalchemy.orm import declarative_base",
    }

    def _auto_fix_stdlib_imports(self, filepath) -> list:
        """
        v0.8.4: Deterministic stdlib import auto-fixer.

        Walks the AST of a Python file, finds all Name references,
        checks which ones are undefined (not imported, not assigned,
        not a builtin), and auto-injects the correct import statement
        for known stdlib modules.

        Returns list of import statements that were injected.
        """
        import ast

        if not filepath.exists():
            return []

        try:
            source = filepath.read_text()
            tree = ast.parse(source)
        except SyntaxError:
            return []  # Can't parse — don't touch it

        # Collect all names that are DEFINED in this file
        defined_names = set()

        # 1. Imports (import X, from X import Y)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    defined_names.add(alias.asname or alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # "from X import Y" defines Y, not X
                    for alias in node.names:
                        defined_names.add(alias.asname or alias.name)
                    # But also mark the module itself as imported
                    defined_names.add(node.module.split('.')[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined_names.add(node.name)
                # Function args are local, add them
                for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                    defined_names.add(arg.arg)
            elif isinstance(node, ast.ClassDef):
                defined_names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_names.add(target.id)
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                defined_names.add(elt.id)
            elif isinstance(node, ast.For):
                if isinstance(node.target, ast.Name):
                    defined_names.add(node.target.id)
            elif isinstance(node, ast.With):
                for item in node.items:
                    if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                        defined_names.add(item.optional_vars.id)
            elif isinstance(node, ast.ExceptHandler):
                if node.name:
                    defined_names.add(node.name)
            elif isinstance(node, ast.Global):
                for name in node.names:
                    defined_names.add(name)

        # Python builtins — never need importing
        builtins = {
            'print', 'len', 'range', 'int', 'str', 'float', 'bool', 'list',
            'dict', 'set', 'tuple', 'type', 'isinstance', 'issubclass',
            'getattr', 'setattr', 'hasattr', 'delattr', 'property',
            'staticmethod', 'classmethod', 'super', 'object',
            'True', 'False', 'None', 'Exception', 'ValueError',
            'TypeError', 'KeyError', 'IndexError', 'AttributeError',
            'RuntimeError', 'NotImplementedError', 'StopIteration',
            'FileNotFoundError', 'IOError', 'OSError', 'ImportError',
            'NameError', 'ZeroDivisionError', 'OverflowError',
            'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
            'min', 'max', 'sum', 'abs', 'round', 'any', 'all',
            'open', 'input', 'id', 'hash', 'repr', 'format',
            'chr', 'ord', 'hex', 'oct', 'bin', 'bytes', 'bytearray',
            'memoryview', 'complex', 'frozenset', 'slice', 'iter', 'next',
            'callable', 'vars', 'dir', 'globals', 'locals', 'exec', 'eval',
            'compile', 'breakpoint', 'exit', 'quit', '__name__', '__file__',
            '__doc__', '__import__', 'self', 'cls',
        }
        defined_names.update(builtins)

        # 2. Collect all Name references used in the code
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # For `datetime.now()`, the Name is `datetime`
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)

        # 3. Find undefined names that match known stdlib modules
        missing = used_names - defined_names
        injected = []

        for name in sorted(missing):
            if name in self.STDLIB_IMPORT_MAP:
                import_stmt = self.STDLIB_IMPORT_MAP[name]
                # Don't duplicate — check if already present as text
                if import_stmt not in source:
                    injected.append(import_stmt)

        if not injected:
            return []

        # 4. Inject imports at the top of the file (after any existing imports or docstrings)
        lines = source.split('\n')
        insert_pos = 0

        # Skip shebang, encoding declarations, and docstrings
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                insert_pos = idx + 1
                # Handle multi-line docstrings
                if stripped.startswith('"""') and stripped.count('"""') == 1:
                    for j in range(idx + 1, len(lines)):
                        if '"""' in lines[j]:
                            insert_pos = j + 1
                            break
                elif stripped.startswith("'''") and stripped.count("'''") == 1:
                    for j in range(idx + 1, len(lines)):
                        if "'''" in lines[j]:
                            insert_pos = j + 1
                            break
            elif stripped.startswith('import ') or stripped.startswith('from '):
                insert_pos = idx + 1  # Insert after last existing import
            elif stripped and not stripped.startswith('#'):
                break  # Hit actual code — stop looking

        # Find the true last import line
        for idx in range(len(lines) - 1, -1, -1):
            stripped = lines[idx].strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                insert_pos = idx + 1
                break

        # Insert the missing imports
        import_block = '\n'.join(injected)
        lines.insert(insert_pos, import_block)
        filepath.write_text('\n'.join(lines))

        logger.info(f"    🔧 AUTO-IMPORT: injected {len(injected)} stdlib imports into {filepath.name}: {', '.join(injected)}")
        return injected

    # ──────────────────────────────────────────────────────────────
    # v0.9.3: Comprehensive post-build import resolver
    #
    # Runs AFTER all files in the micro-build sequence are written.
    # Has the complete picture: what's defined where, what imports
    # what. Fixes three classes of import errors deterministically:
    #
    # 1. WRONG MODULE PATH: from tracker.models import X → from models import X
    #    LLMs hallucinate package prefixes that don't exist.
    #
    # 2. MISSING INTER-FILE IMPORTS: uses TrackerDB but never imports it.
    #    AST-walk finds undefined names, export map resolves them.
    #
    # 3. MISSING STDLIB IMPORTS: uses datetime but never imports it.
    #    Delegates to _auto_fix_stdlib_imports for each file.
    #
    # Inspired by pyflyby (D.E. Shaw) which uses AST to find and
    # insert missing imports with 100% precision.
    # ──────────────────────────────────────────────────────────────

    def _resolve_project_imports(self, filenames: list) -> int:
        """
        v0.9.3: Deterministic post-build import resolver.

        After all files in the build sequence are written, this method:
        1. Builds an export map: {symbol_name: module_name} for all project files
        2. Fixes wrong module paths (hallucinated package prefixes)
        3. Adds missing inter-file imports
        4. Runs stdlib import fix on each file

        Returns total number of fixes applied.
        """
        import ast
        import re as re_mod

        total_fixes = 0

        # --- Phase 1: Build the export map ---
        # Scan all .py files to learn what's defined where
        export_map = {}  # {symbol_name: module_name}
        module_files = {}  # {module_name: filepath}
        all_modules = set()  # All module names in the project

        for py_file in self.working_dir.rglob("*.py"):
            # Skip venv, __pycache__, .agents
            rel = py_file.relative_to(self.working_dir)
            rel_str = str(rel)
            if any(skip in rel_str for skip in ['venv/', '__pycache__/', '.agents/', 'node_modules/']):
                continue

            # Module name: routes/projects.py → routes.projects
            # But also just: models.py → models
            module_name = str(rel).replace('.py', '').replace('/', '.').replace('\\', '.')
            simple_name = py_file.stem  # Just the filename without extension
            all_modules.add(module_name)
            all_modules.add(simple_name)
            module_files[module_name] = py_file
            module_files[simple_name] = py_file

            try:
                source = py_file.read_text()
                tree = ast.parse(source)
            except SyntaxError:
                continue

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    export_map[node.name] = module_name
                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    if not node.name.startswith('_'):
                        export_map[node.name] = module_name
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            # Top-level CONSTANTS
                            export_map[target.id] = module_name

        # Also map simple module stems for subdirectory modules
        # e.g., routes/projects.py: export_map entries use module_name "routes.projects"
        # but we also want to resolve "projects" → "routes.projects"

        # --- Phase 2: Fix each file ---
        for filename in filenames:
            filepath = self.working_dir / filename
            if not filepath.exists() or not filepath.name.endswith('.py'):
                continue

            try:
                source = filepath.read_text()
                tree = ast.parse(source)
            except SyntaxError:
                # Try stdlib fix anyway (it handles parse errors gracefully)
                stdlib_fixes = self._auto_fix_stdlib_imports(filepath)
                total_fixes += len(stdlib_fixes)
                continue

            this_module = filename.replace('.py', '').replace('/', '.').replace('\\', '.')
            lines = source.split('\n')
            changed = False

            # --- Phase 2a: Fix wrong module paths in import statements ---
            # e.g., "from tracker.models import Project" when "models.py" exists
            # e.g., "from project.tracker_db import TrackerDB" when "tracker_db.py" exists
            new_lines = []
            for line in lines:
                m = re_mod.match(r'^(from\s+)([\w.]+)(\s+import\s+.+)$', line)
                if m:
                    prefix, module_path, suffix = m.groups()
                    parts = module_path.split('.')

                    # Check if the full module path exists
                    if module_path not in all_modules:
                        # Try progressively shorter suffixes
                        # e.g., tracker.models → models, project.tracker_db → tracker_db
                        fixed_module = None
                        for i in range(1, len(parts)):
                            candidate = '.'.join(parts[i:])
                            if candidate in all_modules:
                                fixed_module = candidate
                                break

                        # Also try: if "from tracker import app" and "app" is a module
                        if not fixed_module:
                            imported_names = [n.strip().split(' as ')[0] for n in suffix.replace('import', '').split(',')]
                            for name in imported_names:
                                if name.strip() in all_modules:
                                    fixed_module = name.strip()
                                    break

                        if fixed_module and fixed_module != module_path:
                            logger.info(f"    🔧 IMPORT FIX: {module_path} → {fixed_module} in {filename}")
                            line = f"{prefix}{fixed_module}{suffix}"
                            changed = True

                # Also check "import X" statements
                m2 = re_mod.match(r'^(import\s+)([\w.]+)(.*)$', line)
                if m2 and not line.strip().startswith('from'):
                    imp_prefix, module_path, imp_suffix = m2.groups()
                    parts = module_path.split('.')
                    if module_path not in all_modules and len(parts) > 1:
                        for i in range(1, len(parts)):
                            candidate = '.'.join(parts[i:])
                            if candidate in all_modules:
                                logger.info(f"    🔧 IMPORT FIX: import {module_path} → import {candidate} in {filename}")
                                line = f"{imp_prefix}{candidate}{imp_suffix}"
                                changed = True
                                break

                new_lines.append(line)

            if changed:
                source = '\n'.join(new_lines)
                filepath.write_text(source)
                total_fixes += 1
                # Re-parse after changes
                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    continue

            # --- Phase 2b: Find missing inter-file imports ---
            # AST-walk to find all used names, compare against defined names
            defined_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        defined_names.add(alias.asname or alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            defined_names.add(alias.asname or alias.name)
                        defined_names.add(node.module.split('.')[0])
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defined_names.add(node.name)
                    for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                        defined_names.add(arg.arg)
                elif isinstance(node, ast.ClassDef):
                    defined_names.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined_names.add(target.id)
                        elif isinstance(target, ast.Tuple):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    defined_names.add(elt.id)
                elif isinstance(node, ast.For):
                    if isinstance(node.target, ast.Name):
                        defined_names.add(node.target.id)
                elif isinstance(node, ast.With):
                    for item in node.items:
                        if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                            defined_names.add(item.optional_vars.id)
                elif isinstance(node, ast.ExceptHandler):
                    if node.name:
                        defined_names.add(node.name)
                elif isinstance(node, ast.ListComp) or isinstance(node, ast.SetComp) or isinstance(node, ast.GeneratorExp) or isinstance(node, ast.DictComp):
                    for gen in node.generators:
                        if isinstance(gen.target, ast.Name):
                            defined_names.add(gen.target.id)

            # Python builtins
            builtins = {
                'print', 'len', 'range', 'int', 'str', 'float', 'bool', 'list',
                'dict', 'set', 'tuple', 'type', 'isinstance', 'issubclass',
                'getattr', 'setattr', 'hasattr', 'delattr', 'property',
                'staticmethod', 'classmethod', 'super', 'object',
                'True', 'False', 'None', 'Exception', 'ValueError',
                'TypeError', 'KeyError', 'IndexError', 'AttributeError',
                'RuntimeError', 'NotImplementedError', 'StopIteration',
                'FileNotFoundError', 'IOError', 'OSError', 'ImportError',
                'NameError', 'ZeroDivisionError', 'OverflowError',
                'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
                'min', 'max', 'sum', 'abs', 'round', 'any', 'all',
                'open', 'input', 'id', 'hash', 'repr', 'format',
                'chr', 'ord', 'hex', 'oct', 'bin', 'bytes', 'bytearray',
                'memoryview', 'complex', 'frozenset', 'slice', 'iter', 'next',
                'callable', 'vars', 'dir', 'globals', 'locals', 'exec', 'eval',
                'compile', 'breakpoint', 'exit', 'quit', '__name__', '__file__',
                '__doc__', '__import__', 'self', 'cls',
                # v0.9.4: REMOVED Flask names from builtins! They need real imports.
                # Flask, request, jsonify, etc. are NOT builtins — the resolver
                # must add `from flask import ...` when they're used.
                # Also add common pytest fixtures and test helpers as pseudo-builtins
                # (they get injected by the test framework, not imported)
                'client', 'app', 'fixture', 'monkeypatch', 'capsys', 'capfd',
                'tmp_path', 'tmpdir', 'caplog',
            }
            defined_names.update(builtins)

            # Collect all Name references
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        used_names.add(node.value.id)

            # Find undefined names that exist in the export map
            missing = used_names - defined_names
            injected_imports = []

            for name in sorted(missing):
                if name in export_map:
                    src_module = export_map[name]
                    # Don't import from self
                    if src_module == this_module:
                        continue
                    # Use simple module name if possible
                    simple = src_module.split('.')[-1] if '.' in src_module else src_module
                    import_line = f"from {simple} import {name}"
                    # Check for the full dotted path too
                    full_import = f"from {src_module} import {name}"
                    if import_line not in source and full_import not in source and f"import {name}" not in source:
                        injected_imports.append(import_line)

            if injected_imports:
                source = filepath.read_text()
                lines = source.split('\n')

                # Find insertion point (after last import)
                insert_pos = 0
                for idx, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith('import ') or stripped.startswith('from '):
                        insert_pos = idx + 1
                    elif stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''") and insert_pos > 0:
                        break

                # Also check from the end for the true last import
                for idx in range(len(lines) - 1, -1, -1):
                    stripped = lines[idx].strip()
                    if stripped.startswith('import ') or stripped.startswith('from '):
                        insert_pos = max(insert_pos, idx + 1)
                        break

                import_block = '\n'.join(injected_imports)
                lines.insert(insert_pos, import_block)
                filepath.write_text('\n'.join(lines))
                total_fixes += len(injected_imports)
                logger.info(f"    🔧 AUTO-IMPORT: added {len(injected_imports)} project imports to {filename}: "
                           f"{', '.join(injected_imports)}")

            # --- Phase 2c: Stdlib imports (delegate to existing method) ---
            stdlib_fixes = self._auto_fix_stdlib_imports(filepath)
            total_fixes += len(stdlib_fixes)

            # --- Phase 2d: Third-party imports (v0.9.4) ---
            # Re-read source after previous fixes
            try:
                source = filepath.read_text()
                tree = ast.parse(source)
            except SyntaxError:
                continue

            # Re-collect defined and used names after previous fixes
            defined_after = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        defined_after.add(alias.asname or alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            defined_after.add(alias.asname or alias.name)
                        defined_after.add(node.module.split('.')[0])
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defined_after.add(node.name)
                    for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                        defined_after.add(arg.arg)
                elif isinstance(node, ast.ClassDef):
                    defined_after.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined_after.add(target.id)

            # Python builtins (compact set for phase 2d)
            phase2d_builtins = {
                'print', 'len', 'range', 'int', 'str', 'float', 'bool', 'list',
                'dict', 'set', 'tuple', 'type', 'isinstance', 'issubclass',
                'getattr', 'setattr', 'hasattr', 'delattr', 'property',
                'staticmethod', 'classmethod', 'super', 'object',
                'True', 'False', 'None', 'Exception', 'ValueError',
                'TypeError', 'KeyError', 'IndexError', 'AttributeError',
                'RuntimeError', 'NotImplementedError', 'StopIteration',
                'FileNotFoundError', 'IOError', 'OSError', 'ImportError',
                'NameError', 'enumerate', 'zip', 'map', 'filter', 'sorted',
                'reversed', 'min', 'max', 'sum', 'abs', 'round', 'any', 'all',
                'open', 'input', 'id', 'hash', 'repr', 'format', 'bytes',
                'bytearray', 'callable', 'vars', 'dir', 'globals', 'locals',
                'exec', 'eval', 'compile', 'iter', 'next', 'chr', 'ord',
                'self', 'cls',
                # Pytest fixtures (injected, not imported)
                'client', 'app', 'fixture', 'monkeypatch', 'capsys', 'tmp_path',
            }
            defined_after.update(phase2d_builtins)

            used_after = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_after.add(node.id)
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        used_after.add(node.value.id)

            still_missing = used_after - defined_after
            tp_injected = []

            for name in sorted(still_missing):
                if name in self.THIRD_PARTY_IMPORT_MAP:
                    import_stmt = self.THIRD_PARTY_IMPORT_MAP[name]
                    if import_stmt not in source and f"import {name}" not in source:
                        tp_injected.append(import_stmt)

            if tp_injected:
                # Deduplicate (e.g., multiple Flask imports → one `from flask import X, Y`)
                # Group by source module
                module_imports: Dict[str, list] = {}  # {module: [names]}
                standalone_imports = []
                for stmt in tp_injected:
                    m = re_mod.match(r'from\s+(\S+)\s+import\s+(.+)', stmt)
                    if m:
                        mod, names = m.groups()
                        if mod not in module_imports:
                            module_imports[mod] = []
                        module_imports[mod].extend([n.strip() for n in names.split(',')])
                    else:
                        standalone_imports.append(stmt)

                # Build consolidated import lines
                final_imports = list(set(standalone_imports))
                for mod, names in module_imports.items():
                    unique_names = sorted(set(names))
                    # Check which names are actually missing from existing imports
                    needed = [n for n in unique_names if f"import {n}" not in source]
                    if needed:
                        final_imports.append(f"from {mod} import {', '.join(needed)}")

                if final_imports:
                    lines = source.split('\n')
                    insert_pos = 0
                    for idx, line in enumerate(lines):
                        stripped = line.strip()
                        if stripped.startswith('import ') or stripped.startswith('from '):
                            insert_pos = idx + 1
                    for idx in range(len(lines) - 1, -1, -1):
                        stripped = lines[idx].strip()
                        if stripped.startswith('import ') or stripped.startswith('from '):
                            insert_pos = max(insert_pos, idx + 1)
                            break

                    import_block = '\n'.join(final_imports)
                    lines.insert(insert_pos, import_block)
                    filepath.write_text('\n'.join(lines))
                    total_fixes += len(final_imports)
                    logger.info(f"    🔧 THIRD-PARTY IMPORT: added {len(final_imports)} imports to {filename}: "
                               f"{', '.join(final_imports)}")

        return total_fixes

    # ──────────────────────────────────────────────────────────────
    # v0.9.4: Fix datetime module/class confusion
    #
    # The #1 type_error from local models: they write `import datetime`
    # (the module) then call `datetime.strptime()` or `datetime.now()`
    # which only exist on datetime.datetime (the class).
    #
    # This is a deterministic text transform — no LLM needed.
    # ──────────────────────────────────────────────────────────────

    def _fix_datetime_confusion(self, filenames: list) -> int:
        """
        v0.9.4: Fix datetime module vs class confusion.

        When model writes:
            import datetime
            ...
            datetime.strptime(x, fmt)   # WRONG — strptime is on datetime.datetime
            datetime.now()              # WRONG — now() is on datetime.datetime

        Fix to:
            import datetime
            ...
            datetime.datetime.strptime(x, fmt)
            datetime.datetime.now()

        Also handles the reverse: if model already has `from datetime import datetime`
        then `datetime.strptime()` is correct and we don't touch it.

        Returns number of files fixed.
        """
        import re as re_mod

        # Methods that exist on datetime.datetime CLASS but not datetime MODULE
        CLASS_METHODS = {
            'strptime', 'now', 'utcnow', 'today', 'fromtimestamp',
            'utcfromtimestamp', 'fromisoformat', 'combine', 'strftime',
        }
        # Attributes on datetime.datetime instances
        INSTANCE_ATTRS = {
            'year', 'month', 'day', 'hour', 'minute', 'second',
            'microsecond', 'tzinfo', 'isoformat', 'timestamp',
            'date', 'time', 'replace', 'timetuple',
        }

        fixes = 0

        for filename in filenames:
            filepath = self.working_dir / filename
            if not filepath.exists() or not filepath.name.endswith('.py'):
                continue

            try:
                content = filepath.read_text()
            except Exception:
                continue

            # Skip if they already have `from datetime import datetime` — then datetime.X is correct
            if re_mod.search(r'^from\s+datetime\s+import\s+.*\bdatetime\b', content, re_mod.MULTILINE):
                continue

            # Only fix if they have `import datetime` (the module import)
            if not re_mod.search(r'^import\s+datetime\b', content, re_mod.MULTILINE):
                continue

            changed = False

            # Fix datetime.METHOD() → datetime.datetime.METHOD()
            for method in CLASS_METHODS:
                pattern = rf'(?<!\.)datetime\.{method}\b'
                replacement = f'datetime.datetime.{method}'
                if re_mod.search(pattern, content):
                    content = re_mod.sub(pattern, replacement, content)
                    changed = True

            # Fix variable.strftime() when variable came from datetime.strptime → leave alone
            # But fix datetime(year, month, day) → datetime.datetime(year, month, day)
            # Pattern: datetime( but not datetime.datetime(
            pattern = r'(?<!\.)(?<!datetime\.)datetime\s*\('
            if re_mod.search(pattern, content):
                # Check it's being used as a constructor (not as module reference)
                for match in re_mod.finditer(r'(?<!\.)(?<!datetime\.)datetime\s*\(\s*\d', content):
                    content = content[:match.start()] + 'datetime.datetime(' + content[match.end()-1:]
                    changed = True
                    break  # Re-do from scratch since positions shifted
                # Simpler: just ensure any bare datetime( becomes datetime.datetime(
                # where it's clearly a constructor call
                content = re_mod.sub(
                    r'(?<!\.)(?<!datetime\.)(?<!from )(?<!import )datetime\(',
                    'datetime.datetime(',
                    content
                )
                changed = True

            if changed:
                filepath.write_text(content)
                fixes += 1
                logger.info(f"    🔧 DATETIME FIX: fixed module/class confusion in {filename}")

        return fixes

    def _sample_test_file(
        self,
        task_state: TaskState,
        step: dict,
        manifest_text: str,
        step_number: int,
        total_steps: int,
        num_candidates: int = 4,
    ) -> tuple:
        """
        Multi-patch sampling for test files (v0.6.1 + v0.6.2).

        v0.6.1: Generate N candidates at different temperatures, validate each.
        v0.6.2 Fix 2: If all candidates fail, capture error output and inject
                       into a second wave of candidates with error context.
        v0.6.2 Fix 3: Second wave uses 4 more candidates with error-aware prompts.

        Inspired by Agentless (SWE-bench) which generates up to 40 samples,
        and research showing that error context significantly improves
        test generation reliability.

        Returns (AgentResult, verification_dict)
        """
        from standalone_models import AgentResult

        filename = step["file"]
        filepath = self.working_dir / filename
        tests_for = step.get("tests_for", "")

        # v0.8.0: Proactive KB lookup — get relevant patterns before building tests
        kb_build_context = self.kb.get_build_context(
            f"{task_state.goal} {filename} test {tests_for}"
        )
        if kb_build_context:
            logger.info(f"  📚 KB provided proactive context for {filename}")

        # === WAVE 1: Standard sampling (no error context) ===
        temperatures_wave1 = [0.3, 0.6, 0.8, 1.0][:num_candidates]
        logger.info(f"  🎲 Sampling {num_candidates} candidates for {filename}")

        best_score = -1
        best_content = None
        best_verification = {"exists": False, "status": "NO CANDIDATES", "exports": "none"}
        collected_errors = []  # Collect error output for wave 2

        for idx, temp in enumerate(temperatures_wave1):
            logger.info(f"    Candidate {idx+1}/{num_candidates} (temp={temp})...")

            # v0.9.3: Plain-text build mode
            result = self.agent_runner.run_build_single_file_plain(
                state=task_state,
                filename=filename,
                step_info=step,
                manifest=manifest_text,
                step_number=step_number,
                total_steps=total_steps,
                temperature=temp,
                kb_context=kb_build_context,
            )

            if not result.success:
                logger.info(f"    ❌ Candidate {idx+1}: build failed")
                collected_errors.append(f"Candidate {idx+1}: build agent failed — {result.error}")
                continue

            # v0.9.3: Parse from plain-text markers
            parsed_content = self.agent_runner.parse_plain_file_content(result.output or "")
            if parsed_content:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(parsed_content)
            elif not filepath.exists():
                logger.info(f"    ❌ Candidate {idx+1}: file not created")
                collected_errors.append(f"Candidate {idx+1}: file was not created")
                continue

            candidate_content = filepath.read_text()

            # v0.7.1: Import hygiene — auto-fix missing imports before testing
            self._auto_fix_imports_precheck(filepath)
            candidate_content = filepath.read_text()  # re-read after potential fixes

            # Syntax check
            verification = self._verify_single_file(filename, True)
            if "SYNTAX ERROR" in verification.get("status", ""):
                logger.info(f"    ❌ Candidate {idx+1}: syntax error")
                collected_errors.append(f"Candidate {idx+1}: syntax error — {verification['status']}")
                continue

            # v0.9.7: Pre-flight type error fix — run tests once, if TypeError
            # about wrong kwargs, auto-fix before the real test run.
            preflight = self._run_test_file(filename, tests_for)
            if preflight.get("failed", 0) > 0 or preflight.get("errors", 0) > 0:
                preflight_output = preflight.get("output", "")
                if "unexpected keyword argument" in preflight_output:
                    if self._auto_fix_test_type_errors(filepath, preflight_output):
                        logger.info(f"    🔧 Pre-flight: fixed constructor kwargs in candidate {idx+1}")
                        candidate_content = filepath.read_text()

            # Run tests
            test_result = self._run_test_file(filename, tests_for)
            passed = test_result.get("passed", 0)
            failed = test_result.get("failed", 0)
            errors = test_result.get("errors", 0)
            total = passed + failed + errors
            error_output = test_result.get("output", "")

            # v0.7.4: Import hygiene fix-and-rerun.
            # If tests fail with NameError/ImportError, attempt deterministic fixes
            # on BOTH the test file AND any source files mentioned in tracebacks,
            # then re-run ONCE before rejecting the candidate.
            # This catches the 70B's most common mistake: writing correct logic
            # but forgetting `import uuid` or `from datetime import datetime`.
            if (failed > 0 or errors > 0) and error_output:
                fixed_any = False

                # Fix missing stdlib imports in the TEST file
                if self._auto_fix_imports(filepath, error_output):
                    fixed_any = True

                # Fix missing project imports in the TEST file
                if self._auto_fix_project_imports(filepath, error_output):
                    fixed_any = True

                # Also fix NameErrors in SOURCE files referenced by tracebacks.
                # e.g., storage.py:34: in load_tasks  (pytest --tb=short)
                # e.g., File "/tmp/.../storage.py", line 34  (full traceback)
                import re as _re
                # Match both formats: 'storage.py:34: in func' and 'File ".../storage.py", line 34'
                source_errors_short = _re.findall(r'(\w+\.py):\d+: in \w+', error_output)
                source_errors_long = _re.findall(r'File ".*?/([^/\"]+\.py)", line \d+', error_output)
                all_source_files = list(dict.fromkeys(source_errors_short + source_errors_long))
                for src_file in all_source_files:
                    if src_file != filename:  # Don't re-fix the test file
                        src_path = self.working_dir / src_file
                        if src_path.exists():
                            if self._auto_fix_imports(src_path, error_output):
                                fixed_any = True
                                logger.info(f"    🔧 Fixed source file: {src_file}")

                if fixed_any:
                    test_result2 = self._run_test_file(filename, tests_for)
                    passed = test_result2.get("passed", 0)
                    failed = test_result2.get("failed", 0)
                    errors = test_result2.get("errors", 0)
                    total = passed + failed + errors
                    error_output = test_result2.get("output", "")

                    if failed == 0 and errors == 0 and passed > 0:
                        logger.info(f"    ✅ Candidate {idx+1}: fixed via import hygiene — ALL {passed} TESTS PASS!")
                        verification["status"] = "OK"
                        verification["test_results"] = f"{passed}/{total} tests pass"
                        return result, verification

            if failed == 0 and errors == 0 and passed > 0:
                logger.info(f"    ✅ Candidate {idx+1}: ALL {passed} TESTS PASS!")
                verification["status"] = "OK"
                verification["test_results"] = f"{passed}/{total} tests pass"
                return result, verification

            logger.info(f"    ⚠️ Candidate {idx+1}: {passed}/{total} pass, {failed} fail, {errors} error")

            # Record trace for co-training loop
            source_code = ""
            if tests_for:
                # tests_for might be "cli.py" or "cli" — normalize
                source_base = tests_for.replace('.py', '')
                source_path = self.working_dir / f"{source_base}.py"
                if source_path.exists():
                    try:
                        source_code = source_path.read_text()
                    except Exception:
                        pass

            self.trace_collector.record_test_failure(
                test_filename=filename,
                source_filename=f"{tests_for.replace('.py', '')}.py" if tests_for else "",
                prompt_excerpt=f"Step {step_number}/{total_steps}: Create {filename}",
                generated_test_code=candidate_content,
                source_code=source_code,
                test_output=error_output,
                passed=passed,
                failed=failed,
                errors=errors,
                error_category=self._classify_test_error(error_output),
                model_used=self.config.get_agent("build").model.model_id,
                temperature=temp,
                iteration=task_state.iteration,
                task_goal=task_state.goal or "",
                candidate_number=idx + 1,
                total_candidates=num_candidates,
            )

            # Collect error for wave 2
            collected_errors.append(
                f"Candidate {idx+1} (temp={temp}): {passed}/{total} pass. "
                f"Error output:\n{error_output[-300:]}"
            )

            if passed > best_score:
                best_score = passed
                best_content = candidate_content
                best_verification = verification
                best_verification["test_results"] = f"{passed}/{total} tests pass"

        # === WAVE 2: Edit-based repair (v0.9.8, replaces full regeneration) ===
        # Research: Aider, OpenHands, CodeAct all show edit-based repair beats
        # regeneration for iterative fixing. Only regenerate if total garbage (0/N).
        # v1.0: Skip edit repair for small files — whole-file regen is more
        # reliable when the file fits easily in context. Aider uses "whole"
        # format for small files for the same reason.
        best_line_count = best_content.count('\n') + 1 if best_content else 0
        use_edit_repair = (best_score > 0 and best_content and collected_errors
                          and best_line_count > 80)
        if best_score > 0 and best_content and not use_edit_repair and collected_errors:
            logger.info(f"  📝 Small file ({best_line_count} lines) — skipping edit repair, using Wave 2 regen")
        if use_edit_repair:
            # Partial pass — switch to surgical edit mode
            logger.info(f"  🔧 Edit repair: {best_score} tests pass, fixing failures surgically")
            filepath.write_text(best_content)

            # Get source context for the repair agent
            repair_source_ctx = ""
            repair_contract = ""
            if tests_for:
                src_name = tests_for if tests_for.endswith('.py') else tests_for + '.py'
                src_path = self.working_dir / src_name
                if src_path.exists():
                    try:
                        src_content = src_path.read_text()[:6000]  # v1.0: 3000→6000
                        repair_source_ctx = f"## Source Module: `{src_name}`\n```python\n{src_content}\n```"
                        repair_contract = self.agent_runner._extract_api_contract(src_name, src_content)
                    except Exception:
                        pass

            for edit_round in range(3):
                current_content = filepath.read_text()
                test_result_edit = self._run_test_file(filename, tests_for)
                edit_passed = test_result_edit.get("passed", 0)
                edit_failed = test_result_edit.get("failed", 0)
                edit_errors = test_result_edit.get("errors", 0)
                edit_output = test_result_edit.get("output", "")

                if edit_failed == 0 and edit_errors == 0 and edit_passed > 0:
                    logger.info(f"    ✅ Edit round {edit_round}: ALL {edit_passed} TESTS PASS!")
                    verification = self._verify_single_file(filename, True)
                    verification["status"] = "OK"
                    verification["test_results"] = f"{edit_passed}/{edit_passed} tests pass"
                    return AgentResult(success=True, output="Edit repair: all pass"), verification

                logger.info(f"    🔧 Edit round {edit_round + 1}/3: {edit_passed} pass, {edit_failed} fail, {edit_errors} error")

                # Ask model for surgical edits
                edit_result = self.agent_runner.run_edit_repair(
                    state=task_state,
                    filename=filename,
                    current_content=current_content,
                    test_output=edit_output,
                    source_context=repair_source_ctx,
                    contract_section=repair_contract,
                    temperature=0.2 + (edit_round * 0.15),  # Slightly increase temp each round
                )

                if not edit_result.success or not edit_result.output:
                    logger.info(f"    ❌ Edit round {edit_round + 1}: model produced no output")
                    break

                # Parse and apply edits
                edits = self.agent_runner.parse_search_replace(edit_result.output)
                if not edits:
                    logger.info(f"    ❌ Edit round {edit_round + 1}: no SEARCH/REPLACE blocks found")
                    # Fallback: try parsing as full file (model might have ignored edit format)
                    parsed = self.agent_runner.parse_plain_file_content(edit_result.output)
                    if parsed and len(parsed) > 100:
                        filepath.write_text(parsed)
                        logger.info(f"    🔄 Edit round {edit_round + 1}: model output full file instead of edits, using it")
                        continue
                    break

                apply_result = self.agent_runner.apply_search_replace(filepath, edits)
                # v0.9.9: apply_search_replace returns (count, feedback) tuple
                if isinstance(apply_result, tuple):
                    num_applied, edit_feedback = apply_result
                else:
                    num_applied, edit_feedback = apply_result, []

                logger.info(f"    📝 Edit round {edit_round + 1}: applied {num_applied}/{len(edits)} edits")

                if num_applied == 0:
                    if edit_feedback:
                        # v0.9.9: Store feedback for next edit round (Aider-style)
                        task_state.edit_feedback.extend(edit_feedback)
                        logger.info(f"    🔍 Edit round {edit_round + 1}: {len(edit_feedback)} match failures logged for retry")
                        # Don't break — let the next round use the feedback
                        continue
                    logger.info(f"    ❌ Edit round {edit_round + 1}: no edits matched — trying full regen fallback")
                    break
                else:
                    # Clear feedback on successful edits
                    task_state.edit_feedback = []

                # v0.9.7: Auto-fix imports after edits
                self._auto_fix_imports_precheck(filepath)

                # v0.9.7: Auto-fix type errors after edits
                post_edit_test = self._run_test_file(filename, tests_for)
                if post_edit_test.get("failed", 0) > 0:
                    pe_output = post_edit_test.get("output", "")
                    if "unexpected keyword argument" in pe_output:
                        self._auto_fix_test_type_errors(filepath, pe_output)

            # After edit rounds, check final state
            final_test = self._run_test_file(filename, tests_for)
            final_passed = final_test.get("passed", 0)
            final_failed = final_test.get("failed", 0)
            final_errors = final_test.get("errors", 0)

            if final_failed == 0 and final_errors == 0 and final_passed > 0:
                logger.info(f"    ✅ Edit repair complete: ALL {final_passed} TESTS PASS!")
                verification = self._verify_single_file(filename, True)
                verification["status"] = "OK"
                verification["test_results"] = f"{final_passed}/{final_passed} tests pass"
                return AgentResult(success=True, output="Edit repair: all pass"), verification

            # Update best if edits improved things
            if final_passed > best_score:
                best_score = final_passed
                best_content = filepath.read_text()
                best_verification["test_results"] = f"{final_passed}/{final_passed + final_failed + final_errors} tests pass"

        # === WAVE 2 FALLBACK: Full regeneration (only if edit repair failed or 0/N) ===
        # v1.0: Also run Wave 2 for small files that skipped edit repair
        needs_wave2 = (best_score <= 0 and collected_errors) or (
            not use_edit_repair and best_score > 0 and collected_errors)
        if needs_wave2:
            logger.info(f"  🔬 Wave 2: error-aware re-sampling for {filename}")

            # Build error context from wave 1 failures
            error_context = "\n".join(collected_errors[:3])  # Top 3 errors

            # v0.8.0: Query KB for known fix patterns
            kb_context = ""
            for err_text in collected_errors[:2]:
                kb_fix = self.kb.get_fix_for_error(err_text)
                if kb_fix:
                    kb_context += kb_fix
                    break  # One good fix is enough
            if kb_context:
                error_context += "\n" + kb_context
                logger.info(f"  📚 KB provided fix context for {filename}")

            # v0.7.2: Inject the best Wave 1 candidate code so Wave 2 can iterate
            # instead of starting from scratch. This turns random sampling into
            # iterative refinement — much more efficient use of inference compute.
            if best_content:
                error_context += f"""

## 📝 BEST ATTEMPT FROM PREVIOUS WAVE ({best_score} tests passed)
The code below was the closest to working. Fix ONLY the specific failures listed above.
Keep the overall structure, imports, and passing test methods intact.

```python
{best_content[:2500]}
```

Do NOT rewrite from scratch. Start from this code and fix the failing parts.
"""

            temperatures_wave2 = [0.2, 0.3, 0.5, 0.7]  # v0.9.6: Lower temps for repair (TestART)

            for idx, temp in enumerate(temperatures_wave2):
                logger.info(f"    Wave2 Candidate {idx+1}/4 (temp={temp})...")

                # v0.9.3: Plain-text build mode
                result = self.agent_runner.run_build_single_file_plain(
                    state=task_state,
                    filename=filename,
                    step_info=step,
                    manifest=manifest_text,
                    step_number=step_number,
                    total_steps=total_steps,
                    temperature=temp,
                    error_context=error_context,
                    kb_context=kb_build_context,
                )

                if not result.success:
                    logger.info(f"    ❌ Wave2 Candidate {idx+1}: failed")
                    continue

                # v0.9.3: Parse from plain-text markers
                parsed_content = self.agent_runner.parse_plain_file_content(result.output or "")
                if parsed_content:
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    filepath.write_text(parsed_content)
                elif not filepath.exists():
                    logger.info(f"    ❌ Wave2 Candidate {idx+1}: file not created")
                    continue

                candidate_content = filepath.read_text()

                # v0.7.1: Import hygiene — auto-fix missing imports before testing
                self._auto_fix_imports_precheck(filepath)
                candidate_content = filepath.read_text()  # re-read after potential fixes

                verification = self._verify_single_file(filename, True)
                if "SYNTAX ERROR" in verification.get("status", ""):
                    logger.info(f"    ❌ Wave2 Candidate {idx+1}: syntax error")
                    continue

                # v0.9.7: Pre-flight type error fix for Wave 2
                preflight_w2 = self._run_test_file(filename, tests_for)
                if preflight_w2.get("failed", 0) > 0 or preflight_w2.get("errors", 0) > 0:
                    pf_output = preflight_w2.get("output", "")
                    if "unexpected keyword argument" in pf_output:
                        if self._auto_fix_test_type_errors(filepath, pf_output):
                            logger.info("    🔧 Wave2 pre-flight: fixed constructor kwargs")

                test_result = self._run_test_file(filename, tests_for)
                passed = test_result.get("passed", 0)
                failed = test_result.get("failed", 0)
                errors = test_result.get("errors", 0)
                total = passed + failed + errors

                # v0.7.4: Import hygiene fix-and-rerun (same as Wave 1)
                if (failed > 0 or errors > 0):
                    w2_error = test_result.get("output", "")
                    fixed_any = False

                    if self._auto_fix_imports(filepath, w2_error):
                        fixed_any = True
                    if self._auto_fix_project_imports(filepath, w2_error):
                        fixed_any = True

                    # Fix source files referenced in tracebacks
                    import re as _re
                    source_errors_short = _re.findall(r'(\w+\.py):\d+: in \w+', w2_error)
                    source_errors_long = _re.findall(r'File ".*?/([^/\"]+\.py)", line \d+', w2_error)
                    all_source_files = list(dict.fromkeys(source_errors_short + source_errors_long))
                    for src_file in all_source_files:
                        if src_file != filename:
                            src_path = self.working_dir / src_file
                            if src_path.exists():
                                if self._auto_fix_imports(src_path, w2_error):
                                    fixed_any = True
                                    logger.info(f"    🔧 Fixed source file: {src_file}")

                    if fixed_any:
                        test_result2 = self._run_test_file(filename, tests_for)
                        passed = test_result2.get("passed", 0)
                        failed = test_result2.get("failed", 0)
                        errors = test_result2.get("errors", 0)
                        total = passed + failed + errors

                        if failed == 0 and errors == 0 and passed > 0:
                            logger.info(f"    ✅ Wave2 Candidate {idx+1}: fixed via import hygiene — ALL {passed} TESTS PASS!")
                            verification["status"] = "OK"
                            verification["test_results"] = f"{passed}/{total} tests pass"
                            # v0.8.0: Auto-capture the pattern
                            if collected_errors:
                                self.kb.capture_pattern(
                                    error_pattern=collected_errors[0][:200],
                                    solution=f"Fixed {filename} via Wave2 + import hygiene ({passed} tests pass)",
                                    example=filepath.read_text()[:500] if filepath.exists() else "",
                                    source="auto_test_wave2",
                                )
                            return result, verification

                if failed == 0 and errors == 0 and passed > 0:
                    logger.info(f"    ✅ Wave2 Candidate {idx+1}: ALL {passed} TESTS PASS!")
                    verification["status"] = "OK"
                    verification["test_results"] = f"{passed}/{total} tests pass"
                    # v0.8.0: Auto-capture the pattern
                    if collected_errors:
                        self.kb.capture_pattern(
                            error_pattern=collected_errors[0][:200],
                            solution=f"Fixed {filename} via Wave2 error-aware sampling ({passed} tests pass)",
                            example=filepath.read_text()[:500] if filepath.exists() else "",
                            source="auto_test_wave2",
                        )
                    return result, verification

                logger.info(f"    ⚠️ Wave2 Candidate {idx+1}: {passed}/{total} pass, {failed} fail, {errors} error")

                # Record wave 2 trace
                w2_error_output = test_result.get("output", "")
                source_code_w2 = ""
                if tests_for:
                    source_base = tests_for.replace('.py', '')
                    source_path_w2 = self.working_dir / f"{source_base}.py"
                    if source_path_w2.exists():
                        try:
                            source_code_w2 = source_path_w2.read_text()
                        except Exception:
                            pass

                self.trace_collector.record_test_failure(
                    test_filename=filename,
                    source_filename=f"{tests_for.replace('.py', '')}.py" if tests_for else "",
                    prompt_excerpt=f"Wave2 Step {step_number}/{total_steps}: Create {filename} (with error context)",
                    generated_test_code=candidate_content,
                    source_code=source_code_w2,
                    test_output=w2_error_output,
                    passed=passed,
                    failed=failed,
                    errors=errors,
                    error_category=self._classify_test_error(w2_error_output),
                    model_used=self.config.get_agent("build").model.model_id,
                    temperature=temp,
                    iteration=task_state.iteration,
                    task_goal=task_state.goal or "",
                    candidate_number=idx + 5,  # wave 2 starts at 5
                    total_candidates=8,
                )

                if passed > best_score:
                    best_score = passed
                    best_content = candidate_content
                    best_verification = verification
                    best_verification["test_results"] = f"{passed}/{total} tests pass"

        # Use best candidate from either wave
        total_attempts = num_candidates + (4 if collected_errors and best_score <= 0 else 0)
        if best_content is not None:
            filepath.write_text(best_content)
            status = f"BEST OF {total_attempts}: {best_verification.get('test_results', 'unknown')}"
            best_verification["status"] = status
            logger.info(f"  📊 No perfect candidate. Using best: {status}")

            # v0.8.0: Blame detection — if test fails, find which SOURCE file caused it
            if best_score < (passed + failed + errors if 'passed' in dir() else 999):
                all_error_text = " ".join(collected_errors[-3:])
                blamed = self._extract_blamed_source_files(all_error_text, filename)
                if blamed:
                    best_verification["blamed_source_files"] = list(blamed)
                    logger.warning(f"  🔍 Blame detection: {filename} failures trace to: {blamed}")

            return AgentResult(success=True, output=f"Test sampling: {status}"), best_verification

        logger.warning(f"  ❌ All {total_attempts} candidates failed for {filename}")
        return (
            AgentResult(success=False, output="All test candidates failed", error="sampling_exhausted"),
            {"exists": False, "status": "ALL CANDIDATES FAILED", "exports": "none"}
        )

    def _run_test_file(self, test_filename: str, source_module: str = "") -> dict:
        """
        Run a test file and return structured results.

        Returns dict with: passed, failed, errors, output
        """
        import re

        # v0.7.4: Clean test artifacts before running to ensure clean slate.
        # Without this, test_storage.py might leave tasks.json that causes
        # test_cli.py to fail (or vice versa). Each test must start clean.
        # v0.9.9b: Only clean common temp patterns, not all .db files (which
        # may be intentional project databases needed across test runs).
        try:
            self._safe_run(
                "rm -f *.json.bak test_output.* .tmp_*",
                shell=True, cwd=str(self.working_dir),
                capture_output=True, timeout=5
            )
        except Exception:
            pass  # Best effort cleanup

        # Try pytest first, fall back to unittest if pytest isn't available
        commands = [
            ("pytest", f"cd {self.working_dir} && python3 -m pytest {test_filename} -v --tb=short 2>&1"),
            ("unittest", f"cd {self.working_dir} && python3 -m unittest {test_filename.replace('.py', '')} -v 2>&1"),
        ]

        for runner_name, cmd in commands:
            try:
                result = self._safe_run(
                    cmd, shell=True, capture_output=True, text=True, timeout=30,
                    cwd=str(self.working_dir)
                )
                output = result.stdout + result.stderr

                # v0.9.9b: If pytest itself isn't available, fall through to unittest
                if runner_name == "pytest" and "No module named" in output and "pytest" in output:
                    continue

                # pytest summary line parsing
                # CRITICAL: pytest lists items in ANY ORDER (often failures first).
                # We must search for each count INDEPENDENTLY.
                passed_m = re.search(r'(\d+) passed', output)
                failed_m = re.search(r'(\d+) failed', output)
                error_m = re.search(r'(\d+) error', output)

                if passed_m or failed_m or error_m:
                    passed = int(passed_m.group(1)) if passed_m else 0
                    failed = int(failed_m.group(1)) if failed_m else 0
                    errors = int(error_m.group(1)) if error_m else 0
                    return {"passed": passed, "failed": failed, "errors": errors, "output": output[-500:]}

                # unittest style: "Ran X tests" + "OK" or "FAILED"
                unittest_ran = re.search(r'Ran (\d+) test', output)
                if unittest_ran:
                    total = int(unittest_ran.group(1))
                    if "OK" in output and "FAILED" not in output:
                        return {"passed": total, "failed": 0, "errors": 0, "output": output[-500:]}
                    fail_match = re.search(r'failures=(\d+)', output)
                    err_match = re.search(r'errors=(\d+)', output)
                    failures = int(fail_match.group(1)) if fail_match else 0
                    errs = int(err_match.group(1)) if err_match else 0
                    passed_count = total - failures - errs
                    return {"passed": max(0, passed_count), "failed": failures, "errors": errs, "output": output[-500:]}

                # Exit code 0 with no parseable output — assume success
                if result.returncode == 0:
                    return {"passed": 1, "failed": 0, "errors": 0, "output": output[-500:]}

                # v0.9.9b: Non-zero exit, unparseable output — if this was pytest,
                # fall through to unittest instead of returning generic error
                if runner_name == "pytest":
                    continue

                # Last resort: unparseable failure
                return {"passed": 0, "failed": 0, "errors": 1, "output": output[-500:]}

            except Exception as e:
                if runner_name == "pytest":
                    continue  # Try unittest
                return {"passed": 0, "failed": 0, "errors": 1, "output": str(e)[-500:]}

        # Both runners failed entirely
        return {"passed": 0, "failed": 0, "errors": 1, "output": "Both pytest and unittest failed to run"}

    def _snapshot_passing_files(self, task_state: TaskState) -> dict:
        """
        v0.6.2 Fix 1: Snapshot Protection (Augment/Verdent pattern).

        Before a retry iteration, snapshot all .py files that currently
        have valid syntax and pass their tests. If the retry build breaks
        any of them, we can roll back.

        Returns: {filename: content_string} for each passing file
        """
        snapshots = {}

        for pyfile in self.working_dir.glob("*.py"):
            filename = pyfile.name
            if filename.startswith('.') or filename.startswith('__'):
                continue

            # Syntax check
            result = self._safe_run(
                ["python3", "-c", f"import py_compile; py_compile.compile('{pyfile}', doraise=True)"],
                capture_output=True, text=True, timeout=10,
                cwd=str(self.working_dir)
            )
            if result.returncode == 0:
                try:
                    snapshots[filename] = pyfile.read_text()
                except Exception:
                    pass

        if snapshots:
            logger.info(f"  🔒 Snapshot: protected {len(snapshots)} passing files from regression")

        return snapshots

    def _rollback_regressions(self, snapshots: dict):
        """
        v0.6.2 Fix 1: Rollback Regressions.

        After a retry build, check if any previously-passing files now have
        syntax errors or import failures. If so, restore the snapshot.
        """
        rolled_back = []

        for filename, original_content in snapshots.items():
            filepath = self.working_dir / filename

            if not filepath.exists():
                # File was deleted — restore it
                filepath.write_text(original_content)
                rolled_back.append(f"{filename} (deleted → restored)")
                continue

            # Syntax check the new version
            result = self._safe_run(
                ["python3", "-c", f"import py_compile; py_compile.compile('{filepath}', doraise=True)"],
                capture_output=True, text=True, timeout=10,
                cwd=str(self.working_dir)
            )
            if result.returncode != 0:
                # Retry broke this file — roll back
                filepath.write_text(original_content)
                rolled_back.append(f"{filename} (syntax broken → restored)")
                continue

            # For test files, check if tests still pass
            if filename.startswith("test_"):
                test_result = self._run_test_file(filename)
                old_test = self._run_test_file_with_content(filename, original_content)

                new_passed = test_result.get("passed", 0)
                old_passed = old_test.get("passed", 0)

                # If new version passes fewer tests, roll back
                if new_passed < old_passed:
                    filepath.write_text(original_content)
                    rolled_back.append(f"{filename} ({new_passed} < {old_passed} tests → restored)")

        if rolled_back:
            logger.info(f"  🔄 Rollback: restored {len(rolled_back)} regressed files:")
            for rb in rolled_back:
                logger.info(f"    → {rb}")

    def _run_test_file_with_content(self, filename: str, content: str) -> dict:
        """Run a test file with specific content (for regression comparison)."""
        filepath = self.working_dir / filename
        original = filepath.read_text() if filepath.exists() else None
        try:
            filepath.write_text(content)
            return self._run_test_file(filename)
        finally:
            if original is not None:
                filepath.write_text(original)
            elif filepath.exists():
                filepath.unlink()

    def _perform_root_cause_analysis(self, task_state: TaskState, result: IterationResult):
        """
        Analyze why the last iteration failed using LLM-based 5 Whys analysis.

        v0.7.0 Fix 1: RCA Veto — if failure is NameError/SyntaxError in a test file,
        bypass LLM RCA entirely and use deterministic fix (repair the test file only).
        This prevents the __eq__ death spiral where RCA keeps proposing changes to
        models.py when the actual problem is in test_cli.py.

        v0.7.1: RCA Veto now returns structured dict (same format as LLM RCA),
        so concrete_edits flow uniformly into the build prompt.

        Falls back to heuristic RCA if the LLM call fails.
        """
        last_failure = task_state.failure_history[-1] if task_state.failure_history else {}
        phase = last_failure.get("phase", "unknown")
        error = last_failure.get("error", "unknown")

        # v0.9.6: Source blame detection before RCA
        self._detect_source_blame(task_state)

        # v0.7.1: RCA Veto — returns structured dict or None
        vetoed = self._try_rca_veto(task_state, result)
        if vetoed:
            # Vetoed RCA returns structured dict — process it like LLM RCA
            rca_result = vetoed
            logger.info("  🛡️ RCA VETO: deterministic fix applied (skipped LLM RCA)")
        else:
            # Try LLM-based RCA
            rca_result = self.agent_runner.run_rca(task_state, result)  # type: ignore[assignment]

            # v0.7.3: Post-RCA filters to prevent hallucination spirals
            if rca_result:
                rca_result = self._filter_rca_hallucinations(rca_result, task_state)

        if rca_result:
            # Structured RCA succeeded (either from veto or LLM)
            rca = f"ROOT CAUSE: {rca_result.get('root_cause', 'unknown')}"

            why_chain = rca_result.get("why_chain", [])
            if why_chain:
                rca += "\n  5 Whys:"
                for i, why in enumerate(why_chain, 1):
                    rca += f"\n    {i}. {why}"

            action = rca_result.get("what_to_change", "")
            if action:
                rca += f"\n  ACTION: {action}"

            severity = rca_result.get("severity", "medium")
            rca += f"\n  SEVERITY: {severity}"

            # Let RCA control re-exploration
            if rca_result.get("needs_re_exploration", False):
                task_state.needs_re_exploration = True
                logger.info("  RCA recommends re-exploration")
        else:
            # Fallback to heuristic RCA
            rca = self._heuristic_rca(phase, error, task_state)

        # Store RCA in last failure entry
        if task_state.failure_history:
            task_state.failure_history[-1]["rca"] = rca
            # Store raw structured data for direct injection into build prompt
            if rca_result:
                task_state.failure_history[-1]["rca_data"] = rca_result

        logger.info(f"RCA: {rca}")

    def _heuristic_rca(self, phase: str, error: str, task_state: TaskState) -> str:
        """Fallback heuristic RCA when LLM-based analysis is unavailable."""
        rca = f"Phase '{phase}' failed: {error}"

        if "timeout" in error.lower():
            rca += " | RCA: Agent took too long. Consider simplifying the plan or increasing timeout."
            task_state.needs_re_exploration = True
        elif "connection" in error.lower():
            rca += " | RCA: Cannot reach LLM endpoint. Check Ollama is running."
        elif "0/" in error:
            rca += " | RCA: No DoD criteria passed. Build agent may not have executed tools. Check model tool-use capability."
        elif "dod criteria passed" in error.lower():
            rca += " | RCA: Partial DoD pass. Review failed criteria and focus plan on those."

        return rca

    def _detect_source_blame(self, task_state: TaskState) -> list:
        """
        v0.9.6: Source blame detection (MAST taxonomy research).

        When >80% of test failures share the same exception type AND tracebacks
        point to source files, blame the source — not the test.
        """
        import re as _re
        last_failure = task_state.failure_history[-1] if task_state.failure_history else {}
        test_output = last_failure.get("test_output", "") or last_failure.get("error", "")
        if not test_output or len(test_output) < 20:
            return []

        exc_pattern = _re.findall(
            r'(TypeError|AttributeError|NameError|ImportError|ValueError|RuntimeError|KeyError): .+',
            test_output
        )
        if len(exc_pattern) < 2:
            return []

        most_common = max(set(exc_pattern), key=lambda x: exc_pattern.count(x.split(":")[0]))
        mc_type = most_common.split(":")[0]
        ratio = sum(1 for e in exc_pattern if e.startswith(mc_type)) / len(exc_pattern)
        if ratio < 0.7:
            return []

        # Extract source files from tracebacks
        source_in_tb = _re.findall(r'File ".*?/([^/"\s]+\.py)", line \d+', test_output)
        blamed = []
        for sf in source_in_tb:
            if not sf.startswith('test_') and sf not in blamed:
                blamed.append(sf)

        if blamed:
            logger.warning(f"  \U0001F3AF BLAME: {ratio*100:.0f}% failures are {mc_type} in source: {blamed}")
            task_state.blamed_source_files = blamed
        return blamed

    def _filter_rca_hallucinations(self, rca_result: dict, task_state: TaskState) -> dict:
        """
        v0.7.3: Post-LLM-RCA filter to catch common hallucination patterns.

        Problem #1: __eq__ on @dataclass
        Python's @dataclass decorator auto-generates __eq__ based on all fields.
        The 70B model halluccinates "add __eq__" as root cause when the REAL
        problem is that test files can't even import (NameError). This wastes
        every retry iteration on a non-problem.

        Problem #2: Repeated identical RCA
        If the same root cause has been proposed 2+ times and hasn't fixed
        anything, the RCA is wrong. Redirect to "regenerate test files" instead.
        """
        concrete_edits = rca_result.get("concrete_edits", [])
        filtered_edits = []
        eq_filtered = False

        for edit in concrete_edits:
            action = edit.get("action", "")
            details = edit.get("details", "")
            target_file = edit.get("file", "")

            # Filter 1: __eq__ on @dataclass files
            if "__eq__" in details or "__eq__" in action:
                # Check if the target file uses @dataclass
                target_path = self.working_dir / target_file
                if target_path.exists():
                    try:
                        content = target_path.read_text()
                        if "@dataclass" in content:
                            logger.info(f"  🛡️ RCA FILTER: blocked __eq__ edit on @dataclass file {target_file}")
                            eq_filtered = True
                            continue  # Skip this edit — @dataclass already has __eq__
                    except Exception:
                        pass

            # v0.8.4 Filter 3: PROTECT VERIFIED-OK SOURCE FILES
            # v0.9.2 UPDATE: Only protect files whose dependent tests ALSO pass.
            # Research insight: compile+import is a pre-filter, not correctness proof.
            # A file that compiles but has missing methods SHOULD be edited by RCA.
            # SWE-agent v0.7.1 fix: linter should only block NEW errors.
            # Agentless: test execution is the sole correctness signal.
            if target_file and not target_file.startswith("test_") and target_file.endswith(".py"):
                target_path = self.working_dir / target_file
                if target_path.exists():
                    verify = self._verify_single_file(target_file, is_test=False)
                    if verify.get("status") == "OK":
                        # v0.9.2: Before blocking, check if dependent tests pass
                        source_base = target_file.replace('.py', '')
                        dep_tests_pass = True
                        for tf in self.working_dir.glob("test_*.py"):
                            # Check test files that import from this source
                            try:
                                test_content = tf.read_text()
                                if f"from {source_base}" in test_content or f"import {source_base}" in test_content:
                                    test_result = self._run_test_file(tf.name, source_base)
                                    if test_result.get("failed", 0) > 0 or test_result.get("errors", 0) > 0:
                                        dep_tests_pass = False
                                        break
                            except Exception:
                                continue
                        if dep_tests_pass:
                            logger.info(f"  🛡️ RCA GUARD: blocked edit on VERIFIED-OK file {target_file} "
                                       f"(compiles OK + dependent tests pass)")
                            continue  # Skip — this file truly works
                        else:
                            logger.info(f"  ⚠️ RCA GUARD OVERRIDE: {target_file} compiles OK but "
                                       f"dependent tests FAIL — allowing RCA edit")

            filtered_edits.append(edit)

        # Filter 2: Repeated identical root cause detection
        root_cause = rca_result.get("root_cause", "")
        repeat_count = 0
        for failure in task_state.failure_history[:-1]:  # Exclude current
            prev_rca = failure.get("rca_data", {})
            if prev_rca:
                prev_root = prev_rca.get("root_cause", "")
                # Fuzzy match — check if the key phrases overlap
                if prev_root and self._rca_is_similar(prev_root, root_cause):
                    repeat_count += 1

        if repeat_count >= 2:
            logger.warning(f"  🔄 RCA REPEAT DETECTED ({repeat_count + 1}x same diagnosis)")
            logger.warning(f"  Previous diagnosis has failed {repeat_count} times — redirecting to test regeneration")

            # Override: the RCA is clearly wrong. The real problem is probably
            # that test files can't run, not that source files are wrong.
            # Redirect to regenerating test files from scratch.
            test_files = [f.name for f in self.working_dir.glob("test_*.py")]
            new_edits = []
            for tf_name in test_files:
                test_result = self._run_test_file(tf_name)
                if test_result.get("errors", 0) > 0 or test_result.get("passed", 0) == 0:
                    source_file = tf_name.replace("test_", "")
                    new_edits.append({
                        "file": tf_name,
                        "action": "regenerate",
                        "details": f"Delete and rewrite {tf_name} from scratch. Read {source_file} first to get correct class/function names and imports."
                    })

            if new_edits:
                return {
                    "root_cause": f"Previous RCA was wrong (repeated {repeat_count + 1}x without success). Test files need regeneration.",
                    "why_chain": [
                        f"The same root cause has been proposed {repeat_count + 1} times",
                        "Each time the fix was applied, it didn't resolve the issue",
                        "This means the diagnosed root cause is incorrect",
                        "The actual problem is likely that test files have wrong imports or outdated assumptions",
                        "Regenerating test files from the current source code should fix this"
                    ],
                    "what_to_change": "Regenerate failing test files from scratch based on actual source code.",
                    "severity": "high",
                    "needs_re_exploration": False,
                    "concrete_edits": new_edits,
                }

        if eq_filtered and not filtered_edits:
            # All edits were __eq__ hallucinations — check what's actually failing
            test_files = [f.name for f in self.working_dir.glob("test_*.py")]
            for tf_name in test_files:
                test_result = self._run_test_file(tf_name)
                if test_result.get("errors", 0) > 0:
                    error_output = test_result.get("output", "")
                    if "NameError" in error_output or "ImportError" in error_output:
                        filtered_edits.append({
                            "file": tf_name,
                            "action": "fix_imports",
                            "details": "The test file has import/name errors. Read the source module and fix the imports."
                        })

            rca_result["root_cause"] = "Test files have import errors (not source module issues)"
            rca_result["what_to_change"] = "Fix imports in test files. Do NOT modify source modules that use @dataclass."

        rca_result["concrete_edits"] = filtered_edits
        return rca_result

    def _rca_is_similar(self, rca1: str, rca2: str) -> bool:
        """Check if two RCA root causes are substantially similar."""
        # Normalize and extract key phrases
        def normalize(s):
            s = s.lower()
            # Remove common filler words
            for w in ["the", "a", "an", "is", "not", "does", "in", "and", "or", "to", "of"]:
                s = s.replace(f" {w} ", " ")
            return set(s.split())

        words1 = normalize(rca1)
        words2 = normalize(rca2)

        if not words1 or not words2:
            return False

        # Jaccard similarity > 0.5 = similar enough
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return (intersection / union) > 0.5 if union > 0 else False

    def _try_rca_veto(self, task_state: TaskState, result: IterationResult) -> Optional[dict]:
        """
        v0.7.4 RCA Veto: Short-circuit LLM RCA for trivial errors in ANY file.

        v0.7.1 only triggered on test files. But trivial NameErrors can occur
        in source files too (e.g., storage.py missing `from datetime import datetime`).
        Mixed failures (source + test) caused the veto to skip, letting LLM RCA
        spiral on phantom issues.

        v0.7.4: Detect trivial errors in ANY file (test or source) from tracebacks.
        Generate concrete edits for all affected files. Skip LLM RCA entirely.
        """
        if not result or not result.dod_results:
            return None

        import re

        trivial_categories = {"name_error", "import_error", "syntax_error", "attribute_error", "flask_context_error"}
        concrete_edits: List[Dict[str, str]] = []
        why_chain: List[str] = []
        found_any_trivial = False
        seen_files: Set[str] = set()  # Deduplicate edits per file

        for cid, res in result.dod_results.items():
            if not isinstance(res, dict) or res.get("passed"):
                continue

            evidence = res.get("evidence", "") or ""
            cmd = res.get("command", "") or ""
            error_cat = self._classify_test_error(evidence)

            if error_cat not in trivial_categories:
                continue

            found_any_trivial = True

            # Extract ALL python files from tracebacks in the evidence.
            # This catches both test files AND source files.
            # Pytest --tb=short: 'storage.py:34: in load_tasks'
            # Full traceback: 'File "/tmp/.../storage.py", line 34'
            traceback_short = re.findall(r'(\w+\.py):\d+: in \w+', evidence)
            traceback_long = re.findall(r'File ".*?/([^/\"]+\.py)", line \d+', evidence)
            traceback_files = list(dict.fromkeys(traceback_short + traceback_long))

            # Also try to find test file from the command
            cmd_match = re.search(r'(test_\w+\.py)', cmd + " " + evidence)

            # Build target file list: all files from tracebacks + command
            target_files = list(dict.fromkeys(traceback_files))  # preserve order, dedupe
            if cmd_match and cmd_match.group(1) not in target_files:
                target_files.append(cmd_match.group(1))

            # If no files found at all, try generic pattern
            if not target_files:
                generic_match = re.search(r'(\w+\.py)', cmd)
                if generic_match:
                    target_files = [generic_match.group(1)]

            for target_file in target_files:
                if target_file in seen_files:
                    continue
                seen_files.add(target_file)

                # Skip orchestrator files
                if target_file.startswith("standalone_"):
                    continue

                is_test = target_file.startswith("test_")
                source_file = target_file.replace("test_", "") if is_test else target_file

                if error_cat == "name_error":
                    why_chain.append(f"{target_file}: NameError (missing import / undefined name)")
                    concrete_edits.append({
                        "file": target_file,
                        "action": "fix_name_error_imports",
                        "details": "Add missing imports required by the traceback (e.g., uuid, datetime). Keep changes minimal."
                    })
                elif error_cat == "import_error":
                    why_chain.append(f"{target_file}: ImportError (bad module path)")
                    concrete_edits.append({
                        "file": target_file,
                        "action": "fix_import_paths",
                        "details": "Match imports to actual module filenames in workspace."
                    })
                elif error_cat == "syntax_error":
                    why_chain.append(f"{target_file}: SyntaxError")
                    concrete_edits.append({
                        "file": target_file,
                        "action": "fix_syntax",
                        "details": "Fix the exact syntax error shown by py_compile/traceback."
                    })
                elif error_cat == "attribute_error":
                    why_chain.append(f"{target_file}: AttributeError (wrong method/attribute)")
                    attr_match = re.search(
                        r"AttributeError:\s*'?(\w+)'?\s+object has no attribute\s+'?(\w+)'?",
                        evidence
                    )
                    if attr_match:
                        concrete_edits.append({
                            "file": target_file,
                            "action": "fix_attribute",
                            "details": f"`{attr_match.group(1)}` has no attribute `{attr_match.group(2)}`. Read `{source_file}` for correct names."
                        })
                    else:
                        concrete_edits.append({
                            "file": target_file,
                            "action": "fix_attribute",
                            "details": f"Read the source module `{source_file}` to find correct class/function/method names."
                        })
                elif error_cat == "flask_context_error":
                    why_chain.append(f"{target_file}: RuntimeError — calling Flask route functions directly instead of using test_client")
                    concrete_edits.append({
                        "file": target_file,
                        "action": "rewrite_flask_tests",
                        "details": (
                            "REWRITE the test file to use Flask test_client. "
                            "NEVER import or call route functions directly. Use: "
                            "`from app import app; client = app.test_client(); "
                            "response = client.get('/endpoint'); data = response.get_json()`. "
                            "Use setUp() to create self.client = app.test_client(). "
                            "Use self.client.post('/path', json={...}) for POST, "
                            "self.client.get('/path') for GET, self.client.delete('/path') for DELETE."
                        )
                    })

        if not found_any_trivial or not concrete_edits:
            return None

        why_chain.append("Deterministic fix: trivial errors should be fixed directly; no LLM RCA needed.")

        return {
            "root_cause": "Trivial runtime errors detected (NameError/ImportError/SyntaxError)",
            "why_chain": why_chain,
            "what_to_change": "Apply minimal import/syntax fixes to the exact file(s) referenced by traceback.",
            "severity": "low",
            "needs_re_exploration": False,
            "concrete_edits": concrete_edits,
            "_veto_applied": True,
        }

    def _get_rca_edits_for_micro_build(self, task_state: TaskState) -> str:
        """
        v0.7.3: Extract RCA concrete edits as error_context for micro-builds.

        On retry iterations, this provides targeted fix instructions to the
        per-file build prompts instead of relying on the monolithic build
        to interpret them from a giant prompt.
        """
        if not task_state.failure_history:
            return ""

        last_rca = task_state.failure_history[-1].get("rca_data", {})
        if not last_rca:
            return ""

        parts = []
        root_cause = last_rca.get("root_cause", "")
        if root_cause:
            parts.append(f"ROOT CAUSE (from previous iteration): {root_cause}")

        action = last_rca.get("what_to_change", "")
        if action:
            parts.append(f"REQUIRED ACTION: {action}")

        edits = last_rca.get("concrete_edits", [])
        if edits:
            parts.append("CONCRETE EDITS TO APPLY:")
            for edit in edits:
                f = edit.get("file", "?")
                a = edit.get("action", "?")
                d = edit.get("details", "?")
                parts.append(f"  - {f}: [{a}] {d}")

        return "\n".join(parts) if parts else ""

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
        self._report_playbook_feedback(was_successful=True)
        self._sync_session_to_pve()
        logger.info(f"Task ID: {task_state.task_id}")
        logger.info(f"Iterations: {task_state.iteration}")
        logger.info(f"Duration: {task_state.started_at} → {task_state.completed_at}")
        logger.info(f"{'='*60}")

        # v0.9.0: Run librarian to curate this session's knowledge
        if self.librarian:
            if self.librarian_db_path:
                init_librarian_tables(self.librarian_db_path)
            try:
                summary = build_session_summary(
                    task_state=task_state,
                    working_dir=self.working_dir,
                    memory_records=self.memory.records,
                )
                curation_result = self.librarian.curate_session(summary)
                logger.info(f"📚 Librarian curated: patterns={curation_result.get('patterns', 0)}, "
                           f"journal={curation_result.get('journal', 0)}, "
                           f"snippets={curation_result.get('snippets', 0)}")
            except Exception as e:
                logger.warning(f"📚 Librarian curation failed (non-fatal): {e}")

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
        self._report_playbook_feedback(was_successful=False)
        self._sync_session_to_pve()
        logger.error(f"Reason: {reason}")
        logger.error(f"Handoff report: {handoff_path}")
        logger.error(f"{'='*60}")

        # v0.9.0: Run librarian even on failures — lessons from failures are valuable
        if self.librarian:
            if self.librarian_db_path:
                init_librarian_tables(self.librarian_db_path)
            try:
                summary = build_session_summary(
                    task_state=task_state,
                    working_dir=self.working_dir,
                    memory_records=self.memory.records,
                )
                curation_result = self.librarian.curate_session(summary)
                logger.info(f"📚 Librarian curated (from failure): patterns={curation_result.get('patterns', 0)}, "
                           f"journal={curation_result.get('journal', 0)}, "
                           f"snippets={curation_result.get('snippets', 0)}")
            except Exception as e:
                logger.warning(f"📚 Librarian curation failed (non-fatal): {e}")

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

    def restore_backup(self, backup_name: Optional[str] = None):
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
            self._safe_run(
                ["git", "add", "-A"],
                cwd=self.working_dir, capture_output=True, timeout=30
            )
            self._safe_run(
                ["git", "commit", "-m", message, "--allow-empty"],
                cwd=self.working_dir, capture_output=True, timeout=30
            )
        except Exception as e:
            logger.warning(f"Git commit failed: {e}")
