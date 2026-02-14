#!/usr/bin/env python3
"""
benchmark.py — Run standardized benchmark tasks and collect results.

Usage:
    python3 benchmark.py                    # Run standard suite (5 tasks)
    python3 benchmark.py --task 5           # Run only task #5
    python3 benchmark.py --suite standard   # Run standard suite
    python3 benchmark.py --list             # List available tasks
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Benchmark task definitions (ordered by difficulty level)
# ---------------------------------------------------------------------------

TASKS = {
    1: {
        "name": "Calculator",
        "level": 2,
        "description": "Single class with tests",
        "prompt": (
            "Build a calculator module. Structure: calculator.py (Calculator class "
            "with add, subtract, multiply, divide methods, raising ValueError on "
            "division by zero), test_calculator.py (tests for all operations "
            "including edge cases like division by zero, negative numbers, floats)."
        ),
        "source_files": 1,
        "test_files": 1,
    },
    2: {
        "name": "Miniqueue",
        "level": 3,
        "description": "Multi-file, known patterns",
        "prompt": (
            "Build a priority queue system. Structure: models.py (Task dataclass "
            "with id, title, priority, status, created_at), queue.py (TaskQueue "
            "class with add, pop, peek, remove, list_by_priority methods), "
            "test_models.py, test_queue.py."
        ),
        "source_files": 2,
        "test_files": 2,
    },
    3: {
        "name": "Task Tracker CLI",
        "level": 4,
        "description": "Inter-module state, file I/O",
        "prompt": (
            "Build a task tracker with JSON persistence. Structure: models.py "
            "(Task dataclass with id, title, status, priority, created_at, "
            "completed_at), storage.py (JSONStorage class — load/save tasks to "
            "JSON file, handle missing file, atomic writes), tracker.py "
            "(TaskTracker class — add, complete, delete, list with filters by "
            "status/priority, statistics summary), test_models.py, test_storage.py, "
            "test_tracker.py."
        ),
        "source_files": 3,
        "test_files": 3,
    },
    4: {
        "name": "Bookmark Manager API",
        "level": 5,
        "description": "REST API + database + validation",
        "prompt": (
            "Build a bookmark manager REST API with Flask. Features: "
            "add/remove/update bookmarks with URL, title, tags. Search by tag. "
            "Pagination. Input validation. SQLite storage. Structure: models.py "
            "(Bookmark dataclass), database.py (CRUD operations), validators.py "
            "(input validation), app.py (Flask routes). Test files: test_models.py, "
            "test_database.py, test_validators.py, test_app.py."
        ),
        "source_files": 4,
        "test_files": 4,
    },
    5: {
        "name": "Expense Tracker with Auth",
        "level": 6,
        "description": "Complex business logic, auth, edge cases",
        "prompt": (
            "Build an expense tracker API with Flask. Features: JWT authentication "
            "(register, login, token refresh). Users can only see their own "
            "expenses. CRUD for expenses with amount, category, description, date. "
            "Budget limits per category with overspend warnings. Monthly spending "
            "summaries grouped by category. CSV export of expenses with date range "
            "filtering. Edge cases: reject negative amounts, reject future dates, "
            "enforce category name uniqueness per user, handle concurrent budget "
            "updates. SQLite storage. Structure: models.py (User, Expense, Budget "
            "dataclasses), auth.py (JWT token creation/verification, password "
            "hashing with bcrypt), database.py (all CRUD operations, user "
            "isolation), validators.py (input validation including amount bounds, "
            "date logic, category rules), app.py (Flask routes with auth "
            "middleware). Test files: test_models.py, test_auth.py, "
            "test_database.py, test_validators.py, test_app.py."
        ),
        "source_files": 5,
        "test_files": 5,
    },
}

STANDARD_SUITE = [1, 2, 3, 4, 5]


@dataclass
class BenchmarkResult:
    task_id: int
    task_name: str
    level: int
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0
    iterations: int = 0
    dod_passed: int = 0
    dod_total: int = 0
    tests_passed: int = 0
    tests_total: int = 0
    source_first_candidate: int = 0
    source_total: int = 0
    outcome: str = "not_run"  # pass, fail, crash, timeout
    error: str = ""
    log_file: str = ""


def run_task(task_id: int, base_dir: str = "/tmp/bench", max_iterations: int = 3,
             timeout: int = 7200) -> BenchmarkResult:
    """Run a single benchmark task and parse results from the log."""
    task = TASKS[task_id]
    result = BenchmarkResult(
        task_id=task_id,
        task_name=task["name"],
        level=task["level"],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = f"{base_dir}/task{task_id}_{timestamp}"
    log_file = f"{base_dir}/task{task_id}_{timestamp}.log"
    result.log_file = log_file

    cmd = [
        sys.executable, "standalone_main.py",
        task["prompt"],
        "--max-iterations", str(max_iterations),
        "--working-dir", work_dir,
    ]

    result.started_at = datetime.now().isoformat()
    start = time.time()

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as e:
        result.outcome = "timeout"
        result.error = f"Exceeded {timeout}s timeout"
        result.duration_seconds = timeout
        result.finished_at = datetime.now().isoformat()
        # v1.1: Capture partial output for debugging (was lost before)
        partial = (e.stdout or "") + (e.stderr or "") if hasattr(e, 'stdout') else ""
        if partial:
            log_file = f"{base_dir}/task{task_id}_{timestamp}.log"
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            Path(log_file).write_text(f"=== TIMEOUT after {timeout}s ===\n{partial}")
            result.log_file = log_file
            print(f"\n  📋 Partial log saved to {log_file}")
            # Show last 20 lines for quick debugging
            last_lines = partial.strip().split('\n')[-20:]
            print(f"  📋 Last output before timeout:")
            for line in last_lines:
                print(f"     {line}")
        return result
    except Exception as e:
        result.outcome = "crash"
        result.error = str(e)
        result.duration_seconds = time.time() - start
        result.finished_at = datetime.now().isoformat()
        return result

    result.duration_seconds = time.time() - start
    result.finished_at = datetime.now().isoformat()

    # Save log
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    Path(log_file).write_text(output)

    # Parse results from log
    result = _parse_log(output, result)
    return result


def _parse_log(output: str, result: BenchmarkResult) -> BenchmarkResult:
    """Extract metrics from orchestrator log output."""
    import re

    # Outcome
    if "TASK COMPLETED SUCCESSFULLY" in output:
        result.outcome = "pass"
    elif "Fatal error" in output or "Traceback" in output:
        result.outcome = "crash"
        # Extract error
        for line in output.split("\n"):
            if "Fatal error:" in line:
                result.error = line.split("Fatal error:")[-1].strip()
                break
    else:
        result.outcome = "fail"

    # Iterations
    iteration_matches = re.findall(r"ITERATION (\d+)", output)
    if iteration_matches:
        result.iterations = max(int(i) for i in iteration_matches)

    # DoD
    dod_match = re.search(r"DoD final count: (\d+)/(\d+)", output)
    # Get the LAST DoD match (final iteration)
    dod_matches = re.findall(r"DoD final count: (\d+)/(\d+)", output)
    if dod_matches:
        result.dod_passed = int(dod_matches[-1][0])
        result.dod_total = int(dod_matches[-1][1])

    # Individual tests
    test_matches = re.findall(r"Individual test functions: (\d+)/(\d+)", output)
    if test_matches:
        result.tests_passed = int(test_matches[-1][0])
        result.tests_total = int(test_matches[-1][1])

    # Source first-candidate (count "Source candidate 1" → "score 3/3")
    source_1st = len(re.findall(r"Source candidate 1.*?score 3/3", output))
    source_total_matches = re.findall(r"Sampling 3 source candidates", output)
    # Only count from iteration 1 (before first retry)
    result.source_first_candidate = source_1st
    result.source_total = len(source_total_matches)

    return result


def print_results(results: list[BenchmarkResult]):
    """Print a formatted results table."""
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'#':<3} {'Task':<25} {'Lvl':<4} {'DoD':<8} {'Tests':<12} {'Src 1st':<8} {'Iter':<5} {'Time':<8} {'Result':<8}")
    print("-" * 90)

    for r in results:
        time_str = f"{r.duration_seconds / 60:.0f}m" if r.duration_seconds > 0 else "—"
        dod_str = f"{r.dod_passed}/{r.dod_total}" if r.dod_total > 0 else "—"
        test_str = f"{r.tests_passed}/{r.tests_total}" if r.tests_total > 0 else "—"
        src_str = f"{r.source_first_candidate}/{r.source_total}" if r.source_total > 0 else "—"
        iter_str = str(r.iterations) if r.iterations > 0 else "—"

        icon = {"pass": "✅", "fail": "❌", "crash": "💥", "timeout": "⏰", "not_run": "⬜"}
        outcome_str = icon.get(r.outcome, "?") + " " + r.outcome

        print(f"{r.task_id:<3} {r.task_name:<25} {r.level:<4} {dod_str:<8} {test_str:<12} {src_str:<8} {iter_str:<5} {time_str:<8} {outcome_str}")

    print("-" * 90)

    # Summary
    passed = sum(1 for r in results if r.outcome == "pass")
    total = len(results)
    total_time = sum(r.duration_seconds for r in results) / 60
    print(f"\nSuite: {passed}/{total} tasks passed | Total time: {total_time:.0f} minutes")
    print()


def save_results(results: list[BenchmarkResult], output_file: str = "benchmark_results.json"):
    """Save results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [r.__dict__ for r in results],
    }
    Path(output_file).write_text(json.dumps(data, indent=2))
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run orchestrator benchmarks")
    parser.add_argument("--task", type=int, help="Run a specific task by number")
    parser.add_argument("--suite", choices=["standard"], default="standard",
                        help="Run a predefined suite")
    parser.add_argument("--list", action="store_true", help="List available tasks")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=7200, help="Per-task timeout in seconds")
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable benchmark tasks:\n")
        for tid, task in TASKS.items():
            print(f"  {tid}. [{task['level']}] {task['name']}: {task['description']}")
            print(f"     {task['source_files']} source + {task['test_files']} test files")
        print()
        return

    if args.task:
        task_ids = [args.task]
    else:
        task_ids = STANDARD_SUITE

    print(f"\n🏁 Running {len(task_ids)} benchmark task(s)...\n")

    results = []
    for tid in task_ids:
        task = TASKS[tid]
        print(f"{'=' * 60}")
        print(f"Task {tid}: {task['name']} (Level {task['level']})")
        print(f"{'=' * 60}")
        result = run_task(tid, max_iterations=args.max_iterations, timeout=args.timeout)
        results.append(result)

        icon = {"pass": "✅", "fail": "❌", "crash": "💥", "timeout": "⏰"}
        print(f"\n  → {icon.get(result.outcome, '?')} {result.outcome.upper()} "
              f"({result.duration_seconds / 60:.0f}m, {result.iterations} iterations, "
              f"{result.tests_passed}/{result.tests_total} tests)\n")

    print_results(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
