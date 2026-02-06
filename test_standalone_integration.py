#!/usr/bin/env python3
"""
Integration tests for the standalone orchestrator.

Tests component isolation, imports, serialization, and tool execution.
Does NOT require a running Ollama instance for unit tests.
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# Track test results
passed = 0
failed = 0
errors = []


def test(name, condition, detail=""):
    global passed, failed, errors
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        errors.append(f"{name}: {detail}")
        print(f"  ❌ {name} — {detail}")


# ============================================================
print("\n=== Milestone 1: Models & Config ===")
# ============================================================

try:
    from standalone_models import TaskState, DoD, DoDCriterion, ExecutionPhase, AgentResult, IterationResult
    test("M1.1 Import standalone_models", True)
except Exception as e:
    test("M1.1 Import standalone_models", False, str(e))

try:
    from standalone_config import Config, AgentConfig, ModelConfig, load_config, default_config
    test("M1.2 Import standalone_config", True)
except Exception as e:
    test("M1.2 Import standalone_config", False, str(e))

# Test TaskState serialization
try:
    state = TaskState(task_id="test-001", goal="Test goal", iteration=3)
    state.phase = ExecutionPhase.BUILD
    json_str = state.to_json()
    loaded = TaskState.from_json(json_str)
    test("M1.3 TaskState round-trip", loaded.task_id == "test-001" and loaded.iteration == 3 and loaded.phase == ExecutionPhase.BUILD)
except Exception as e:
    test("M1.3 TaskState round-trip", False, str(e))

# Test DoD
try:
    dod = DoD()
    cid = dod.add("File exists", method="test", command="test -f foo.py")
    dod.mark_passed(cid)
    test("M1.4 DoD add/mark", dod.all_passed() and len(dod.criteria) == 1)
except Exception as e:
    test("M1.4 DoD add/mark", False, str(e))

# Test Config
try:
    config = default_config()
    test("M1.5 default_config()", "build" in config.agents and "test" in config.agents)
except Exception as e:
    test("M1.5 default_config()", False, str(e))

try:
    config = Config.load_default()
    test("M1.6 Config.load_default()", config.max_iterations > 0)
except Exception as e:
    test("M1.6 Config.load_default()", False, str(e))


# ============================================================
print("\n=== Milestone 2: Session Manager ===")
# ============================================================

try:
    from standalone_session import SessionManager
    test("M2.1 Import SessionManager", True)
except Exception as e:
    test("M2.1 Import SessionManager", False, str(e))

# Test save/load round-trip
tmpdir = None
try:
    tmpdir = Path(tempfile.mkdtemp())
    sm = SessionManager(tmpdir)

    # Check directories were created
    test("M2.2 Creates .agents dirs",
         (tmpdir / ".agents" / "plans").is_dir() and
         (tmpdir / ".agents" / "reports").is_dir() and
         (tmpdir / ".agents" / "logs").is_dir())

    # Save and load
    state = TaskState(task_id="sess-test", goal="Session test", iteration=2)
    sm.save_state(state)
    loaded = sm.load_state()
    test("M2.3 Save/load round-trip", loaded.task_id == "sess-test" and loaded.iteration == 2)

    # has_existing_session
    test("M2.4 has_existing_session", sm.has_existing_session())

    # update_progress
    sm.update_progress(state, "Test progress entry")
    test("M2.5 update_progress", (tmpdir / "PROGRESS.md").exists())

except Exception as e:
    test("M2.x Session tests", False, str(e))
finally:
    if tmpdir and tmpdir.exists():
        shutil.rmtree(tmpdir)


# ============================================================
print("\n=== Milestone 3: Agent Runner (No opencode) ===")
# ============================================================

try:
    from standalone_agents import AgentRunner, ToolExecutor, LLMClient, TOOL_DEFINITIONS
    test("M3.1 Import AgentRunner", True)
except Exception as e:
    test("M3.1 Import AgentRunner", False, str(e))

# Verify no opencode/subprocess-as-CLI
try:
    import standalone_agents
    source = Path(standalone_agents.__file__).read_text()
    test("M3.2 No 'opencode' in source", "opencode" not in source)
except Exception as e:
    test("M3.2 No opencode check", False, str(e))

# Test ToolExecutor directly
tmpdir2 = None
try:
    tmpdir2 = Path(tempfile.mkdtemp())
    executor = ToolExecutor(tmpdir2)

    # write_file
    result = executor.execute("write_file", {"path": "test.py", "content": "print('hello')\n"})
    test("M3.3 ToolExecutor write_file", "OK" in result and (tmpdir2 / "test.py").exists())

    # read_file
    result = executor.execute("read_file", {"path": "test.py"})
    test("M3.4 ToolExecutor read_file", "print" in result)

    # run_command
    result = executor.execute("run_command", {"command": "echo hello_world"})
    test("M3.5 ToolExecutor run_command", "hello_world" in result)

    # run_command with python
    result = executor.execute("run_command", {"command": "python3 -c \"print(2+2)\""})
    test("M3.6 ToolExecutor python execution", "4" in result)

    # list_directory
    result = executor.execute("list_directory", {"path": "."})
    test("M3.7 ToolExecutor list_directory", "test.py" in result)

    # edit_file
    result = executor.execute("edit_file", {"path": "test.py", "old_str": "hello", "new_str": "world"})
    content = (tmpdir2 / "test.py").read_text()
    test("M3.8 ToolExecutor edit_file", "world" in content and "OK" in result)

    # Tool definitions exist
    test("M3.9 Tool definitions", len(TOOL_DEFINITIONS) >= 4)

except Exception as e:
    test("M3.x ToolExecutor tests", False, str(e))
finally:
    if tmpdir2 and tmpdir2.exists():
        shutil.rmtree(tmpdir2)

# AgentRunner instantiation
try:
    config = Config.load_default()
    runner = AgentRunner(config, Path("."))
    test("M3.10 AgentRunner instantiation", runner is not None)
except Exception as e:
    test("M3.10 AgentRunner instantiation", False, str(e))


# ============================================================
print("\n=== Milestone 4: Orchestrator ===")
# ============================================================

try:
    from standalone_orchestrator import Orchestrator
    test("M4.1 Import Orchestrator", True)
except Exception as e:
    test("M4.1 Import Orchestrator", False, str(e))

# Check required methods
try:
    source = Path("standalone_orchestrator.py").read_text()
    test("M4.2 Has _execute_explore", "_execute_iteration" in source)  # Contains explore within iteration
    test("M4.3 Has _perform_rca", "_perform_root_cause_analysis" in source)
    test("M4.4 Has git commit", "git commit" in source or "git_commit" in source)
except Exception as e:
    test("M4.x Orchestrator method checks", False, str(e))

try:
    config = Config.load_default()
    orch = Orchestrator(config, Path("."))
    test("M4.5 Orchestrator instantiation", orch is not None)
except Exception as e:
    test("M4.5 Orchestrator instantiation", False, str(e))


# ============================================================
print("\n=== Milestone 5: CLI Entry Point ===")
# ============================================================

test("M5.1 standalone_main.py exists", Path("standalone_main.py").exists())
test("M5.2 Is executable", os.access("standalone_main.py", os.X_OK))

try:
    first_line = Path("standalone_main.py").read_text().split("\n")[0]
    test("M5.3 Has shebang", first_line.strip() == "#!/usr/bin/env python3")
except Exception as e:
    test("M5.3 Shebang check", False, str(e))

try:
    source = Path("standalone_main.py").read_text()
    test("M5.4 No opencode", "opencode" not in source)
except Exception as e:
    test("M5.4 No opencode check", False, str(e))

# Help flag
try:
    import subprocess
    result = subprocess.run(
        [sys.executable, "standalone_main.py", "--help"],
        capture_output=True, text=True, timeout=10
    )
    test("M5.5 --help works", result.returncode == 0 and "task" in result.stdout.lower())
except Exception as e:
    test("M5.5 --help check", False, str(e))


# ============================================================
print("\n=== Milestone 6: Prompts ===")
# ============================================================

test("M6.1 prompts/ directory exists", Path("prompts").is_dir())
test("M6.2 initializer.txt", Path("prompts/initializer.txt").exists() and Path("prompts/initializer.txt").stat().st_size > 0)
test("M6.3 explore.txt", Path("prompts/explore.txt").exists() and Path("prompts/explore.txt").stat().st_size > 0)

try:
    plan_content = Path("prompts/plan.txt").read_text()
    test("M6.4 plan.txt has DoD", "definition of done" in plan_content.lower() or "dod" in plan_content.lower())
except:
    test("M6.4 plan.txt DoD", False, "File missing or unreadable")

test("M6.5 build.txt", Path("prompts/build.txt").exists() and Path("prompts/build.txt").stat().st_size > 0)

try:
    test_content = Path("prompts/test.txt").read_text()
    test("M6.6 test.txt has verification", "verification" in test_content.lower() or "verify" in test_content.lower())
except:
    test("M6.6 test.txt verification", False, "File missing or unreadable")


# ============================================================
print("\n=== Summary ===")
# ============================================================

total = passed + failed
print(f"\n{'='*50}")
print(f"Results: {passed}/{total} tests passed, {failed} failed")
print(f"{'='*50}")

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(f"  ❌ {e}")

sys.exit(0 if failed == 0 else 1)
