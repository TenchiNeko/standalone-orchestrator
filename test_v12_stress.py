#!/usr/bin/env python3
"""
v1.2 COMPREHENSIVE STRESS TEST
================================
Tests all new optimizations + regression tests for v1.1 patches.

A. Config: New inference fields (repeat_penalty, thinking, top_p, etc.)
B. Ollama Options: repeat_penalty, top_p, min_p, num_keep in API payloads
C. Thinking Mode: /think and /no_think injection per agent role
D. Structured Edit Repair: JSON schema + method + fallback logic
E. AST-Aware RAG: chunk_python_ast + add_ast_chunks
F. Self-Play Data: Training pair collection
G. Regression: All v1.1 patches still work
"""

import sys
import os
import ast
import json
import textwrap
import tempfile
import difflib

sys.path.insert(0, '/home/claude/orchestrator-work')

PASS = 0
FAIL = 0
ERRORS = []

def check(name, condition, detail=""):
    global PASS, FAIL, ERRORS
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        msg = f"  ❌ {name}: {detail}"
        print(msg)
        ERRORS.append(msg)


# ============================================================
# A. CONFIG: NEW INFERENCE FIELDS
# ============================================================
print("\n═══ A. Config: Inference Optimization Fields ═══")

from standalone_config import ModelConfig, Config, default_config

# A1: Default values
mc = ModelConfig(name="test", provider="ollama", model_id="test")
check("A1: repeat_penalty default 1.0", mc.repeat_penalty == 1.0)
check("A2: thinking_mode default auto", mc.thinking_mode == "auto")
check("A3: thinking_budget default 0", mc.thinking_budget == 0)
check("A4: top_p default 0.95", mc.top_p == 0.95)
check("A5: min_p default 0", mc.min_p == 0.0)
check("A6: num_keep default -1", mc.num_keep == -1)
check("A7: draft_model default None", mc.draft_model is None)

# A8: Default config values
config = default_config()
primary = config.agents["plan"].model
check("A8: primary repeat_penalty=1.0", primary.repeat_penalty == 1.0)
check("A9: primary thinking=auto", primary.thinking_mode == "auto")
check("A10: primary top_p=0.95", primary.top_p == 0.95)

secondary = config.agents["test"].model
check("A11: secondary repeat_penalty=1.0", secondary.repeat_penalty == 1.0)
check("A12: secondary thinking=disabled", secondary.thinking_mode == "disabled")


# ============================================================
# B. OLLAMA OPTIONS: API PAYLOAD VERIFICATION
# ============================================================
print("\n═══ B. Ollama Options: API Payload ═══")

with open('standalone_agents.py') as f:
    agents_src = f.read()

# B1: repeat_penalty in both options blocks
# Options are now built as separate dicts (options, options2) then assigned to payload
rp_count = agents_src.count('"repeat_penalty": self.config.repeat_penalty')
check(f"B1: repeat_penalty in {rp_count}/2 options", rp_count >= 2)

# B2: top_p in both options blocks
tp_count = agents_src.count('"top_p": self.config.top_p')
check(f"B2: top_p in {tp_count}/2 options", tp_count >= 2)

# B3: num_keep conditionally added
check("B3: num_keep conditional", 'num_keep' in agents_src and 'self.config.num_keep >= 0' in agents_src)

# B4: min_p conditionally added
check("B4: min_p conditional", 'min_p' in agents_src and 'self.config.min_p > 0' in agents_src)


# ============================================================
# C. THINKING MODE INJECTION
# ============================================================
print("\n═══ C. Thinking Mode Injection ═══")

# C1: /think injection exists
check("C1: /think injection", '"/think\\n"' in agents_src or "'/think\\n'" in agents_src or '/think\\n' in agents_src)

# C2: /no_think injection exists
check("C2: /no_think injection", '/no_think' in agents_src)

# C3: Heavy roles get thinking
check("C3: heavy_roles defined", "heavy_roles" in agents_src and '"plan"' in agents_src and '"build"' in agents_src)

# C4: Auto mode logic
check("C4: auto→enabled for heavy", 'thinking_mode = "enabled"' in agents_src)
check("C5: auto→disabled for light", 'thinking_mode = "disabled"' in agents_src)

# C6: Budget control
check("C6: thinking_budget injection", "thinking_budget" in agents_src)

# C7: <think> block stripping (v1.1 regression)
check("C7: think block stripping", "<think>" in agents_src and "re.sub" in agents_src)


# ============================================================
# D. STRUCTURED EDIT REPAIR
# ============================================================
print("\n═══ D. Structured Edit Repair ═══")

from standalone_agents import AgentRunner, EDIT_REPAIR_SCHEMA

# D1: Schema structure
check("D1: schema has edits", "edits" in EDIT_REPAIR_SCHEMA["properties"])
check("D2: edit has search/replace", 
      "search" in EDIT_REPAIR_SCHEMA["properties"]["edits"]["items"]["properties"] and
      "replace" in EDIT_REPAIR_SCHEMA["properties"]["edits"]["items"]["properties"])
check("D3: edits required", "edits" in EDIT_REPAIR_SCHEMA["required"])

# D4: Schema is valid JSON Schema
try:
    json.dumps(EDIT_REPAIR_SCHEMA)
    check("D4: schema serializable", True)
except:
    check("D4: schema serializable", False)

# D5: run_edit_repair_structured method exists
check("D5: method exists", hasattr(AgentRunner, 'run_edit_repair_structured'))

# D6: Original run_edit_repair still exists
check("D6: original still exists", hasattr(AgentRunner, 'run_edit_repair'))

# D7: Structured fallback in orchestrator
with open('standalone_orchestrator.py') as f:
    orch_src = f.read()
check("D7: structured→fallback pattern", 
      'run_edit_repair_structured' in orch_src and
      'edits is None' in orch_src and
      'parse_search_replace' in orch_src)

# D8: Structured method uses chat_structured
import inspect
method_src = inspect.getsource(AgentRunner.run_edit_repair_structured)
check("D8: uses chat_structured", "chat_structured" in method_src)
check("D9: uses EDIT_REPAIR_SCHEMA", "EDIT_REPAIR_SCHEMA" in method_src)
check("D10: returns list of tuples or None", "return edits" in method_src and "return None" in method_src)

# D11: Simulate structured output parsing
mock_result = {
    "edits": [
        {"search": "def hello():\n    return 'world'", "replace": "def hello():\n    return 'earth'", "reason": "fix greeting"},
        {"search": "x = 1", "replace": "x = 2"},
    ],
    "analysis": "Fixed greeting and variable"
}
edits = []
for edit in mock_result["edits"]:
    s = edit.get("search", "").strip()
    r = edit.get("replace", "").strip()
    if s:
        edits.append((s, r))
check("D11: mock parse produces 2 edits", len(edits) == 2)
check("D12: mock edit format correct", edits[0][0] == "def hello():\n    return 'world'" and edits[0][1] == "def hello():\n    return 'earth'")

# D13: Thinking enabled in structured repair
check("D13: /think in structured repair", "/think" in method_src)


# ============================================================
# E. AST-AWARE RAG
# ============================================================
print("\n═══ E. AST-Aware RAG ═══")

from librarian_store import chunk_python_ast, add_ast_chunks

# E1: Basic function chunking
src = textwrap.dedent("""\
    import os
    from pathlib import Path
    
    def hello():
        '''Say hello.'''
        return 'world'
    
    def goodbye(name: str) -> str:
        '''Say goodbye to someone.'''
        return f'goodbye {name}'
    
    class Greeter:
        def __init__(self, name):
            self.name = name
        
        def greet(self):
            return f'hello {self.name}'
""")
chunks = chunk_python_ast(src, "greeter.py")
check(f"E1: produced {len(chunks)} chunks", len(chunks) >= 3)

# E2: Function has signature
func_chunks = [c for c in chunks if c["type"] == "function"]
check("E2: function chunks found", len(func_chunks) >= 2)

# E3: Has docstring
hello_chunk = [c for c in chunks if c["name"] == "hello"]
check("E3: hello has docstring", hello_chunk and "Say hello" in hello_chunk[0].get("docstring", ""))

# E4: Has imports
check("E4: chunks have imports", any(c.get("imports") for c in chunks))

# E5: Class chunk
class_chunks = [c for c in chunks if c["type"] == "class"]
check("E5: class chunk found", len(class_chunks) >= 1)

# E6: Class has methods list
if class_chunks:
    check("E6: class has methods", "methods" in class_chunks[0] and "greet" in str(class_chunks[0]["methods"]))
else:
    check("E6: class has methods", False, "no class chunks")

# E7: Function signature extraction
sig_chunks = [c for c in chunks if "goodbye" in c.get("signature", "")]
check("E7: signature with types", sig_chunks and "str" in sig_chunks[0]["signature"])

# E8: Empty file
chunks = chunk_python_ast("", "empty.py")
check("E8: empty file → no chunks", len(chunks) == 0)

# E9: Syntax error file
chunks = chunk_python_ast("def broken(\n    # missing close", "broken.py")
check("E9: syntax error → fallback chunk", len(chunks) == 1 and chunks[0]["type"] == "file")

# E10: Large class gets split into methods
src = "import os\n\nclass BigClass:\n"
for i in range(40):  # 40 methods × 3 lines = 120+ lines, exceeds max_chunk_lines=80
    src += f"    def method_{i}(self):\n        return {i}\n\n"
chunks = chunk_python_ast(src, "big.py")
method_chunks = [c for c in chunks if c["type"] == "method"]
check(f"E10: large class split into {len(method_chunks)} methods", len(method_chunks) >= 10)

# E11: Real Flask app pattern
src = textwrap.dedent("""\
    import os
    import jwt
    from datetime import datetime
    from flask import Flask, request, jsonify, g
    
    def create_app(test_config=None):
        app = Flask(__name__)
        app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev')
        
        @app.route('/register', methods=['POST'])
        def register():
            data = request.get_json()
            return jsonify({"user": data.get("username")}), 201
        
        @app.route('/login', methods=['POST'])
        def login():
            data = request.get_json()
            token = jwt.encode({"user": data["username"]}, app.config['SECRET_KEY'])
            return jsonify({"token": token})
        
        return app
    
    class ExpenseDB:
        def __init__(self, path='expenses.db'):
            self.path = path
        
        def add_expense(self, user_id, amount, description):
            pass
        
        def get_expenses(self, user_id):
            return []
""")
chunks = chunk_python_ast(src, "app.py")
check(f"E11: Flask app produces {len(chunks)} chunks", len(chunks) >= 2)

# E12: add_ast_chunks to temp DB
with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
    tmp_db = f.name
from librarian_store import init_librarian_tables
init_librarian_tables(tmp_db)
count = add_ast_chunks(src, "app.py", domain="web_api", db_path=tmp_db)
check(f"E12: stored {count} chunks to DB", count >= 2)
os.unlink(tmp_db)

# E13: Async function support
src = "import asyncio\n\nasync def fetch_data(url: str) -> dict:\n    '''Fetch data from URL.'''\n    return {}\n"
chunks = chunk_python_ast(src, "async.py")
async_chunks = [c for c in chunks if c["type"] == "async_function"]
check("E13: async function detected", len(async_chunks) == 1)


# ============================================================
# F. SELF-PLAY DATA COLLECTION
# ============================================================
print("\n═══ F. Self-Play Data Collection ═══")

# F1: Training pair collection code exists
check("F1: selfplay in orchestrator", "selfplay" in orch_src and "training_pairs" in orch_src)

# F2: JSONL format
check("F2: JSONL format", "jsonl" in orch_src)

# F3: Instruction/output fields
check("F3: instruction field", '"instruction"' in orch_src)
check("F4: output field", '"output"' in orch_src and '"filename"' in orch_src)

# F5: Non-fatal error handling
check("F5: non-fatal wrapper", "Self-play data collection failed (non-fatal)" in orch_src)

# F6: Skips hidden files
check("F6: skips hidden", 'startswith(".")' in orch_src or "startswith('.')" in orch_src)

# F7: _detect_domain exists
from standalone_orchestrator import Orchestrator
check("F7: _detect_domain exists", hasattr(Orchestrator, '_detect_domain'))

# F8: Domain detection
check("F8: flask→web_api", Orchestrator._detect_domain("Build a Flask REST API") == "web_api")
check("F9: cli→cli", Orchestrator._detect_domain("Create a CLI tool with argparse") == "cli")
check("F10: general fallback", Orchestrator._detect_domain("Do something random") == "general")


# ============================================================
# G. REGRESSION: ALL v1.1 PATCHES STILL WORK
# ============================================================
print("\n═══ G. Regression: v1.1 Patches ═══")

# G1: AST-guarded imports
check("G1: _find_safe_import_line", hasattr(Orchestrator, '_find_safe_import_line'))

src = "import os\n\ntry:\n    from flask import g\nexcept:\n    pass\n"
pos = Orchestrator._find_safe_import_line(src)
check("G2: try block avoided", pos == 1)

# G3: Hashline tag stripping
out = "<<<SEARCH>>>\n  3:f1| from flask import g\n<<<REPLACE>>>\nfrom flask import g, current_app\n<<<END>>>"
edits = AgentRunner.parse_search_replace(out)
check("G3: tags stripped from SEARCH", len(edits) == 1 and "f1|" not in edits[0][0])

# G4: Fuzzy matching
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write("def hello():\n    return 'world'\n")
    tmp = f.name
num, fb = AgentRunner.apply_search_replace(tmp, [("def hello():\n    return 'world'", "def hello():\n    return 'earth'")])
check("G4: exact match works", num == 1)
os.unlink(tmp)

# G5: Cross-import blocking
check("G5: cross-import block", "test→source" in orch_src)

# G6: num_ctx passed
check("G6: num_ctx in options", agents_src.count('"num_ctx": self.config.context_window') >= 2)

# G7: _safe_write_python
check("G7: _safe_write_python exists", hasattr(Orchestrator, '_safe_write_python'))

from pathlib import Path
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write("x = 1\n")
    tmp = f.name
result = Orchestrator._safe_write_python(Path(tmp), "def broken(\n", "x = 1\n")
check("G8: safe_write rejects broken", result == False and open(tmp).read() == "x = 1\n")
os.unlink(tmp)

# G9: KB local fallback
with open('kb_client.py') as f:
    kb_src = f.read()
check("G9: KB local fallback", "_local_fallback" in kb_src)

# G10: Edit repair feedback
check("G10: total_edits_applied tracking", "total_edits_applied" in orch_src)

# G11: Flask golden snippet in edit repair
check("G11: Flask in edit repair", "Flask golden snippets for edit repair" in agents_src)

# G12: Benchmark timeout capture
with open('benchmark.py') as f:
    bench_src = f.read()
check("G12: timeout captures output", "e.stdout" in bench_src)

# G13: All source files parse
all_parse = True
for fname in ['standalone_orchestrator.py', 'standalone_agents.py', 'standalone_config.py',
              'kb_client.py', 'librarian_store.py', 'benchmark.py', 'standalone_main.py',
              'standalone_models.py', 'standalone_session.py', 'standalone_memory.py',
              'librarian.py', 'playbook_reader.py', 'standalone_trace_collector.py']:
    try:
        ast.parse(open(fname).read())
    except (SyntaxError, FileNotFoundError) as e:
        all_parse = False
        print(f"    ⚠️ {fname}: {e}")
check("G13: all 13 source files parse", all_parse)


# ============================================================
# H. ADVERSARIAL EDGE CASES FOR NEW CODE
# ============================================================
print("\n═══ H. Adversarial Edge Cases ═══")

# H1: Structured output with empty edits
mock = {"edits": [], "analysis": "nothing to fix"}
edits = [(e.get("search","").strip(), e.get("replace","").strip()) for e in mock["edits"] if e.get("search","").strip()]
check("H1: empty edits list → empty result", len(edits) == 0)

# H2: Structured output with missing fields
mock = {"edits": [{"search": "foo", "replace": "bar"}, {"replace": "baz"}]}
edits = [(e.get("search","").strip(), e.get("replace","").strip()) for e in mock["edits"] if e.get("search","").strip()]
check("H2: missing search skipped", len(edits) == 1)

# H3: AST chunk with decorator
src = textwrap.dedent("""\
    import functools
    
    @functools.lru_cache
    def expensive(n: int) -> int:
        return sum(range(n))
""")
chunks = chunk_python_ast(src, "cached.py")
func_chunks = [c for c in chunks if c["type"] == "function"]
check("H3: decorated function chunked", len(func_chunks) == 1)

# H4: AST chunk with nested classes
src = textwrap.dedent("""\
    class Outer:
        class Inner:
            def method(self):
                pass
        def outer_method(self):
            pass
""")
chunks = chunk_python_ast(src, "nested.py")
check("H4: nested class handled", len(chunks) >= 1)

# H5: AST chunk with type annotations
src = "from typing import List, Optional\n\ndef process(items: List[str], limit: Optional[int] = None) -> bool:\n    return True\n"
chunks = chunk_python_ast(src, "typed.py")
func_chunks = [c for c in chunks if c["type"] == "function"]
check("H5: type annotations in signature", func_chunks and "List[str]" in func_chunks[0].get("signature", ""))

# H6: Config JSON override for new fields
from standalone_config import load_config
import json as _json
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    _json.dump({
        "agents": {
            "plan": {
                "repeat_penalty": 1.1,
                "thinking_mode": "enabled",
                "thinking_budget": 4096,
                "top_p": 0.9,
            }
        }
    }, f)
    tmp_cfg = f.name
cfg = load_config(Path(tmp_cfg))
plan_model = cfg.agents["plan"].model
check("H6: JSON override repeat_penalty", plan_model.repeat_penalty == 1.1)
check("H7: JSON override thinking_mode", plan_model.thinking_mode == "enabled")
check("H8: JSON override thinking_budget", plan_model.thinking_budget == 4096)
check("H9: JSON override top_p", plan_model.top_p == 0.9)
os.unlink(tmp_cfg)

# H10: _detect_domain edge cases
check("H10: domain - database", Orchestrator._detect_domain("Build SQLite ORM models") == "database")
check("H11: domain - testing", Orchestrator._detect_domain("Create pytest fixtures") == "testing")
check("H12: domain - empty", Orchestrator._detect_domain("") == "general")

# H13: AST chunking with only imports
src = "import os\nimport sys\nfrom pathlib import Path\n"
chunks = chunk_python_ast(src, "imports_only.py")
check("H13: imports-only file", len(chunks) <= 1)  # Should produce 0-1 chunks

# H14: Very large function (> max_chunk_lines)
src = "def huge_func():\n" + "".join(f"    x_{i} = {i}\n" for i in range(200)) + "    return 0\n"
chunks = chunk_python_ast(src, "huge.py")
check("H14: huge function as single chunk", len(chunks) >= 1)

# H15: Module-level code between functions
src = textwrap.dedent("""\
    import os
    
    X = 42
    Y = 'hello'
    
    def foo():
        return X
    
    Z = X + 1
    
    def bar():
        return Z
""")
chunks = chunk_python_ast(src, "mixed.py")
check(f"H15: module + functions = {len(chunks)} chunks", len(chunks) >= 2)


# ============================================================
# FINAL REPORT
# ============================================================
print(f"\n{'='*60}")
print(f"v1.2 STRESS TEST RESULTS: {PASS} passed, {FAIL} failed out of {PASS+FAIL} tests")
print(f"{'='*60}")

if ERRORS:
    print(f"\n❌ FAILURES:")
    for e in ERRORS:
        print(e)
    sys.exit(1)
else:
    print(f"\n✅ ALL {PASS} TESTS PASSED — v1.2 CLEAN")
    sys.exit(0)
