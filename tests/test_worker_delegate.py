import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "orchestrator_v2" / "worker_delegate.py"
spec = importlib.util.spec_from_file_location("worker_delegate", MODULE_PATH)
worker_delegate = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(worker_delegate)


class WorkerDelegateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        subprocess.run(["git", "-C", str(self.root), "init", "-q"], check=True)
        (self.root / "src").mkdir()
        (self.root / "src" / "bug.py").write_text("def calculate(value):\n    return value + 1\n", encoding="utf-8")
        (self.root / ".env").write_text("TOKEN=not-for-workers\n", encoding="utf-8")
        outside = Path(self.temp.name) / "outside.txt"
        outside.write_text("outside\n", encoding="utf-8")
        try:
            (self.root / "src" / "escape.txt").symlink_to(outside)
        except OSError:
            pass

    def tearDown(self):
        self.temp.cleanup()

    def test_root_confinement_and_secret_rejection(self):
        for value in ("../outside.txt", "/etc/passwd", ".env"):
            with self.assertRaises(worker_delegate.WorkerError):
                worker_delegate._safe_path(self.root, value)
        if (self.root / "src" / "escape.txt").is_symlink():
            with self.assertRaises(worker_delegate.WorkerError):
                worker_delegate._safe_path(self.root, "src/escape.txt")

    def test_bounded_read_search_glob_git_and_no_write(self):
        examined = set()
        before = (self.root / "src" / "bug.py").read_bytes()
        read = worker_delegate.dispatch_tool(self.root, "worker_read_file", {"path": "src/bug.py", "start_line": 1, "end_line": 2}, examined)
        self.assertIn("return value + 1", read["content"])
        search = worker_delegate.dispatch_tool(self.root, "worker_search", {"pattern": "calculate", "max_results": 2}, examined)
        self.assertEqual(search["matches"][0]["path"], "src/bug.py")
        glob = worker_delegate.dispatch_tool(self.root, "worker_glob", {"pattern": "src/*.py", "max_results": 2}, examined)
        self.assertEqual(glob["matches"], ["src/bug.py"])
        status = worker_delegate.dispatch_tool(self.root, "worker_git", {"action": "status"}, examined)
        self.assertEqual(status["exit_code"], 0)
        self.assertNotIn(".env", status["output"])
        with self.assertRaises(worker_delegate.WorkerError):
            worker_delegate.dispatch_tool(self.root, "worker_git", {"action": "diff"}, examined)
        self.assertEqual((self.root / "src" / "bug.py").read_bytes(), before)

    def test_unknown_malformed_and_result_bounds(self):
        with self.assertRaises(worker_delegate.WorkerError):
            worker_delegate.dispatch_tool(self.root, "bash", {"command": "touch should-not-exist"}, set())
        with self.assertRaises(worker_delegate.WorkerError):
            worker_delegate.dispatch_tool(self.root, "worker_read_file", "not-an-object", set())
        self.assertFalse((self.root / "should-not-exist").exists())
        result = worker_delegate.bound_result({
            "status": "complete", "finding": "x" * 5000,
            "evidence": [{"path": "src/bug.py", "lines": "1-2", "note": "y" * 1000} for _ in range(20)],
            "recommended_next_action": "z" * 2000, "uncertainty": "u" * 1000,
        }, "scout", {"files_examined": 1, "worker_steps": 2, "worker_input_tokens": 3, "worker_output_tokens": 4, "elapsed_seconds": 0.1})
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()), worker_delegate.MAX_RESULT_BYTES)
        self.assertLessEqual(len(result["evidence"]), worker_delegate.MAX_EVIDENCE)

    def test_evidence_is_validated_against_the_repository(self):
        result = worker_delegate._validate_evidence(self.root, {
            "evidence": [
                {"path": "src/bug.py", "lines": "1", "note": "real"},
                {"path": "security/data-loss", "lines": "3", "note": "prose"},
                {"path": "importlib.import_mod", "lines": "3", "note": "not a file"},
            ]
        })
        self.assertEqual([item["path"] for item in result["evidence"]], ["src/bug.py"])

    def test_worker_schema_is_small(self):
        self.assertLess(len(json.dumps(worker_delegate._tool_schemas(), separators=(",", ":"))), 1400)


if __name__ == "__main__":
    unittest.main()
