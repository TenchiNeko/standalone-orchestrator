import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILTER = ROOT / "scripts" / "agentmemory-mcp-filter.mjs"
SKILLS = ROOT / "skills"
ALLOWED = {
    "memory_smart_search", "memory_recall", "memory_save",
    "memory_lesson_recall", "memory_lesson_save",
}


class LeanStackTests(unittest.TestCase):
    def test_filter_preserves_configured_argv_and_exposes_only_daily_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "argv.json"
            server = root / "fake.mjs"
            server.write_text(
                "import fs from 'node:fs';\n"
                f"fs.writeFileSync({json.dumps(str(log))}, JSON.stringify(process.argv.slice(2)));\n"
                "let b=''; process.stdin.on('data', c => { b += c; let i; while ((i=b.indexOf('\\n')) >= 0) { const line=b.slice(0,i); b=b.slice(i+1); if (!line) continue; const q=JSON.parse(line); if (q.method==='tools/list') { console.log(JSON.stringify({jsonrpc:'2.0',id:q.id,result:{tools:[...['memory_smart_search','memory_recall','memory_save','memory_lesson_recall','memory_lesson_save','memory_reflect'].map(name=>({name}) )]}})); } else if (q.id !== undefined) console.log(JSON.stringify({jsonrpc:'2.0',id:q.id,result:{}})); } });\n"
            )
            env = os.environ.copy()
            env.update({
                "V2_AGENTMEMORY_MCP_COMMAND": json.dumps(["node", str(server), "preserved-arg"]),
                "V2_AGENTMEMORY_ALLOWED_TOOLS": json.dumps(sorted(ALLOWED)),
            })
            proc = subprocess.Popen(["node", str(FILTER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
            assert proc.stdin and proc.stdout
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n")
            proc.stdin.flush()
            result = json.loads(proc.stdout.readline())
            proc.terminate()
            proc.wait(timeout=3)
            self.assertEqual({tool["name"] for tool in result["result"]["tools"]}, ALLOWED)
            self.assertEqual(json.loads(log.read_text()), ["preserved-arg"])
            proc.stdin.close()
            proc.stdout.close()

    def test_skill_catalog_is_small_and_named(self):
        expected = {"lean-debug", "lean-test", "lean-review"}
        found = set()
        for path in SKILLS.glob("*/SKILL.md"):
            found.add(path.parent.name)
            text = path.read_text()
            self.assertLessEqual(len(text.split()), 600)
            self.assertIn("name:", text.split("---", 2)[1])
            self.assertIn("description:", text.split("---", 2)[1])
        self.assertEqual(found, expected)


if __name__ == "__main__":
    unittest.main()
