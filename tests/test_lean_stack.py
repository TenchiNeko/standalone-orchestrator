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
FAKE_SERVER = r'''
import fs from "node:fs";
const log = process.argv[2];
fs.writeFileSync(log, JSON.stringify({argv: process.argv.slice(2)}) + "\n");
const huge = ("α😀 quoted \\\"json\\\"\\n {\\\"memory\\\":true} ").repeat(700);
let buffer = "";
function reply(message) { process.stdout.write(JSON.stringify(message) + "\n"); }
process.stdin.on("data", (chunk) => {
  buffer += chunk;
  let index;
  while ((index = buffer.indexOf("\n")) >= 0) {
    const line = buffer.slice(0, index);
    buffer = buffer.slice(index + 1);
    if (!line.trim()) continue;
    const request = JSON.parse(line);
    fs.appendFileSync(log, JSON.stringify(request) + "\n");
    if (request.method === "tools/list") {
      reply({jsonrpc: "2.0", id: request.id, result: {tools: [
        ...["memory_smart_search", "memory_recall", "memory_save", "memory_lesson_recall", "memory_lesson_save", "memory_reflect"].map(name => ({name}))
      ]}});
      continue;
    }
    if (request.method !== "tools/call" || request.id === undefined) continue;
    const name = request.params?.name;
    const args = request.params?.arguments || {};
    if (name === "memory_recall" && args.query === "rpc-error") {
      reply({jsonrpc: "2.0", id: request.id, error: {code: -32001, message: "upstream fake error"}});
    } else if (["memory_recall", "memory_smart_search", "memory_lesson_recall"].includes(name)) {
      reply({jsonrpc: "2.0", id: request.id, result: {
        isError: args.query === "iserror",
        content: [{type: "text", text: huge}],
        structuredContent: {source: "fake", payload: huge}
      }});
    } else if (["memory_save", "memory_lesson_save"].includes(name)) {
      reply({jsonrpc: "2.0", id: request.id, result: {content: [{type: "text", text: huge} ]}});
    } else {
      reply({jsonrpc: "2.0", id: request.id, result: {}});
    }
  }
});
'''


class LeanStackTests(unittest.TestCase):
    def start_filter(self, root, server, log):
        env = os.environ.copy()
        env.update({
            "V2_AGENTMEMORY_MCP_COMMAND": json.dumps(["node", str(server), str(log), "preserved-arg"]),
            "V2_AGENTMEMORY_ALLOWED_TOOLS": json.dumps(sorted(ALLOWED)),
        })
        proc = subprocess.Popen(
            ["node", str(FILTER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=env,
        )
        self.addCleanup(self.stop_filter, proc)
        return proc

    @staticmethod
    def stop_filter(proc):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream and not stream.closed:
                stream.close()

    @staticmethod
    def send(proc, request):
        assert proc.stdin and proc.stdout
        proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        return json.loads(proc.stdout.readline())

    @staticmethod
    def visible_size(result):
        visible = {"content": result.get("content")}
        if "structuredContent" in result:
            visible["structuredContent"] = result["structuredContent"]
        return len(json.dumps(visible, ensure_ascii=False, separators=(",", ":")).encode())

    def make_fixture(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        server = root / "fake.mjs"
        server.write_text(FAKE_SERVER)
        return root, server, root / "requests.jsonl"

    def test_filter_clamps_retrieval_and_preserves_transport_and_daily_tools(self):
        root, server, log = self.make_fixture()
        proc = self.start_filter(root, server, log)

        listed = self.send(proc, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertEqual({tool["name"] for tool in listed["result"]["tools"]}, ALLOWED)

        self.send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "memory_recall", "arguments": {"query": "large", "limit": 50, "format": "full", "token_budget": 10000}
        }})
        self.send(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "memory_recall", "arguments": {"query": "small", "limit": 2, "format": "narrative", "token_budget": 80}
        }})
        self.send(proc, {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
            "name": "memory_smart_search", "arguments": {"query": "search", "limit": 50, "expandIds": "a,b,c,d,e"}
        }})
        self.send(proc, {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {
            "name": "memory_lesson_recall", "arguments": {"query": "lesson", "project": "p", "minConfidence": 0.7, "limit": 50}
        }})
        save_args = {"agentId": "qwen", "content": "keep", "project": "p", "type": "note"}
        lesson_save_args = {"confidence": 0.8, "content": "keep lesson", "project": "p", "tags": ["x"]}
        self.send(proc, {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "memory_save", "arguments": save_args}})
        self.send(proc, {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "memory_lesson_save", "arguments": lesson_save_args}})

        records = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(records[0]["argv"][1:], ["preserved-arg"])
        by_id = {record["id"]: record for record in records[1:] if "id" in record}
        recall = by_id[2]["params"]["arguments"]
        self.assertEqual((recall["limit"], recall["format"], recall["token_budget"]), (3, "compact", 900))
        small = by_id[3]["params"]["arguments"]
        self.assertEqual((small["limit"], small["format"], small["token_budget"]), (2, "compact", 80))
        smart = by_id[4]["params"]["arguments"]
        self.assertEqual(smart["limit"], 3)
        self.assertEqual(smart["expandIds"], "a,b,c")
        lesson = by_id[5]["params"]["arguments"]
        self.assertEqual(lesson["limit"], 3)
        self.assertEqual((lesson["query"], lesson["project"], lesson["minConfidence"]), ("lesson", "p", 0.7))
        self.assertEqual(by_id[6]["params"]["arguments"], save_args)
        self.assertEqual(by_id[7]["params"]["arguments"], lesson_save_args)

    def test_filter_bounds_unicode_payload_preserves_errors_and_nonretrieval(self):
        root, server, log = self.make_fixture()
        proc = self.start_filter(root, server, log)
        bounded = self.send(proc, {"jsonrpc": "2.0", "id": 11, "method": "tools/call", "params": {
            "name": "memory_recall", "arguments": {"query": "large"}
        }})
        self.assertEqual(bounded["id"], 11)
        self.assertEqual(bounded["jsonrpc"], "2.0")
        self.assertIn("truncated by v2 lean-context guard", bounded["result"]["content"][0]["text"])
        self.assertLessEqual(self.visible_size(bounded["result"]), 4096)
        json.dumps(bounded, ensure_ascii=False).encode("utf-8")

        iserror = self.send(proc, {"jsonrpc": "2.0", "id": 12, "method": "tools/call", "params": {
            "name": "memory_smart_search", "arguments": {"query": "iserror"}
        }})
        self.assertIs(iserror["result"]["isError"], True)
        self.assertLessEqual(self.visible_size(iserror["result"]), 4096)

        upstream = self.send(proc, {"jsonrpc": "2.0", "id": 13, "method": "tools/call", "params": {
            "name": "memory_recall", "arguments": {"query": "rpc-error"}
        }})
        self.assertEqual(upstream["error"]["code"], -32001)
        self.assertEqual(upstream["error"]["message"], "upstream fake error")

        nonretrieval = self.send(proc, {"jsonrpc": "2.0", "id": 14, "method": "tools/call", "params": {
            "name": "memory_save", "arguments": {"content": "unchanged"}
        }})
        self.assertGreater(len(json.dumps(nonretrieval, ensure_ascii=False).encode()), 4096)
        self.assertNotIn("truncated by v2 lean-context guard", nonretrieval["result"]["content"][0]["text"])

        # Responses consume metadata; many sequential requests remain usable.
        for request_id in range(1000, 2100):
            response = self.send(proc, {"jsonrpc": "2.0", "id": request_id, "method": "tools/list"})
            self.assertEqual(response["id"], request_id)
        malformed = proc.stdin
        assert malformed
        malformed.write("{not-json\n")
        malformed.flush()
        clean = self.send(proc, {"jsonrpc": "2.0", "id": 2101, "method": "tools/list"})
        self.assertEqual({tool["name"] for tool in clean["result"]["tools"]}, ALLOWED)

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
