import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_opencode_config.py"


class ConfigContextTests(unittest.TestCase):
    def _build(self, config: dict, **env_overrides: str) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opencode.json"
            path.write_text(json.dumps(config))
            env = os.environ.copy()
            env.update({"OPENCODE_USER_CONFIG": str(path), "V2_POLICY_MODE": "light", "V2_SUPERVISOR_ISOLATE_PLUGINS": "1", **env_overrides})
            return json.loads(subprocess.check_output(["python3", str(BUILDER)], env=env, text=True))

    def test_light_keeps_memory_capture_but_omits_prompt_heavy_plugins(self):
        config = {
            "plugin": ["@dietrichgebert/ponytail@4.10.0", "superpowers@git+https://example.invalid/superpowers", "/home/brandon/.config/opencode/plugins/agentmemory-capture.ts"],
            "mcp": {"agentmemory": {"type": "local", "command": ["npx", "-y", "@agentmemory/mcp"], "enabled": True}},
        }
        built = self._build(config)
        self.assertEqual(built["plugin"], ["/home/brandon/.config/opencode/plugins/agentmemory-capture.ts", str(ROOT / "opencode-plugin" / "orchestrator-supervisor.ts")])
        self.assertEqual(built["mcp"]["agentmemory"]["command"], ["node", str(ROOT / "scripts" / "agentmemory-mcp-filter.mjs")])

    def test_strict_isolation_remains_plugin_free_except_supervisor(self):
        config = {"plugin": ["@dietrichgebert/ponytail@4.10.0", "/home/brandon/.config/opencode/plugins/agentmemory-capture.ts"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opencode.json"
            path.write_text(json.dumps(config))
            env = os.environ.copy()
            env.update({"OPENCODE_USER_CONFIG": str(path), "V2_POLICY_MODE": "strict", "V2_SUPERVISOR_ISOLATE_PLUGINS": "1"})
            built = json.loads(subprocess.check_output(["python3", str(BUILDER)], env=env, text=True))
        self.assertEqual(built["plugin"], [str(ROOT / "opencode-plugin" / "orchestrator-supervisor.ts")])

    def test_compaction_anchor_is_structured_not_raw_string_slice(self):
        plugin = (ROOT / "opencode-plugin" / "orchestrator-supervisor.ts").read_text()
        self.assertIn("compactFacts(summary)", plugin)
        self.assertNotIn("compact.slice(0, 1800)", plugin)


if __name__ == "__main__":
    unittest.main()
