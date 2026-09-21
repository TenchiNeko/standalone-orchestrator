import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class OpenCodeConfigTests(unittest.TestCase):
    def test_v2_compaction_policy_is_local_and_explicit(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "build_opencode_config.py"
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "opencode.json"
            config_path.write_text(json.dumps({"compaction": {"reserved": 12000}, "model": "other/model"}))
            env = {
                **os.environ,
                "OPENCODE_USER_CONFIG": str(config_path),
                "V2_COMPACTION_RESERVED": "8000",
            }
            output = subprocess.check_output([sys.executable, str(script)], env=env, text=True)
            generated = json.loads(output)
            original_reserved = json.loads(config_path.read_text())["compaction"]["reserved"]

        self.assertEqual(generated["compaction"], {"reserved": 8000, "auto": True, "prune": True})
        self.assertEqual(original_reserved, 12000)
        self.assertEqual(generated["provider"]["orca-q6"]["models"]["orca27b-ultra-q6-mtp"]["limit"], {
            "context": 32768,
            "input": 28000,
            "output": 2048,
        })


if __name__ == "__main__":
    unittest.main()
