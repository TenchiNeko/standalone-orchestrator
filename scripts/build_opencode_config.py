#!/usr/bin/env python3
"""Build launcher-only OpenCode config without changing the user's config file."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config_path = Path(os.environ.get("OPENCODE_USER_CONFIG", "~/.config/opencode/opencode.json")).expanduser()
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    # OpenCode also merges its XDG config.  The daily launcher opts into an
    # isolated config home and must not re-add global AgentMemory/Superpowers
    # plugins (their large instructions and independent tools defeat the
    # single supervised loop).  The user's global config remains untouched.
    plugins = [] if os.environ.get("V2_SUPERVISOR_ISOLATE_PLUGINS") == "1" else list(config.get("plugin") or [])
    plugin = str(root / "opencode-plugin" / "orchestrator-supervisor.ts")
    if plugin not in plugins:
        plugins.append(plugin)
    config["plugin"] = plugins
    providers = dict(config.get("provider") or {})
    providers["orca-q6"] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Current local Qwen (v2 supervised)",
        "options": {
            "baseURL": "http://127.0.0.1:18089/v1",
            "apiKey": "{file:/home/brandon/.config/opencode/secrets/orca-q6.key}",
            "timeout": 900000,
        },
        "models": {"orca27b-ultra-q6-mtp": {"name": "Qwen3.8-27B Ultra Q6"}},
    }
    config["provider"] = providers
    # A command-line -m/--model still wins in OpenCode; otherwise use the
    # verified production local model for this launcher only.
    if not any(arg == "-m" or arg == "--model" or arg.startswith("--model=") for arg in sys.argv[1:]):
        config["model"] = "orca-q6/orca27b-ultra-q6-mtp"
        config["small_model"] = "orca-q6/orca27b-ultra-q6-mtp"
    print(json.dumps(config, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
