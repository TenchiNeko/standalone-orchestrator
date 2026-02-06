# Standalone Agent Orchestrator

A fully local, multi-agent coding system that uses Ollama-hosted LLMs to autonomously plan, build, and verify software tasks. No external API dependencies required.

## Architecture

```
EXPLORE → PLAN → BUILD → TEST → ✅ (or retry)
```

| Agent | Role | Model |
|---|---|---|
| **Initializer** | Set up project, create feature list | 7B (lightweight) |
| **Explorer** | Read codebase, gather context | 7B (secondary node) |
| **Planner** | Create implementation plan + DoD criteria | 32B (primary) |
| **Builder** | Write code, run commands via tool execution | 32B (primary) |
| **Verifier** | Run verification commands against DoD | Direct execution |

### Key Design Decisions

- **Direct Ollama API** — No CLI wrappers. HTTP calls to `/api/chat` with tool definitions.
- **Fallback tool parsing** — When models embed tool calls as JSON in text content (common with Qwen via Ollama), the runner extracts and executes them automatically.
- **Direct verification** — Test phase runs verification commands directly without LLM involvement for reliability.
- **Auto-backup** — Working directory is snapshotted before each build phase.
- **Safety rails** — Protected files/directories cannot be deleted by agents.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally
- `requests` Python package
- At least one code-capable model pulled in Ollama

## Quick Start

```bash
# Pull models
ollama pull qwen2.5-coder:32b
ollama pull qwen2.5-coder:7b

# Install dependency
pip install requests

# Run tests
python3 test_standalone_integration.py

# Run a task
mkdir ~/my-project
python3 standalone_main.py "Create a hello world function" --working-dir ~/my-project -v
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_PRIMARY_URL` | `http://127.0.0.1:11434` | Primary Ollama endpoint (for 32B + 7B models) |
| `OLLAMA_SECONDARY_URL` | `http://127.0.0.1:11434` | Secondary Ollama endpoint (for offloaded agents) |

### Config File Override

```bash
python3 standalone_main.py "task" --config my_config.json
```

```json
{
  "max_iterations": 10,
  "agents": {
    "build": {
      "model": "qwen2.5-coder:32b",
      "endpoint": "http://my-gpu-server:11434",
      "timeout": 900
    }
  }
}
```

## CLI Usage

```bash
# New task
python3 standalone_main.py "Build a REST API for user auth" -v

# Resume interrupted task
python3 standalone_main.py --resume

# Custom working directory
python3 standalone_main.py "Fix the bug" --working-dir ~/project

# Limit iterations
python3 standalone_main.py "Refactor module" --max-iterations 5
```

## File Structure

```
standalone_main.py          # CLI entry point
standalone_orchestrator.py  # Main execution loop
standalone_agents.py        # Agent runner + tool execution + LLM client
standalone_config.py        # Model/agent configuration
standalone_models.py        # Data models (TaskState, DoD, etc.)
standalone_session.py       # Session persistence + progress tracking
prompts/                    # Agent system prompts
  initializer.txt
  explore.txt
  plan.txt
  build.txt
  test.txt
```

## How It Works

1. **Orchestrator** receives a task and runs the EXPLORE → PLAN → BUILD → TEST loop
2. **AgentRunner** sends prompts to Ollama with tool definitions via HTTP
3. Models respond with tool calls (or JSON-in-text that gets parsed)
4. **ToolExecutor** runs the tools locally: `write_file`, `read_file`, `run_command`, `edit_file`, `list_directory`
5. Results feed back to the model for the next step
6. **Direct verification** runs DoD commands and checks exit codes
7. On failure, the orchestrator performs root cause analysis and retries

## Restoring from Backup

If a build agent damages your project:

```python
python3 -c "
from standalone_orchestrator import Orchestrator
from standalone_config import load_config
from pathlib import Path
o = Orchestrator(load_config(), Path('./my-project'))
o.restore_backup()
"
```

## License

MIT
