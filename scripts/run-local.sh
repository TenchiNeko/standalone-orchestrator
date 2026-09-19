#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK="${1:-$ROOT/examples/task.json}"
MODE="${2:-off}"
SESSION="orchestrator-v2"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "already running: $SESSION"; exit 0; fi
mkdir -p "$ROOT/.orchestrator-v2"
tmux new-session -d -s "$SESSION" -n run "cd '$ROOT' && QWEN_API_KEY_FILE=/home/brandon/.config/opencode/secrets/orca-q6.key '$ROOT/.venv/bin/python' -m orchestrator_v2.cli run --task '$TASK' --jev '$MODE' > '$ROOT/.orchestrator-v2/last-run.json' 2>&1"
echo "started $SESSION; inspect with: tmux attach -t $SESSION"
