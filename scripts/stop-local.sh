#!/usr/bin/env bash
set -euo pipefail
tmux kill-session -t orchestrator-v2 2>/dev/null || true
