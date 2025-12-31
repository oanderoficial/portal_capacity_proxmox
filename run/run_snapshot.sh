#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$BASE/.venv/bin/python"
LOG="$BASE/logs/collector.log"
LOCK="$BASE/run/collector.lock"
mkdir -p "$BASE/logs" "$BASE/run"
cd "$BASE"
if command -v flock >/dev/null 2>&1; then
  flock -n "$LOCK" "$VENV_PY" -c "from collectors.proxmox_collector import save_snapshot; save_snapshot()" >> "$LOG" 2>&1
else
  "$VENV_PY" -c "from collectors.proxmox_collector import save_snapshot; save_snapshot()" >> "$LOG" 2>&1
fi
