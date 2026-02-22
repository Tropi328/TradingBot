#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/opt/trading-bot}"
CONFIG_PATH="${CONFIG_PATH:-config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

exec "$PYTHON_BIN" "$ROOT_DIR/tools/ops_watchdog.py" --root "$ROOT_DIR" --config "$CONFIG_PATH" --json
