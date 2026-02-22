#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/opt/trading-bot}"
CONFIG_PATH="${CONFIG_PATH:-config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

ARGS=(
  -m bot.ops_runtime
  backup-now
  --root "$ROOT_DIR"
  --config "$CONFIG_PATH"
)

if [[ -n "${RETENTION_DAYS:-}" ]]; then
  ARGS+=(--retention-days "$RETENTION_DAYS")
fi
if [[ -n "${BACKUP_VERIFY_ON_CREATE:-}" ]]; then
  ARGS+=(--verify-on-create "$BACKUP_VERIFY_ON_CREATE")
fi

exec "$PYTHON_BIN" "${ARGS[@]}"
