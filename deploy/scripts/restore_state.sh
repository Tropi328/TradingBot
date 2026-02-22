#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [--verify-only] <backup_dir>"
  exit 1
fi

VERIFY_ONLY="false"
BACKUP_DIR=""
for arg in "$@"; do
  if [[ "$arg" == "--verify-only" ]]; then
    VERIFY_ONLY="true"
    continue
  fi
  BACKUP_DIR="$arg"
done

if [[ -z "$BACKUP_DIR" ]]; then
  echo "Missing backup_dir argument."
  exit 1
fi

ROOT_DIR="${ROOT_DIR:-/opt/trading-bot}"
CONFIG_PATH="${CONFIG_PATH:-config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

if [[ "$VERIFY_ONLY" == "true" ]]; then
  exec "$PYTHON_BIN" -m bot.ops_runtime restore-verify "$BACKUP_DIR"
fi

exec "$PYTHON_BIN" -m bot.ops_runtime restore-apply "$BACKUP_DIR" --root "$ROOT_DIR" --config "$CONFIG_PATH"
