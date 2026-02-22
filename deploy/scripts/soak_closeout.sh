#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/opt/trading-bot}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
SOAK_SINCE_DAYS="${SOAK_SINCE_DAYS:-7}"
SOAK_END_DATE="${SOAK_END_DATE:-$(date -u +%F)}"
VERIFY_ONLY_DRILL_DONE="${VERIFY_ONLY_DRILL_DONE:-false}"
FULL_RESTORE_DRILL_DONE="${FULL_RESTORE_DRILL_DONE:-false}"
OUT_PATH="${OUT_PATH:-$ROOT_DIR/reports/ops/soak_closeout_${SOAK_END_DATE}.json}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

args=(
  "$ROOT_DIR/tools/ops_soak_closeout.py"
  --root "$ROOT_DIR"
  --since-days "$SOAK_SINCE_DAYS"
  --end-date "$SOAK_END_DATE"
  --output "$OUT_PATH"
  --json
)

if [[ "$VERIFY_ONLY_DRILL_DONE" == "true" ]]; then
  args+=(--verify-only-drill-done)
fi
if [[ "$FULL_RESTORE_DRILL_DONE" == "true" ]]; then
  args+=(--full-restore-drill-done)
fi

exec "$PYTHON_BIN" "${args[@]}"
