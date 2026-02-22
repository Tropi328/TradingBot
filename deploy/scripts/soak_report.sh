#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/opt/trading-bot}"
CONFIG_PATH="${CONFIG_PATH:-config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
SOAK_SINCE_HOURS="${SOAK_SINCE_HOURS:-24}"
SOAK_RETENTION_DAYS="${SOAK_RETENTION_DAYS:-30}"
SOAK_REPORTS_DIR="${SOAK_REPORTS_DIR:-$ROOT_DIR/runtime/soak_reports}"
BACKUP_LOG="${BACKUP_LOG:-$ROOT_DIR/logs/backup.log}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

mkdir -p "$SOAK_REPORTS_DIR" "$(dirname "$BACKUP_LOG")"

ts_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
day_utc="$(date -u +%F)"
out_file="${SOAK_REPORTS_DIR}/${day_utc}.json"
tmp_file="${out_file}.tmp"

set +e
"$PYTHON_BIN" "$ROOT_DIR/tools/ops_soak_report.py" \
  --root "$ROOT_DIR" \
  --config "$CONFIG_PATH" \
  --since-hours "$SOAK_SINCE_HOURS" \
  --json > "$tmp_file"
rc=$?
set -e

mv "$tmp_file" "$out_file"

# Keep rolling retention window for soak report artifacts.
find "$SOAK_REPORTS_DIR" -type f -name '*.json' -mtime +"$SOAK_RETENTION_DAYS" -delete || true

if [[ "$rc" -ne 0 ]]; then
  printf '%s SOAK_FAIL report=%s rc=%s\n' "$ts_utc" "$out_file" "$rc" >> "$BACKUP_LOG"

  token="${ALERT_TELEGRAM_BOT_TOKEN:-}"
  chat_id="${ALERT_TELEGRAM_CHAT_ID:-}"
  if [[ -n "$token" && -n "$chat_id" ]]; then
    msg="SOAK_FAIL on $(hostname): rc=${rc}, report=${out_file}"
    curl -fsS --max-time 10 \
      -X POST "https://api.telegram.org/bot${token}/sendMessage" \
      -d "chat_id=${chat_id}" \
      --data-urlencode "text=${msg}" \
      >/dev/null || true
  fi
fi

exit "$rc"
