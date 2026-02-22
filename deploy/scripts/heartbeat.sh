#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/opt/trading-bot}"
RUNTIME_DIR="${RUNTIME_DIR:-$ROOT_DIR/runtime}"
HEARTBEAT_FILE="${HEARTBEAT_FILE:-$RUNTIME_DIR/heartbeat.json}"
HEARTBEAT_FAIL_FILE="${HEARTBEAT_FAIL_FILE:-$RUNTIME_DIR/heartbeat_fail_count}"
CHECK_URL="${HEARTBEAT_CHECK_URL:-https://ifconfig.me}"
NETWORK_TIMEOUT_SECONDS="${NETWORK_TIMEOUT_SECONDS:-8}"

mkdir -p "$RUNTIME_DIR"

send_telegram() {
  local message="$1"
  local token="${ALERT_TELEGRAM_BOT_TOKEN:-}"
  local chat_id="${ALERT_TELEGRAM_CHAT_ID:-}"
  if [[ -z "$token" || -z "$chat_id" ]]; then
    return 0
  fi
  curl -fsS --max-time 10 \
    -X POST "https://api.telegram.org/bot${token}/sendMessage" \
    -d "chat_id=${chat_id}" \
    --data-urlencode "text=${message}" \
    >/dev/null || true
}

# Lightweight CPU activity.
if [[ -f "${ROOT_DIR}/README.md" ]]; then
  sha256sum "${ROOT_DIR}/README.md" >/dev/null 2>&1 || true
else
  printf '%s\n' "$(date -u +%s)" | sha256sum >/dev/null 2>&1 || true
fi

status="ok"
fail_count=0

if curl -fsS --max-time "${NETWORK_TIMEOUT_SECONDS}" "${CHECK_URL}" >/dev/null; then
  fail_count=0
else
  status="network_fail"
  if [[ -f "$HEARTBEAT_FAIL_FILE" ]]; then
    fail_count="$(cat "$HEARTBEAT_FAIL_FILE" 2>/dev/null || echo 0)"
  fi
  fail_count="$((fail_count + 1))"
fi

printf '%s\n' "$fail_count" > "$HEARTBEAT_FAIL_FILE"

timestamp_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
timestamp_epoch="$(date -u +%s)"
hostname="$(hostname)"
tmp_file="${HEARTBEAT_FILE}.tmp"
cat > "$tmp_file" <<EOF
{"timestamp_utc":"${timestamp_utc}","timestamp_epoch":${timestamp_epoch},"hostname":"${hostname}","status":"${status}","fail_count":${fail_count},"check_url":"${CHECK_URL}"}
EOF
mv "$tmp_file" "$HEARTBEAT_FILE"

if [[ "$status" != "ok" && "$fail_count" -eq 3 ]]; then
  send_telegram "bot-heartbeat: network check failed 3 times in a row on $(hostname)"
fi
