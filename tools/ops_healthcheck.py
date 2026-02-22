from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bot.config import load_config
from bot.ops_runtime import run_ops_healthcheck


def _format_text(payload: dict[str, object]) -> str:
    lines = [f"status={payload.get('status')} error_codes={payload.get('error_codes')}"]
    heartbeat = payload.get("heartbeat", {})
    if isinstance(heartbeat, dict):
        lines.append(
            "heartbeat="
            f"exists:{heartbeat.get('exists')} "
            f"age_seconds:{heartbeat.get('age_seconds')} "
            f"status:{heartbeat.get('status')} "
            f"fail_count:{heartbeat.get('fail_count')}"
        )
    services = payload.get("services", {})
    if isinstance(services, dict):
        for name, state in sorted(services.items()):
            lines.append(f"service[{name}]={state}")
    errors = payload.get("errors", [])
    if isinstance(errors, list):
        for entry in errors:
            if not isinstance(entry, dict):
                continue
            lines.append(f"error[{entry.get('code')}] {entry.get('message')}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ops healthcheck (services, heartbeat, DB write, backup freshness).")
    parser.add_argument("--root", default=".", help="Project root (default: current directory).")
    parser.add_argument("--config", default="config.yaml", help="Config path relative to root.")
    parser.add_argument("--json", action="store_true", help="Print JSON result.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    config = load_config(root / args.config)
    payload = run_ops_healthcheck(root=root, config=config)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        print(_format_text(payload))
    sys.exit(0 if payload.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
