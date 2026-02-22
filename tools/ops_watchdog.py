from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bot.config import load_config
from bot.ops_runtime import run_watchdog


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one watchdog cycle and emit runtime/watchdog.json.")
    parser.add_argument("--root", default=".", help="Project root (default: current directory).")
    parser.add_argument("--config", default="config.yaml", help="Config path relative to root.")
    parser.add_argument("--json", action="store_true", help="Print JSON payload.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    config = load_config(root / args.config)
    payload = run_watchdog(root=root, config=config)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        print(f"status={payload.get('status')} issues={payload.get('issues')} notifications={payload.get('notifications')}")
    sys.exit(0 if payload.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
