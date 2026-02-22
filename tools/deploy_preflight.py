from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bot.config import load_config
from bot.ops_runtime import run_deploy_preflight


def _format_text(payload: dict[str, object]) -> str:
    lines = [f"status={payload.get('status')} failed_count={payload.get('failed_count')}"]
    checks = payload.get("checks", [])
    if isinstance(checks, list):
        for entry in checks:
            if not isinstance(entry, dict):
                continue
            marker = "OK" if bool(entry.get("ok")) else "FAIL"
            lines.append(f"[{marker}] {entry.get('name')} -> {entry.get('details')}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy preflight checker (Oracle paper setup).")
    parser.add_argument("--root", default=".", help="Project root (default: current directory).")
    parser.add_argument("--config", default="config.yaml", help="Config path relative to root.")
    parser.add_argument("--json", action="store_true", help="Print JSON result.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    config = load_config(root / args.config)
    payload = run_deploy_preflight(root=root, config=config)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        print(_format_text(payload))
    sys.exit(0 if payload.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
