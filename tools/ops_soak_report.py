from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.config import AppConfig, load_config
from bot.ops_runtime import (
    ERROR_BACKUP_STALE,
    ERROR_DB_WRITE_FAIL,
    ERROR_HEARTBEAT_STALE,
    ERROR_SERVICE_DOWN,
    run_ops_healthcheck,
)

CRITICAL_ERROR_CODES = {
    ERROR_SERVICE_DOWN,
    ERROR_HEARTBEAT_STALE,
    ERROR_DB_WRITE_FAIL,
    ERROR_BACKUP_STALE,
}


def _now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _normalize_codes(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        code = str(item).strip().upper()
        if not code or code in seen:
            continue
        seen.add(code)
        normalized.append(code)
    return normalized


def _extract_codes_from_watchdog_issues(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    codes: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        code = text.split(":", 1)[0].strip().upper()
        if not code or code in seen:
            continue
        seen.add(code)
        codes.append(code)
    return codes


def build_soak_report(
    *,
    root: Path,
    config: AppConfig,
    since_hours: float,
    now_epoch: int | None = None,
) -> dict[str, Any]:
    now_epoch = int(now_epoch if now_epoch is not None else _now_epoch())
    since_seconds = int(max(1.0, float(since_hours)) * 3600.0)

    health = run_ops_healthcheck(root=root, config=config, now_epoch=now_epoch)
    health_error_codes = _normalize_codes(health.get("error_codes", []))
    health_critical_codes = [code for code in health_error_codes if code in CRITICAL_ERROR_CODES]

    runtime_dir = root / "runtime"
    watchdog_path = runtime_dir / "watchdog.json"
    heartbeat_path = runtime_dir / "heartbeat.json"

    watchdog_payload = _read_json(watchdog_path)
    watchdog_exists = watchdog_payload is not None
    watchdog_age_seconds = (
        max(0, now_epoch - int(watchdog_path.stat().st_mtime)) if watchdog_path.exists() else None
    )
    watchdog_in_window = bool(watchdog_age_seconds is not None and watchdog_age_seconds <= since_seconds)

    watchdog_codes: list[str] = []
    watchdog_issue_codes: list[str] = []
    if watchdog_payload is not None and watchdog_in_window:
        watchdog_codes = _normalize_codes(watchdog_payload.get("healthcheck_error_codes", []))
        watchdog_issue_codes = _extract_codes_from_watchdog_issues(watchdog_payload.get("issues", []))

    critical_from_watchdog = sorted(
        {
            code
            for code in (*watchdog_codes, *watchdog_issue_codes)
            if code in CRITICAL_ERROR_CODES
        }
    )

    critical_issues = sorted(
        {*(f"healthcheck:{code}" for code in health_critical_codes), *(f"watchdog:{code}" for code in critical_from_watchdog)}
    )

    heartbeat_age_seconds = None
    if heartbeat_path.exists():
        heartbeat_age_seconds = max(0, now_epoch - int(heartbeat_path.stat().st_mtime))

    backups = health.get("backups", {})
    if not isinstance(backups, dict):
        backups = {}
    services = health.get("services", {})
    if not isinstance(services, dict):
        services = {}

    report: dict[str, Any] = {
        "status": "ok" if not critical_issues else "fail",
        "timestamp_utc": _iso_now(),
        "root": str(root),
        "since_hours": float(since_hours),
        "since_seconds": since_seconds,
        "critical_error_codes": sorted(CRITICAL_ERROR_CODES),
        "critical_issues": critical_issues,
        "healthcheck_status": str(health.get("status", "unknown")),
        "healthcheck_error_codes": health_error_codes,
        "service_states": services,
        "heartbeat_age_seconds": heartbeat_age_seconds,
        "last_backup_age_seconds": backups.get("latest_age_seconds"),
        "latest_backup_dir": backups.get("latest_dir"),
        "watchdog_exists": watchdog_exists,
        "watchdog_in_window": watchdog_in_window,
        "watchdog_age_seconds": watchdog_age_seconds,
        "watchdog_status": (watchdog_payload or {}).get("status"),
        "watchdog_healthcheck_error_codes": watchdog_codes,
        "watchdog_issue_codes": watchdog_issue_codes,
        "watchdog_issues": (watchdog_payload or {}).get("issues", []),
    }
    return report


def _format_text(report: dict[str, Any]) -> str:
    lines = [
        f"status={report.get('status')} since_hours={report.get('since_hours')}",
        f"critical_issues={report.get('critical_issues')}",
        f"healthcheck_error_codes={report.get('healthcheck_error_codes')}",
        f"service_states={report.get('service_states')}",
        f"heartbeat_age_seconds={report.get('heartbeat_age_seconds')}",
        f"last_backup_age_seconds={report.get('last_backup_age_seconds')}",
        (
            "watchdog="
            f"exists:{report.get('watchdog_exists')} "
            f"in_window:{report.get('watchdog_in_window')} "
            f"age_seconds:{report.get('watchdog_age_seconds')} "
            f"status:{report.get('watchdog_status')}"
        ),
    ]
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate daily cloud-paper soak report from ops artifacts.")
    parser.add_argument("--root", default=".", help="Project/runtime root path.")
    parser.add_argument("--config", default="config.yaml", help="Config path relative to root.")
    parser.add_argument("--since-hours", type=float, default=24.0, help="Lookback window in hours.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.root).resolve()
    config = load_config(root / args.config)
    report = build_soak_report(
        root=root,
        config=config,
        since_hours=float(args.since_hours),
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print(_format_text(report))
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
