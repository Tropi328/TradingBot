from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_date(raw: str) -> date:
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{raw}', expected YYYY-MM-DD") from exc


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


def _date_window(end_day: date, since_days: int) -> list[date]:
    total = max(1, int(since_days))
    start_day = end_day - timedelta(days=total - 1)
    return [start_day + timedelta(days=idx) for idx in range(total)]


def _backup_day_from_dir(backup_dir: Path) -> str:
    name = backup_dir.name
    if len(name) >= 8 and name[:8].isdigit():
        return f"{name[:4]}-{name[4:6]}-{name[6:8]}"
    mtime = datetime.fromtimestamp(backup_dir.stat().st_mtime, tz=timezone.utc)
    return mtime.strftime("%Y-%m-%d")


def build_closeout_report(
    *,
    root: Path,
    since_days: int,
    end_day: date,
    verify_only_drill_done: bool,
    full_restore_drill_done: bool,
) -> dict[str, Any]:
    window_days = _date_window(end_day=end_day, since_days=since_days)
    window_days_iso = [day.isoformat() for day in window_days]

    soak_dir = root / "runtime" / "soak_reports"
    missing_report_days: list[str] = []
    critical_report_days: list[dict[str, Any]] = []
    critical_codes_counter: Counter[str] = Counter()

    for day_iso in window_days_iso:
        report_path = soak_dir / f"{day_iso}.json"
        payload = _read_json(report_path)
        if payload is None:
            missing_report_days.append(day_iso)
            continue
        status = str(payload.get("status", "")).strip().lower()
        critical_issues = payload.get("critical_issues", [])
        if not isinstance(critical_issues, list):
            critical_issues = []
        if status != "ok" or critical_issues:
            cleaned_issues: list[str] = []
            for raw_issue in critical_issues:
                issue = str(raw_issue).strip()
                if not issue:
                    continue
                cleaned_issues.append(issue)
                code = issue.split(":", 1)[-1].strip().upper() if ":" in issue else issue.upper()
                if code:
                    critical_codes_counter[code] += 1
            critical_report_days.append(
                {
                    "day_utc": day_iso,
                    "status": status or "unknown",
                    "critical_issues": cleaned_issues,
                    "report_path": str(report_path),
                }
            )

    backups_dir = root / "backups"
    backups_by_day: dict[str, list[dict[str, Any]]] = {}
    if backups_dir.exists():
        for item in sorted(backups_dir.iterdir()):
            if not item.is_dir():
                continue
            manifest_path = item / "manifest.json"
            manifest = _read_json(manifest_path)
            integrity_ok = bool(isinstance(manifest, dict) and manifest.get("integrity_ok") is True)
            day_iso = _backup_day_from_dir(item)
            backups_by_day.setdefault(day_iso, []).append(
                {
                    "backup_dir": str(item),
                    "manifest_exists": manifest_path.exists(),
                    "manifest_path": str(manifest_path),
                    "manifest_integrity_ok": integrity_ok,
                }
            )

    backup_integrity_fail_days: list[str] = []
    for day_iso in window_days_iso:
        day_backups = backups_by_day.get(day_iso, [])
        if not day_backups:
            backup_integrity_fail_days.append(day_iso)
            continue
        if not any(item.get("manifest_integrity_ok") is True for item in day_backups):
            backup_integrity_fail_days.append(day_iso)

    criteria = {
        "daily_reports_present": len(missing_report_days) == 0,
        "daily_reports_clean": len(critical_report_days) == 0,
        "backup_integrity_daily": len(backup_integrity_fail_days) == 0,
        "verify_only_drill_done": bool(verify_only_drill_done),
        "full_restore_drill_done": bool(full_restore_drill_done),
    }
    go = all(criteria.values())

    return {
        "timestamp_utc": _utc_now().isoformat(),
        "window_start_utc": window_days_iso[0],
        "window_end_utc": window_days_iso[-1],
        "expected_days": len(window_days_iso),
        "reports_found_days": len(window_days_iso) - len(missing_report_days),
        "missing_report_days": missing_report_days,
        "critical_report_days": critical_report_days,
        "critical_codes_counter": dict(sorted(critical_codes_counter.items())),
        "backup_integrity_fail_days": backup_integrity_fail_days,
        "criteria": criteria,
        "decision": "GO" if go else "NO-GO",
        "notes": "GO requires all criteria=true for the 7-day cloud-paper soak window.",
    }


def _format_text(payload: dict[str, Any]) -> str:
    criteria = payload.get("criteria", {})
    return "\n".join(
        [
            f"decision={payload.get('decision')}",
            f"window={payload.get('window_start_utc')}..{payload.get('window_end_utc')}",
            f"missing_report_days={payload.get('missing_report_days')}",
            f"backup_integrity_fail_days={payload.get('backup_integrity_fail_days')}",
            f"critical_codes_counter={payload.get('critical_codes_counter')}",
            f"criteria={criteria}",
        ]
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate final GO/NO-GO closeout from daily soak artifacts.")
    parser.add_argument("--root", default=".", help="Project/runtime root path.")
    parser.add_argument("--since-days", type=int, default=7, help="Window length in days.")
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        default=None,
        help="Window end date in UTC (YYYY-MM-DD). Default: today UTC.",
    )
    parser.add_argument(
        "--verify-only-drill-done",
        action="store_true",
        help="Mark verify-only restore drill as completed.",
    )
    parser.add_argument(
        "--full-restore-drill-done",
        action="store_true",
        help="Mark full restore drill (on test copy) as completed.",
    )
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.root).resolve()
    end_day = args.end_date or _utc_now().date()
    payload = build_closeout_report(
        root=root,
        since_days=int(args.since_days),
        end_day=end_day,
        verify_only_drill_done=bool(args.verify_only_drill_done),
        full_restore_drill_done=bool(args.full_restore_drill_done),
    )
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        print(_format_text(payload))
    return 0 if payload.get("decision") == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
