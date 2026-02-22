from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_closeout(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_repo_root() / "tools" / "ops_soak_closeout.py"),
            "--root",
            str(root),
            "--since-days",
            "7",
            "--end-date",
            "2026-03-01",
            *extra,
        ],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )


def _seed_ok_window(root: Path) -> None:
    soak_dir = root / "runtime" / "soak_reports"
    backups_dir = root / "backups"
    soak_dir.mkdir(parents=True, exist_ok=True)
    backups_dir.mkdir(parents=True, exist_ok=True)
    end_day = date(2026, 3, 1)
    for offset in range(7):
        day = end_day - timedelta(days=offset)
        day_iso = day.isoformat()
        (soak_dir / f"{day_iso}.json").write_text(
            json.dumps({"status": "ok", "critical_issues": [], "healthcheck_error_codes": []}),
            encoding="utf-8",
        )
        stamp = day.strftime("%Y%m%d") + "-060500"
        backup_day_dir = backups_dir / stamp
        backup_day_dir.mkdir(parents=True, exist_ok=True)
        (backup_day_dir / "manifest.json").write_text(
            json.dumps({"files": [], "integrity_check": {}, "integrity_ok": True}),
            encoding="utf-8",
        )


def test_ops_soak_closeout_go_when_all_criteria_pass(tmp_path: Path) -> None:
    _seed_ok_window(tmp_path)
    proc = _run_closeout(
        tmp_path,
        "--verify-only-drill-done",
        "--full-restore-drill-done",
        "--json",
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["decision"] == "GO"
    assert payload["criteria"]["daily_reports_clean"] is True
    assert payload["criteria"]["backup_integrity_daily"] is True
    assert payload["criteria"]["verify_only_drill_done"] is True
    assert payload["criteria"]["full_restore_drill_done"] is True


def test_ops_soak_closeout_fails_when_report_day_missing(tmp_path: Path) -> None:
    _seed_ok_window(tmp_path)
    (tmp_path / "runtime" / "soak_reports" / "2026-02-26.json").unlink()
    proc = _run_closeout(
        tmp_path,
        "--verify-only-drill-done",
        "--full-restore-drill-done",
        "--json",
    )
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["decision"] == "NO-GO"
    assert "2026-02-26" in payload["missing_report_days"]


def test_ops_soak_closeout_fails_when_backup_integrity_bad(tmp_path: Path) -> None:
    _seed_ok_window(tmp_path)
    bad_manifest = tmp_path / "backups" / "20260227-060500" / "manifest.json"
    bad_manifest.write_text(
        json.dumps({"files": [], "integrity_check": {}, "integrity_ok": False}),
        encoding="utf-8",
    )
    proc = _run_closeout(
        tmp_path,
        "--verify-only-drill-done",
        "--full-restore-drill-done",
        "--json",
    )
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["decision"] == "NO-GO"
    assert "2026-02-27" in payload["backup_integrity_fail_days"]
