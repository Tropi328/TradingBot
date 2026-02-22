from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _create_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path), timeout=2.0) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.commit()


def _write_min_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ops:",
                "  required_services: []",
                "  heartbeat_stale_seconds: 900",
                "  watchdog_interval_seconds: 60",
                "  alert_cooldown_seconds: 300",
                "  backup_retention_days: 14",
                "  backup_verify_on_create: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_soak(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_repo_root() / "tools" / "ops_soak_report.py"),
            "--root",
            str(root),
            "--config",
            "config.yaml",
            *extra,
        ],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_ops_soak_report_ok_exit_zero(tmp_path: Path) -> None:
    _write_min_config(tmp_path / "config.yaml")
    (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)
    (tmp_path / "backups" / "latest").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runtime" / "heartbeat.json").write_text(
        json.dumps({"timestamp_utc": "2026-02-22T00:00:00Z", "status": "ok", "fail_count": 0}),
        encoding="utf-8",
    )
    (tmp_path / "runtime" / "watchdog.json").write_text(
        json.dumps({"status": "ok", "issues": [], "healthcheck_error_codes": []}),
        encoding="utf-8",
    )
    (tmp_path / "backups" / "latest" / "manifest.json").write_text(
        json.dumps({"files": [], "integrity_check": {}, "integrity_ok": True}),
        encoding="utf-8",
    )
    _create_db(tmp_path / "state" / "bot_state_paper_fx.db")
    _create_db(tmp_path / "state" / "bot_state_paper_btc.db")

    proc = _run_soak(tmp_path, "--since-hours", "24")
    assert proc.returncode == 0
    assert "status=ok" in proc.stdout


def test_ops_soak_report_critical_code_exit_one(tmp_path: Path) -> None:
    _write_min_config(tmp_path / "config.yaml")
    _create_db(tmp_path / "state" / "bot_state_paper_fx.db")
    _create_db(tmp_path / "state" / "bot_state_paper_btc.db")

    proc = _run_soak(tmp_path, "--since-hours", "24")
    assert proc.returncode == 1
    assert "E_HEARTBEAT_STALE" in proc.stdout


def test_ops_soak_report_missing_heartbeat_backup_reason(tmp_path: Path) -> None:
    _write_min_config(tmp_path / "config.yaml")
    _create_db(tmp_path / "state" / "bot_state_paper_fx.db")
    _create_db(tmp_path / "state" / "bot_state_paper_btc.db")

    proc = _run_soak(tmp_path, "--since-hours", "24", "--json")
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["status"] == "fail"
    assert "healthcheck:E_HEARTBEAT_STALE" in payload["critical_issues"]
    assert "healthcheck:E_BACKUP_STALE" in payload["critical_issues"]
