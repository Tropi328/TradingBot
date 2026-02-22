from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from bot.config import AppConfig
from bot.ops_runtime import build_backup_manifest, run_deploy_preflight


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_deploy_required_files_exist() -> None:
    root = _repo_root()
    required = [
        root / "deploy" / "systemd" / "bot-paper-fx.service",
        root / "deploy" / "systemd" / "bot-paper-btc.service",
        root / "deploy" / "systemd" / "bot-heartbeat.service",
        root / "deploy" / "systemd" / "bot-heartbeat.timer",
        root / "deploy" / "systemd" / "bot-soak-report.service",
        root / "deploy" / "systemd" / "bot-soak-report.timer",
        root / "deploy" / "scripts" / "heartbeat.sh",
        root / "deploy" / "scripts" / "soak_report.sh",
        root / "deploy" / "scripts" / "backup_state.sh",
        root / "deploy" / "scripts" / "restore_state.sh",
        root / "deploy" / "env" / "paper_fx.env.example",
        root / "deploy" / "env" / "paper_btc.env.example",
    ]
    missing = [str(path) for path in required if not path.exists()]
    assert missing == []


def test_env_template_syntax() -> None:
    root = _repo_root()
    for env_file in (
        root / "deploy" / "env" / "paper_fx.env.example",
        root / "deploy" / "env" / "paper_btc.env.example",
    ):
        lines = env_file.read_text(encoding="utf-8").splitlines()
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            assert "=" in stripped
            key, _value = stripped.split("=", 1)
            assert key.strip() != ""


def test_manifest_builder_smoke(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")
    manifest = build_backup_manifest(backup_dir=tmp_path, integrity_results={"bot_state_paper_fx.db": "ok"})
    assert isinstance(manifest, dict)
    assert "files" in manifest
    assert manifest["integrity_ok"] is True


def test_preflight_health_logic_smoke(tmp_path: Path) -> None:
    for rel in ("state", "logs", "runtime", "backups", "deploy/env", "deploy/scripts"):
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)
    (tmp_path / "deploy" / "env" / "paper_fx.env").write_text("A=1\n", encoding="utf-8")
    (tmp_path / "deploy" / "env" / "paper_btc.env").write_text("A=1\n", encoding="utf-8")
    (tmp_path / "deploy" / "scripts" / "heartbeat.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (tmp_path / "deploy" / "scripts" / "watchdog.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (tmp_path / "deploy" / "scripts" / "backup_state.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (tmp_path / "deploy" / "scripts" / "restore_state.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    payload = run_deploy_preflight(
        root=tmp_path,
        config=AppConfig(),
        service_state_fn=lambda _name: "active",
    )
    assert payload["status"] in {"ok", "fail"}
    assert "checks" in payload


def test_ops_soak_report_smoke_format_and_exit_code(tmp_path: Path) -> None:
    for rel in ("state", "runtime", "backups/latest"):
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.yaml").write_text("ops:\n  required_services: []\n", encoding="utf-8")
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
    for db_name in ("bot_state_paper_fx.db", "bot_state_paper_btc.db"):
        with sqlite3.connect(str(tmp_path / "state" / db_name), timeout=2.0) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT)")
            conn.commit()

    proc = subprocess.run(
        [
            sys.executable,
            str(_repo_root() / "tools" / "ops_soak_report.py"),
            "--root",
            str(tmp_path),
            "--config",
            "config.yaml",
            "--since-hours",
            "24",
        ],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "status=ok" in proc.stdout
