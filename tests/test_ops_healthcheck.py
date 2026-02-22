from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from bot.config import AppConfig
from bot.ops_runtime import (
    ERROR_HEARTBEAT_STALE,
    ERROR_SERVICE_DOWN,
    run_ops_healthcheck,
)


def _create_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path), timeout=2.0) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.commit()


def _prepare_healthy_runtime(root: Path) -> None:
    (root / "runtime").mkdir(parents=True, exist_ok=True)
    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "backups" / "latest").mkdir(parents=True, exist_ok=True)
    (root / "runtime" / "heartbeat.json").write_text(
        json.dumps({"timestamp_utc": "2026-02-22T00:00:00Z", "status": "ok", "fail_count": 0}),
        encoding="utf-8",
    )
    (root / "backups" / "latest" / "manifest.json").write_text(
        json.dumps({"files": [], "integrity_check": {}, "integrity_ok": True}),
        encoding="utf-8",
    )
    _create_db(root / "state" / "bot_state_paper_fx.db")
    _create_db(root / "state" / "bot_state_paper_btc.db")


def test_ops_healthcheck_services_down(tmp_path: Path) -> None:
    _prepare_healthy_runtime(tmp_path)
    config = AppConfig()

    def _service_state(name: str) -> str:
        if name == "bot-paper-fx.service":
            return "inactive"
        return "active"

    payload = run_ops_healthcheck(root=tmp_path, config=config, service_state_fn=_service_state, now_epoch=1700000000)
    assert payload["status"] == "fail"
    assert ERROR_SERVICE_DOWN in payload["error_codes"]


def test_ops_healthcheck_heartbeat_stale(tmp_path: Path) -> None:
    _prepare_healthy_runtime(tmp_path)
    config = AppConfig()

    payload = run_ops_healthcheck(
        root=tmp_path,
        config=config,
        service_state_fn=lambda _name: "active",
        now_epoch=9999999999,
    )
    assert payload["status"] == "fail"
    assert ERROR_HEARTBEAT_STALE in payload["error_codes"]
