from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from bot.config import AppConfig
from bot.ops_runtime import run_watchdog


def _create_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path), timeout=2.0) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.commit()


def _prepare_root(root: Path) -> None:
    (root / "runtime").mkdir(parents=True, exist_ok=True)
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


def test_alert_dedup_and_recovery(tmp_path: Path) -> None:
    _prepare_root(tmp_path)
    config = AppConfig(ops={"alert_cooldown_seconds": 300})
    notifications: list[str] = []

    service_state = {"bot-paper-fx.service": "inactive", "bot-paper-btc.service": "active"}

    def _service(name: str) -> str:
        return service_state.get(name, "active")

    run1 = run_watchdog(
        root=tmp_path,
        config=config,
        now_epoch=1000,
        service_state_fn=_service,
        notifier=notifications.append,
    )
    assert run1["status"] == "alert"
    assert len(run1["notifications"]) == 1
    assert run1["service_states"]["bot-paper-fx.service"] == "inactive"
    assert "E_SERVICE_DOWN" in run1["healthcheck_error_codes"]
    assert isinstance(run1["last_backup_age_seconds"], (int, float))

    run2 = run_watchdog(
        root=tmp_path,
        config=config,
        now_epoch=1100,
        service_state_fn=_service,
        notifier=notifications.append,
    )
    assert run2["status"] == "alert"
    assert run2["notifications"] == []

    service_state["bot-paper-fx.service"] = "active"
    run3 = run_watchdog(
        root=tmp_path,
        config=config,
        now_epoch=1200,
        service_state_fn=_service,
        notifier=notifications.append,
    )
    assert run3["status"] == "ok"
    assert len(run3["notifications"]) == 1
    assert "recovery" in run3["notifications"][0].lower()
    assert run3["service_states"]["bot-paper-fx.service"] == "active"


def test_dual_db_isolation(tmp_path: Path) -> None:
    db_fx = tmp_path / "bot_state_paper_fx.db"
    db_btc = tmp_path / "bot_state_paper_btc.db"
    _create_db(db_fx)
    _create_db(db_btc)

    conn_fx = sqlite3.connect(str(db_fx), timeout=2.0)
    conn_btc = sqlite3.connect(str(db_btc), timeout=2.0)
    try:
        conn_fx.execute("BEGIN IMMEDIATE")
        conn_btc.execute("BEGIN IMMEDIATE")
        conn_fx.execute("ROLLBACK")
        conn_btc.execute("ROLLBACK")
    finally:
        conn_fx.close()
        conn_btc.close()
