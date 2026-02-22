from __future__ import annotations

import sqlite3
from pathlib import Path

from bot.config import AppConfig
from bot.ops_runtime import run_backup_now, run_restore_verify, verify_backup_manifest


def _create_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path), timeout=2.0) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute("INSERT INTO t(v) VALUES ('x')")
        conn.commit()


def test_backup_manifest_and_integrity(tmp_path: Path) -> None:
    _create_db(tmp_path / "state" / "bot_state_paper_fx.db")
    _create_db(tmp_path / "state" / "bot_state_paper_btc.db")
    (tmp_path / "config.yaml").write_text("timezone: Europe/Warsaw\n", encoding="utf-8")

    config = AppConfig(ops={"backup_verify_on_create": True, "backup_retention_days": 14})
    payload = run_backup_now(root=tmp_path, config=config)
    assert payload["status"] == "ok"

    backup_dir = Path(str(payload["backup_dir"]))
    ok, errors, manifest = verify_backup_manifest(backup_dir)
    assert ok is True
    assert errors == []
    assert manifest is not None
    integrity = manifest.get("integrity_check", {})
    assert isinstance(integrity, dict)
    assert integrity.get("bot_state_paper_fx.db") == "ok"
    assert integrity.get("bot_state_paper_btc.db") == "ok"


def test_restore_verify_only(tmp_path: Path) -> None:
    _create_db(tmp_path / "state" / "bot_state_paper_fx.db")
    _create_db(tmp_path / "state" / "bot_state_paper_btc.db")
    config = AppConfig(ops={"backup_verify_on_create": True})
    backup = run_backup_now(root=tmp_path, config=config)
    backup_dir = Path(str(backup["backup_dir"]))

    result = run_restore_verify(backup_dir=backup_dir)
    assert result["status"] == "ok"
    assert result["errors"] == []
