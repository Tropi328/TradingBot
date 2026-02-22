from __future__ import annotations

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
        root / "deploy" / "scripts" / "heartbeat.sh",
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
