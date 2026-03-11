from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import requests

from bot.config import AppConfig, load_config

LOGGER = logging.getLogger(__name__)

ERROR_SERVICE_DOWN = "E_SERVICE_DOWN"
ERROR_HEARTBEAT_STALE = "E_HEARTBEAT_STALE"
ERROR_DB_WRITE_FAIL = "E_DB_WRITE_FAIL"
ERROR_BACKUP_STALE = "E_BACKUP_STALE"


def _utc_now_epoch() -> int:
    return int(datetime.now(UTC).timestamp())


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _runtime_dir(root: Path) -> Path:
    return root / "runtime"


def _state_dir(root: Path) -> Path:
    return root / "state"


def _backups_dir(root: Path) -> Path:
    return root / "backups"


def _logs_dir(root: Path) -> Path:
    return root / "logs"


def expected_state_db_paths(root: Path) -> list[Path]:
    base = _state_dir(root)
    return [
        base / "bot_state_paper_fx.db",
        base / "bot_state_paper_btc.db",
    ]


def _read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def _append_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _service_state_systemctl(service_name: str) -> str:
    if shutil.which("systemctl") is None:
        return "unavailable"
    try:
        proc = subprocess.run(
            ["systemctl", "is-active", service_name],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return "unknown"
    state = (proc.stdout or proc.stderr or "").strip().lower()
    if proc.returncode == 0:
        return state or "active"
    return state or "inactive"


def _latest_backup_dir(backups_dir: Path) -> Path | None:
    if not backups_dir.exists():
        return None
    candidates = [item for item in backups_dir.iterdir() if item.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _begin_immediate_writable(db_path: Path) -> tuple[bool, str | None]:
    try:
        conn = sqlite3.connect(str(db_path), timeout=2.0)
    except Exception as exc:
        return False, str(exc)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("ROLLBACK")
    except Exception as exc:
        return False, str(exc)
    finally:
        conn.close()
    return True, None


def run_ops_healthcheck(
    *,
    root: Path,
    config: AppConfig,
    now_epoch: int | None = None,
    service_state_fn: Callable[[str], str] | None = None,
) -> dict[str, object]:
    now_epoch = int(now_epoch if now_epoch is not None else _utc_now_epoch())
    runtime_dir = _runtime_dir(root)
    state_dir = _state_dir(root)
    backups_dir = _backups_dir(root)
    service_state_fn = service_state_fn or _service_state_systemctl

    errors: list[dict[str, object]] = []
    services_payload: dict[str, str] = {}
    heartbeat_payload: dict[str, object] = {
        "path": str(runtime_dir / "heartbeat.json"),
        "exists": False,
        "age_seconds": None,
        "status": None,
        "fail_count": None,
    }
    db_payload: dict[str, dict[str, object]] = {}
    backups_payload: dict[str, object] = {
        "path": str(backups_dir),
        "latest_dir": None,
        "latest_age_seconds": None,
        "manifest_exists": False,
        "manifest_path": None,
        "manifest_integrity_ok": None,
    }

    for service_name in config.ops.required_services:
        state = str(service_state_fn(service_name)).strip().lower() or "unknown"
        services_payload[service_name] = state
        if state != "active":
            errors.append(
                {
                    "code": ERROR_SERVICE_DOWN,
                    "message": f"Service {service_name} is not active ({state})",
                    "service": service_name,
                    "state": state,
                }
            )

    heartbeat_path = runtime_dir / "heartbeat.json"
    if heartbeat_path.exists():
        heartbeat_payload["exists"] = True
        age_seconds = max(0, now_epoch - int(heartbeat_path.stat().st_mtime))
        heartbeat_payload["age_seconds"] = age_seconds
        heartbeat_data = _read_json(heartbeat_path, {})
        if isinstance(heartbeat_data, dict):
            heartbeat_payload["status"] = heartbeat_data.get("status")
            heartbeat_payload["fail_count"] = heartbeat_data.get("fail_count")
        if age_seconds > int(config.ops.heartbeat_stale_seconds):
            errors.append(
                {
                    "code": ERROR_HEARTBEAT_STALE,
                    "message": f"Heartbeat is stale ({age_seconds}s)",
                    "path": str(heartbeat_path),
                    "age_seconds": age_seconds,
                }
            )
    else:
        errors.append(
            {
                "code": ERROR_HEARTBEAT_STALE,
                "message": "Heartbeat file missing",
                "path": str(heartbeat_path),
                "age_seconds": None,
            }
        )

    state_dir.mkdir(parents=True, exist_ok=True)
    for db_path in expected_state_db_paths(root):
        db_key = db_path.name
        exists = db_path.exists()
        db_item: dict[str, object] = {
            "path": str(db_path),
            "exists": exists,
            "writable": False,
            "error": None,
        }
        if not exists:
            db_item["error"] = "missing"
            errors.append(
                {
                    "code": ERROR_DB_WRITE_FAIL,
                    "message": f"State DB missing: {db_path.name}",
                    "db": str(db_path),
                }
            )
            db_payload[db_key] = db_item
            continue
        writable, write_error = _begin_immediate_writable(db_path)
        db_item["writable"] = writable
        db_item["error"] = write_error
        if not writable:
            errors.append(
                {
                    "code": ERROR_DB_WRITE_FAIL,
                    "message": f"Cannot acquire write transaction for {db_path.name}",
                    "db": str(db_path),
                    "error": write_error,
                }
            )
        db_payload[db_key] = db_item

    latest_backup = _latest_backup_dir(backups_dir)
    if latest_backup is not None:
        age_seconds = max(0, now_epoch - int(latest_backup.stat().st_mtime))
        backups_payload["latest_dir"] = str(latest_backup)
        backups_payload["latest_age_seconds"] = age_seconds
        manifest_path = latest_backup / "manifest.json"
        backups_payload["manifest_path"] = str(manifest_path)
        backups_payload["manifest_exists"] = manifest_path.exists()
        if not manifest_path.exists():
            errors.append(
                {
                    "code": ERROR_BACKUP_STALE,
                    "message": "Latest backup missing manifest.json",
                    "latest_dir": str(latest_backup),
                    "manifest_path": str(manifest_path),
                }
            )
        else:
            manifest_payload = _read_json(manifest_path, None)
            if not isinstance(manifest_payload, dict):
                errors.append(
                    {
                        "code": ERROR_BACKUP_STALE,
                        "message": "Latest backup manifest is invalid JSON",
                        "manifest_path": str(manifest_path),
                    }
                )
            else:
                raw_integrity_ok = manifest_payload.get("integrity_ok")
                integrity_ok = bool(raw_integrity_ok is True)
                backups_payload["manifest_integrity_ok"] = integrity_ok
                if not integrity_ok:
                    errors.append(
                        {
                            "code": ERROR_BACKUP_STALE,
                            "message": "Latest backup manifest integrity_ok is not true",
                            "manifest_path": str(manifest_path),
                            "integrity_ok": raw_integrity_ok,
                        }
                    )
        if age_seconds > 24 * 3600:
            errors.append(
                {
                    "code": ERROR_BACKUP_STALE,
                    "message": f"Latest backup is stale ({age_seconds}s)",
                    "latest_dir": str(latest_backup),
                    "age_seconds": age_seconds,
                }
            )
    else:
        errors.append(
            {
                "code": ERROR_BACKUP_STALE,
                "message": "No backup directory found",
                "backups_dir": str(backups_dir),
            }
        )

    error_codes = sorted({str(item.get("code")) for item in errors})
    return {
        "status": "ok" if not errors else "fail",
        "timestamp_utc": _utc_now_iso(),
        "error_codes": error_codes,
        "errors": errors,
        "services": services_payload,
        "heartbeat": heartbeat_payload,
        "db_checks": db_payload,
        "backups": backups_payload,
    }


def run_deploy_preflight(
    *,
    root: Path,
    config: AppConfig,
    service_state_fn: Callable[[str], str] | None = None,
) -> dict[str, object]:
    service_state_fn = service_state_fn or _service_state_systemctl
    checks: list[dict[str, object]] = []

    for required_dir in (_state_dir(root), _logs_dir(root), _runtime_dir(root), _backups_dir(root)):
        checks.append(
            {
                "name": f"dir:{required_dir}",
                "ok": required_dir.exists() and required_dir.is_dir(),
                "details": "exists" if required_dir.exists() else "missing",
            }
        )

    env_paths = [
        root / "deploy" / "env" / "paper_fx.env",
        root / "deploy" / "env" / "paper_btc.env",
    ]
    for env_path in env_paths:
        ok = env_path.exists()
        details = "exists"
        if not ok:
            details = "missing"
        elif os.name == "posix":
            mode = env_path.stat().st_mode & 0o777
            if mode != 0o600:
                ok = False
                details = f"permissions {oct(mode)} expected 0o600"
        checks.append({"name": f"env:{env_path}", "ok": ok, "details": details})

    script_paths = [
        root / "deploy" / "scripts" / "heartbeat.sh",
        root / "deploy" / "scripts" / "watchdog.sh",
        root / "deploy" / "scripts" / "soak_report.sh",
        root / "deploy" / "scripts" / "soak_closeout.sh",
        root / "deploy" / "scripts" / "backup_state.sh",
        root / "deploy" / "scripts" / "restore_state.sh",
    ]
    for script_path in script_paths:
        ok = script_path.exists()
        details = "exists"
        if not ok:
            details = "missing"
        elif os.name == "posix" and not os.access(script_path, os.X_OK):
            ok = False
            details = "not executable"
        checks.append({"name": f"script:{script_path}", "ok": ok, "details": details})

    sqlite3_available = shutil.which("sqlite3") is not None
    checks.append(
        {
            "name": "binary:sqlite3",
            "ok": sqlite3_available,
            "details": "found" if sqlite3_available else "missing",
        }
    )

    for service_name in config.ops.required_services:
        state = str(service_state_fn(service_name)).strip().lower() or "unknown"
        checks.append(
            {
                "name": f"service:{service_name}",
                "ok": state == "active",
                "details": state,
            }
        )

    failed = [item for item in checks if not bool(item.get("ok"))]
    return {
        "status": "ok" if not failed else "fail",
        "timestamp_utc": _utc_now_iso(),
        "checks": checks,
        "failed_count": len(failed),
    }


def build_backup_manifest(
    *,
    backup_dir: Path,
    created_at_utc: str | None = None,
    config_path: Path | None = None,
    active_variant_path: Path | None = None,
    integrity_results: dict[str, str] | None = None,
) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for path in sorted(backup_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "manifest.json":
            continue
        rel = str(path.relative_to(backup_dir)).replace("\\", "/")
        files.append(
            {
                "path": rel,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    integrity_results = integrity_results or {}
    integrity_ok = all(str(value).strip().lower() == "ok" for value in integrity_results.values())
    return {
        "created_at_utc": created_at_utc or _utc_now_iso(),
        "backup_dir": str(backup_dir),
        "config_path": str(config_path) if config_path is not None else None,
        "active_variant_path": str(active_variant_path) if active_variant_path is not None else None,
        "files": files,
        "integrity_check": integrity_results,
        "integrity_ok": integrity_ok,
    }


def write_backup_manifest(
    *,
    backup_dir: Path,
    config_path: Path | None = None,
    active_variant_path: Path | None = None,
    integrity_results: dict[str, str] | None = None,
) -> Path:
    payload = build_backup_manifest(
        backup_dir=backup_dir,
        config_path=config_path,
        active_variant_path=active_variant_path,
        integrity_results=integrity_results,
    )
    manifest_path = backup_dir / "manifest.json"
    _write_json(manifest_path, payload)
    return manifest_path


def parse_backup_manifest(backup_dir: Path) -> dict[str, object] | None:
    manifest_path = backup_dir / "manifest.json"
    payload = _read_json(manifest_path, None)
    if not isinstance(payload, dict):
        return None
    return payload


def verify_backup_manifest(backup_dir: Path) -> tuple[bool, list[str], dict[str, object] | None]:
    payload = parse_backup_manifest(backup_dir)
    if payload is None:
        return False, ["manifest_missing_or_invalid"], None
    errors: list[str] = []
    files = payload.get("files", [])
    if not isinstance(files, list):
        return False, ["manifest_files_invalid"], payload
    for entry in files:
        if not isinstance(entry, dict):
            errors.append("manifest_entry_invalid")
            continue
        rel = str(entry.get("path", "")).strip()
        expected_sha = str(entry.get("sha256", "")).strip().lower()
        if not rel or not expected_sha:
            errors.append("manifest_entry_missing_fields")
            continue
        file_path = backup_dir / rel
        if not file_path.exists():
            errors.append(f"missing:{rel}")
            continue
        actual_sha = _sha256(file_path)
        if actual_sha.lower() != expected_sha:
            errors.append(f"checksum:{rel}")

    integrity = payload.get("integrity_check", {})
    if isinstance(integrity, dict):
        for db_name, result in integrity.items():
            if str(result).strip().lower() != "ok":
                errors.append(f"integrity:{db_name}:{result}")
    else:
        errors.append("integrity_section_invalid")

    return len(errors) == 0, errors, payload


def _mask_env_line(line: str) -> str:
    raw = line.rstrip("\n")
    if "=" not in raw:
        return raw
    key, value = raw.split("=", 1)
    key_upper = key.strip().upper()
    sensitive = ("TOKEN", "PASSWORD", "SECRET", "API_KEY", "PRIVATE_KEY", "CHAT_ID")
    if any(fragment in key_upper for fragment in sensitive):
        return f"{key}=***REDACTED***"
    return f"{key}={value}"


def run_backup_now(
    *,
    root: Path,
    config: AppConfig,
    retention_days: int | None = None,
    verify_on_create: bool | None = None,
    active_variant_path: Path | None = None,
) -> dict[str, object]:
    retention_days = int(retention_days if retention_days is not None else config.ops.backup_retention_days)
    verify_on_create = bool(
        verify_on_create if verify_on_create is not None else config.ops.backup_verify_on_create
    )

    backups_dir = _backups_dir(root)
    backups_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup_dir = backups_dir / timestamp
    backup_dir.mkdir(parents=True, exist_ok=False)

    integrity_results: dict[str, str] = {}
    copied_dbs: list[str] = []
    for source_db in expected_state_db_paths(root):
        if not source_db.exists():
            continue
        target_db = backup_dir / source_db.name
        with sqlite3.connect(str(source_db), timeout=5.0) as src_conn:
            with sqlite3.connect(str(target_db), timeout=5.0) as dst_conn:
                src_conn.backup(dst_conn)
        copied_dbs.append(source_db.name)
        if verify_on_create:
            with sqlite3.connect(str(target_db), timeout=5.0) as conn:
                row = conn.execute("PRAGMA integrity_check;").fetchone()
            status = str(row[0]).strip().lower() if row else "unknown"
            integrity_results[source_db.name] = status
        else:
            integrity_results[source_db.name] = "skipped"

    config_path = root / "config.yaml"
    if config_path.exists():
        shutil.copy2(config_path, backup_dir / "config.yaml")

    chosen_variant = (
        active_variant_path
        if active_variant_path is not None
        else Path(
            os.getenv(
                "ACTIVE_CONFIG_PATH",
                str(root / "configs" / "variants" / "config.variant_SAFE_BASE.yaml"),
            )
        )
    )
    if chosen_variant.exists():
        shutil.copy2(chosen_variant, backup_dir / "active_variant.yaml")

    env_target = backup_dir / "env"
    env_target.mkdir(parents=True, exist_ok=True)
    env_files = [
        root / "deploy" / "env" / "paper_fx.env",
        root / "deploy" / "env" / "paper_btc.env",
    ]
    for env_file in env_files:
        if not env_file.exists():
            continue
        masked_lines = [_mask_env_line(line) for line in env_file.read_text(encoding="utf-8").splitlines()]
        (env_target / f"{env_file.name}.masked").write_text(
            "\n".join(masked_lines) + "\n",
            encoding="utf-8",
        )

    manifest_path = write_backup_manifest(
        backup_dir=backup_dir,
        config_path=config_path if config_path.exists() else None,
        active_variant_path=chosen_variant if chosen_variant.exists() else None,
        integrity_results=integrity_results,
    )
    manifest_ok, manifest_errors, _ = verify_backup_manifest(backup_dir)

    if retention_days >= 0:
        now_epoch = _utc_now_epoch()
        for item in backups_dir.iterdir():
            if not item.is_dir():
                continue
            if item == backup_dir:
                continue
            age_days = (now_epoch - int(item.stat().st_mtime)) / 86400.0
            if age_days > retention_days:
                shutil.rmtree(item, ignore_errors=True)

    log_path = _logs_dir(root) / "backup.log"
    _append_line(
        log_path,
        f"{_utc_now_iso()} backup_dir={backup_dir} copied_dbs={','.join(copied_dbs) or '-'} "
        f"manifest_ok={manifest_ok} verify_on_create={verify_on_create}",
    )

    status = "ok"
    errors: list[str] = []
    if verify_on_create:
        for db_name, result in integrity_results.items():
            if str(result).strip().lower() != "ok":
                status = "fail"
                errors.append(f"integrity:{db_name}:{result}")
    if not manifest_ok:
        status = "fail"
        errors.extend(manifest_errors)

    result = {
        "status": status,
        "backup_dir": str(backup_dir),
        "manifest_path": str(manifest_path),
        "copied_dbs": copied_dbs,
        "integrity_results": integrity_results,
        "errors": errors,
    }
    if status != "ok":
        _append_line(log_path, f"{_utc_now_iso()} backup_status=fail errors={'|'.join(errors)}")
    return result


def run_restore_verify(*, backup_dir: Path) -> dict[str, object]:
    ok_manifest, manifest_errors, _ = verify_backup_manifest(backup_dir)
    errors = list(manifest_errors)

    for db_path in sorted(backup_dir.glob("*.db")):
        try:
            with sqlite3.connect(str(db_path), timeout=5.0) as conn:
                row = conn.execute("PRAGMA integrity_check;").fetchone()
            status = str(row[0]).strip().lower() if row else "unknown"
            if status != "ok":
                errors.append(f"integrity:{db_path.name}:{status}")
        except Exception as exc:
            errors.append(f"integrity:{db_path.name}:{exc}")

    return {
        "status": "ok" if ok_manifest and not errors else "fail",
        "backup_dir": str(backup_dir),
        "errors": errors,
    }


def run_restore_apply(
    *,
    root: Path,
    backup_dir: Path,
    services: list[str],
    service_state_fn: Callable[[str], str] | None = None,
) -> dict[str, object]:
    verify = run_restore_verify(backup_dir=backup_dir)
    if verify["status"] != "ok":
        return verify

    if not _state_dir(root).exists():
        _state_dir(root).mkdir(parents=True, exist_ok=True)

    if shutil.which("systemctl") is not None:
        for service_name in services:
            subprocess.run(["systemctl", "stop", service_name], check=False)

    restored: list[str] = []
    for db_path in sorted(backup_dir.glob("*.db")):
        target = _state_dir(root) / db_path.name
        shutil.copy2(db_path, target)
        restored.append(target.name)

    if shutil.which("systemctl") is not None:
        for service_name in services:
            subprocess.run(["systemctl", "start", service_name], check=False)

    _append_line(
        _logs_dir(root) / "backup.log",
        f"{_utc_now_iso()} restore_apply backup_dir={backup_dir} restored={','.join(restored) or '-'}",
    )
    return {"status": "ok", "backup_dir": str(backup_dir), "restored": restored, "errors": []}


def _send_telegram_message(text: str) -> None:
    token = str(os.getenv("ALERT_TELEGRAM_BOT_TOKEN", "")).strip()
    chat_id = str(os.getenv("ALERT_TELEGRAM_CHAT_ID", "")).strip()
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10, verify=True)
    except Exception as exc:
        # Log without leaking the token value
        LOGGER.warning("Telegram message failed (token redacted): %s", type(exc).__name__)


def run_watchdog(
    *,
    root: Path,
    config: AppConfig,
    now_epoch: int | None = None,
    service_state_fn: Callable[[str], str] | None = None,
    notifier: Callable[[str], None] | None = None,
) -> dict[str, object]:
    now_epoch = int(now_epoch if now_epoch is not None else _utc_now_epoch())
    notifier = notifier or _send_telegram_message
    runtime = _runtime_dir(root)
    runtime.mkdir(parents=True, exist_ok=True)

    health = run_ops_healthcheck(
        root=root,
        config=config,
        now_epoch=now_epoch,
        service_state_fn=service_state_fn,
    )
    issues: list[str] = []
    for error in health.get("errors", []):
        if not isinstance(error, dict):
            continue
        code = str(error.get("code", "")).strip().upper()
        if code not in {ERROR_SERVICE_DOWN, ERROR_HEARTBEAT_STALE}:
            continue
        issues.append(f"{code}:{error.get('message', '')}")
    issues = sorted(set(issues))

    state_path = runtime / "watchdog.json"
    global_alert_path = runtime / "watchdog_last_global_alert_epoch"
    issue_alert_path = runtime / "watchdog_issue_alert_state.json"
    previous_issues_path = runtime / "watchdog_previous_issues.json"

    previous_issues = _read_json(previous_issues_path, [])
    if not isinstance(previous_issues, list):
        previous_issues = []
    previous_set = {str(item) for item in previous_issues}
    current_set = set(issues)

    issue_alert_state = _read_json(issue_alert_path, {})
    if not isinstance(issue_alert_state, dict):
        issue_alert_state = {}
    issue_alert_state = {str(k): int(v) for k, v in issue_alert_state.items() if str(v).isdigit()}

    try:
        global_last = int(global_alert_path.read_text(encoding="utf-8").strip() if global_alert_path.exists() else "0")
    except Exception:
        global_last = 0

    notifications: list[str] = []
    cooldown = int(config.ops.alert_cooldown_seconds)
    if issues:
        if (now_epoch - global_last) >= cooldown:
            to_alert: list[str] = []
            for issue in issues:
                issue_last = int(issue_alert_state.get(issue, 0))
                if (now_epoch - issue_last) >= cooldown:
                    to_alert.append(issue)
            if to_alert:
                message = f"bot-watchdog alert on {os.uname().nodename if hasattr(os, 'uname') else 'host'}: {'; '.join(to_alert)}"
                notifier(message)
                notifications.append(message)
                global_last = now_epoch
                for issue in to_alert:
                    issue_alert_state[issue] = now_epoch

    recovered = sorted(previous_set - current_set)
    if recovered:
        message = (
            f"bot-watchdog recovery on {os.uname().nodename if hasattr(os, 'uname') else 'host'}: "
            f"resolved {'; '.join(recovered)}"
        )
        notifier(message)
        notifications.append(message)

    heartbeat_age = health.get("heartbeat", {}).get("age_seconds")
    services = health.get("services", {})
    if not isinstance(services, dict):
        services = {}
    healthcheck_error_codes = health.get("error_codes", [])
    if not isinstance(healthcheck_error_codes, list):
        healthcheck_error_codes = []
    backups = health.get("backups", {})
    if not isinstance(backups, dict):
        backups = {}
    last_backup_age_seconds = backups.get("latest_age_seconds")

    payload = {
        "timestamp_utc": _utc_now_iso(),
        "status": "ok" if not issues else "alert",
        "issues": issues,
        "issue_fingerprints": issues,
        "recovered_issues": recovered,
        "watchdog_interval_seconds": int(config.ops.watchdog_interval_seconds),
        "service_states": services,
        "healthcheck_error_codes": healthcheck_error_codes,
        "last_backup_age_seconds": last_backup_age_seconds,
        "heartbeat_age": heartbeat_age,
        "heartbeat_age_seconds": heartbeat_age,
        "notifications": notifications,
    }
    _write_json(state_path, payload)
    _write_json(previous_issues_path, issues)
    _write_json(issue_alert_path, issue_alert_state)
    global_alert_path.write_text(str(global_last), encoding="utf-8")
    return payload


def _resolve_root(path: str | None) -> Path:
    if path:
        return Path(path).resolve()
    return Path.cwd().resolve()


def _ops_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ops runtime helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    p_health = sub.add_parser("healthcheck")
    p_health.add_argument("--root", default=None)
    p_health.add_argument("--config", default="config.yaml")
    p_health.add_argument("--json", action="store_true")

    p_preflight = sub.add_parser("preflight")
    p_preflight.add_argument("--root", default=None)
    p_preflight.add_argument("--config", default="config.yaml")
    p_preflight.add_argument("--json", action="store_true")

    p_backup = sub.add_parser("backup-now")
    p_backup.add_argument("--root", default=None)
    p_backup.add_argument("--config", default="config.yaml")
    p_backup.add_argument("--retention-days", type=int, default=None)
    p_backup.add_argument("--verify-on-create", choices=["true", "false"], default=None)

    p_restore_verify = sub.add_parser("restore-verify")
    p_restore_verify.add_argument("backup_dir")

    p_restore_apply = sub.add_parser("restore-apply")
    p_restore_apply.add_argument("backup_dir")
    p_restore_apply.add_argument("--root", default=None)
    p_restore_apply.add_argument("--config", default="config.yaml")

    p_watchdog = sub.add_parser("watchdog")
    p_watchdog.add_argument("--root", default=None)
    p_watchdog.add_argument("--config", default="config.yaml")

    return parser


def _print_payload(payload: dict[str, object], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return
    print(f"status={payload.get('status')}")
    for key in ("error_codes", "errors", "issues", "notifications"):
        value = payload.get(key)
        if value:
            print(f"{key}={value}")


def main() -> None:
    args = _ops_parser().parse_args()
    if args.command in {"healthcheck", "preflight", "backup-now", "restore-apply", "watchdog"}:
        root = _resolve_root(getattr(args, "root", None))
    else:
        root = Path.cwd().resolve()

    if args.command == "healthcheck":
        config = load_config(root / args.config)
        payload = run_ops_healthcheck(root=root, config=config)
        _print_payload(payload, bool(args.json))
        sys.exit(0 if payload["status"] == "ok" else 1)
    if args.command == "preflight":
        config = load_config(root / args.config)
        payload = run_deploy_preflight(root=root, config=config)
        _print_payload(payload, bool(args.json))
        sys.exit(0 if payload["status"] == "ok" else 1)
    if args.command == "backup-now":
        config = load_config(root / args.config)
        verify_override = None
        if args.verify_on_create is not None:
            verify_override = str(args.verify_on_create).strip().lower() == "true"
        payload = run_backup_now(
            root=root,
            config=config,
            retention_days=args.retention_days,
            verify_on_create=verify_override,
        )
        _print_payload(payload, True)
        sys.exit(0 if payload["status"] == "ok" else 1)
    if args.command == "restore-verify":
        payload = run_restore_verify(backup_dir=Path(args.backup_dir).resolve())
        _print_payload(payload, True)
        sys.exit(0 if payload["status"] == "ok" else 1)
    if args.command == "restore-apply":
        config = load_config(root / args.config)
        payload = run_restore_apply(
            root=root,
            backup_dir=Path(args.backup_dir).resolve(),
            services=list(config.ops.required_services),
        )
        _print_payload(payload, True)
        sys.exit(0 if payload["status"] == "ok" else 1)
    if args.command == "watchdog":
        config = load_config(root / args.config)
        payload = run_watchdog(root=root, config=config)
        _print_payload(payload, True)
        sys.exit(0 if payload["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
