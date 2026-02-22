from __future__ import annotations

import sys

import main as main_module
from bot.config import AppConfig


def _parse_args(monkeypatch, *argv: str):
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    return main_module.parse_args()


def test_dashboard_is_disabled_by_default(monkeypatch) -> None:
    args = _parse_args(monkeypatch, "--backtest")
    assert args.dashboard is False
    assert args.hold_viewers is False


def test_dashboard_forces_decision_trace(monkeypatch) -> None:
    args = _parse_args(monkeypatch, "--backtest", "--dashboard")
    config = AppConfig()
    config.diagnostics.decision_trace_enabled = False

    main_module._apply_cli_overrides(args, config)
    assert config.diagnostics.decision_trace_enabled is True


def test_backtest_does_not_auto_enable_trace_by_default(monkeypatch) -> None:
    args = _parse_args(monkeypatch, "--backtest")
    config = AppConfig()
    config.diagnostics.decision_trace_enabled = False
    config.diagnostics.decision_trace_auto_enable_backtest = False

    main_module._apply_cli_overrides(args, config)
    assert config.diagnostics.decision_trace_enabled is False


def test_backtest_can_auto_enable_trace_via_config(monkeypatch) -> None:
    args = _parse_args(monkeypatch, "--backtest")
    config = AppConfig()
    config.diagnostics.decision_trace_enabled = False
    config.diagnostics.decision_trace_auto_enable_backtest = True

    main_module._apply_cli_overrides(args, config)
    assert config.diagnostics.decision_trace_enabled is True


def test_parse_research_optimize_flags(monkeypatch) -> None:
    args = _parse_args(
        monkeypatch,
        "--research-optimize",
        "--backtest-start",
        "2024-01-01",
        "--backtest-end",
        "2025-02-01",
        "--research-runtime-budget",
        "deep",
        "--research-benchmark-symbols",
        "XAUUSD",
    )
    assert args.research_optimize is True
    assert args.research_runtime_budget == "deep"
    assert args.research_benchmark_symbols == "XAUUSD"


def test_parse_ops_runtime_flags(monkeypatch) -> None:
    args = _parse_args(
        monkeypatch,
        "--ops-healthcheck",
        "--ops-backup-now",
        "--ops-restore-verify",
        "backups/20260222-000000",
    )
    assert args.ops_healthcheck is True
    assert args.ops_backup_now is True
    assert args.ops_restore_verify == "backups/20260222-000000"
