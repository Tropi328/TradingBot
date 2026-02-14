from __future__ import annotations

from pathlib import Path

import pytest

from bot.reporting.backtest_reporter import BacktestMeta, BacktestReporter, BacktestRun
from bot.reporting.metrics import compute_drawdown_series, compute_metrics


def test_metrics_zero_trades_with_equity() -> None:
    equity = [
        {"ts": "2024-01-01T00:00:00+00:00", "equity": 10000.0},
        {"ts": "2024-01-01T00:05:00+00:00", "equity": 10010.0},
    ]
    metrics = compute_metrics([], equity)

    assert metrics["trades_count"] == 0
    assert metrics["wins"] == 0
    assert metrics["losses"] == 0
    assert metrics["win_rate_pct"] == 0.0
    assert metrics["total_pnl"] == 0.0
    assert metrics["equity_start"] == 10000.0
    assert metrics["equity_end"] == 10010.0
    assert metrics["max_drawdown"] == 0.0
    assert metrics["max_drawdown_pct"] == 0.0


def test_metrics_only_wins_and_only_losses_handle_zero_division() -> None:
    wins_only = [{"pnl": 10.0}, {"pnl": 20.0}]
    metrics_wins = compute_metrics(wins_only, [])
    assert metrics_wins["wins"] == 2
    assert metrics_wins["losses"] == 0
    assert metrics_wins["payoff_ratio"] == 0.0
    assert metrics_wins["profit_factor"] == 0.0

    losses_only = [{"pnl": -5.0}, {"pnl": -10.0}]
    metrics_losses = compute_metrics(losses_only, [])
    assert metrics_losses["wins"] == 0
    assert metrics_losses["losses"] == 2
    assert metrics_losses["payoff_ratio"] == 0.0
    assert metrics_losses["profit_factor"] == 0.0


def test_drawdown_zero_for_rising_equity() -> None:
    equity = [
        {"ts": "2024-01-01T00:00:00+00:00", "equity": 10000.0},
        {"ts": "2024-01-01T00:05:00+00:00", "equity": 10020.0},
        {"ts": "2024-01-01T00:10:00+00:00", "equity": 10100.0},
    ]
    series = compute_drawdown_series(equity)
    assert series
    assert all(point["drawdown"] == 0.0 for point in series)
    assert all(point["drawdown_pct"] == 0.0 for point in series)


def test_metrics_drawdown_with_equity_dip() -> None:
    equity = [
        {"ts": "2024-01-01T00:00:00+00:00", "equity": 10000.0},
        {"ts": "2024-01-01T00:05:00+00:00", "equity": 10100.0},
        {"ts": "2024-01-01T00:10:00+00:00", "equity": 9900.0},
        {"ts": "2024-01-01T00:15:00+00:00", "equity": 10200.0},
    ]
    metrics = compute_metrics([], equity)
    assert metrics["max_drawdown"] == pytest.approx(200.0)
    assert metrics["max_drawdown_pct"] == pytest.approx((200.0 / 10100.0) * 100.0)


def test_reporter_writes_json_and_csv_for_empty_trades(tmp_path: Path) -> None:
    reporter = BacktestReporter(tmp_path / "reports" / "backtest")
    run = BacktestRun(
        meta=BacktestMeta(
            symbol="XAUUSD",
            timeframe="5m",
            start="2024-01-01",
            end="2024-01-02",
            variant="W0",
            mode="backtest",
            initial_equity=10000.0,
        ),
        trades=[],
        equity=[{"ts": "2024-01-01T00:00:00+00:00", "equity": 10000.0}],
    )
    metrics = reporter.generate(run=run, formats=("json", "csv"))
    assert metrics["trades_count"] == 0

    assert reporter.last_output_dir is not None
    outdir = reporter.last_output_dir
    assert (outdir / "report.json").exists()
    assert (outdir / "summary.json").exists()
    assert (outdir / "trades.csv").exists()
    assert (outdir / "equity.csv").exists()
