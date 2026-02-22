from __future__ import annotations

from bot.reporting.metrics import compute_metrics


def test_anomaly_flags_trigger_for_extreme_profile() -> None:
    trades = [
        {"pnl": 1000.0, "pnl_net": 1000.0, "spread_points": 6.0},
        {"pnl": -1.0, "pnl_net": -1.0, "spread_points": 6.0},
        {"pnl": 500.0, "pnl_net": 500.0, "spread_points": 6.0},
    ]
    metrics = compute_metrics(trades, [], initial_equity=10000.0)
    flags = set(metrics.get("anomaly_flags", []))

    assert "PF_EXTREME" in flags
    assert "PAYOFF_EXTREME" in flags
    assert "LOSS_TINY_VS_SPREAD" in flags


def test_anomaly_flags_empty_for_normal_profile() -> None:
    trades = [
        {"pnl": 20.0, "pnl_net": 20.0, "spread_points": 2.0},
        {"pnl": -10.0, "pnl_net": -10.0, "spread_points": 2.0},
        {"pnl": 12.0, "pnl_net": 12.0, "spread_points": 2.0},
        {"pnl": -9.0, "pnl_net": -9.0, "spread_points": 2.0},
    ]
    metrics = compute_metrics(trades, [], initial_equity=10000.0)
    assert metrics.get("anomaly_flags", []) == []
