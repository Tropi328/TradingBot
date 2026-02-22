from __future__ import annotations

import pytest

from bot.reporting.metrics import compute_metrics


def test_drawdown_peak_initial_and_alias_are_consistent() -> None:
    equity = [
        {"ts": "2024-01-01T00:00:00+00:00", "equity": 10000.0},
        {"ts": "2024-01-01T00:05:00+00:00", "equity": 12000.0},
        {"ts": "2024-01-01T00:10:00+00:00", "equity": 9000.0},
    ]
    metrics = compute_metrics([], equity, initial_equity=10000.0)

    assert metrics["max_drawdown"] == pytest.approx(3000.0)
    assert metrics["max_drawdown_pct_peak"] == pytest.approx(25.0)
    assert metrics["max_drawdown_pct_initial"] == pytest.approx(30.0)
    assert metrics["max_drawdown_pct"] == pytest.approx(metrics["max_drawdown_pct_peak"])
