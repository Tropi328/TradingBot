from __future__ import annotations

from bot.research.optimizer import build_time_split


def test_build_time_split_uses_70_30_with_day_boundaries() -> None:
    split = build_time_split(
        backtest_start="2024-01-01",
        backtest_end="2025-02-01",
        split_ratio_is=0.70,
        min_days_is=30,
        min_days_oos=30,
    )

    assert split.days_total > 0
    assert split.days_is + split.days_oos == split.days_total
    assert split.days_is > split.days_oos
    assert split.is_start == "2024-01-01"
    assert split.is_end < split.oos_end
    assert split.fallback_applied is False


def test_build_time_split_falls_back_when_min_days_cannot_fit() -> None:
    split = build_time_split(
        backtest_start="2024-01-01",
        backtest_end="2024-02-15",
        split_ratio_is=0.70,
        min_days_is=40,
        min_days_oos=40,
    )

    assert split.fallback_applied is True
    assert split.days_is >= 1
    assert split.days_oos >= 1
    assert split.days_is + split.days_oos == split.days_total
