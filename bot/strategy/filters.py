from __future__ import annotations

from bot.data.spreads import median_spread


def spread_is_ok(
    current_spread: float | None,
    spread_history: list[float],
    *,
    window: int,
    max_multiple_of_median: float,
) -> bool:
    if current_spread is None:
        return False
    median = median_spread(spread_history, window)
    if median <= 0:
        return True
    return current_spread <= (max_multiple_of_median * median)

