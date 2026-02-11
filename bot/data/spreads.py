from __future__ import annotations

import statistics


def compute_spread(bid: float, ask: float) -> float:
    return max(0.0, ask - bid)


def median_spread(spreads: list[float], window: int) -> float:
    if not spreads:
        return 0.0
    trimmed = spreads[-window:] if len(spreads) > window else spreads
    return float(statistics.median(trimmed))

