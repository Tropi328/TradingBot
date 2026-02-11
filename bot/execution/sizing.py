from __future__ import annotations

import math


def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        raise ValueError("step must be > 0")
    return math.floor(value / step) * step


def position_size_from_risk(
    *,
    equity: float,
    risk_per_trade: float,
    entry_price: float,
    stop_price: float,
    min_size: float,
    size_step: float,
) -> float:
    risk_distance = abs(entry_price - stop_price)
    if risk_distance <= 0:
        return 0.0
    risk_amount = equity * risk_per_trade
    raw_size = risk_amount / risk_distance
    sized = floor_to_step(raw_size, size_step)
    if sized < min_size:
        return 0.0
    return round(sized, 8)


def r_multiple(side: str, entry_price: float, stop_price: float, current_price: float) -> float:
    risk = abs(entry_price - stop_price)
    if risk == 0:
        return 0.0
    if side == "LONG":
        return (current_price - entry_price) / risk
    return (entry_price - current_price) / risk

