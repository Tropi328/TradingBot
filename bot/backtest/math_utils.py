"""Small math/utility helpers — extracted from engine.py."""

from __future__ import annotations

import math

from bot.strategy.contracts import DecisionAction
from bot.strategy.decision_core import clamp_value as _clamp_core


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    qq = min(1.0, max(0.0, float(q)))
    pos = qq * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    w = pos - lo
    return ordered[lo] + ((ordered[hi] - ordered[lo]) * w)


def _spread_point_stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    vals = [float(v) for v in values]
    avg = sum(vals) / len(vals)
    med = _quantile(vals, 0.5)
    p90 = _quantile(vals, 0.9)
    return avg, med, p90


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return _clamp_core(value, min_value, max_value)


def _quantile_from_sorted(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    q_norm = _clamp(float(q), 0.0, 1.0)
    pos = q_norm * float(len(values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    frac = pos - float(lo)
    return float(values[lo]) + (float(values[hi]) - float(values[lo])) * frac


def _action_priority(action: DecisionAction) -> int:
    if action == DecisionAction.TRADE:
        return 3
    if action == DecisionAction.SMALL:
        return 2
    if action == DecisionAction.MANAGE:
        return 1
    return 0
