"""Shared TP-profile helpers used by both live and backtest pipelines."""

from __future__ import annotations


def tp2_r_for_target_total_r(
    *,
    target_total_r: float,
    tp1_trigger_r: float,
    tp1_fraction: float,
    mode: str = "strict_tp_price",
) -> float:
    """Compute TP2 R so that with partial TP1 the total trade payoff stays on target.

    Example:
    - TP1 at 1R with 50% size, target total=2R  -> TP2 must be 3R
    - TP1 at 1R with 50% size, target total=3R  -> TP2 must be 5R
    """
    total_r = max(0.1, float(target_total_r))
    mode_norm = str(mode).strip().lower()
    if mode_norm == "strict_tp_price":
        return total_r
    frac = max(0.0, min(0.99, float(tp1_fraction)))
    trigger_r = max(0.0, float(tp1_trigger_r))
    if frac <= 0.0:
        return total_r
    tp2_r = (total_r - (frac * trigger_r)) / max(1e-9, 1.0 - frac)
    return max(total_r, tp2_r)
