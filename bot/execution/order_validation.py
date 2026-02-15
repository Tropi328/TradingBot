from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from bot.config import BacktestTuningConfig, RiskConfig


@dataclass(slots=True)
class RiskCashPlan:
    base_risk_cash: float
    target_risk_cash: float
    max_risk_cash: float


def compute_risk_cash_plan(
    *,
    risk: RiskConfig,
    equity: float,
    effective_risk_per_trade: float,
) -> RiskCashPlan:
    eq = max(0.0, float(equity))
    percent_risk_cash = eq * max(0.0, float(effective_risk_per_trade))
    mode = str(risk.risk_mode or "percent").strip().lower()
    if mode == "cash" and risk.max_risk_cash_per_trade is not None:
        base_risk_cash = max(0.0, float(risk.max_risk_cash_per_trade))
    else:
        base_risk_cash = percent_risk_cash

    min_risk_cash = max(0.0, float(risk.min_risk_cash_per_trade))
    target_risk_cash = max(base_risk_cash, min_risk_cash)
    max_risk_cash_cfg = risk.max_risk_cash_per_trade
    max_risk_cash = float(max_risk_cash_cfg) if max_risk_cash_cfg is not None else max(base_risk_cash, target_risk_cash)
    max_risk_cash = max(0.0, max_risk_cash)
    return RiskCashPlan(
        base_risk_cash=base_risk_cash,
        target_risk_cash=target_risk_cash,
        max_risk_cash=max_risk_cash,
    )


def price_to_points(price_delta: float, *, point_size: float) -> float:
    step = float(point_size)
    if step <= 0:
        return float(price_delta)
    return float(price_delta) / step


def expected_move_too_small(
    *,
    expected_move_points: float,
    spread_points: float,
    min_edge_to_cost_ratio: float,
) -> bool:
    expected_move_points = max(0.0, float(expected_move_points))
    estimated_cost_points = max(0.0, float(spread_points))
    required = float(min_edge_to_cost_ratio) * estimated_cost_points
    return expected_move_points < required


def minutes_to_next_rollover(ts: datetime, *, hour: int, minute: int) -> float:
    ts_utc = ts.astimezone(timezone.utc)
    rollover = ts_utc.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if ts_utc >= rollover:
        rollover += timedelta(days=1)
    return max(0.0, (rollover - ts_utc).total_seconds() / 60.0)


def in_rollover_entry_block_window(
    *,
    ts: datetime,
    swap_hour: int,
    swap_minute: int,
    cfg: BacktestTuningConfig,
) -> bool:
    before = max(0, int(cfg.rollover_block_minutes_before))
    after = max(0, int(cfg.rollover_block_minutes_after))
    if before <= 0 and after <= 0:
        return False

    ts_utc = ts.astimezone(timezone.utc)
    next_roll = ts_utc.replace(hour=swap_hour, minute=swap_minute, second=0, microsecond=0)
    if ts_utc >= next_roll:
        next_roll += timedelta(days=1)
    prev_roll = next_roll - timedelta(days=1)

    mins_before = (next_roll - ts_utc).total_seconds() / 60.0
    mins_after = (ts_utc - prev_roll).total_seconds() / 60.0
    return (0.0 <= mins_before <= float(before)) or (0.0 <= mins_after <= float(after))
