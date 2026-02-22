from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(converted) or math.isinf(converted):
        return default
    return converted


def _as_optional_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(converted) or math.isinf(converted):
        return None
    return converted


def _max_consecutive(pnls: Sequence[float], *, positive: bool) -> int:
    longest = 0
    current = 0
    for pnl in pnls:
        is_match = pnl > 0 if positive else pnl < 0
        if is_match:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


def compute_drawdown_series(equity: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not equity:
        return out

    peak: float | None = None
    for idx, point in enumerate(equity):
        eq = _as_float(point.get("equity"))
        if peak is None or eq > peak:
            peak = eq
        drawdown = max(0.0, peak - eq)
        drawdown_pct = ((drawdown / peak) * 100.0) if peak > 0 else 0.0
        out.append(
            {
                "idx": idx,
                "ts": str(point.get("ts", "") or ""),
                "equity": eq,
                "drawdown": drawdown,
                "drawdown_pct": drawdown_pct,
            }
        )
    return out


def compute_anomaly_flags(metrics: Mapping[str, Any]) -> list[str]:
    flags: list[str] = []
    profit_factor_net = _as_float(metrics.get("profit_factor_net"), default=0.0)
    payoff_ratio = _as_float(metrics.get("payoff_ratio"), default=0.0)
    avg_loss = _as_float(metrics.get("avg_loss"), default=0.0)
    median_spread_points_raw = metrics.get("median_spread_points")
    median_spread_points = _as_optional_float(median_spread_points_raw)

    if profit_factor_net > 20.0:
        flags.append("PF_EXTREME")
    if payoff_ratio > 30.0:
        flags.append("PAYOFF_EXTREME")
    if median_spread_points is not None and median_spread_points > 0 and avg_loss < 0:
        if abs(avg_loss) < (0.25 * median_spread_points):
            flags.append("LOSS_TINY_VS_SPREAD")
    return flags


def compute_metrics(
    trades: Sequence[Mapping[str, Any]],
    equity: Sequence[Mapping[str, Any]],
    *,
    initial_equity: float | None = None,
) -> dict[str, Any]:
    pnl_values = [_as_float(trade.get("pnl")) for trade in trades]
    pnl_gross_values = [_as_float(trade.get("pnl_gross", trade.get("pnl"))) for trade in trades]
    pnl_net_values = [_as_float(trade.get("pnl_net", trade.get("pnl"))) for trade in trades]
    spread_cost_sum = sum(_as_float(trade.get("spread_cost")) for trade in trades)
    slippage_cost_sum = sum(_as_float(trade.get("slippage_cost")) for trade in trades)
    commission_cost_sum = sum(_as_float(trade.get("commission_cost")) for trade in trades)
    swap_cost_sum = sum(_as_float(trade.get("swap_cost")) for trade in trades)
    fx_cost_sum = sum(_as_float(trade.get("fx_cost")) for trade in trades)
    spread_points_values = [
        value
        for value in (_as_optional_float(trade.get("spread_points")) for trade in trades)
        if value is not None and value >= 0
    ]
    trades_count = len(pnl_values)
    wins = sum(1 for pnl in pnl_values if pnl > 0)
    losses = sum(1 for pnl in pnl_values if pnl < 0)
    gross_profit = sum(pnl for pnl in pnl_values if pnl > 0)
    gross_loss = sum(pnl for pnl in pnl_values if pnl < 0)

    avg_pnl = (sum(pnl_values) / trades_count) if trades_count else 0.0
    median_pnl = statistics.median(pnl_values) if pnl_values else 0.0

    win_values = [pnl for pnl in pnl_values if pnl > 0]
    loss_values = [pnl for pnl in pnl_values if pnl < 0]
    avg_win = (sum(win_values) / len(win_values)) if win_values else 0.0
    avg_loss = (sum(loss_values) / len(loss_values)) if loss_values else 0.0

    payoff_ratio = (avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else 0.0

    max_consecutive_wins = _max_consecutive(pnl_values, positive=True)
    max_consecutive_losses = _max_consecutive(pnl_values, positive=False)

    total_pnl_gross = sum(pnl_gross_values) if pnl_gross_values else 0.0
    total_pnl_net = sum(pnl_net_values) if pnl_net_values else 0.0

    wins_net = sum(1 for pnl in pnl_net_values if pnl > 0)
    losses_net = sum(1 for pnl in pnl_net_values if pnl < 0)
    gross_profit_net = sum(pnl for pnl in pnl_net_values if pnl > 0)
    gross_loss_net = sum(pnl for pnl in pnl_net_values if pnl < 0)
    profit_factor_net = (gross_profit_net / abs(gross_loss_net)) if gross_loss_net < 0 else 0.0

    entry_timestamps = []
    for trade in trades:
        raw = trade.get("entry_ts") or trade.get("entry_time")
        if raw is not None:
            entry_timestamps.append(str(raw))
    if len(entry_timestamps) >= 2:
        from datetime import datetime

        sorted_ts = sorted(entry_timestamps)
        first = _parse_timestamp(sorted_ts[0])
        last = _parse_timestamp(sorted_ts[-1])
        if first and last:
            span_days = max(1.0, (last - first).total_seconds() / 86400.0)
            trades_per_day = trades_count / span_days
        else:
            trades_per_day = 0.0
    else:
        trades_per_day = 0.0

    equity_series = compute_drawdown_series(equity)
    if equity_series:
        equity_start = _as_float(equity_series[0].get("equity"))
        equity_end = _as_float(equity_series[-1].get("equity"))
        max_drawdown = max((_as_float(point.get("drawdown")) for point in equity_series), default=0.0)
        max_drawdown_pct_peak = max((_as_float(point.get("drawdown_pct")) for point in equity_series), default=0.0)
    else:
        equity_start = 0.0
        equity_end = 0.0
        max_drawdown = 0.0
        max_drawdown_pct_peak = 0.0

    if initial_equity is not None and float(initial_equity) > 0:
        initial_base = float(initial_equity)
    elif equity_start > 0:
        initial_base = float(equity_start)
    else:
        initial_base = 0.0

    max_drawdown_pct_initial = (max_drawdown / initial_base * 100.0) if initial_base > 0 else 0.0
    median_spread_points = statistics.median(spread_points_values) if spread_points_values else None

    metrics = {
        "trades_count": trades_count,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": ((wins / trades_count) * 100.0) if trades_count else 0.0,
        "total_pnl": sum(pnl_values),
        "total_pnl_gross": total_pnl_gross,
        "total_pnl_net": total_pnl_net,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "avg_pnl": avg_pnl,
        "median_pnl": median_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "profit_factor_net": profit_factor_net,
        "wins_net": wins_net,
        "losses_net": losses_net,
        "spread_cost_sum": spread_cost_sum,
        "slippage_cost_sum": slippage_cost_sum,
        "commission_cost_sum": commission_cost_sum,
        "swap_cost_sum": swap_cost_sum,
        "fx_cost_sum": fx_cost_sum,
        "cost_breakdown_net": {
            "spread_cost_sum": spread_cost_sum,
            "slippage_cost_sum": slippage_cost_sum,
            "commission_cost_sum": commission_cost_sum,
            "swap_cost_sum": swap_cost_sum,
            "fx_cost_sum": fx_cost_sum,
        },
        "blocked_by_reason": {},
        "oos_pass": None,
        "constraint_dd_cap_pass": None,
        "objective_value": None,
        "largest_win": max(pnl_values, default=0.0),
        "largest_loss": min(pnl_values, default=0.0),
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "trades_per_day": round(trades_per_day, 4),
        "equity_start": equity_start,
        "equity_end": equity_end,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct_peak": max_drawdown_pct_peak,
        "max_drawdown_pct_initial": max_drawdown_pct_initial,
        "max_drawdown_pct": max_drawdown_pct_peak,
        "median_spread_points": median_spread_points,
    }
    metrics["anomaly_flags"] = compute_anomaly_flags(metrics)
    return metrics


def _parse_timestamp(raw: str):
    from datetime import datetime

    raw = raw.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None
