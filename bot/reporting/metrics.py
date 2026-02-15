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


def compute_metrics(trades: Sequence[Mapping[str, Any]], equity: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pnl_values = [_as_float(trade.get("pnl")) for trade in trades]
    spread_cost_sum = sum(_as_float(trade.get("spread_cost")) for trade in trades)
    slippage_cost_sum = sum(_as_float(trade.get("slippage_cost")) for trade in trades)
    commission_cost_sum = sum(_as_float(trade.get("commission_cost")) for trade in trades)
    swap_cost_sum = sum(_as_float(trade.get("swap_cost")) for trade in trades)
    fx_cost_sum = sum(_as_float(trade.get("fx_cost")) for trade in trades)
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

    equity_series = compute_drawdown_series(equity)
    if equity_series:
        equity_start = _as_float(equity_series[0].get("equity"))
        equity_end = _as_float(equity_series[-1].get("equity"))
        max_drawdown = max((_as_float(point.get("drawdown")) for point in equity_series), default=0.0)
        max_drawdown_pct = max((_as_float(point.get("drawdown_pct")) for point in equity_series), default=0.0)
    else:
        equity_start = 0.0
        equity_end = 0.0
        max_drawdown = 0.0
        max_drawdown_pct = 0.0

    return {
        "trades_count": trades_count,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": ((wins / trades_count) * 100.0) if trades_count else 0.0,
        "total_pnl": sum(pnl_values),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "avg_pnl": avg_pnl,
        "median_pnl": median_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "spread_cost_sum": spread_cost_sum,
        "slippage_cost_sum": slippage_cost_sum,
        "commission_cost_sum": commission_cost_sum,
        "swap_cost_sum": swap_cost_sum,
        "fx_cost_sum": fx_cost_sum,
        "largest_win": max(pnl_values, default=0.0),
        "largest_loss": min(pnl_values, default=0.0),
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "equity_start": equity_start,
        "equity_end": equity_end,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
    }
