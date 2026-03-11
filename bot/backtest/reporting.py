"""Trade quality metrics, report aggregation — extracted from engine.py."""

from __future__ import annotations

from collections import Counter
from typing import Any

from bot.backtest.math_utils import _action_priority, _spread_point_stats
from bot.backtest.models import BacktestReport, BacktestTrade
from bot.capital_ramp import (
    START_EQUITY_PLN,
    CapitalRampEvent,
    CapitalRampRuntime,
)
from bot.config import AppConfig, AssetConfig
from bot.strategy.contracts import StrategyOutcome


def _trade_quality_metrics(trades: list[BacktestTrade]) -> tuple[float, float, float, float]:
    if not trades:
        return 0.0, 0.0, 0.0, 0.0
    win_values = [float(trade.pnl) for trade in trades if trade.pnl > 0]
    loss_values = [abs(float(trade.pnl)) for trade in trades if trade.pnl < 0]
    avg_win = (sum(win_values) / len(win_values)) if win_values else 0.0
    avg_loss = (sum(loss_values) / len(loss_values)) if loss_values else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    gross_profit = sum(win_values)
    gross_loss = sum(loss_values)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
    return avg_win, avg_loss, payoff_ratio, profit_factor


def _trade_r_quality_metrics(trades: list[BacktestTrade]) -> tuple[float, float, float]:
    if not trades:
        return 0.0, 0.0, 0.0
    win_values = [float(trade.r_multiple) for trade in trades if trade.r_multiple > 0]
    loss_values = [abs(float(trade.r_multiple)) for trade in trades if trade.r_multiple < 0]
    avg_win_r = (sum(win_values) / len(win_values)) if win_values else 0.0
    avg_loss_r = (sum(loss_values) / len(loss_values)) if loss_values else 0.0
    payoff_r = (avg_win_r / avg_loss_r) if avg_loss_r > 0 else 0.0
    return avg_win_r, avg_loss_r, payoff_r


def _net_quality_metrics(trades: list[BacktestTrade], equity_start: float) -> tuple[float, float, float]:
    """Return (expectancy_net, profit_factor_net, max_drawdown_net) using pnl_net."""
    if not trades:
        return 0.0, 0.0, 0.0
    total_pnl_net = sum(float(t.pnl_net) for t in trades)
    expectancy_net = total_pnl_net / len(trades)
    net_wins = [float(t.pnl_net) for t in trades if t.pnl_net > 0]
    net_losses = [abs(float(t.pnl_net)) for t in trades if t.pnl_net < 0]
    gross_profit_net = sum(net_wins)
    gross_loss_net = sum(net_losses)
    profit_factor_net = (gross_profit_net / gross_loss_net) if gross_loss_net > 0 else 0.0
    equity = equity_start
    peak = equity
    max_dd_net = 0.0
    for t in trades:
        equity += float(t.pnl_net)
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd_net:
            max_dd_net = dd
    return expectancy_net, profit_factor_net, max_dd_net


def _exit_reason_distribution(trades: list[BacktestTrade]) -> dict[str, int]:
    out: Counter[str] = Counter()
    for trade in trades:
        reason = str(trade.reason_close or trade.reason or "UNKNOWN").upper()
        out[reason] += 1
    return dict(out)


def _per_bias_trade_metrics(trades: list[BacktestTrade]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[BacktestTrade]] = {"LONG": [], "SHORT": [], "FLAT": [], "UNKNOWN": []}
    for trade in trades:
        key = str(trade.gate_bias or "UNKNOWN").upper()
        if key not in grouped:
            key = "UNKNOWN"
        grouped[key].append(trade)

    out: dict[str, dict[str, float]] = {}
    for key, bucket in grouped.items():
        if not bucket:
            continue
        count = len(bucket)
        wins = sum(1 for trade in bucket if float(trade.pnl) > 0)
        total_pnl = sum(float(trade.pnl) for trade in bucket)
        out[key] = {
            "trades": float(count),
            "wins": float(wins),
            "losses": float(count - wins),
            "win_rate": (wins / count) if count else 0.0,
            "total_pnl": total_pnl,
            "expectancy": (total_pnl / count) if count else 0.0,
            "avg_r": (sum(float(trade.r_multiple) for trade in bucket) / count) if count else 0.0,
        }
    return out


def _gate_counts_from_blockers(blockers: Counter[str]) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in blockers.items()
        if key.startswith("GATE_")
        or key.startswith("DAILY_GATE_")
        or key.startswith("EXEC_FAIL_")
        or key.startswith("PIPELINE_NOT_READY")
    }


def _capital_ramp_event_to_dict(event: CapitalRampEvent) -> dict[str, Any]:
    from datetime import UTC

    return {
        "event_type": str(event.event_type),
        "event_ts_utc": event.event_ts_utc.astimezone(UTC).isoformat(),
        "local_date": event.local_date.isoformat(),
        "amount": float(event.amount),
        "model_equity": float(event.model_equity),
        "payload": dict(event.payload or {}),
    }


def _capital_ramp_summary(
    runtime: CapitalRampRuntime | None,
    events: list[CapitalRampEvent],
) -> tuple[bool, float, int, str | None, list[dict[str, Any]]]:
    if runtime is None:
        return False, 0.0, 0, None, []
    state = runtime.state
    return (
        True,
        float(state.topups_total),
        int(state.topups_count),
        state.stopped_reason,
        [_capital_ramp_event_to_dict(item) for item in events],
    )


def _merge_wait_metrics(reports: list[BacktestReport]) -> dict[str, float]:
    keys: set[str] = set()
    for report in reports:
        keys.update(report.wait_metrics.keys())
    merged: dict[str, float] = {}
    for key in keys:
        values = [float(report.wait_metrics.get(key, 0.0)) for report in reports]
        if key.endswith("_max_bars"):
            merged[key] = float(max(values)) if values else 0.0
        else:
            merged[key] = round((sum(values) / len(values)) if values else 0.0, 3)
    return merged


def aggregate_backtest_reports(
    *,
    config: AppConfig,
    asset: AssetConfig,
    reports: list[BacktestReport],
) -> BacktestReport:
    if not reports:
        return BacktestReport(
            epic=asset.epic,
            trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            total_pnl=0.0,
            expectancy=0.0,
            avg_r=0.0,
            max_drawdown=0.0,
            time_in_market_bars=0,
            equity_end=config.risk.equity,
            trade_log=[],
            fx_conversion_pct_used=float(config.backtest_tuning.fx_conversion_pct),
            account_currency=str(config.account_currency).upper(),
            instrument_currency=str(asset.instrument_currency or asset.currency).upper(),
            fx_conversion_fee_rate_used=float(config.fx_conversion_fee_rate),
            capital_ramp_enabled=bool(config.capital_ramp.enabled),
        )

    all_trades = sorted(
        [trade for report in reports for trade in report.trade_log],
        key=lambda item: (item.exit_time, item.entry_time),
    )
    trade_count = len(all_trades)
    wins = sum(1 for trade in all_trades if trade.pnl > 0)
    losses = sum(1 for trade in all_trades if trade.pnl <= 0)
    total_pnl = sum(float(trade.pnl) for trade in all_trades)
    expectancy = (total_pnl / trade_count) if trade_count else 0.0
    avg_r = (sum(float(trade.r_multiple) for trade in all_trades) / trade_count) if trade_count else 0.0
    win_rate = (wins / trade_count) if trade_count else 0.0
    avg_win, avg_loss, payoff_ratio, profit_factor = _trade_quality_metrics(all_trades)
    avg_win_r, avg_loss_r, payoff_r = _trade_r_quality_metrics(all_trades)
    exit_reason_distribution = _exit_reason_distribution(all_trades)

    equity_start_for_agg = float(
        START_EQUITY_PLN
        if any(bool(getattr(report, "capital_ramp_enabled", False)) for report in reports)
        else config.risk.equity
    )
    equity = equity_start_for_agg
    peak = equity
    max_drawdown = 0.0
    for trade in all_trades:
        equity += float(trade.pnl)
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    expectancy_net, profit_factor_net, max_drawdown_net = _net_quality_metrics(all_trades, equity_start_for_agg)

    decision_counts: Counter[str] = Counter()
    blockers: Counter[str] = Counter()
    gate_blocks: Counter[str] = Counter()
    gate_reason_blocks: Counter[str] = Counter()
    execution_fail: Counter[str] = Counter()
    missing_feature_counts: Counter[str] = Counter()
    timeout_resets: Counter[str] = Counter()
    score_bins: Counter[str] = Counter()
    spread_adjustments: Counter[str] = Counter()
    gate_bias_bars: Counter[str] = Counter()
    gate_bias_days: Counter[str] = Counter()
    gate_modes: Counter[str] = Counter()
    spread_modes: set[str] = set()
    assumed_spread_values: list[float] = []
    fx_conversion_pct_values: list[float] = []
    score_values: list[float] = []
    signal_candidates = 0
    time_in_market_bars = 0
    count_be_moves = 0
    count_tp1_hits = 0
    blocked_by_gate = 0
    orders_submitted = 0
    trades_filled = 0
    rejected_by_reason: Counter[str] = Counter()
    spread_cost_sum = 0.0
    slippage_cost_sum = 0.0
    commission_cost_sum = 0.0
    swap_cost_sum = 0.0
    fx_cost_sum = 0.0
    spread_points_samples: list[float] = []
    forced_closes_count = 0
    min_size_overrides_count = 0
    margin_capped_count = 0
    account_currencies: Counter[str] = Counter()
    instrument_currencies: Counter[str] = Counter()
    fx_fee_rate_values: list[float] = []
    capital_ramp_enabled = False
    capital_ramp_topups_total = 0.0
    capital_ramp_topups_count = 0
    capital_ramp_stopped_reason: str | None = None
    capital_ramp_events: list[dict[str, Any]] = []
    for report in reports:
        decision_counts.update(report.decision_counts)
        blockers.update(report.top_blockers)
        gate_blocks.update(report.gate_block_counts)
        gate_reason_blocks.update(report.blocked_by_gate_reasons)
        execution_fail.update(report.execution_fail_breakdown)
        missing_feature_counts.update(report.missing_feature_counts)
        timeout_resets.update(report.wait_timeout_resets)
        score_bins.update(report.score_bins)
        spread_adjustments.update(report.spread_gate_adjustments)
        gate_bias_bars.update(report.daily_gate_bias_bars)
        gate_bias_days.update(report.daily_gate_bias_days)
        gate_modes.update([str(report.daily_gate_mode or "off").lower()])
        spread_modes.add(report.spread_mode)
        assumed_spread_values.append(float(report.assumed_spread_used))
        fx_conversion_pct_values.append(float(getattr(report, "fx_conversion_pct_used", 0.0)))
        if report.avg_score is not None:
            score_values.append(float(report.avg_score))
        signal_candidates += int(report.signal_candidates)
        time_in_market_bars += int(report.time_in_market_bars)
        count_be_moves += int(report.count_be_moves)
        count_tp1_hits += int(report.count_tp1_hits)
        blocked_by_gate += int(getattr(report, "blocked_by_gate", 0))
        orders_submitted += int(getattr(report, "orders_submitted", 0))
        trades_filled += int(getattr(report, "trades_filled", 0))
        rejected_by_reason.update(getattr(report, "rejected_by_reason", {}) or {})
        spread_cost_sum += float(getattr(report, "spread_cost_sum", 0.0) or 0.0)
        slippage_cost_sum += float(getattr(report, "slippage_cost_sum", 0.0) or 0.0)
        commission_cost_sum += float(getattr(report, "commission_cost_sum", 0.0) or 0.0)
        swap_cost_sum += float(getattr(report, "swap_cost_sum", 0.0) or 0.0)
        fx_cost_sum += float(getattr(report, "fx_cost_sum", 0.0) or 0.0)
        avg_spread_pts = float(getattr(report, "avg_spread_points", 0.0) or 0.0)
        spread_points_samples.extend([avg_spread_pts] * max(1, int(getattr(report, "trades", 0) or 0)))
        forced_closes_count += int(getattr(report, "forced_closes_count", 0) or 0)
        min_size_overrides_count += int(getattr(report, "min_size_overrides_count", 0) or 0)
        margin_capped_count += int(getattr(report, "margin_capped_count", 0) or 0)
        account_currencies.update([str(getattr(report, "account_currency", config.account_currency)).upper()])
        instrument_currencies.update(
            [str(getattr(report, "instrument_currency", asset.instrument_currency or asset.currency)).upper()]
        )
        fx_fee_rate_values.append(float(getattr(report, "fx_conversion_fee_rate_used", config.fx_conversion_fee_rate)))
        if bool(getattr(report, "capital_ramp_enabled", False)):
            capital_ramp_enabled = True
        capital_ramp_topups_total = max(
            capital_ramp_topups_total,
            float(getattr(report, "capital_ramp_topups_total", 0.0) or 0.0),
        )
        capital_ramp_topups_count = max(
            capital_ramp_topups_count,
            int(getattr(report, "capital_ramp_topups_count", 0) or 0),
        )
        reason = getattr(report, "capital_ramp_stopped_reason", None)
        if reason:
            capital_ramp_stopped_reason = str(reason)
        events = getattr(report, "capital_ramp_events", None)
        if isinstance(events, list) and events:
            capital_ramp_events.extend(events)

    _agg_spread = _spread_point_stats(spread_points_samples)
    return BacktestReport(
        epic=asset.epic,
        trades=trade_count,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        expectancy=expectancy,
        avg_r=avg_r,
        max_drawdown=max_drawdown,
        time_in_market_bars=time_in_market_bars,
        equity_end=equity,
        trade_log=all_trades,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff_ratio,
        profit_factor=profit_factor,
        avg_win_r=avg_win_r,
        avg_loss_r=avg_loss_r,
        payoff_r=payoff_r,
        count_be_moves=count_be_moves,
        count_tp1_hits=count_tp1_hits,
        exit_reason_distribution=exit_reason_distribution,
        top_blockers=dict(blockers.most_common(10)),
        gate_block_counts=dict(gate_blocks if gate_blocks else _gate_counts_from_blockers(blockers)),
        missing_feature_counts=dict(missing_feature_counts),
        decision_counts=dict(decision_counts),
        signal_candidates=signal_candidates,
        wait_timeout_resets={
            "reaction": int(timeout_resets.get("reaction", timeout_resets.get("REACTION", 0))),
            "mitigation": int(timeout_resets.get("mitigation", timeout_resets.get("MITIGATION", 0))),
            "total": int(timeout_resets.get("total", timeout_resets.get("REACTION_TIMEOUT_RESET", 0))),
        },
        wait_metrics=_merge_wait_metrics(reports),
        execution_fail_breakdown=dict(execution_fail),
        avg_score=round(sum(score_values) / len(score_values), 4) if score_values else None,
        score_bins=dict(score_bins),
        spread_mode=next(iter(spread_modes)) if len(spread_modes) == 1 else "MIXED",
        assumed_spread_used=round(sum(assumed_spread_values) / len(assumed_spread_values), 6)
        if assumed_spread_values
        else 0.0,
        spread_gate_adjustments=dict(spread_adjustments),
        fx_conversion_pct_used=round(sum(fx_conversion_pct_values) / len(fx_conversion_pct_values), 6)
        if fx_conversion_pct_values
        else 0.0,
        daily_gate_mode=gate_modes.most_common(1)[0][0] if gate_modes else "off",
        daily_gate_bias_bars=dict(gate_bias_bars),
        daily_gate_bias_days=dict(gate_bias_days),
        blocked_by_gate=blocked_by_gate,
        blocked_by_gate_reasons=dict(gate_reason_blocks),
        per_bias_trade_metrics=_per_bias_trade_metrics(all_trades),
        orders_submitted=orders_submitted,
        trades_filled=trades_filled,
        rejected_by_reason=dict(rejected_by_reason),
        spread_cost_sum=spread_cost_sum,
        slippage_cost_sum=slippage_cost_sum,
        commission_cost_sum=commission_cost_sum,
        swap_cost_sum=swap_cost_sum,
        fx_cost_sum=fx_cost_sum,
        total_pnl_net=sum(float(t.pnl_net) for t in all_trades) if all_trades else total_pnl,
        total_pnl_gross=sum(float(t.pnl_gross) for t in all_trades) if all_trades else total_pnl,
        expectancy_net=expectancy_net,
        profit_factor_net=profit_factor_net,
        max_drawdown_net=max_drawdown_net,
        account_currency=account_currencies.most_common(1)[0][0]
        if account_currencies
        else str(config.account_currency).upper(),
        instrument_currency=instrument_currencies.most_common(1)[0][0]
        if instrument_currencies
        else str(asset.instrument_currency or asset.currency).upper(),
        fx_conversion_fee_rate_used=(
            round(sum(fx_fee_rate_values) / len(fx_fee_rate_values), 8)
            if fx_fee_rate_values
            else float(config.fx_conversion_fee_rate)
        ),
        avg_spread_points=_agg_spread[0],
        median_spread_points=_agg_spread[1],
        p90_spread_points=_agg_spread[2],
        forced_closes_count=forced_closes_count,
        min_size_overrides_count=min_size_overrides_count,
        margin_capped_count=margin_capped_count,
        capital_ramp_enabled=capital_ramp_enabled,
        capital_ramp_topups_total=capital_ramp_topups_total,
        capital_ramp_topups_count=capital_ramp_topups_count,
        capital_ramp_stopped_reason=capital_ramp_stopped_reason,
        capital_ramp_events=capital_ramp_events,
    )


def _is_better_outcome(
    *,
    current: StrategyOutcome,
    current_rank: float,
    best: StrategyOutcome | None,
    best_rank: float,
) -> bool:
    if best is None:
        return True
    current_action = _action_priority(current.evaluation.action)
    best_action = _action_priority(best.evaluation.action)
    if current_action != best_action:
        return current_action > best_action
    current_has_order = current.order_request is not None
    best_has_order = best.order_request is not None
    if current_has_order != best_has_order:
        return current_has_order
    return current_rank > best_rank
