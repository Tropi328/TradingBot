"""Backtest data models — extracted from engine.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class BacktestTrade:
    epic: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    r_multiple: float
    reason: str
    fees: float = 0.0
    score: float | None = None
    forced_exit: bool = False
    reason_open: str = "LIMIT_ENTRY"
    reason_close: str = ""
    gate_bias: str | None = None
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    commission_cost: float = 0.0
    swap_cost: float = 0.0
    fx_cost: float = 0.0
    pnl_gross: float = 0.0
    pnl_net: float = 0.0
    equity_before: float = 0.0
    equity_after: float = 0.0
    margin_capped: bool = False


@dataclass(slots=True)
class BacktestReport:
    epic: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    expectancy: float
    avg_r: float
    max_drawdown: float
    time_in_market_bars: int
    equity_end: float
    trade_log: list[BacktestTrade] = field(default_factory=list)
    avg_win: float = 0.0
    avg_loss: float = 0.0
    payoff_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    payoff_r: float = 0.0
    count_be_moves: int = 0
    count_tp1_hits: int = 0
    exit_reason_distribution: dict[str, int] = field(default_factory=dict)
    top_blockers: dict[str, int] = field(default_factory=dict)
    gate_block_counts: dict[str, int] = field(default_factory=dict)
    missing_feature_counts: dict[str, int] = field(default_factory=dict)
    decision_counts: dict[str, int] = field(default_factory=dict)
    signal_candidates: int = 0
    wait_timeout_resets: dict[str, int] = field(default_factory=dict)
    wait_metrics: dict[str, float] = field(default_factory=dict)
    execution_fail_breakdown: dict[str, int] = field(default_factory=dict)
    avg_score: float | None = None
    score_bins: dict[str, int] = field(default_factory=dict)
    spread_mode: str = "REAL_BIDASK"
    assumed_spread_used: float = 0.0
    spread_gate_adjustments: dict[str, int] = field(default_factory=dict)
    fx_conversion_pct_used: float = 0.0
    daily_gate_mode: str = "off"
    daily_gate_bias_bars: dict[str, int] = field(default_factory=dict)
    daily_gate_bias_days: dict[str, int] = field(default_factory=dict)
    blocked_by_gate: int = 0
    blocked_by_gate_reasons: dict[str, int] = field(default_factory=dict)
    per_bias_trade_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    orders_submitted: int = 0
    trades_filled: int = 0
    rejected_by_reason: dict[str, int] = field(default_factory=dict)
    spread_cost_sum: float = 0.0
    slippage_cost_sum: float = 0.0
    commission_cost_sum: float = 0.0
    swap_cost_sum: float = 0.0
    fx_cost_sum: float = 0.0
    total_pnl_net: float = 0.0
    total_pnl_gross: float = 0.0
    expectancy_net: float = 0.0
    profit_factor_net: float = 0.0
    max_drawdown_net: float = 0.0
    account_currency: str = "USD"
    instrument_currency: str = "USD"
    fx_conversion_fee_rate_used: float = 0.0
    avg_spread_points: float = 0.0
    median_spread_points: float = 0.0
    p90_spread_points: float = 0.0
    forced_closes_count: int = 0
    min_size_overrides_count: int = 0
    margin_capped_count: int = 0
    decision_funnel: dict[str, Any] = field(default_factory=dict)
    score_v3_summary: dict[str, Any] = field(default_factory=dict)
    shadow_summary: dict[str, Any] = field(default_factory=dict)
    capital_ramp_enabled: bool = False
    capital_ramp_topups_total: float = 0.0
    capital_ramp_topups_count: int = 0
    capital_ramp_stopped_reason: str | None = None
    capital_ramp_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epic": self.epic,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "expectancy": self.expectancy,
            "avg_r": self.avg_r,
            "max_drawdown": self.max_drawdown,
            "time_in_market_bars": self.time_in_market_bars,
            "equity_end": self.equity_end,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "payoff_ratio": self.payoff_ratio,
            "profit_factor": self.profit_factor,
            "avg_win_R": self.avg_win_r,
            "avg_loss_R": self.avg_loss_r,
            "payoff_R": self.payoff_r,
            "count_BE_moves": self.count_be_moves,
            "count_TP1_hits": self.count_tp1_hits,
            "exit_reason_distribution": self.exit_reason_distribution,
            "signal_candidates": self.signal_candidates,
            "decision_counts": self.decision_counts,
            "top_blockers": self.top_blockers,
            "gate_block_counts": self.gate_block_counts,
            "missing_feature_counts": self.missing_feature_counts,
            "wait_timeout_resets": self.wait_timeout_resets,
            "wait_metrics": self.wait_metrics,
            "execution_fail_breakdown": self.execution_fail_breakdown,
            "avg_score": self.avg_score,
            "count_score_bins": self.score_bins,
            "spread_mode": self.spread_mode,
            "assumed_spread_used": self.assumed_spread_used,
            "spread_gate_adjustments": self.spread_gate_adjustments,
            "fx_conversion_pct_used": self.fx_conversion_pct_used,
            "daily_gate_mode": self.daily_gate_mode,
            "daily_gate_bias_bars": self.daily_gate_bias_bars,
            "daily_gate_bias_days": self.daily_gate_bias_days,
            "blocked_by_gate": self.blocked_by_gate,
            "blocked_by_gate_reasons": self.blocked_by_gate_reasons,
            "per_bias_trade_metrics": self.per_bias_trade_metrics,
            "orders_submitted": self.orders_submitted,
            "trades_filled": self.trades_filled,
            "rejected_by_reason": self.rejected_by_reason,
            "spread_cost_sum": self.spread_cost_sum,
            "slippage_cost_sum": self.slippage_cost_sum,
            "commission_cost_sum": self.commission_cost_sum,
            "swap_cost_sum": self.swap_cost_sum,
            "fx_cost_sum": self.fx_cost_sum,
            "total_pnl_net": self.total_pnl_net,
            "total_pnl_gross": self.total_pnl_gross,
            "expectancy_net": self.expectancy_net,
            "profit_factor_net": self.profit_factor_net,
            "max_drawdown_net": self.max_drawdown_net,
            "account_currency": self.account_currency,
            "instrument_currency": self.instrument_currency,
            "fx_conversion_fee_rate_used": self.fx_conversion_fee_rate_used,
            "avg_spread_points": self.avg_spread_points,
            "median_spread_points": self.median_spread_points,
            "p90_spread_points": self.p90_spread_points,
            "forced_closes_count": self.forced_closes_count,
            "min_size_overrides_count": self.min_size_overrides_count,
            "margin_capped_count": self.margin_capped_count,
            "decision_funnel": self.decision_funnel,
            "score_v3_summary": self.score_v3_summary,
            "shadow_summary": self.shadow_summary,
            "capital_ramp_enabled": self.capital_ramp_enabled,
            "capital_ramp_topups_total": self.capital_ramp_topups_total,
            "capital_ramp_topups_count": self.capital_ramp_topups_count,
            "capital_ramp_stopped_reason": self.capital_ramp_stopped_reason,
            "capital_ramp_events": self.capital_ramp_events,
        }


@dataclass(slots=True)
class WalkForwardReport:
    epic: str
    splits: list[BacktestReport]
    aggregate: BacktestReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "epic": self.epic,
            "splits": [split.to_dict() for split in self.splits],
            "aggregate": self.aggregate.to_dict(),
        }


@dataclass(slots=True)
class _PendingOrder:
    side: str
    entry: float
    stop: float
    tp: float
    size: float
    expiry_index: int
    created_at: datetime
    reason_open: str = "SIGNAL"
    score: float | None = None
    gate_bias: str | None = None
    margin_capped: bool = False


@dataclass(slots=True)
class _OpenPosition:
    side: str
    entry: float
    stop: float
    tp: float
    size: float
    opened_at: datetime
    initial_stop: float
    initial_risk: float
    initial_size: float = 0.0
    max_loss_r_cap: float = 1.0
    tp1_trigger_r: float = 0.5
    tp1_fraction: float = 0.5
    be_offset_r: float = 0.0
    be_delay_bars_after_tp1: int = 0
    trailing_after_tp1: bool = True
    trailing_window_bars: int = 8
    trailing_buffer_r: float = 0.05
    be_moved: bool = False
    tp1_taken: bool = False
    tp1_hit_index: int | None = None
    realized_partial: float = 0.0
    swap_total: float = 0.0
    fx_conversion_total: float = 0.0
    spread_cost_total: float = 0.0
    slippage_cost_total: float = 0.0
    commission_total: float = 0.0
    swap_cost_total: float = 0.0
    fx_cost_total: float = 0.0
    next_swap_ts: datetime | None = None
    reason_open: str = "SIGNAL"
    score: float | None = None
    gate_bias: str | None = None
    margin_capped: bool = False
    _skip_first_bar: bool = True


@dataclass(slots=True)
class BacktestVariant:
    code: str = "W0"
    reaction_timeout_reset: bool = False
    soft_reason_penalties: bool = False
    thresholds_v2: bool = False
    dynamic_threshold_bump: bool = False


@dataclass(slots=True)
class _WaitGateState:
    wait_type: str
    enter_bar_index: int
    enter_ts: datetime
    enter_reason: str | None = None
    timed_out_soft: bool = False


@dataclass(slots=True)
class _ExecutionFailSample:
    ts_utc: str
    symbol: str
    strategy: str
    reason: str
    spread_ratio: float | None
    atr_m5: float | None
    missing_features: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _NoPriceSample:
    ts_utc: str
    symbol: str
    timeframe: str
    strategy: str
    price_mode: str
    missing_fields: list[str]
    source_files: list[str]
    source_datasets: list[str]
    record: dict[str, object]


@dataclass(slots=True)
class _ReactionTimeoutSample:
    ts_utc: str
    symbol: str
    strategy: str
    state: str
    waited_bars: int
    reason: str
