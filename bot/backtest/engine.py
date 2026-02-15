from __future__ import annotations

from collections import Counter, deque
import csv
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from bot.config import AppConfig, AssetConfig
from bot.data.candles import Candle
from bot.execution.feasibility import RejectReason, validate_order
from bot.execution.fx import FxConverter
from bot.gating.daily_gate import DailyGateProvider
from bot.strategy.candidate_queue import CandidateQueue
from bot.strategy.contracts import (
    BiasState,
    DecisionAction,
    SetupCandidate,
    StrategyDataBundle,
    StrategyEvaluation,
    StrategyOutcome,
    StrategyPlugin,
)
from bot.strategy.indicators import atr
from bot.strategy.index_existing import IndexExistingStrategy
from bot.strategy.orb_h4_retest import OrbH4RetestStrategy
from bot.strategy.orderflow import CompositeOrderflowProvider, OrderflowSnapshot
from bot.strategy.ranker import rank_score
from bot.strategy.risk import RiskEngine
from bot.strategy.router import StrategyRoute, StrategyRouter
from bot.strategy.scalp_ict_pa import ScalpIctPriceActionStrategy
from bot.strategy.schedule import is_schedule_open
from bot.strategy.state_machine import StrategyEngine
from bot.strategy.trend_pullback_m15 import TrendPullbackM15Strategy

LOGGER = logging.getLogger(__name__)


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
    expectancy_net: float = 0.0
    profit_factor_net: float = 0.0
    max_drawdown_net: float = 0.0
    account_currency: str = "USD"
    instrument_currency: str = "USD"
    fx_conversion_fee_rate_used: float = 0.0

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
            "total_pnl_net": self.total_pnl_net or self.total_pnl,
            "expectancy_net": self.expectancy_net or self.expectancy,
            "profit_factor_net": self.profit_factor_net or self.profit_factor,
            "max_drawdown_net": self.max_drawdown_net or self.max_drawdown,
            "account_currency": self.account_currency,
            "instrument_currency": self.instrument_currency,
            "fx_conversion_fee_rate_used": self.fx_conversion_fee_rate_used,
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


def _parse_dt(value: str) -> datetime:
    normalized = value.replace(" ", "T")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_candles_csv(path: str | Path) -> list[Candle]:
    csv_path = Path(path)
    candles: list[Candle] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames:
            reader.fieldnames = [name.lstrip("\ufeff").strip() for name in reader.fieldnames]
        required = {"timestamp", "open", "high", "low", "close"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must include: timestamp,open,high,low,close")
        for row in reader:
            candles.append(
                Candle(
                    timestamp=_parse_dt(str(row["timestamp"])),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume") or 0.0),
                )
            )
    return sorted(candles, key=lambda c: c.timestamp)


def _bucket_time(dt: datetime, minutes: int) -> datetime:
    unix = int(dt.timestamp())
    size = minutes * 60
    return datetime.fromtimestamp(unix - (unix % size), tz=timezone.utc)


def aggregate_candles(candles: list[Candle], timeframe_minutes: int) -> list[Candle]:
    if not candles:
        return []
    result: list[Candle] = []
    bucket_start = _bucket_time(candles[0].timestamp, timeframe_minutes)
    open_price = candles[0].open
    high = candles[0].high
    low = candles[0].low
    close = candles[0].close
    volume = candles[0].volume or 0.0

    for candle in candles[1:]:
        current_bucket = _bucket_time(candle.timestamp, timeframe_minutes)
        if current_bucket != bucket_start:
            result.append(
                Candle(
                    timestamp=bucket_start,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                )
            )
            bucket_start = current_bucket
            open_price = candle.open
            high = candle.high
            low = candle.low
            close = candle.close
            volume = candle.volume or 0.0
            continue
        high = max(high, candle.high)
        low = min(low, candle.low)
        close = candle.close
        volume += candle.volume or 0.0

    result.append(
        Candle(
            timestamp=bucket_start,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
    )
    return result


def _calc_exit(
    position: _OpenPosition,
    candle: Candle,
    *,
    assumed_spread: float,
    slippage: float,
) -> tuple[bool, float, str]:
    if position.side == "LONG":
        stop_hit = candle.low <= position.stop
        tp_hit = candle.high >= position.tp
    else:
        stop_hit = candle.high >= position.stop
        tp_hit = candle.low <= position.tp
    if not stop_hit and not tp_hit:
        return False, 0.0, ""
    # Conservative fill order when both happen in one bar.
    reason = "STOP" if stop_hit else "TP"
    if reason == "STOP":
        be_stop = False
        if position.be_moved:
            if position.side == "LONG":
                be_stop = position.stop >= (position.entry - 1e-9)
            else:
                be_stop = position.stop <= (position.entry + 1e-9)
        if be_stop:
            reason = "BE"
        if position.side == "LONG":
            fill_price = position.stop - slippage
            max_loss_price = position.entry - (position.initial_risk * position.max_loss_r_cap)
            fill_price = max(fill_price, max_loss_price)
        else:
            fill_price = position.stop + slippage
            max_loss_price = position.entry + (position.initial_risk * position.max_loss_r_cap)
            fill_price = min(fill_price, max_loss_price)
        return True, fill_price, reason

    if position.side == "LONG":
        base_tp_fill = candle.bid if candle.bid is not None else (position.tp - (assumed_spread * 0.5))
        fill_price = base_tp_fill - slippage
    else:
        base_tp_fill = candle.ask if candle.ask is not None else (position.tp + (assumed_spread * 0.5))
        fill_price = base_tp_fill + slippage
    return True, fill_price, reason


def _default_observe_evaluation(*, symbol: str, reason: str) -> StrategyEvaluation:
    return StrategyEvaluation(
        action=DecisionAction.OBSERVE,
        score_total=0.0,
        reasons_blocking=[reason],
        would_enter_if=["VALID_CANDIDATE"],
        snapshot={"symbol": symbol},
    )


def _pick_best_candidate(
    *,
    strategy: StrategyPlugin,
    symbol: str,
    candidates: list[SetupCandidate],
    data: StrategyDataBundle,
) -> tuple[SetupCandidate | None, StrategyEvaluation]:
    if not candidates:
        return None, _default_observe_evaluation(symbol=symbol, reason="NO_CANDIDATE")
    best_candidate = candidates[0]
    best_eval = strategy.evaluate_candidate(symbol, best_candidate, data)
    for candidate in candidates[1:]:
        current = strategy.evaluate_candidate(symbol, candidate, data)
        current_score = current.score_total if current.score_total is not None else -1.0
        best_score = best_eval.score_total if best_eval.score_total is not None else -1.0
        if current.action == DecisionAction.TRADE and best_eval.action != DecisionAction.TRADE:
            best_candidate, best_eval = candidate, current
            continue
        if current.action == best_eval.action and current_score > best_score:
            best_candidate, best_eval = candidate, current
    return best_candidate, best_eval


def _normalize_action_for_score(
    *,
    evaluation: StrategyEvaluation,
    config: AppConfig,
    trade_threshold: float | None = None,
    small_min: float | None = None,
    small_max: float | None = None,
) -> StrategyEvaluation:
    if evaluation.score_total is None:
        return evaluation
    if evaluation.reasons_blocking:
        evaluation.action = DecisionAction.OBSERVE
        return evaluation
    if trade_threshold is None or small_min is None or small_max is None:
        trade_threshold = float(config.decision_policy.trade_score_threshold)
        small_min = float(config.decision_policy.small_score_min)
        small_max = float(config.decision_policy.small_score_max)
    score = float(evaluation.score_total)
    if score >= trade_threshold:
        evaluation.action = DecisionAction.TRADE
    elif small_min <= score <= small_max:
        evaluation.action = DecisionAction.SMALL
    else:
        evaluation.action = DecisionAction.OBSERVE
        if "SCORE_BELOW_MIN" not in evaluation.reasons_blocking:
            evaluation.reasons_blocking.append("SCORE_BELOW_MIN")
    return evaluation


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _quantile_from_sorted(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    q_norm = _clamp(float(q), 0.0, 1.0)
    pos = q_norm * float(len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    frac = pos - float(lo)
    return float(values[lo]) + (float(values[hi]) - float(values[lo])) * frac


def _resolve_dynamic_spread_bounds(
    *,
    config: AppConfig,
    symbol: str,
    fallback_spread: float,
) -> tuple[float, float] | None:
    tuning = config.backtest_tuning
    if not bool(tuning.dynamic_assumed_spread_enabled):
        return None
    symbol_norm = str(symbol).strip().upper()
    min_map = tuning.dynamic_assumed_spread_min_by_symbol
    max_map = tuning.dynamic_assumed_spread_max_by_symbol
    min_spread = min_map.get(symbol_norm)
    max_spread = max_map.get(symbol_norm)
    if min_spread is None and max_spread is None:
        return None
    fallback = max(0.0, float(fallback_spread))
    lo = float(min_spread) if min_spread is not None else fallback
    hi = float(max_spread) if max_spread is not None else fallback
    lo = max(0.0, lo)
    hi = max(0.0, hi)
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _build_dynamic_assumed_spread_series(
    *,
    candles_m5: list[Candle],
    atr_values: list[float | None],
    min_spread: float,
    max_spread: float,
) -> list[float]:
    count = len(candles_m5)
    if count == 0:
        return []
    lo = max(0.0, float(min_spread))
    hi = max(lo, float(max_spread))
    if hi - lo <= 1e-12:
        return [lo] * count

    atr_clean: list[float] = []
    for value in atr_values:
        if value is None:
            continue
        try:
            atr_value = float(value)
        except (TypeError, ValueError):
            continue
        if atr_value > 0:
            atr_clean.append(atr_value)

    if not atr_clean:
        mid = (lo + hi) * 0.5
        return [mid] * count

    atr_sorted = sorted(atr_clean)
    q10 = _quantile_from_sorted(atr_sorted, 0.10)
    q90 = _quantile_from_sorted(atr_sorted, 0.90)
    if q90 <= q10:
        mid = (lo + hi) * 0.5
        return [mid] * count

    out: list[float] = []
    for index in range(count):
        atr_value_raw = atr_values[index] if index < len(atr_values) else None
        ratio = 0.5
        if atr_value_raw is not None:
            try:
                atr_value = float(atr_value_raw)
                if atr_value > 0:
                    ratio = _clamp((atr_value - q10) / (q90 - q10), 0.0, 1.0)
            except (TypeError, ValueError):
                ratio = 0.5
        out.append(lo + ((hi - lo) * ratio))
    return out


def _parse_swap_time_utc(value: str) -> tuple[int, int]:
    raw = str(value).strip()
    parts = raw.split(":")
    if len(parts) != 2:
        return 23, 0
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return 23, 0
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return 23, 0
    return hour, minute


def _next_rollover_timestamp(ts: datetime, *, hour: int, minute: int) -> datetime:
    ts_utc = ts.astimezone(timezone.utc)
    rollover = ts_utc.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if ts_utc >= rollover:
        rollover += timedelta(days=1)
    return rollover


def _convert_cash_to_account(
    *,
    amount: float,
    category: str,
    fx_converter: FxConverter | None,
    instrument_currency: str,
    account_currency: str,
    fx_apply_to: set[str],
) -> tuple[float, float]:
    if fx_converter is None or str(instrument_currency).upper() == str(account_currency).upper():
        return float(amount), 0.0
    apply_fee = str(category).strip().lower() in fx_apply_to
    converted = fx_converter.convert(
        amount=float(amount),
        from_currency=instrument_currency,
        to_currency=account_currency,
        apply_fee=apply_fee,
    )
    return float(converted.converted_amount), float(converted.fx_cost)


def _apply_overnight_swap_if_due(
    *,
    position: _OpenPosition,
    candle_ts: datetime,
    swap_hour: int,
    swap_minute: int,
    long_swap_pct: float,
    short_swap_pct: float,
    fx_converter: FxConverter | None = None,
    instrument_currency: str = "USD",
    account_currency: str = "USD",
    fx_apply_to: set[str] | None = None,
) -> float:
    if position.next_swap_ts is None:
        position.next_swap_ts = _next_rollover_timestamp(candle_ts, hour=swap_hour, minute=swap_minute)
        return 0.0

    ts_utc = candle_ts.astimezone(timezone.utc)
    applied = 0.0
    while ts_utc >= position.next_swap_ts:
        rate_pct = float(long_swap_pct) if position.side == "LONG" else float(short_swap_pct)
        swap_instr = float(position.entry) * float(position.size) * (rate_pct / 100.0)
        fx_apply = fx_apply_to or {"pnl", "swap", "commission"}
        swap_account, swap_fx_cost = _convert_cash_to_account(
            amount=swap_instr,
            category="swap",
            fx_converter=fx_converter,
            instrument_currency=instrument_currency,
            account_currency=account_currency,
            fx_apply_to=fx_apply,
        )
        position.realized_partial += swap_account
        position.swap_total += swap_account
        position.swap_cost_total += -swap_account
        position.fx_conversion_total += swap_fx_cost
        position.fx_cost_total += swap_fx_cost
        applied += swap_account
        position.next_swap_ts = position.next_swap_ts + timedelta(days=1)
    return applied


def _resolve_orderflow_mode(*, symbol: str, route_params: dict[str, object], default_mode: str, full_symbols: set[str]) -> str:
    params = route_params.get("orderflow")
    if isinstance(params, dict):
        mode = str(params.get("mode", "")).strip().upper()
        if mode in {"LITE", "FULL"}:
            return mode
    if symbol.strip().upper() in full_symbols:
        return "FULL"
    mode = default_mode.strip().upper()
    return mode if mode in {"LITE", "FULL"} else "LITE"


def _orderflow_param(
    *,
    route_params: dict[str, object],
    settings: dict[str, float] | None,
    key: str,
    default: float,
) -> float:
    params = route_params.get("orderflow")
    if isinstance(params, dict):
        try:
            if key in params:
                return float(params[key])
        except (TypeError, ValueError):
            pass
    if settings is not None and key in settings:
        try:
            return float(settings[key])
        except (TypeError, ValueError):
            return default
    return default


def _soft_reason_penalty_map(
    config: AppConfig,
    *,
    route_params: dict[str, object] | None = None,
) -> dict[str, float]:
    tuning = config.backtest_tuning
    penalties: dict[str, float] = {
        "ORB_NO_RETEST": tuning.penalty_orb_no_retest,
        "ORB_CONFIRMATIONS_LOW": tuning.penalty_orb_confirm_low,
        "SCALP_NO_DISPLACEMENT": tuning.penalty_scalp_no_displacement,
        "SCALP_NO_MSS": tuning.penalty_scalp_no_mss,
        "SCALP_NO_FVG": tuning.penalty_scalp_no_fvg,
    }
    if isinstance(route_params, dict):
        raw = route_params.get("soft_penalties")
        if isinstance(raw, dict):
            for key, value in raw.items():
                reason = str(key).strip().upper()
                if not reason:
                    continue
                try:
                    penalty = float(value)
                except (TypeError, ValueError):
                    continue
                if penalty > 0:
                    penalty = -penalty
                penalties[reason] = penalty
    return penalties


def _apply_soft_reason_penalties(
    *,
    evaluation: StrategyEvaluation,
    config: AppConfig,
    route_params: dict[str, object] | None = None,
    enabled: bool,
) -> StrategyEvaluation:
    if not enabled:
        return evaluation
    penalties = _soft_reason_penalty_map(config, route_params=route_params)
    soft_reasons: list[str] = []
    remaining: list[str] = []
    for reason in evaluation.reasons_blocking:
        penalty = penalties.get(reason)
        if penalty is None:
            remaining.append(reason)
            continue
        key = f"penalty_soft_{reason.lower()}"
        existing = float(evaluation.score_breakdown.get(key, 0.0))
        evaluation.score_breakdown[key] = existing + float(penalty)
        soft_reasons.append(reason)
    if soft_reasons:
        evaluation.metadata["soft_reasons"] = list(dict.fromkeys(soft_reasons))
    evaluation.reasons_blocking = remaining
    return evaluation


def _thresholds_for_variant(config: AppConfig, variant: BacktestVariant) -> tuple[float, float, float]:
    if variant.thresholds_v2:
        tuning = config.backtest_tuning
        return (
            float(tuning.thresholds_v2_trade),
            float(tuning.thresholds_v2_small_min),
            float(tuning.thresholds_v2_small_max),
        )
    return (
        float(config.decision_policy.trade_score_threshold),
        float(config.decision_policy.small_score_min),
        float(config.decision_policy.small_score_max),
    )


def _adjust_thresholds_dynamic(
    *,
    trade_threshold: float,
    small_min: float,
    small_max: float,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    config: AppConfig,
    enabled: bool,
) -> tuple[float, float, float, list[str]]:
    if not enabled:
        return trade_threshold, small_min, small_max, []
    reasons: list[str] = []
    tuning = config.backtest_tuning
    gates_cfg = route_params.get("quality_gates")
    spread_ratio_max = None
    min_atr_m5 = None
    if isinstance(gates_cfg, dict):
        try:
            spread_ratio_max = float(gates_cfg.get("spread_ratio_max")) if gates_cfg.get("spread_ratio_max") is not None else None
        except (TypeError, ValueError):
            spread_ratio_max = None
        try:
            min_atr_m5 = float(gates_cfg.get("min_atr_m5")) if gates_cfg.get("min_atr_m5") is not None else None
        except (TypeError, ValueError):
            min_atr_m5 = None
    spread_ratio = evaluation.metadata.get("spread_ratio")
    atr_m5 = evaluation.metadata.get("atr_m5", evaluation.snapshot.get("atr_m5"))

    if spread_ratio is not None and spread_ratio_max is not None:
        try:
            spread_ratio_f = float(spread_ratio)
            if spread_ratio_f > (float(tuning.dynamic_spread_ratio_frac) * spread_ratio_max):
                bump = float(tuning.dynamic_spread_score_penalty)
                trade_threshold += bump
                small_min += bump
                small_max += bump
                reasons.append("DYN_THRESHOLD_SPREAD")
        except (TypeError, ValueError):
            pass

    if atr_m5 is not None and min_atr_m5 is not None:
        try:
            atr_f = float(atr_m5)
            if atr_f < (float(tuning.dynamic_atr_buffer_mult) * min_atr_m5):
                bump = float(tuning.dynamic_atr_score_penalty)
                trade_threshold += bump
                small_min += bump
                small_max += bump
                reasons.append("DYN_THRESHOLD_ATR")
        except (TypeError, ValueError):
            pass

    return trade_threshold, small_min, small_max, reasons


def _compute_v2_score(
    *,
    strategy_name: str,
    bias: BiasState,
    route_params: dict[str, object],
    config: AppConfig,
    evaluation: StrategyEvaluation,
    news_blocked: bool,
    schedule_open: bool,
    orderflow_snapshot: OrderflowSnapshot | None = None,
    setup_side: str | None = None,
    orderflow_settings: dict[str, float] | None = None,
) -> StrategyEvaluation:
    raw = dict(evaluation.score_breakdown)
    evaluation.metadata["raw_score_breakdown"] = raw

    bias_raw = float(max(raw.get("bias", 0.0), raw.get("trend_strength", 0.0), raw.get("breakout_quality", 0.0)))
    sweep_raw = float(max(raw.get("sweep", 0.0), raw.get("liquidity_setup", 0.0), raw.get("retest_quality", 0.0)))
    mss_raw = float(max(raw.get("mss", 0.0), raw.get("confirmation_strength", 0.0), raw.get("trigger_quality", 0.0)))
    displacement_raw = float(max(raw.get("displacement", 0.0), raw.get("trigger_quality", 0.0), raw.get("confirmation_strength", 0.0)))
    fvg_raw = float(max(raw.get("fvg", 0.0), raw.get("mitigation_quality", 0.0), raw.get("retest_quality", 0.0)))

    bias_regime = _clamp((bias_raw / 20.0) * 15.0, 0.0, 15.0)
    location_score = 10.0
    pd_eq = evaluation.metadata.get("h1_pd_eq", evaluation.snapshot.get("h1_pd_eq"))
    h1_close = evaluation.metadata.get("h1_close", evaluation.snapshot.get("h1_close"))
    side = str(setup_side or evaluation.metadata.get("side", "")).upper()
    if pd_eq is not None and h1_close is not None and side in {"LONG", "SHORT"}:
        try:
            eq_float = float(pd_eq)
            close_float = float(h1_close)
            if side == "LONG":
                location_score = 15.0 if close_float < eq_float else 6.0
            else:
                location_score = 15.0 if close_float > eq_float else 6.0
        except (TypeError, ValueError):
            location_score = 10.0
    if strategy_name == "INDEX_EXISTING":
        location_score = 12.0 if not evaluation.reasons_blocking else 6.0
    liquidity_score = _clamp((sweep_raw / 20.0) * 15.0, 0.0, 15.0)
    edge_score = _clamp(bias_regime + location_score + liquidity_score, 0.0, 45.0)

    mitigation_quality = _clamp((fvg_raw / 15.0) * 15.0, 0.0, 15.0)
    trigger_confirmations = int(evaluation.metadata.get("trigger_confirmations", 0))
    reaction_confirmed = _clamp((trigger_confirmations / 3.0) * 15.0, 0.0, 15.0)
    trigger_clean = _clamp(((mss_raw / 20.0) * 5.0) + ((displacement_raw / 20.0) * 5.0), 0.0, 10.0)
    trigger_score = _clamp(mitigation_quality + reaction_confirmed + trigger_clean, 0.0, 40.0)

    atr_value = evaluation.metadata.get("atr_m5", evaluation.snapshot.get("atr_m5"))
    spread_value = evaluation.snapshot.get("spread", evaluation.metadata.get("spread"))
    spread_mode = str(evaluation.metadata.get("spread_mode", "REAL_BIDASK")).upper()
    spread_ratio = None
    if atr_value is not None and spread_value is not None:
        try:
            atr_float = float(atr_value)
            spread_float = float(spread_value)
            if atr_float > 0:
                spread_ratio = spread_float / atr_float
        except (TypeError, ValueError):
            spread_ratio = None
    max_spread_ratio = 0.15
    gates_cfg = route_params.get("quality_gates")
    if isinstance(gates_cfg, dict):
        max_spread_ratio = float(gates_cfg.get("spread_ratio_max", 0.15))
    if spread_ratio is None:
        spread_ratio_score = 3.0
        slippage_risk_score = 2.0
    elif spread_ratio <= max_spread_ratio:
        spread_ratio_score = 8.0
        slippage_risk_score = 4.0
    elif spread_ratio <= max_spread_ratio * 1.25:
        spread_ratio_score = 4.0
        slippage_risk_score = 2.0
    else:
        spread_ratio_score = 0.0
        slippage_risk_score = 0.0
    market_state_score = 3.0 if schedule_open else 0.0
    execution_score = _clamp(spread_ratio_score + slippage_risk_score + market_state_score, 0.0, 15.0)

    of_trigger_bonus = 0.0
    of_execution_bonus = 0.0
    of_divergence_penalty = 0.0
    if orderflow_snapshot is not None:
        of_dict = orderflow_snapshot.to_dict()
        evaluation.metadata["orderflow_snapshot"] = of_dict
        evaluation.snapshot["orderflow"] = of_dict

        confidence = float(orderflow_snapshot.confidence)
        chop_score = float(orderflow_snapshot.metrics.chop_score)
        of_spread_ratio = float(orderflow_snapshot.metrics.spread_ratio)
        of_pressure = float(orderflow_snapshot.pressure)
        of_direction = str(orderflow_snapshot.direction).upper()

        trigger_bonus_cap = _orderflow_param(
            route_params=route_params,
            settings=orderflow_settings,
            key="trigger_bonus_max",
            default=10.0,
        )
        execution_bonus_cap = _orderflow_param(
            route_params=route_params,
            settings=orderflow_settings,
            key="execution_bonus_max",
            default=5.0,
        )
        divergence_penalty_min = _orderflow_param(
            route_params=route_params,
            settings=orderflow_settings,
            key="divergence_penalty_min",
            default=6.0,
        )
        divergence_penalty_max = _orderflow_param(
            route_params=route_params,
            settings=orderflow_settings,
            key="divergence_penalty_max",
            default=10.0,
        )
        if divergence_penalty_min > divergence_penalty_max:
            divergence_penalty_min, divergence_penalty_max = divergence_penalty_max, divergence_penalty_min

        flow_alignment = _clamp(abs(of_pressure), 0.0, 1.0)
        of_trigger_bonus = _clamp(
            confidence * (1.0 - chop_score) * (0.5 + (0.5 * flow_alignment)) * trigger_bonus_cap,
            0.0,
            trigger_bonus_cap,
        )
        execution_quality = _clamp(1.0 - (of_spread_ratio / max(max_spread_ratio, 1e-9)), 0.0, 1.0)
        of_execution_bonus = _clamp(confidence * execution_quality * execution_bonus_cap, 0.0, execution_bonus_cap)

        if side in {"LONG", "SHORT"} and of_direction in {"LONG", "SHORT"} and of_direction != side:
            of_divergence_penalty = _clamp(
                divergence_penalty_min + ((divergence_penalty_max - divergence_penalty_min) * flow_alignment),
                divergence_penalty_min,
                divergence_penalty_max,
            )

    penalties: dict[str, float] = {}
    if bias.direction == "NEUTRAL":
        penalties["NEUTRAL_BIAS"] = 5.0
    if bool(evaluation.metadata.get("near_adr_exhausted")):
        penalties["NEAR_ADR_EXHAUSTED"] = 6.0
    if bool(evaluation.metadata.get("news_medium_window")):
        penalties["NEWS_MEDIUM_WINDOW"] = 8.0
    if bool(evaluation.metadata.get("correlation_exposure")):
        penalties["CORRELATION_EXPOSURE"] = 6.0
    if bool(evaluation.metadata.get("late_retest")):
        penalties["LATE_RETEST"] = 5.0
    if news_blocked:
        penalties["NEWS_MEDIUM_WINDOW"] = max(penalties.get("NEWS_MEDIUM_WINDOW", 0.0), 8.0)
    if of_divergence_penalty > 0:
        penalties["OF_DIVERGENCE"] = max(penalties.get("OF_DIVERGENCE", 0.0), of_divergence_penalty)
    if spread_mode == "ASSUMED_OHLC" and spread_ratio is not None and spread_ratio > max_spread_ratio:
        soft_penalty = float(config.backtest_tuning.ohlc_only_spread_soft_penalty)
        penalties["ASSUMED_OHLC_SPREAD"] = max(penalties.get("ASSUMED_OHLC_SPREAD", 0.0), soft_penalty)
        evaluation.metadata["spread_gate_soft_penalty_applied"] = True
        soft_reasons = evaluation.metadata.get("soft_reasons")
        if not isinstance(soft_reasons, list):
            soft_reasons = []
        if "ASSUMED_OHLC_SPREAD" not in soft_reasons:
            soft_reasons.append("ASSUMED_OHLC_SPREAD")
        evaluation.metadata["soft_reasons"] = soft_reasons
    for key, value in raw.items():
        if key.startswith("penalty_") and value < 0:
            mapped_key = key.replace("penalty_", "").upper()
            penalties[mapped_key] = max(penalties.get(mapped_key, 0.0), abs(float(value)))

    penalty_total = sum(penalties.values())
    score_pre_penalty = edge_score + trigger_score + execution_score + of_trigger_bonus + of_execution_bonus
    score_total = _clamp(score_pre_penalty - penalty_total, 0.0, 100.0)

    evaluation.score_layers = {
        "edge": round(edge_score, 2),
        "trigger": round(trigger_score, 2),
        "execution": round(execution_score, 2),
        "orderflow": round(of_trigger_bonus + of_execution_bonus, 2),
    }
    evaluation.penalties = {key: round(value, 2) for key, value in penalties.items()}
    evaluation.score_total = round(score_total, 2)
    evaluation.score_breakdown = {
        "edge_total": round(edge_score, 2),
        "trigger_total": round(trigger_score, 2),
        "execution_total": round(execution_score, 2),
        "penalty_total": round(penalty_total, 2),
        "score_pre_penalty": round(score_pre_penalty, 2),
        "score_total": round(score_total, 2),
    }
    if spread_ratio is not None:
        evaluation.metadata["spread_ratio"] = spread_ratio
    return evaluation


def _evaluate_hard_gates(
    *,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    now: datetime,
    timezone_name: str,
) -> tuple[dict[str, bool], list[str]]:
    gates = {
        "ExecutionGate": True,
        "ScheduleGate": True,
        "ReactionGate": True,
        "RiskGate": True,
    }
    reasons: list[str] = []

    schedule_cfg = route_params.get("schedule")
    schedule_open = True
    if isinstance(schedule_cfg, dict):
        schedule_open = is_schedule_open(now, schedule_cfg, timezone_name)
    if not schedule_open:
        gates["ScheduleGate"] = False
        reasons.append("EXEC_FAIL_MARKET_CLOSED")

    gates_cfg = route_params.get("quality_gates")
    if isinstance(gates_cfg, dict):
        max_spread_ratio = float(gates_cfg.get("spread_ratio_max", 0.15))
        try:
            min_confirm = max(0, int(gates_cfg.get("min_confirm", 0)))
        except (TypeError, ValueError):
            min_confirm = 0
        try:
            min_confirm_trade = max(0, int(gates_cfg.get("min_confirm_trade", min_confirm)))
        except (TypeError, ValueError):
            min_confirm_trade = min_confirm
        try:
            min_confirm_small = max(0, int(gates_cfg.get("min_confirm_small", max(0, min_confirm_trade - 1))))
        except (TypeError, ValueError):
            min_confirm_small = max(0, min_confirm_trade - 1)
        if min_confirm_small > min_confirm_trade:
            min_confirm_small = min_confirm_trade
    else:
        max_spread_ratio = 0.15
        min_confirm = 0
        min_confirm_trade = 0
        min_confirm_small = 0
    spread_ratio = evaluation.metadata.get("spread_ratio")
    atr_value = evaluation.metadata.get("atr_m5", evaluation.snapshot.get("atr_m5"))
    spread_value = evaluation.snapshot.get("spread", evaluation.metadata.get("spread"))
    spread_mode = str(evaluation.metadata.get("spread_mode", "REAL_BIDASK")).upper()
    missing_features: list[str] = []
    if spread_value is None:
        quote = evaluation.metadata.get("quote")
        if isinstance(quote, tuple) and len(quote) >= 3:
            spread_value = quote[2]
    close_value = evaluation.snapshot.get("close", evaluation.metadata.get("close"))
    if atr_value is None:
        gates["ExecutionGate"] = False
        reasons.append("EXEC_FAIL_MISSING_FEATURES")
        missing_features.append("atr_m5")
    else:
        try:
            if float(atr_value) <= 0:
                gates["ExecutionGate"] = False
                reasons.append("EXEC_FAIL_INVALID_ATR")
                missing_features.append("atr_m5")
        except (TypeError, ValueError):
            gates["ExecutionGate"] = False
            reasons.append("EXEC_FAIL_INVALID_ATR")
            missing_features.append("atr_m5")
    if spread_ratio is None:
        if atr_value is not None and spread_value is not None:
            try:
                atr_float = float(atr_value)
                spread_float = float(spread_value)
                if atr_float > 0:
                    spread_ratio = spread_float / atr_float
                    evaluation.metadata["spread_ratio"] = spread_ratio
            except (TypeError, ValueError):
                spread_ratio = None
    if spread_ratio is None:
        if spread_value is None and close_value is not None:
            # No BID/ASK spread available: keep pipeline alive for mid-only data.
            spread_ratio = 0.0
            evaluation.metadata["spread_ratio"] = spread_ratio
        elif spread_value is None:
            gates["ExecutionGate"] = False
            reasons.append("EXEC_FAIL_NO_PRICE")
        elif atr_value is not None and (isinstance(atr_value, (float, int)) and float(atr_value) > 0):
            gates["ExecutionGate"] = False
            reasons.append("EXEC_FAIL_MISSING_FEATURES")
            missing_features.append("spread_or_spread_ratio")
    elif float(spread_ratio) > max_spread_ratio:
        if spread_mode == "ASSUMED_OHLC":
            evaluation.metadata["spread_gate_ohlc_hard_skipped"] = True
            soft_reasons = evaluation.metadata.get("soft_reasons")
            if not isinstance(soft_reasons, list):
                soft_reasons = []
            if "ASSUMED_OHLC_SPREAD" not in soft_reasons:
                soft_reasons.append("ASSUMED_OHLC_SPREAD")
            evaluation.metadata["soft_reasons"] = soft_reasons
        else:
            gates["ExecutionGate"] = False
            reasons.append("EXEC_FAIL_SPREAD_TOO_HIGH")

    has_candidate = (
        evaluation.metadata.get("candidate_id") is not None
        or evaluation.metadata.get("setup_id") is not None
    )
    required_confirm = 0
    if has_candidate:
        if evaluation.action == DecisionAction.TRADE:
            required_confirm = min_confirm_trade
        elif evaluation.action == DecisionAction.SMALL:
            required_confirm = min_confirm_small
    if required_confirm > 0 and has_candidate:
        trigger_raw = evaluation.metadata.get("trigger_confirmations", evaluation.snapshot.get("trigger_confirmations"))
        if trigger_raw is None:
            missing_features.append("trigger_confirmations")
            gates["ExecutionGate"] = False
            reasons.append("EXEC_FAIL_MISSING_FEATURES")
        else:
            try:
                trigger_int = int(trigger_raw)
            except (TypeError, ValueError):
                trigger_int = -1
            if trigger_int < required_confirm:
                can_downgrade_to_small = (
                    evaluation.action == DecisionAction.TRADE
                    and min_confirm_small > 0
                    and trigger_int >= min_confirm_small
                )
                if can_downgrade_to_small:
                    evaluation.action = DecisionAction.SMALL
                    evaluation.metadata["confirmations_downgrade"] = {
                        "from": "TRADE",
                        "to": "SMALL",
                        "trigger_confirmations": trigger_int,
                        "required_trade": min_confirm_trade,
                        "required_small": min_confirm_small,
                    }
                    soft_reasons = evaluation.metadata.get("soft_reasons")
                    if not isinstance(soft_reasons, list):
                        soft_reasons = []
                    if "CONFIRMATIONS_DOWNGRADE_SMALL" not in soft_reasons:
                        soft_reasons.append("CONFIRMATIONS_DOWNGRADE_SMALL")
                    evaluation.metadata["soft_reasons"] = soft_reasons
                else:
                    gates["ExecutionGate"] = False
                    reasons.append("EXEC_FAIL_CONFIRMATIONS_LOW")

    if close_value is None:
        missing_features.append("close")
    if missing_features:
        evaluation.metadata["missing_features"] = list(dict.fromkeys(missing_features))

    evaluation.gates = gates
    if reasons:
        for key, value in gates.items():
            if not value:
                evaluation.gate_blocked = key
                break
    return gates, reasons


def _quality_gate_reasons(
    *,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    now: datetime,
    timezone_name: str,
) -> list[str]:
    _, reasons = _evaluate_hard_gates(
        route_params=route_params,
        evaluation=evaluation,
        now=now,
        timezone_name=timezone_name,
    )
    return reasons


def _missing_execution_features(
    *,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
) -> list[str]:
    missing: list[str] = []
    metadata = evaluation.metadata if isinstance(evaluation.metadata, dict) else {}
    snapshot = evaluation.snapshot if isinstance(evaluation.snapshot, dict) else {}

    atr_value = metadata.get("atr_m5", snapshot.get("atr_m5"))
    try:
        if atr_value is None or float(atr_value) <= 0:
            missing.append("atr_m5")
    except (TypeError, ValueError):
        missing.append("atr_m5")

    close_value = snapshot.get("close", metadata.get("close"))
    try:
        if close_value is None or float(close_value) != float(close_value):
            missing.append("close")
    except (TypeError, ValueError):
        missing.append("close")

    gates_cfg = route_params.get("quality_gates")
    min_confirm = 0
    if isinstance(gates_cfg, dict):
        try:
            min_confirm = max(0, int(gates_cfg.get("min_confirm", 0)))
        except (TypeError, ValueError):
            min_confirm = 0
    if min_confirm > 0 and metadata.get("trigger_confirmations") is None:
        missing.append("trigger_confirmations")

    deduped = list(dict.fromkeys(missing))
    if deduped:
        evaluation.metadata["missing_features"] = deduped
        evaluation.metadata["is_ready"] = False
    else:
        evaluation.metadata["is_ready"] = True
    return deduped


def _apply_wait_timeout_soft_mode(
    *,
    evaluation: StrategyEvaluation,
    config: AppConfig,
) -> StrategyEvaluation:
    if not bool(evaluation.metadata.get("wait_timeout_soft_mode")):
        return evaluation

    soft_penalty = max(0.0, float(config.backtest_tuning.wait_timeout_soft_penalty))
    if evaluation.score_total is not None and soft_penalty > 0:
        score_now = max(0.0, float(evaluation.score_total) - soft_penalty)
        evaluation.score_total = round(score_now, 2)
        evaluation.score_breakdown["penalty_wait_timeout_soft"] = -round(soft_penalty, 4)

    if evaluation.action == DecisionAction.TRADE:
        evaluation.action = DecisionAction.SMALL

    current_override = evaluation.metadata.get("risk_multiplier_override")
    try:
        current_value = float(current_override) if current_override is not None else 1.0
    except (TypeError, ValueError):
        current_value = 1.0
    timeout_small = max(0.01, min(1.0, float(config.backtest_tuning.wait_timeout_small_risk_multiplier)))
    evaluation.metadata["risk_multiplier_override"] = min(current_value, timeout_small)

    soft_reasons = evaluation.metadata.get("soft_reasons")
    if not isinstance(soft_reasons, list):
        soft_reasons = []
    if "WAIT_TIMEOUT_SOFT_MODE" not in soft_reasons:
        soft_reasons.append("WAIT_TIMEOUT_SOFT_MODE")
    evaluation.metadata["soft_reasons"] = soft_reasons
    return evaluation


def _apply_reaction_gate_with_timeout(
    *,
    strategy_key: str,
    bar_index: int,
    now: datetime,
    evaluation: StrategyEvaluation,
    wait_states: dict[str, _WaitGateState],
    variant: BacktestVariant,
    config: AppConfig,
    timeout_resets: Counter[str],
    wait_durations: dict[str, list[int]],
    reset_block_bar: dict[str, int],
    timeout_samples: list[_ReactionTimeoutSample],
) -> list[str]:
    setup_state = str(evaluation.metadata.get("setup_state", "READY")).upper()
    if setup_state == "WAIT_REACTION":
        wait_type = "REACTION"
        timeout_bars = int(config.backtest_tuning.wait_reaction_timeout_bars)
        base_reason = "GATE_REACTION_WAIT_REACTION"
        reset_reason = "REACTION_TIMEOUT_SOFT_REACTION"
    elif setup_state == "WAIT_MITIGATION":
        wait_type = "MITIGATION"
        timeout_bars = int(config.backtest_tuning.wait_mitigation_timeout_bars)
        base_reason = "GATE_REACTION_WAIT_MITIGATION"
        reset_reason = "REACTION_TIMEOUT_SOFT_MITIGATION"
    else:
        state = wait_states.pop(strategy_key, None)
        if state is not None and not state.timed_out_soft:
            wait_durations.setdefault(state.wait_type, []).append(max(0, bar_index - state.enter_bar_index))
        return []

    locked_bar = reset_block_bar.get(strategy_key)
    if locked_bar is not None and locked_bar < bar_index:
        reset_block_bar.pop(strategy_key, None)
        locked_bar = None
    if locked_bar is not None and bar_index <= locked_bar:
        evaluation.metadata["wait_soft_grace_active"] = True
        evaluation.metadata["wait_timeout_soft_mode"] = True
        evaluation.metadata["setup_state"] = "SOFT_READY"
        return []

    state = wait_states.get(strategy_key)
    if state is None or state.wait_type != wait_type:
        wait_states[strategy_key] = _WaitGateState(
            wait_type=wait_type,
            enter_bar_index=bar_index,
            enter_ts=now,
            enter_reason=base_reason,
        )
        evaluation.metadata["wait_enter_bar_index"] = bar_index
        evaluation.metadata["wait_type"] = wait_type
        evaluation.metadata["wait_enter_reason"] = base_reason
        return [base_reason]

    elapsed = max(0, bar_index - state.enter_bar_index)
    evaluation.metadata["wait_enter_bar_index"] = state.enter_bar_index
    evaluation.metadata["wait_type"] = state.wait_type
    evaluation.metadata["wait_enter_reason"] = state.enter_reason
    evaluation.metadata["wait_elapsed_bars"] = elapsed
    if state.timed_out_soft:
        soft_decay_bars = max(0, int(config.backtest_tuning.wait_timeout_soft_grace_bars))
        if elapsed > (timeout_bars + soft_decay_bars):
            wait_states.pop(strategy_key, None)
            if soft_decay_bars > 0:
                reset_block_bar[strategy_key] = bar_index + soft_decay_bars
            evaluation.metadata["wait_soft_decay_cleared"] = True
            evaluation.metadata["setup_state"] = "SOFT_READY"
            soft_reasons = evaluation.metadata.get("soft_reasons")
            if not isinstance(soft_reasons, list):
                soft_reasons = []
            clear_code = f"REACTION_SOFT_DECAY_CLEAR_{wait_type}"
            if clear_code not in soft_reasons:
                soft_reasons.append(clear_code)
            evaluation.metadata["soft_reasons"] = soft_reasons
            return []
        evaluation.metadata["wait_timeout_soft_mode"] = True
        evaluation.metadata["wait_timeout_type"] = wait_type
        soft_reasons = evaluation.metadata.get("soft_reasons")
        if not isinstance(soft_reasons, list):
            soft_reasons = []
        soft_code = f"REACTION_TIMEOUT_SOFT_{wait_type}"
        if soft_code not in soft_reasons:
            soft_reasons.append(soft_code)
        evaluation.metadata["soft_reasons"] = soft_reasons
        return []
    timeouts_enabled = bool(config.backtest_tuning.reaction_timeout_force_enable or variant.reaction_timeout_reset)
    if timeouts_enabled and elapsed > timeout_bars:
        state.timed_out_soft = True
        wait_states[strategy_key] = state
        wait_durations.setdefault(wait_type, []).append(elapsed)
        timeout_resets[wait_type] = int(timeout_resets.get(wait_type, 0)) + 1
        timeout_resets["REACTION_TIMEOUT_RESET"] = int(timeout_resets.get("REACTION_TIMEOUT_RESET", 0)) + 1
        evaluation.metadata["reaction_timeout_reset"] = False
        evaluation.metadata["reaction_timeout_bars"] = elapsed
        evaluation.metadata["wait_timeout_soft_mode"] = True
        evaluation.metadata["wait_timeout_type"] = wait_type
        soft_reasons = evaluation.metadata.get("soft_reasons")
        if not isinstance(soft_reasons, list):
            soft_reasons = []
        soft_code = f"REACTION_TIMEOUT_SOFT_{wait_type}"
        if soft_code not in soft_reasons:
            soft_reasons.append(soft_code)
        evaluation.metadata["soft_reasons"] = soft_reasons
        if len(timeout_samples) < 50:
            symbol, strategy = strategy_key.split(":", 1) if ":" in strategy_key else (strategy_key, "UNKNOWN")
            timeout_samples.append(
                _ReactionTimeoutSample(
                    ts_utc=now.isoformat(),
                    symbol=symbol,
                    strategy=strategy,
                    state=wait_type,
                    waited_bars=int(elapsed),
                    reason=reset_reason,
                )
            )
        return []
    hard_block_bars = max(0, int(config.backtest_tuning.wait_hard_block_bars))
    if elapsed > hard_block_bars:
        evaluation.metadata["wait_timeout_soft_mode"] = True
        evaluation.metadata["wait_timeout_type"] = wait_type
        evaluation.metadata["setup_state"] = "SOFT_READY"
        soft_reasons = evaluation.metadata.get("soft_reasons")
        if not isinstance(soft_reasons, list):
            soft_reasons = []
        progress_code = f"REACTION_WAIT_SOFT_{wait_type}"
        if progress_code not in soft_reasons:
            soft_reasons.append(progress_code)
        evaluation.metadata["soft_reasons"] = soft_reasons
        return []
    return [base_reason]


def _collect_execution_fail_sample(
    *,
    samples: list[_ExecutionFailSample],
    max_samples: int,
    ts: datetime,
    symbol: str,
    strategy: str,
    reason: str,
    evaluation: StrategyEvaluation,
) -> None:
    if len(samples) >= max_samples:
        return
    spread_ratio = evaluation.metadata.get("spread_ratio")
    atr_value = evaluation.metadata.get("atr_m5", evaluation.snapshot.get("atr_m5"))
    try:
        spread_ratio_float = float(spread_ratio) if spread_ratio is not None else None
    except (TypeError, ValueError):
        spread_ratio_float = None
    try:
        atr_float = float(atr_value) if atr_value is not None else None
    except (TypeError, ValueError):
        atr_float = None
    samples.append(
        _ExecutionFailSample(
            ts_utc=ts.isoformat(),
            symbol=symbol,
            strategy=strategy,
            reason=reason,
            spread_ratio=spread_ratio_float,
            atr_m5=atr_float,
            missing_features=[
                str(item)
                for item in evaluation.metadata.get("missing_features", [])
                if str(item).strip()
            ]
            if isinstance(evaluation.metadata.get("missing_features"), list)
            else [],
        )
    )


def _write_execution_fail_debug(path: Path | None, samples: list[_ExecutionFailSample]) -> None:
    if path is None or not samples:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = {
                "ts_utc": sample.ts_utc,
                "symbol": sample.symbol,
                "strategy": sample.strategy,
                "reason": sample.reason,
                "spread_ratio": sample.spread_ratio,
                "atr_m5": sample.atr_m5,
                "missing_features": sample.missing_features,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _collect_no_price_sample(
    *,
    samples: list[_NoPriceSample],
    max_samples: int,
    ts: datetime,
    symbol: str,
    strategy: str,
    evaluation: StrategyEvaluation,
    data_context: dict[str, Any] | None,
) -> None:
    if len(samples) >= max_samples:
        return
    ctx = data_context or {}
    snapshot = evaluation.snapshot if isinstance(evaluation.snapshot, dict) else {}
    metadata = evaluation.metadata if isinstance(evaluation.metadata, dict) else {}
    missing_fields: list[str] = []
    for field_name, value in (
        ("snapshot.spread", snapshot.get("spread", metadata.get("spread"))),
        ("snapshot.close", snapshot.get("close", metadata.get("close"))),
        ("metadata.atr_m5", metadata.get("atr_m5", snapshot.get("atr_m5"))),
    ):
        if value is None or (isinstance(value, float) and value != value):
            missing_fields.append(field_name)

    record = {
        "close": snapshot.get("close", metadata.get("close")),
        "spread": snapshot.get("spread", metadata.get("spread")),
        "atr_m5": metadata.get("atr_m5", snapshot.get("atr_m5")),
        "spread_ratio": metadata.get("spread_ratio"),
        "bid": metadata.get("bid"),
        "ask": metadata.get("ask"),
        "price_mode_requested": ctx.get("price_mode_requested"),
    }
    source_files_raw = ctx.get("source_files")
    source_files = source_files_raw if isinstance(source_files_raw, list) else []
    source_datasets_raw = ctx.get("source_datasets")
    source_datasets = source_datasets_raw if isinstance(source_datasets_raw, list) else []
    timeframe = str(ctx.get("timeframe") or "5m")
    price_mode = str(ctx.get("price_mode_requested") or metadata.get("price_mode") or "unknown")
    samples.append(
        _NoPriceSample(
            ts_utc=ts.isoformat(),
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            price_mode=price_mode,
            missing_fields=missing_fields,
            source_files=[str(item) for item in source_files],
            source_datasets=[str(item) for item in source_datasets],
            record=record,
        )
    )


def _write_no_price_debug(path: Path | None, samples: list[_NoPriceSample]) -> None:
    if path is None or not samples:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = {
                "ts_utc": sample.ts_utc,
                "symbol": sample.symbol,
                "timeframe": sample.timeframe,
                "strategy": sample.strategy,
                "price_mode": sample.price_mode,
                "missing_fields": sample.missing_fields,
                "source_files": sample.source_files,
                "source_datasets": sample.source_datasets,
                "record": sample.record,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _write_reaction_timeout_debug(path: Path | None, samples: list[_ReactionTimeoutSample]) -> None:
    if path is None or not samples:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = {
                "ts_utc": sample.ts_utc,
                "symbol": sample.symbol,
                "strategy": sample.strategy,
                "state": sample.state,
                "waited_bars": sample.waited_bars,
                "reason": sample.reason,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _apply_orderflow_small_soft_gate(
    *,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    orderflow_settings: dict[str, float] | None,
) -> StrategyEvaluation:
    if evaluation.action != DecisionAction.SMALL:
        return evaluation
    snapshot_raw = evaluation.metadata.get("orderflow_snapshot")
    if not isinstance(snapshot_raw, dict):
        return evaluation
    metrics = snapshot_raw.get("metrics")
    if not isinstance(metrics, dict):
        return evaluation

    try:
        confidence = float(snapshot_raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        chop_score = float(metrics.get("chop_score", 0.0))
    except (TypeError, ValueError):
        chop_score = 0.0

    conf_threshold = _orderflow_param(
        route_params=route_params,
        settings=orderflow_settings,
        key="small_soft_gate_confidence",
        default=0.75,
    )
    chop_threshold = _orderflow_param(
        route_params=route_params,
        settings=orderflow_settings,
        key="small_soft_gate_chop",
        default=0.75,
    )
    if confidence >= conf_threshold and chop_score >= chop_threshold:
        evaluation.action = DecisionAction.OBSERVE
        if "OF_SOFT_GATE_CHOP" not in evaluation.reasons_blocking:
            evaluation.reasons_blocking.append("OF_SOFT_GATE_CHOP")
        if "ORDERFLOW_CHOP_CLEARED" not in evaluation.would_enter_if:
            evaluation.would_enter_if.append("ORDERFLOW_CHOP_CLEARED")
        evaluation.gates.setdefault("OrderflowSoftGate", True)
        evaluation.gates["OrderflowSoftGate"] = False
        if evaluation.gate_blocked is None:
            evaluation.gate_blocked = "OrderflowSoftGate"
    return evaluation


def _risk_multiplier_for(
    *,
    evaluation: StrategyEvaluation,
    route_risk: dict[str, object],
    config: AppConfig,
) -> float:
    signal_override = evaluation.metadata.get("risk_multiplier_override")
    if signal_override is not None:
        return max(0.01, min(1.0, float(signal_override)))
    if evaluation.action == DecisionAction.SMALL:
        value = float(route_risk.get("small_risk_multiplier", config.decision_policy.small_risk_multiplier_default))
        return max(0.01, min(1.0, value))
    value = float(route_risk.get("trade_risk_multiplier", config.decision_policy.trade_risk_multiplier))
    return max(0.01, min(1.0, value))


def _append_live_placeholder(candles: list[Candle], timeframe_minutes: int) -> list[Candle]:
    if not candles:
        return []
    last = candles[-1]
    return candles + [_live_placeholder_from(last, timeframe_minutes)]


def _live_placeholder_from(last: Candle, timeframe_minutes: int) -> Candle:
    live_ts = datetime.fromtimestamp(last.timestamp.timestamp() + (timeframe_minutes * 60), tz=timezone.utc)
    return Candle(
        timestamp=live_ts,
        open=last.close,
        high=last.close,
        low=last.close,
        close=last.close,
        bid=last.bid,
        ask=last.ask,
        volume=0.0,
    )


def _action_priority(action: DecisionAction) -> int:
    if action == DecisionAction.TRADE:
        return 3
    if action == DecisionAction.SMALL:
        return 2
    if action == DecisionAction.MANAGE:
        return 1
    return 0


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


def _estimate_structure_target(
    *,
    side: str,
    entry: float,
    candles: list[Candle],
    lookback_bars: int,
) -> float | None:
    if not candles:
        return None
    recent = candles[-max(2, int(lookback_bars)) :]
    if side == "LONG":
        candidates = [float(candle.high) for candle in recent if float(candle.high) > entry]
        return max(candidates) if candidates else None
    candidates = [float(candle.low) for candle in recent if float(candle.low) < entry]
    return min(candidates) if candidates else None


def _normalize_tp_by_r(
    *,
    side: str,
    entry: float,
    stop: float,
    requested_tp: float,
    min_r: float,
    max_r: float,
) -> tuple[float, float]:
    risk = abs(entry - stop)
    if risk <= 0:
        return requested_tp, 0.0
    if side == "LONG":
        requested_r = (requested_tp - entry) / risk
    else:
        requested_r = (entry - requested_tp) / risk
    target_r = max(min_r, min(max_r, float(requested_r)))
    if side == "LONG":
        tp = entry + (risk * target_r)
    else:
        tp = entry - (risk * target_r)
    return tp, target_r


def _tp2_r_for_target_total_r(
    *,
    target_total_r: float,
    tp1_trigger_r: float,
    tp1_fraction: float,
    mode: str = "strict_tp_price",
) -> float:
    """
    Compute TP2 R so that with partial TP1 the total trade payoff stays on target.

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


def _expected_rr(
    *,
    side: str,
    entry: float,
    stop: float,
    target: float,
) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    if side == "LONG":
        reward = target - entry
    else:
        reward = entry - target
    if reward <= 0:
        return 0.0
    return reward / risk


def _manage_open_position(
    *,
    position: _OpenPosition,
    candle: Candle,
    candles_m5: list[Candle],
    index: int,
    spread: float = 0.0,
    slippage: float,
    fx_converter: FxConverter | None = None,
    instrument_currency: str = "USD",
    account_currency: str = "USD",
    fx_apply_to: set[str] | None = None,
) -> tuple[bool, bool]:
    risk_dist = max(0.0, float(position.initial_risk))
    if risk_dist <= 0:
        return False, False

    tp1_hit = False
    be_moved_now = False
    if not position.tp1_taken:
        if position.side == "LONG":
            tp1_level = position.entry + (risk_dist * position.tp1_trigger_r)
            reached_tp1 = candle.high >= tp1_level
            partial_fill = tp1_level - slippage
        else:
            tp1_level = position.entry - (risk_dist * position.tp1_trigger_r)
            reached_tp1 = candle.low <= tp1_level
            partial_fill = tp1_level + slippage
        if reached_tp1:
            close_size = position.size * position.tp1_fraction
            close_size = max(0.0, min(position.size, close_size))
            if close_size > 0:
                position.spread_cost_total += max(0.0, float(spread)) * 0.5 * close_size
                position.slippage_cost_total += abs(float(slippage)) * close_size
                if position.side == "LONG":
                    partial_pnl_instr = (partial_fill - position.entry) * close_size
                else:
                    partial_pnl_instr = (position.entry - partial_fill) * close_size
                partial_pnl, partial_fx_cost = _convert_cash_to_account(
                    amount=partial_pnl_instr,
                    category="pnl",
                    fx_converter=fx_converter,
                    instrument_currency=instrument_currency,
                    account_currency=account_currency,
                    fx_apply_to=(fx_apply_to or {"pnl", "swap", "commission"}),
                )
                position.realized_partial += partial_pnl
                position.fx_conversion_total += partial_fx_cost
                position.fx_cost_total += partial_fx_cost
                position.size -= close_size
                position.tp1_taken = True
                position.tp1_hit_index = int(index)
                tp1_hit = True

    if position.tp1_taken and not position.be_moved and position.tp1_hit_index is not None:
        elapsed_since_tp1 = int(index - position.tp1_hit_index)
        if elapsed_since_tp1 >= int(max(0, position.be_delay_bars_after_tp1)):
            if position.side == "LONG":
                confirm_ok = candle.close >= position.entry
                be_price = position.entry + (risk_dist * position.be_offset_r)
                if confirm_ok and be_price > position.stop:
                    position.stop = be_price
                    be_moved_now = True
            else:
                confirm_ok = candle.close <= position.entry
                be_price = position.entry - (risk_dist * position.be_offset_r)
                if confirm_ok and be_price < position.stop:
                    position.stop = be_price
                    be_moved_now = True
            if be_moved_now:
                position.be_moved = True

    if position.trailing_after_tp1 and position.tp1_taken and position.size > 0 and position.be_moved:
        window = max(2, int(position.trailing_window_bars))
        recent = candles_m5[max(0, index - window + 1) : index + 1]
        if recent:
            buffer_value = risk_dist * max(0.0, float(position.trailing_buffer_r))
            if position.side == "LONG":
                swing_low = min(float(item.low) for item in recent)
                trail_stop = swing_low - buffer_value
                if trail_stop > position.stop:
                    position.stop = trail_stop
            else:
                swing_high = max(float(item.high) for item in recent)
                trail_stop = swing_high + buffer_value
                if trail_stop < position.stop:
                    position.stop = trail_stop

    return tp1_hit, be_moved_now


def _gate_counts_from_blockers(blockers: Counter[str]) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in blockers.items()
        if key.startswith("GATE_")
        or key.startswith("DAILY_GATE_")
        or key.startswith("EXEC_FAIL_")
        or key.startswith("PIPELINE_NOT_READY")
    }


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

    equity = float(config.risk.equity)
    peak = equity
    max_drawdown = 0.0
    for trade in all_trades:
        equity += float(trade.pnl)
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

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
    account_currencies: Counter[str] = Counter()
    instrument_currencies: Counter[str] = Counter()
    fx_fee_rate_values: list[float] = []
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
        account_currencies.update([str(getattr(report, "account_currency", config.account_currency)).upper()])
        instrument_currencies.update([str(getattr(report, "instrument_currency", asset.instrument_currency or asset.currency)).upper()])
        fx_fee_rate_values.append(float(getattr(report, "fx_conversion_fee_rate_used", config.fx_conversion_fee_rate)))

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
        assumed_spread_used=round(sum(assumed_spread_values) / len(assumed_spread_values), 6) if assumed_spread_values else 0.0,
        spread_gate_adjustments=dict(spread_adjustments),
        fx_conversion_pct_used=round(sum(fx_conversion_pct_values) / len(fx_conversion_pct_values), 6) if fx_conversion_pct_values else 0.0,
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
        total_pnl_net=total_pnl,
        expectancy_net=expectancy,
        profit_factor_net=profit_factor,
        max_drawdown_net=max_drawdown,
        account_currency=account_currencies.most_common(1)[0][0] if account_currencies else str(config.account_currency).upper(),
        instrument_currency=instrument_currencies.most_common(1)[0][0]
        if instrument_currencies
        else str(asset.instrument_currency or asset.currency).upper(),
        fx_conversion_fee_rate_used=(
            round(sum(fx_fee_rate_values) / len(fx_fee_rate_values), 8) if fx_fee_rate_values else float(config.fx_conversion_fee_rate)
        ),
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


def run_backtest_multi_strategy(
    *,
    config: AppConfig,
    asset: AssetConfig,
    candles_m5: list[Candle],
    assumed_spread: float = 0.2,
    slippage_points: float = 0.0,
    slippage_atr_multiplier: float = 0.0,
    variant: BacktestVariant | None = None,
    execution_debug_path: str | Path | None = None,
    no_price_debug_path: str | Path | None = None,
    reaction_timeout_debug_path: str | Path | None = None,
    data_context: dict[str, Any] | None = None,
    trade_start_utc: datetime | None = None,
    flatten_at_chunk_end: bool = False,
    daily_gate: DailyGateProvider | None = None,
    daily_gate_prepared: bool = False,
) -> BacktestReport:
    variant_cfg = variant or BacktestVariant()
    debug_path = Path(execution_debug_path) if execution_debug_path is not None else None
    no_price_path = Path(no_price_debug_path) if no_price_debug_path is not None else None
    reaction_timeout_path = Path(reaction_timeout_debug_path) if reaction_timeout_debug_path is not None else None
    backtest_context: dict[str, Any] = dict(data_context or {})
    trade_start = trade_start_utc.astimezone(timezone.utc) if trade_start_utc is not None else None
    segment_id = str(backtest_context.get("segment_index", "1"))
    segment_start_index = 0
    segment_start_raw = backtest_context.get("segment_start_utc")
    if isinstance(segment_start_raw, str) and segment_start_raw:
        try:
            seg_ts = _parse_dt(segment_start_raw)
            for idx, candle in enumerate(candles_m5):
                if candle.timestamp >= seg_ts:
                    segment_start_index = idx
                    break
        except ValueError:
            segment_start_index = 0
    spread_mode = str(backtest_context.get("spread_mode", "REAL_BIDASK")).upper()
    assumed_spread_used = float(backtest_context.get("assumed_spread_used", assumed_spread))
    risk_engine = RiskEngine(config.risk)
    router = StrategyRouter(config)
    strategy_plugins: dict[str, StrategyPlugin] = {
        "INDEX_EXISTING": IndexExistingStrategy(config),
        "SCALP_ICT_PA": ScalpIctPriceActionStrategy(config),
        "ORB_H4_RETEST": OrbH4RetestStrategy(config),
        "TREND_PULLBACK_M15": TrendPullbackM15Strategy(config),
    }
    orderflow_provider = CompositeOrderflowProvider(
        default_mode=config.orderflow.default_mode,
        symbol_modes={symbol: "FULL" for symbol in config.orderflow.full_symbols},
    )
    orderflow_settings = {
        "trigger_bonus_max": float(config.orderflow.trigger_bonus_max),
        "execution_bonus_max": float(config.orderflow.execution_bonus_max),
        "divergence_penalty_min": float(config.orderflow.divergence_penalty_min),
        "divergence_penalty_max": float(config.orderflow.divergence_penalty_max),
        "small_soft_gate_confidence": float(config.orderflow.small_soft_gate_confidence),
        "small_soft_gate_chop": float(config.orderflow.small_soft_gate_chop),
    }
    orderflow_full_symbols = set(config.orderflow.full_symbols)
    orderflow_default_mode = config.orderflow.default_mode
    orderflow_default_window = int(config.orderflow.default_window)
    candidate_queue = CandidateQueue()
    daily_gate_mode = str(daily_gate.mode).lower() if daily_gate is not None else "off"
    if daily_gate is not None and daily_gate.enabled and not daily_gate_prepared:
        daily_gate.refresh_from_candles(candles_m5)

    candles_m15 = aggregate_candles(candles_m5, 15)
    candles_h1 = aggregate_candles(candles_m5, 60)
    atr_values = atr(candles_m5, config.indicators.atr_period)
    spread_bounds = _resolve_dynamic_spread_bounds(
        config=config,
        symbol=asset.epic,
        fallback_spread=assumed_spread_used,
    )
    dynamic_assumed_spread: list[float] | None = None
    if spread_mode == "ASSUMED_OHLC" and spread_bounds is not None:
        dynamic_assumed_spread = _build_dynamic_assumed_spread_series(
            candles_m5=candles_m5,
            atr_values=atr_values,
            min_spread=spread_bounds[0],
            max_spread=spread_bounds[1],
        )
        if dynamic_assumed_spread:
            assumed_spread_used = float(sum(dynamic_assumed_spread) / len(dynamic_assumed_spread))

    equity = config.risk.equity
    peak_equity = equity
    max_drawdown = 0.0
    trades: list[BacktestTrade] = []
    pending: _PendingOrder | None = None
    open_pos: _OpenPosition | None = None
    time_in_market_bars = 0
    count_be_moves = 0
    count_tp1_hits = 0
    decision_counts: Counter[str] = Counter()
    blockers: Counter[str] = Counter()
    signal_candidates = 0
    timeout_resets: Counter[str] = Counter()
    wait_states: dict[str, _WaitGateState] = {}
    wait_reset_block_bar: dict[str, int] = {}
    wait_durations: dict[str, list[int]] = {"REACTION": [], "MITIGATION": []}
    reaction_timeout_samples: list[_ReactionTimeoutSample] = []
    execution_fail_breakdown: Counter[str] = Counter()
    missing_feature_counts: Counter[str] = Counter()
    execution_fail_samples: list[_ExecutionFailSample] = []
    no_price_samples: list[_NoPriceSample] = []
    spread_gate_adjustments: Counter[str] = Counter()
    score_values: list[float] = []
    score_bins: Counter[str] = Counter()
    daily_gate_bias_bars: Counter[str] = Counter()
    daily_gate_bias_days: Counter[str] = Counter()
    seen_day_bias: dict[date, str] = {}
    blocked_by_gate_reasons: Counter[str] = Counter()
    blocked_by_gate = 0
    missing_feature_debug_logged = 0
    feature_warmup_bars = max(3, int(config.indicators.atr_period) + 2)
    trade_thr_base, small_min_base, small_max_base = _thresholds_for_variant(config, variant_cfg)
    timeouts_enabled = bool(config.backtest_tuning.reaction_timeout_force_enable or variant_cfg.reaction_timeout_reset)

    daily_trades: dict[str, int] = {}
    daily_pnl: dict[str, float] = {}

    m15_ptr = -1
    h1_ptr = -1
    last_h1_closed_ts: datetime | None = None
    last_m15_closed_ts: datetime | None = None
    swap_hour, swap_minute = _parse_swap_time_utc(config.backtest_tuning.overnight_swap_time_utc)
    long_swap_pct = float(config.backtest_tuning.overnight_swap_long_pct)
    short_swap_pct = float(config.backtest_tuning.overnight_swap_short_pct)
    account_currency = str(config.account_currency).strip().upper()
    instrument_currency = str(asset.instrument_currency or asset.currency).strip().upper()
    fx_apply_to = {str(item).strip().lower() for item in config.fx_apply_to}
    fx_converter: FxConverter | None = None
    if instrument_currency != account_currency:
        fx_converter = FxConverter(
            fee_rate=float(config.fx_conversion_fee_rate),
            fee_mode=str(config.fx_fee_mode),
            rate_source=str(config.fx_rate_source),
            static_rates=config.fx_static_rates,
        )
    rejected_by_reason: Counter[str] = Counter()
    orders_submitted = 0
    trades_filled = 0

    def _spread_for(index: int) -> float:
        candle = candles_m5[index]
        if candle.bid is not None and candle.ask is not None:
            return max(0.0, candle.ask - candle.bid)
        if dynamic_assumed_spread is not None and index < len(dynamic_assumed_spread):
            return max(0.0, float(dynamic_assumed_spread[index]))
        return max(0.0, assumed_spread)

    def _slippage_for(index: int) -> float:
        atr_term = 0.0
        if 0 <= index < len(atr_values):
            atr_val = atr_values[index]
            if atr_val is not None:
                atr_term = max(0.0, slippage_atr_multiplier * atr_val)
        return max(0.0, slippage_points) + atr_term

    def _cap_size_by_margin(entry_price: float, requested_size: float) -> float:
        if entry_price <= 0:
            return 0.0
        margin_requirement_pct = float(config.backtest_tuning.broker_margin_requirement_pct)
        leverage = float(config.backtest_tuning.broker_leverage)
        caps: list[float] = []
        if margin_requirement_pct > 0:
            caps.append((equity / (margin_requirement_pct / 100.0)) / entry_price)
        if leverage > 0:
            caps.append((equity * leverage) / entry_price)
        if not caps:
            return max(0.0, requested_size)
        max_size = max(0.0, min(caps))
        step = asset.size_step if asset.size_step > 0 else 0.01
        max_size = math.floor(max_size / step) * step
        if max_size < asset.min_size:
            return 0.0
        return min(max(0.0, requested_size), max_size)

    start_idx = max(config.indicators.ema_period_h1 + 10, 250)
    spread_window = max(1, int(config.spread_filter.window)) + 1
    spread_history_window: deque[float] = deque(maxlen=spread_window)
    slice_m5_live = _append_live_placeholder(candles_m5[: start_idx + 1], 5)
    slice_m15_live: list[Candle] = []
    slice_h1_live: list[Candle] = []

    for i in range(start_idx, len(candles_m5)):
        candle = candles_m5[i]
        if i > start_idx:
            slice_m5_live[-1] = candle
            slice_m5_live.append(_live_placeholder_from(candle, 5))
        spread_now = _spread_for(i)
        slippage_now = _slippage_for(i)
        spread_history_window.append(spread_now)
        spread_history = list(spread_history_window)
        day_key = candle.timestamp.date().isoformat()
        daily_trades.setdefault(day_key, 0)
        daily_pnl.setdefault(day_key, 0.0)
        gate_result = None
        if daily_gate is not None and daily_gate.enabled:
            gate_result = daily_gate.evaluate(
                ts=candle.timestamp,
                symbol=asset.epic,
                spread=spread_now,
            )
            gate_bias = str(gate_result.bias).upper()
            daily_gate_bias_bars[gate_bias] += 1
            gate_day = candle.timestamp.astimezone(timezone.utc).date()
            if gate_day not in seen_day_bias:
                seen_day_bias[gate_day] = gate_bias
                daily_gate_bias_days[gate_bias] += 1

        if timeouts_enabled and wait_states:
            for key, state in list(wait_states.items()):
                timeout_bars = (
                    int(config.backtest_tuning.wait_reaction_timeout_bars)
                    if state.wait_type == "REACTION"
                    else int(config.backtest_tuning.wait_mitigation_timeout_bars)
                )
                elapsed = max(0, i - state.enter_bar_index)
                if state.timed_out_soft:
                    soft_decay_bars = max(0, int(config.backtest_tuning.wait_timeout_soft_grace_bars))
                    if elapsed > (timeout_bars + soft_decay_bars):
                        wait_states.pop(key, None)
                        if soft_decay_bars > 0:
                            wait_reset_block_bar[key] = i + soft_decay_bars
                    continue
                if elapsed <= timeout_bars:
                    continue
                state.timed_out_soft = True
                wait_states[key] = state
                wait_durations.setdefault(state.wait_type, []).append(elapsed)
                timeout_resets[state.wait_type] = int(timeout_resets.get(state.wait_type, 0)) + 1
                timeout_resets["REACTION_TIMEOUT_RESET"] = int(timeout_resets.get("REACTION_TIMEOUT_RESET", 0)) + 1
                reason = (
                    "REACTION_TIMEOUT_SOFT_REACTION"
                    if state.wait_type == "REACTION"
                    else "REACTION_TIMEOUT_SOFT_MITIGATION"
                )
                if len(reaction_timeout_samples) < 50:
                    symbol, strategy = key.split(":", 1) if ":" in key else (key, "UNKNOWN")
                    reaction_timeout_samples.append(
                        _ReactionTimeoutSample(
                            ts_utc=candle.timestamp.isoformat(),
                            symbol=symbol,
                            strategy=strategy,
                            state=state.wait_type,
                            waited_bars=int(elapsed),
                            reason=reason,
                        )
                    )

        if trade_start is not None and candle.timestamp < trade_start:
            continue

        if pending is not None and i > pending.expiry_index:
            pending = None

        if pending is not None:
            touched = pending.entry >= candle.low and pending.entry <= candle.high
            if touched:
                slippage = slippage_now
                if pending.side == "LONG":
                    base_entry = candle.ask if candle.ask is not None else (candle.close + (spread_now * 0.5))
                    entry_fill = base_entry + slippage
                else:
                    base_entry = candle.bid if candle.bid is not None else (candle.close - (spread_now * 0.5))
                    entry_fill = base_entry - slippage
                initial_risk = abs(pending.entry - pending.stop)
                entry_spread_cost = max(0.0, float(spread_now)) * 0.5 * float(pending.size)
                entry_slippage_cost = abs(float(slippage)) * float(pending.size)
                open_pos = _OpenPosition(
                    side=pending.side,
                    entry=entry_fill,
                    stop=pending.stop,
                    tp=pending.tp,
                    size=pending.size,
                    opened_at=candle.timestamp,
                    initial_stop=pending.stop,
                    initial_risk=initial_risk,
                    max_loss_r_cap=float(config.backtest_tuning.max_loss_r_cap),
                    tp1_trigger_r=float(config.backtest_tuning.tp1_trigger_r),
                    tp1_fraction=float(config.backtest_tuning.tp1_fraction),
                    be_offset_r=float(config.backtest_tuning.be_offset_r),
                    be_delay_bars_after_tp1=int(config.backtest_tuning.be_delay_bars_after_tp1),
                    trailing_after_tp1=bool(config.backtest_tuning.trailing_after_tp1),
                    trailing_window_bars=int(config.backtest_tuning.trailing_swing_window_bars),
                    trailing_buffer_r=float(config.backtest_tuning.trailing_buffer_r),
                    next_swap_ts=_next_rollover_timestamp(
                        candle.timestamp,
                        hour=swap_hour,
                        minute=swap_minute,
                    ),
                    realized_partial=0.0,
                    spread_cost_total=entry_spread_cost,
                    slippage_cost_total=entry_slippage_cost,
                    reason_open=pending.reason_open,
                    score=pending.score,
                    gate_bias=pending.gate_bias,
                )
                trades_filled += 1
                pending = None

        if open_pos is not None:
            time_in_market_bars += 1
            _apply_overnight_swap_if_due(
                position=open_pos,
                candle_ts=candle.timestamp,
                swap_hour=swap_hour,
                swap_minute=swap_minute,
                long_swap_pct=long_swap_pct,
                short_swap_pct=short_swap_pct,
                fx_converter=fx_converter,
                instrument_currency=instrument_currency,
                account_currency=account_currency,
                fx_apply_to=fx_apply_to,
            )
            tp1_hit, be_moved_now = _manage_open_position(
                position=open_pos,
                candle=candle,
                candles_m5=candles_m5,
                index=i,
                spread=spread_now,
                slippage=slippage_now,
                fx_converter=fx_converter,
                instrument_currency=instrument_currency,
                account_currency=account_currency,
                fx_apply_to=fx_apply_to,
            )
            if tp1_hit:
                count_tp1_hits += 1
            if be_moved_now:
                count_be_moves += 1

            should_close, exit_price, reason = _calc_exit(
                open_pos,
                candle,
                assumed_spread=spread_now,
                slippage=slippage_now,
            )
            if should_close:
                close_size = float(open_pos.size)
                open_pos.spread_cost_total += max(0.0, float(spread_now)) * 0.5 * close_size
                open_pos.slippage_cost_total += abs(float(slippage_now)) * close_size
                if open_pos.side == "LONG":
                    remaining_pnl_instr = (exit_price - open_pos.entry) * close_size
                else:
                    remaining_pnl_instr = (open_pos.entry - exit_price) * close_size
                remaining_pnl, close_fx_cost = _convert_cash_to_account(
                    amount=remaining_pnl_instr,
                    category="pnl",
                    fx_converter=fx_converter,
                    instrument_currency=instrument_currency,
                    account_currency=account_currency,
                    fx_apply_to=fx_apply_to,
                )
                open_pos.fx_conversion_total += close_fx_cost
                open_pos.fx_cost_total += close_fx_cost
                total_pnl = open_pos.realized_partial + remaining_pnl
                equity += total_pnl
                daily_pnl[day_key] += total_pnl
                r_denom = risk_engine.per_trade_risk_amount(equity=equity)
                r_mult = (total_pnl / r_denom) if r_denom > 0 else 0.0
                fees_total = open_pos.swap_total + open_pos.fx_conversion_total
                trades.append(
                    BacktestTrade(
                        epic=asset.epic,
                        side=open_pos.side,
                        entry_time=open_pos.opened_at,
                        exit_time=candle.timestamp,
                        entry_price=open_pos.entry,
                        exit_price=exit_price,
                        size=open_pos.size,
                        pnl=total_pnl,
                        fees=fees_total,
                        r_multiple=r_mult,
                        reason=reason,
                        score=open_pos.score,
                        reason_open=open_pos.reason_open,
                        reason_close=reason,
                        gate_bias=open_pos.gate_bias,
                        spread_cost=open_pos.spread_cost_total,
                        slippage_cost=open_pos.slippage_cost_total,
                        commission_cost=open_pos.commission_total,
                        swap_cost=open_pos.swap_cost_total,
                        fx_cost=open_pos.fx_cost_total,
                    )
                )
                open_pos = None
                peak_equity = max(peak_equity, equity)
                drawdown = peak_equity - equity
                max_drawdown = max(max_drawdown, drawdown)

        if open_pos is not None or pending is not None:
            continue

        if daily_trades[day_key] >= risk_engine.effective_max_trades_per_day(equity=equity):
            blockers["RISK_MAX_TRADES_DAY"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue
        if risk_engine.should_turn_off_for_day(daily_pnl[day_key], equity=equity):
            blockers["RISK_DAILY_STOP"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        t = candle.timestamp
        prev_m15_ptr = m15_ptr
        prev_h1_ptr = h1_ptr
        while (m15_ptr + 1) < len(candles_m15) and candles_m15[m15_ptr + 1].timestamp <= t:
            m15_ptr += 1
        while (h1_ptr + 1) < len(candles_h1) and candles_h1[h1_ptr + 1].timestamp <= t:
            h1_ptr += 1
        if m15_ptr < 20 or h1_ptr < 50:
            blockers["PIPELINE_WARMUP"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        if m15_ptr != prev_m15_ptr:
            if not slice_m15_live:
                slice_m15_live = _append_live_placeholder(candles_m15[: m15_ptr + 1], 15)
            else:
                for idx_m15 in range(prev_m15_ptr + 1, m15_ptr + 1):
                    latest_m15 = candles_m15[idx_m15]
                    slice_m15_live[-1] = latest_m15
                    slice_m15_live.append(_live_placeholder_from(latest_m15, 15))
        if h1_ptr != prev_h1_ptr:
            if not slice_h1_live:
                slice_h1_live = _append_live_placeholder(candles_h1[: h1_ptr + 1], 60)
            else:
                for idx_h1 in range(prev_h1_ptr + 1, h1_ptr + 1):
                    latest_h1 = candles_h1[idx_h1]
                    slice_h1_live[-1] = latest_h1
                    slice_h1_live.append(_live_placeholder_from(latest_h1, 60))

        m15_closed_ts = candles_m15[m15_ptr].timestamp if m15_ptr >= 0 else None
        h1_closed_ts = candles_h1[h1_ptr].timestamp if h1_ptr >= 0 else None
        m15_new_close = m15_closed_ts is not None and m15_closed_ts != last_m15_closed_ts
        h1_new_close = h1_closed_ts is not None and h1_closed_ts != last_h1_closed_ts
        if m15_new_close:
            last_m15_closed_ts = m15_closed_ts
        if h1_new_close:
            last_h1_closed_ts = h1_closed_ts

        slice_m5 = slice_m5_live
        slice_m15 = slice_m15_live
        slice_h1 = slice_h1_live
        quote = None
        if candle.bid is not None and candle.ask is not None:
            quote = (candle.bid, candle.ask, spread_now)
        gate_required_side: str | None = None
        if daily_gate is not None and daily_gate.enabled and gate_result is not None:
            bias_now = str(gate_result.bias).upper()
            if bias_now in {"LONG", "SHORT", "FLAT"}:
                gate_required_side = bias_now

        routes = router.routes_for(asset.epic)
        best_outcome: StrategyOutcome | None = None
        best_route: StrategyRoute | None = None
        best_rank = float("-inf")
        for route in routes:
            strategy = strategy_plugins.get(route.strategy)
            if strategy is None:
                unknown = StrategyOutcome(
                    symbol=asset.epic,
                    strategy_name=route.strategy,
                    bias=BiasState(
                        symbol=asset.epic,
                        strategy_name=route.strategy,
                        direction="NEUTRAL",
                        timeframe=config.timeframes.m5,
                        updated_at=t,
                        metadata={},
                    ),
                    candidate=None,
                    evaluation=_default_observe_evaluation(symbol=asset.epic, reason=f"UNKNOWN_STRATEGY_{route.strategy}"),
                    order_request=None,
                    reason_codes=[f"UNKNOWN_STRATEGY_{route.strategy}"],
                    payload={},
                )
                rank = -1000.0
                if _is_better_outcome(current=unknown, current_rank=rank, best=best_outcome, best_rank=best_rank):
                    best_outcome = unknown
                    best_route = route
                    best_rank = rank
                continue

            bundle = StrategyDataBundle(
                symbol=asset.epic,
                now=t,
                candles_h1=slice_h1,
                candles_m15=slice_m15,
                candles_m5=slice_m5,
                spread=spread_now,
                spread_history=spread_history,
                news_blocked=False,
                entry_state="WAIT",
                h1_new_close=h1_new_close,
                m15_new_close=m15_new_close,
                m5_new_close=True,
                quote=quote,
                extra={
                    "minimal_tick_buffer": asset.minimal_tick_buffer,
                    "strategy_params": route.params,
                    "strategy_risk": route.risk,
                    "origin_strategy": route.strategy,
                    "suppress_missed_opportunity_logs": True,
                },
            )
            strategy.preprocess(asset.epic, bundle)
            bias = strategy.compute_bias(asset.epic, bundle)
            if gate_required_side == "FLAT":
                gate_reason = "DAILY_GATE_FLAT"
                gated_eval = _default_observe_evaluation(symbol=asset.epic, reason=gate_reason)
                gated_outcome = StrategyOutcome(
                    symbol=asset.epic,
                    strategy_name=route.strategy,
                    bias=bias,
                    candidate=None,
                    evaluation=gated_eval,
                    order_request=None,
                    reason_codes=[gate_reason],
                    payload={"score_total": gated_eval.score_total, "route_priority": route.priority},
                )
                rank = -850.0 + (route.priority * 0.01)
                if _is_better_outcome(current=gated_outcome, current_rank=rank, best=best_outcome, best_rank=best_rank):
                    best_outcome = gated_outcome
                    best_route = route
                    best_rank = rank
                continue
            if gate_required_side in {"LONG", "SHORT"}:
                bias_dir = str(getattr(bias, "direction", "NEUTRAL")).upper()
                side_mismatch = (gate_required_side == "LONG" and bias_dir == "SHORT") or (
                    gate_required_side == "SHORT" and bias_dir == "LONG"
                )
                if side_mismatch:
                    gate_reason = "DAILY_GATE_LONG_ONLY" if gate_required_side == "LONG" else "DAILY_GATE_SHORT_ONLY"
                    gated_eval = _default_observe_evaluation(symbol=asset.epic, reason=gate_reason)
                    gated_outcome = StrategyOutcome(
                        symbol=asset.epic,
                        strategy_name=route.strategy,
                        bias=bias,
                        candidate=None,
                        evaluation=gated_eval,
                        order_request=None,
                        reason_codes=[gate_reason],
                        payload={"score_total": gated_eval.score_total, "route_priority": route.priority},
                    )
                    rank = -840.0 + (route.priority * 0.01)
                    if _is_better_outcome(current=gated_outcome, current_rank=rank, best=best_outcome, best_rank=best_rank):
                        best_outcome = gated_outcome
                        best_route = route
                        best_rank = rank
                    continue
            raw_candidates = strategy.detect_candidates(asset.epic, bundle)
            candidates = candidate_queue.put_many(
                symbol=asset.epic,
                strategy=route.strategy,
                candidates=raw_candidates,
                now=t,
            )
            candidate, evaluation = _pick_best_candidate(
                strategy=strategy,
                symbol=asset.epic,
                candidates=candidates,
                data=bundle,
            )
            schedule_cfg = route.params.get("schedule")
            schedule_open = True
            if isinstance(schedule_cfg, dict):
                schedule_open = is_schedule_open(t, schedule_cfg, config.timezone)
            mode_override = _resolve_orderflow_mode(
                symbol=asset.epic,
                route_params=route.params,
                default_mode=orderflow_default_mode,
                full_symbols=orderflow_full_symbols,
            )
            route_of = route.params.get("orderflow")
            window = orderflow_default_window
            if isinstance(route_of, dict):
                try:
                    window = max(8, int(route_of.get("window", orderflow_default_window)))
                except (TypeError, ValueError):
                    window = orderflow_default_window
            atr_for_of = evaluation.metadata.get("atr_m5", evaluation.snapshot.get("atr_m5"))
            atr_for_of_float: float | None
            try:
                atr_for_of_float = float(atr_for_of) if atr_for_of is not None else None
            except (TypeError, ValueError):
                atr_for_of_float = None
            orderflow_snapshot = orderflow_provider.get_snapshot(
                symbol=asset.epic,
                tf=config.timeframes.m5,
                window=window,
                candles=slice_m5,
                spread=spread_now,
                quote=quote,
                atr_value=atr_for_of_float,
                extra=bundle.extra,
                mode_override=mode_override,
            )
            evaluation.snapshot.setdefault("spread", spread_now)
            evaluation.snapshot.setdefault("close", candle.close)
            evaluation.metadata["spread"] = spread_now
            evaluation.metadata["close"] = candle.close
            evaluation.metadata["price_mode"] = str(backtest_context.get("price_mode_requested") or "unknown")
            evaluation.metadata["timeframe"] = str(backtest_context.get("timeframe") or config.timeframes.m5)
            evaluation.metadata["spread_mode"] = spread_mode
            evaluation.metadata["assumed_spread_used"] = assumed_spread_used
            evaluation.metadata["data_context"] = backtest_context
            bars_since_segment_start = max(0, i - segment_start_index)
            evaluation.metadata["bars_since_segment_start"] = bars_since_segment_start
            atr_runtime = atr_values[i] if 0 <= i < len(atr_values) else None
            if bars_since_segment_start >= feature_warmup_bars and atr_runtime is not None:
                try:
                    atr_runtime_f = float(atr_runtime)
                except (TypeError, ValueError):
                    atr_runtime_f = None
                if atr_runtime_f is not None and atr_runtime_f > 0:
                    evaluation.metadata.setdefault("atr_m5", atr_runtime_f)
                    evaluation.snapshot.setdefault("atr_m5", atr_runtime_f)
            trigger_value = evaluation.metadata.get("trigger_confirmations")
            if trigger_value is None:
                alt_trigger = evaluation.metadata.get("confirmations")
                if alt_trigger is None:
                    alt_trigger = evaluation.snapshot.get("trigger_confirmations")
                try:
                    trigger_int = int(alt_trigger) if alt_trigger is not None else 0
                except (TypeError, ValueError):
                    trigger_int = 0
                evaluation.metadata["trigger_confirmations"] = max(0, trigger_int)
            if quote is not None:
                evaluation.metadata["quote"] = quote
                evaluation.metadata["bid"] = quote[0]
                evaluation.metadata["ask"] = quote[1]
            evaluation = _apply_soft_reason_penalties(
                evaluation=evaluation,
                config=config,
                route_params=route.params,
                enabled=variant_cfg.soft_reason_penalties,
            )
            evaluation = _compute_v2_score(
                strategy_name=route.strategy,
                bias=bias,
                route_params=route.params,
                config=config,
                evaluation=evaluation,
                news_blocked=False,
                schedule_open=schedule_open,
                orderflow_snapshot=orderflow_snapshot,
                setup_side=candidate.side if candidate is not None else None,
                orderflow_settings=orderflow_settings,
            )
            trade_thr, small_min, small_max, dynamic_reasons = _adjust_thresholds_dynamic(
                trade_threshold=trade_thr_base,
                small_min=small_min_base,
                small_max=small_max_base,
                route_params=route.params,
                evaluation=evaluation,
                config=config,
                enabled=variant_cfg.dynamic_threshold_bump,
            )
            if dynamic_reasons:
                evaluation.metadata["dynamic_threshold_reasons"] = dynamic_reasons
            _ = _normalize_action_for_score(
                evaluation=evaluation,
                config=config,
                trade_threshold=trade_thr,
                small_min=small_min,
                small_max=small_max,
            )
            if candidate is None or bars_since_segment_start < feature_warmup_bars:
                missing_features = []
                evaluation.metadata["is_ready"] = True
            else:
                missing_features = _missing_execution_features(
                    route_params=route.params,
                    evaluation=evaluation,
                )
                if missing_features and missing_feature_debug_logged < 10:
                    missing_feature_debug_logged += 1
                    LOGGER.info(
                        "Missing features | ts=%s segment=%s bars_since_segment_start=%d missing=%s",
                        t.isoformat(),
                        segment_id,
                        bars_since_segment_start,
                        ",".join(str(item) for item in missing_features),
                    )
            if missing_features:
                gate_reasons = ["PIPELINE_NOT_READY_MISSING_FEATURES"]
            else:
                gate_reasons = _quality_gate_reasons(
                    route_params=route.params,
                    evaluation=evaluation,
                    now=t,
                    timezone_name=config.timezone,
                )
            reaction_reasons = _apply_reaction_gate_with_timeout(
                strategy_key=f"{asset.epic}:{route.strategy}",
                bar_index=i,
                now=t,
                evaluation=evaluation,
                wait_states=wait_states,
                variant=variant_cfg,
                config=config,
                timeout_resets=timeout_resets,
                wait_durations=wait_durations,
                reset_block_bar=wait_reset_block_bar,
                timeout_samples=reaction_timeout_samples,
            )
            gate_reasons.extend(reaction_reasons)
            evaluation = _apply_wait_timeout_soft_mode(
                evaluation=evaluation,
                config=config,
            )
            if gate_reasons:
                for code in gate_reasons:
                    if code not in evaluation.reasons_blocking:
                        evaluation.reasons_blocking.append(code)
                evaluation.action = DecisionAction.OBSERVE
            evaluation = _apply_orderflow_small_soft_gate(
                route_params=route.params,
                evaluation=evaluation,
                orderflow_settings=orderflow_settings,
            )
            signal_request = (
                strategy.generate_order(asset.epic, evaluation, candidate, bundle)
                if candidate is not None and evaluation.action in {DecisionAction.TRADE, DecisionAction.SMALL}
                else None
            )
            outcome = StrategyOutcome(
                symbol=asset.epic,
                strategy_name=route.strategy,
                bias=bias,
                candidate=candidate,
                evaluation=evaluation,
                order_request=signal_request,
                reason_codes=list(dict.fromkeys(evaluation.reasons_blocking)),
                payload={"score_total": evaluation.score_total, "route_priority": route.priority},
            )
            soft_reasons = evaluation.metadata.get("soft_reasons")
            if isinstance(soft_reasons, list):
                for soft_reason in soft_reasons:
                    code = f"SOFT_REASON_{str(soft_reason).upper()}"
                    if code not in outcome.reason_codes:
                        outcome.reason_codes.append(code)
            rank = rank_score(evaluation) + (route.priority * 0.01)
            if _is_better_outcome(current=outcome, current_rank=rank, best=best_outcome, best_rank=best_rank):
                best_outcome = outcome
                best_route = route
                best_rank = rank

        if best_outcome is None or best_route is None:
            blockers["NO_ROUTE"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        if best_outcome.evaluation.score_total is not None:
            score_now = float(best_outcome.evaluation.score_total)
            score_values.append(score_now)
            if score_now >= trade_thr_base:
                score_bins["trade_bin"] += 1
            elif small_min_base <= score_now <= small_max_base:
                score_bins["small_bin"] += 1
            else:
                score_bins["observe_bin"] += 1

        meta = best_outcome.evaluation.metadata if isinstance(best_outcome.evaluation.metadata, dict) else {}
        if bool(meta.get("spread_gate_soft_penalty_applied")):
            spread_gate_adjustments["ASSUMED_OHLC_SOFT_PENALTY_APPLIED"] += 1
        if bool(meta.get("spread_gate_ohlc_hard_skipped")):
            spread_gate_adjustments["ASSUMED_OHLC_HARD_GATE_SKIPPED"] += 1

        if best_outcome.order_request is None:
            reasons = best_outcome.reason_codes or ["NO_SIGNAL"]
            if "PIPELINE_NOT_READY_MISSING_FEATURES" in reasons:
                missing_items = best_outcome.evaluation.metadata.get("missing_features")
                if isinstance(missing_items, list) and missing_items:
                    for item in missing_items:
                        key = str(item).strip() or "UNKNOWN"
                        missing_feature_counts[key] += 1
                else:
                    missing_feature_counts["UNKNOWN"] += 1
            for reason in reasons:
                blockers[reason] += 1
                if reason.startswith("EXEC_FAIL_"):
                    execution_fail_breakdown[reason] += 1
                    _collect_execution_fail_sample(
                        samples=execution_fail_samples,
                        max_samples=200,
                        ts=t,
                        symbol=asset.epic,
                        strategy=best_outcome.strategy_name,
                        reason=reason,
                        evaluation=best_outcome.evaluation,
                    )
                    if reason == "EXEC_FAIL_NO_PRICE":
                        _collect_no_price_sample(
                            samples=no_price_samples,
                            max_samples=50,
                            ts=t,
                            symbol=asset.epic,
                            strategy=best_outcome.strategy_name,
                            evaluation=best_outcome.evaluation,
                            data_context=backtest_context,
                        )
            decision_counts["NO_SIGNAL"] += 1
            continue

        order_request = best_outcome.order_request
        if daily_gate is not None and daily_gate.enabled and gate_result is not None:
            gate_reasons: list[str] = list(gate_result.reasons)
            gate_bias = str(gate_result.bias).upper()
            if gate_bias == "FLAT":
                gate_reasons.append("DAILY_GATE_FLAT")
            elif gate_bias == "LONG" and str(order_request.side).upper() != "LONG":
                gate_reasons.append("DAILY_GATE_LONG_ONLY")
            elif gate_bias == "SHORT" and str(order_request.side).upper() != "SHORT":
                gate_reasons.append("DAILY_GATE_SHORT_ONLY")
            if gate_result.allowed_strategies:
                allowed = {str(item).upper() for item in gate_result.allowed_strategies}
                if str(best_route.strategy).upper() not in allowed:
                    gate_reasons.append("DAILY_GATE_STRATEGY_BLOCKED")
            if gate_reasons:
                blocked_by_gate += 1
                for reason in list(dict.fromkeys(gate_reasons)):
                    blockers[reason] += 1
                    blocked_by_gate_reasons[reason] += 1
                decision_counts["NO_SIGNAL"] += 1
                continue

        risk_dist = abs(float(order_request.entry_price) - float(order_request.stop_price))
        if risk_dist <= 0:
            blockers["ORDER_INVALID_RISK"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        expected_rr_min = float(config.backtest_tuning.expected_rr_min)
        expected_rr_lookback = max(10, int(config.backtest_tuning.expected_rr_lookback_bars))
        structure_target = _estimate_structure_target(
            side=order_request.side,
            entry=float(order_request.entry_price),
            candles=candles_m5[max(0, i - expected_rr_lookback + 1) : i + 1],
            lookback_bars=expected_rr_lookback,
        )
        requested_tp = float(order_request.take_profit)
        if structure_target is None:
            rr_target = requested_tp
        elif order_request.side == "LONG":
            rr_target = max(requested_tp, float(structure_target))
        else:
            rr_target = min(requested_tp, float(structure_target))
        expected_rr_value = _expected_rr(
            side=order_request.side,
            entry=float(order_request.entry_price),
            stop=float(order_request.stop_price),
            target=rr_target,
        )
        signal_rr = getattr(order_request, "rr", None)
        try:
            if signal_rr is not None:
                expected_rr_value = max(expected_rr_value, float(signal_rr))
        except (TypeError, ValueError):
            pass
        best_outcome.evaluation.metadata["expected_rr"] = round(expected_rr_value, 4)
        if structure_target is not None:
            best_outcome.evaluation.metadata["expected_rr_target_structure"] = float(structure_target)
        if expected_rr_value < expected_rr_min:
            blockers["EXPECTED_RR_TOO_LOW"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        is_a_plus = bool(getattr(order_request, "a_plus", False))
        if is_a_plus:
            target_total_r = float(config.backtest_tuning.tp_target_a_plus_r)
            best_outcome.evaluation.metadata["tp_target_profile"] = "A_PLUS_3R"
        else:
            target_total_r = float(config.backtest_tuning.tp_target_min_r)
            best_outcome.evaluation.metadata["tp_target_profile"] = "STANDARD_2R"
        target_min_r = _tp2_r_for_target_total_r(
            target_total_r=target_total_r,
            tp1_trigger_r=float(config.backtest_tuning.tp1_trigger_r),
            tp1_fraction=float(config.backtest_tuning.tp1_fraction),
            mode=str(config.backtest_tuning.tp_profile_mode),
        )
        target_max_r = target_min_r
        best_outcome.evaluation.metadata["target_r_profile_total"] = round(target_total_r, 4)
        best_outcome.evaluation.metadata["target_r_tp2"] = round(target_min_r, 4)
        tp_source = rr_target
        normalized_tp, normalized_r = _normalize_tp_by_r(
            side=order_request.side,
            entry=float(order_request.entry_price),
            stop=float(order_request.stop_price),
            requested_tp=tp_source,
            min_r=target_min_r,
            max_r=target_max_r,
        )
        order_request.take_profit = normalized_tp
        best_outcome.evaluation.metadata["target_r_normalized"] = round(normalized_r, 4)

        signal_candidates += 1
        risk_multiplier = _risk_multiplier_for(
            evaluation=best_outcome.evaluation,
            route_risk=best_route.risk,
            config=config,
        )
        effective_risk_per_trade = risk_engine.effective_risk_per_trade(
            risk_multiplier=risk_multiplier,
            equity=equity,
        )
        risk_distance = abs(float(order_request.entry_price) - float(order_request.stop_price))
        if risk_distance <= 0:
            blockers["M5_INVALID_RISK_DISTANCE"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue
        max_risk_cash = float(equity) * float(effective_risk_per_trade)
        raw_size = max_risk_cash / risk_distance if risk_distance > 0 else 0.0
        feasibility = validate_order(
            raw_size=raw_size,
            entry_price=float(order_request.entry_price),
            stop_price=float(order_request.stop_price),
            take_profit=float(order_request.take_profit),
            min_size=float(asset.min_size),
            size_step=float(asset.size_step),
            max_risk_cash=max_risk_cash,
            equity=float(equity),
            open_positions_count=0,
            max_positions=int(config.risk.max_positions),
            spread=float(spread_now),
            spread_limit=(float(config.daily_gate.max_spread) if config.daily_gate.max_spread is not None else None),
            min_stop_distance=float(asset.minimal_tick_buffer),
            free_margin=float(equity),
            margin_requirement_pct=float(config.backtest_tuning.broker_margin_requirement_pct),
            max_leverage=float(config.backtest_tuning.broker_leverage),
            margin_safety_factor=1.0,
        )
        if not feasibility.ok:
            reject = feasibility.reason.value if feasibility.reason is not None else "UNKNOWN_REJECT"
            rejected_by_reason[reject] += 1
            blockers[reject] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue
        size = float(feasibility.details.get("rounded_size", 0.0))
        if size <= 0:
            blockers["SIZE_MARGIN_LIMIT"] += 1
            blockers["SIZE_INVALID"] += 1
            blockers["INSUFFICIENT_EQUITY"] += 1
            rejected_by_reason[RejectReason.SIZE_TOO_SMALL.value] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        pending = _PendingOrder(
            side=order_request.side,
            entry=order_request.entry_price,
            stop=order_request.stop_price,
            tp=order_request.take_profit,
            size=size,
            expiry_index=i + config.execution.limit_ttl_bars,
            created_at=t,
            reason_open=",".join(best_outcome.reason_codes) if best_outcome.reason_codes else "SIGNAL",
            score=best_outcome.evaluation.score_total,
            gate_bias=(str(gate_result.bias).upper() if gate_result is not None else None),
        )
        orders_submitted += 1
        daily_trades[day_key] += 1
        decision_counts[best_outcome.evaluation.action.value] += 1

    if candles_m5:
        last_index = len(candles_m5) - 1
        for state in wait_states.values():
            if state.timed_out_soft:
                continue
            wait_durations.setdefault(state.wait_type, []).append(max(0, last_index - state.enter_bar_index))

    _write_execution_fail_debug(debug_path, execution_fail_samples)
    _write_no_price_debug(no_price_path, no_price_samples)
    _write_reaction_timeout_debug(reaction_timeout_path, reaction_timeout_samples)

    wait_metrics: dict[str, float] = {}
    for wait_type, durations in wait_durations.items():
        if not durations:
            wait_metrics[f"{wait_type.lower()}_avg_bars"] = 0.0
            wait_metrics[f"{wait_type.lower()}_max_bars"] = 0.0
            continue
        wait_metrics[f"{wait_type.lower()}_avg_bars"] = round(sum(durations) / len(durations), 3)
        wait_metrics[f"{wait_type.lower()}_max_bars"] = float(max(durations))

    avg_score = round(sum(score_values) / len(score_values), 4) if score_values else None

    if flatten_at_chunk_end and open_pos is not None and candles_m5:
        last_candle = candles_m5[-1]
        spread_now = _spread_for(len(candles_m5) - 1)
        close_size = float(open_pos.size)
        open_pos.spread_cost_total += max(0.0, float(spread_now)) * 0.5 * close_size
        open_pos.slippage_cost_total += abs(float(_slippage_for(len(candles_m5) - 1))) * close_size
        if open_pos.side == "LONG":
            exit_price = last_candle.bid if last_candle.bid is not None else (last_candle.close - (spread_now * 0.5))
            remaining_pnl_instr = (exit_price - open_pos.entry) * close_size
        else:
            exit_price = last_candle.ask if last_candle.ask is not None else (last_candle.close + (spread_now * 0.5))
            remaining_pnl_instr = (open_pos.entry - exit_price) * close_size
        remaining_pnl, close_fx_cost = _convert_cash_to_account(
            amount=remaining_pnl_instr,
            category="pnl",
            fx_converter=fx_converter,
            instrument_currency=instrument_currency,
            account_currency=account_currency,
            fx_apply_to=fx_apply_to,
        )
        open_pos.fx_conversion_total += close_fx_cost
        open_pos.fx_cost_total += close_fx_cost
        total_pnl = open_pos.realized_partial + remaining_pnl
        equity += total_pnl
        r_denom = risk_engine.per_trade_risk_amount(equity=equity)
        r_mult = (total_pnl / r_denom) if r_denom > 0 else 0.0
        fees_total = open_pos.swap_total + open_pos.fx_conversion_total
        trades.append(
            BacktestTrade(
                epic=asset.epic,
                side=open_pos.side,
                entry_time=open_pos.opened_at,
                exit_time=last_candle.timestamp,
                entry_price=open_pos.entry,
                exit_price=exit_price,
                size=close_size,
                pnl=total_pnl,
                fees=fees_total,
                r_multiple=r_mult,
                reason="FORCED_CHUNK_END",
                forced_exit=True,
                score=open_pos.score,
                reason_open=open_pos.reason_open,
                reason_close="FORCED_CHUNK_END",
                gate_bias=open_pos.gate_bias,
                spread_cost=open_pos.spread_cost_total,
                slippage_cost=open_pos.slippage_cost_total,
                commission_cost=open_pos.commission_total,
                swap_cost=open_pos.swap_cost_total,
                fx_cost=open_pos.fx_cost_total,
            )
        )
        peak_equity = max(peak_equity, equity)
        drawdown = peak_equity - equity
        max_drawdown = max(max_drawdown, drawdown)
        open_pos = None

    wins = sum(1 for trade in trades if trade.pnl > 0)
    losses = sum(1 for trade in trades if trade.pnl <= 0)
    total_pnl = sum(trade.pnl for trade in trades)
    trade_count = len(trades)
    expectancy = (total_pnl / trade_count) if trade_count else 0.0
    avg_r = (sum(trade.r_multiple for trade in trades) / trade_count) if trade_count else 0.0
    win_rate = (wins / trade_count) if trade_count else 0.0
    avg_win, avg_loss, payoff_ratio, profit_factor = _trade_quality_metrics(trades)
    avg_win_r, avg_loss_r, payoff_r = _trade_r_quality_metrics(trades)
    exit_reason_distribution = _exit_reason_distribution(trades)
    spread_cost_sum = sum(float(trade.spread_cost) for trade in trades)
    slippage_cost_sum = sum(float(trade.slippage_cost) for trade in trades)
    commission_cost_sum = sum(float(trade.commission_cost) for trade in trades)
    swap_cost_sum = sum(float(trade.swap_cost) for trade in trades)
    fx_cost_sum = sum(float(trade.fx_cost) for trade in trades)

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
        equity_end=config.risk.equity + total_pnl,
        trade_log=trades,
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
        gate_block_counts=_gate_counts_from_blockers(blockers),
        missing_feature_counts=dict(missing_feature_counts),
        decision_counts=dict(decision_counts),
        signal_candidates=signal_candidates,
        wait_timeout_resets={
            "reaction": int(timeout_resets.get("REACTION", 0)),
            "mitigation": int(timeout_resets.get("MITIGATION", 0)),
            "total": int(timeout_resets.get("REACTION_TIMEOUT_RESET", 0)),
        },
        wait_metrics=wait_metrics,
        execution_fail_breakdown=dict(execution_fail_breakdown),
        avg_score=avg_score,
        score_bins=dict(score_bins),
        spread_mode=spread_mode,
        assumed_spread_used=float(assumed_spread_used),
        spread_gate_adjustments=dict(spread_gate_adjustments),
        fx_conversion_pct_used=float(config.backtest_tuning.fx_conversion_pct),
        daily_gate_mode=daily_gate_mode,
        daily_gate_bias_bars=dict(daily_gate_bias_bars),
        daily_gate_bias_days=dict(daily_gate_bias_days),
        blocked_by_gate=blocked_by_gate,
        blocked_by_gate_reasons=dict(blocked_by_gate_reasons),
        per_bias_trade_metrics=_per_bias_trade_metrics(trades),
        orders_submitted=orders_submitted,
        trades_filled=trades_filled,
        rejected_by_reason=dict(rejected_by_reason),
        spread_cost_sum=spread_cost_sum,
        slippage_cost_sum=slippage_cost_sum,
        commission_cost_sum=commission_cost_sum,
        swap_cost_sum=swap_cost_sum,
        fx_cost_sum=fx_cost_sum,
        total_pnl_net=total_pnl,
        expectancy_net=expectancy,
        profit_factor_net=profit_factor,
        max_drawdown_net=max_drawdown,
        account_currency=account_currency,
        instrument_currency=instrument_currency,
        fx_conversion_fee_rate_used=float(config.fx_conversion_fee_rate),
    )


def run_backtest(
    *,
    config: AppConfig,
    asset: AssetConfig,
    candles_m5: list[Candle],
    assumed_spread: float = 0.2,
    slippage_points: float = 0.0,
    slippage_atr_multiplier: float = 0.0,
    daily_gate: DailyGateProvider | None = None,
) -> BacktestReport:
    strategy_engine = StrategyEngine(config)
    risk_engine = RiskEngine(config.risk)
    candles_m15 = aggregate_candles(candles_m5, 15)
    candles_h1 = aggregate_candles(candles_m5, 60)
    atr_values = atr(candles_m5, config.indicators.atr_period)
    daily_gate_mode = str(daily_gate.mode).lower() if daily_gate is not None else "off"
    if daily_gate is not None and daily_gate.enabled:
        daily_gate.refresh_from_candles(candles_m5)
    spread_mode = "REAL_BIDASK" if any(c.bid is not None and c.ask is not None for c in candles_m5) else "ASSUMED_OHLC"
    assumed_spread_used = float(max(0.0, assumed_spread))
    spread_bounds = _resolve_dynamic_spread_bounds(
        config=config,
        symbol=asset.epic,
        fallback_spread=assumed_spread_used,
    )
    dynamic_assumed_spread: list[float] | None = None
    if spread_mode == "ASSUMED_OHLC" and spread_bounds is not None:
        dynamic_assumed_spread = _build_dynamic_assumed_spread_series(
            candles_m5=candles_m5,
            atr_values=atr_values,
            min_spread=spread_bounds[0],
            max_spread=spread_bounds[1],
        )
        if dynamic_assumed_spread:
            assumed_spread_used = float(sum(dynamic_assumed_spread) / len(dynamic_assumed_spread))

    equity = config.risk.equity
    peak_equity = equity
    max_drawdown = 0.0
    trades: list[BacktestTrade] = []
    pending: _PendingOrder | None = None
    open_pos: _OpenPosition | None = None
    time_in_market_bars = 0
    count_be_moves = 0
    count_tp1_hits = 0
    daily_gate_bias_bars: Counter[str] = Counter()
    daily_gate_bias_days: Counter[str] = Counter()
    seen_day_bias: dict[date, str] = {}
    blocked_by_gate_reasons: Counter[str] = Counter()
    blocked_by_gate = 0

    daily_trades: dict[str, int] = {}
    daily_pnl: dict[str, float] = {}
    swap_hour, swap_minute = _parse_swap_time_utc(config.backtest_tuning.overnight_swap_time_utc)
    long_swap_pct = float(config.backtest_tuning.overnight_swap_long_pct)
    short_swap_pct = float(config.backtest_tuning.overnight_swap_short_pct)
    account_currency = str(config.account_currency).strip().upper()
    instrument_currency = str(asset.instrument_currency or asset.currency).strip().upper()
    fx_apply_to = {str(item).strip().lower() for item in config.fx_apply_to}
    fx_converter: FxConverter | None = None
    if instrument_currency != account_currency:
        fx_converter = FxConverter(
            fee_rate=float(config.fx_conversion_fee_rate),
            fee_mode=str(config.fx_fee_mode),
            rate_source=str(config.fx_rate_source),
            static_rates=config.fx_static_rates,
        )
    rejected_by_reason: Counter[str] = Counter()
    orders_submitted = 0
    trades_filled = 0
    m15_ptr = -1
    h1_ptr = -1

    def _spread_for(index: int) -> float:
        candle = candles_m5[index]
        if candle.bid is not None and candle.ask is not None:
            return max(0.0, candle.ask - candle.bid)
        if dynamic_assumed_spread is not None and index < len(dynamic_assumed_spread):
            return max(0.0, float(dynamic_assumed_spread[index]))
        return max(0.0, assumed_spread)

    def _slippage_for(index: int) -> float:
        atr_term = 0.0
        if 0 <= index < len(atr_values):
            atr_val = atr_values[index]
            if atr_val is not None:
                atr_term = max(0.0, slippage_atr_multiplier * atr_val)
        return max(0.0, slippage_points) + atr_term

    def _cap_size_by_margin(entry_price: float, requested_size: float) -> float:
        if entry_price <= 0:
            return 0.0
        margin_requirement_pct = float(config.backtest_tuning.broker_margin_requirement_pct)
        leverage = float(config.backtest_tuning.broker_leverage)
        caps: list[float] = []
        if margin_requirement_pct > 0:
            caps.append((equity / (margin_requirement_pct / 100.0)) / entry_price)
        if leverage > 0:
            caps.append((equity * leverage) / entry_price)
        if not caps:
            return max(0.0, requested_size)
        max_size = max(0.0, min(caps))
        step = asset.size_step if asset.size_step > 0 else 0.01
        max_size = math.floor(max_size / step) * step
        if max_size < asset.min_size:
            return 0.0
        return min(max(0.0, requested_size), max_size)

    start_idx = max(config.indicators.ema_period_h1 + 10, 250)
    spread_window = max(1, int(config.spread_filter.window)) + 1
    spread_history_window: deque[float] = deque(maxlen=spread_window)
    slice_m5: list[Candle] = list(candles_m5[:start_idx])
    slice_m15: list[Candle] = []
    slice_h1: list[Candle] = []
    for i in range(start_idx, len(candles_m5)):
        candle = candles_m5[i]
        spread_now = _spread_for(i)
        slippage_now = _slippage_for(i)
        spread_history_window.append(spread_now)
        spread_history = list(spread_history_window)
        slice_m5.append(candle)
        day_key = candle.timestamp.date().isoformat()
        daily_trades.setdefault(day_key, 0)
        daily_pnl.setdefault(day_key, 0.0)
        gate_result = None
        if daily_gate is not None and daily_gate.enabled:
            gate_result = daily_gate.evaluate(
                ts=candle.timestamp,
                symbol=asset.epic,
                spread=spread_now,
            )
            gate_bias = str(gate_result.bias).upper()
            daily_gate_bias_bars[gate_bias] += 1
            gate_day = candle.timestamp.astimezone(timezone.utc).date()
            if gate_day not in seen_day_bias:
                seen_day_bias[gate_day] = gate_bias
                daily_gate_bias_days[gate_bias] += 1

        if pending is not None and i > pending.expiry_index:
            pending = None

        if pending is not None:
            touched = pending.entry >= candle.low and pending.entry <= candle.high
            if touched:
                slippage = slippage_now
                if pending.side == "LONG":
                    base_entry = candle.ask if candle.ask is not None else (candle.close + (spread_now * 0.5))
                    entry_fill = base_entry + slippage
                else:
                    base_entry = candle.bid if candle.bid is not None else (candle.close - (spread_now * 0.5))
                    entry_fill = base_entry - slippage
                initial_risk = abs(pending.entry - pending.stop)
                entry_spread_cost = max(0.0, float(spread_now)) * 0.5 * float(pending.size)
                entry_slippage_cost = abs(float(slippage)) * float(pending.size)
                open_pos = _OpenPosition(
                    side=pending.side,
                    entry=entry_fill,
                    stop=pending.stop,
                    tp=pending.tp,
                    size=pending.size,
                    opened_at=candle.timestamp,
                    initial_stop=pending.stop,
                    initial_risk=initial_risk,
                    max_loss_r_cap=float(config.backtest_tuning.max_loss_r_cap),
                    tp1_trigger_r=float(config.backtest_tuning.tp1_trigger_r),
                    tp1_fraction=float(config.backtest_tuning.tp1_fraction),
                    be_offset_r=float(config.backtest_tuning.be_offset_r),
                    be_delay_bars_after_tp1=int(config.backtest_tuning.be_delay_bars_after_tp1),
                    trailing_after_tp1=bool(config.backtest_tuning.trailing_after_tp1),
                    trailing_window_bars=int(config.backtest_tuning.trailing_swing_window_bars),
                    trailing_buffer_r=float(config.backtest_tuning.trailing_buffer_r),
                    next_swap_ts=_next_rollover_timestamp(
                        candle.timestamp,
                        hour=swap_hour,
                        minute=swap_minute,
                    ),
                    realized_partial=0.0,
                    spread_cost_total=entry_spread_cost,
                    slippage_cost_total=entry_slippage_cost,
                    reason_open=pending.reason_open,
                    score=pending.score,
                    gate_bias=pending.gate_bias,
                )
                trades_filled += 1
                pending = None

        if open_pos is not None:
            time_in_market_bars += 1
            _apply_overnight_swap_if_due(
                position=open_pos,
                candle_ts=candle.timestamp,
                swap_hour=swap_hour,
                swap_minute=swap_minute,
                long_swap_pct=long_swap_pct,
                short_swap_pct=short_swap_pct,
                fx_converter=fx_converter,
                instrument_currency=instrument_currency,
                account_currency=account_currency,
                fx_apply_to=fx_apply_to,
            )

            should_close, exit_price, reason = _calc_exit(
                open_pos,
                candle,
                assumed_spread=spread_now,
                slippage=slippage_now,
            )
            if should_close:
                close_size = float(open_pos.size)
                open_pos.spread_cost_total += max(0.0, float(spread_now)) * 0.5 * close_size
                open_pos.slippage_cost_total += abs(float(slippage_now)) * close_size
                if open_pos.side == "LONG":
                    remaining_pnl_instr = (exit_price - open_pos.entry) * close_size
                else:
                    remaining_pnl_instr = (open_pos.entry - exit_price) * close_size
                remaining_pnl, close_fx_cost = _convert_cash_to_account(
                    amount=remaining_pnl_instr,
                    category="pnl",
                    fx_converter=fx_converter,
                    instrument_currency=instrument_currency,
                    account_currency=account_currency,
                    fx_apply_to=fx_apply_to,
                )
                open_pos.fx_conversion_total += close_fx_cost
                open_pos.fx_cost_total += close_fx_cost
                total_pnl = open_pos.realized_partial + remaining_pnl
                equity += total_pnl
                daily_pnl[day_key] += total_pnl
                r_denom = risk_engine.per_trade_risk_amount(equity=equity)
                r_mult = (total_pnl / r_denom) if r_denom > 0 else 0.0
                fees_total = open_pos.swap_total + open_pos.fx_conversion_total
                trades.append(
                    BacktestTrade(
                        epic=asset.epic,
                        side=open_pos.side,
                        entry_time=open_pos.opened_at,
                        exit_time=candle.timestamp,
                        entry_price=open_pos.entry,
                        exit_price=exit_price,
                        size=open_pos.size,
                        pnl=total_pnl,
                        fees=fees_total,
                        r_multiple=r_mult,
                        reason=reason,
                        score=open_pos.score,
                        reason_open=open_pos.reason_open,
                        reason_close=reason,
                        gate_bias=open_pos.gate_bias,
                        spread_cost=open_pos.spread_cost_total,
                        slippage_cost=open_pos.slippage_cost_total,
                        commission_cost=open_pos.commission_total,
                        swap_cost=open_pos.swap_cost_total,
                        fx_cost=open_pos.fx_cost_total,
                    )
                )
                open_pos = None
                peak_equity = max(peak_equity, equity)
                drawdown = peak_equity - equity
                max_drawdown = max(max_drawdown, drawdown)

        if open_pos is not None or pending is not None:
            continue

        if daily_trades[day_key] >= risk_engine.effective_max_trades_per_day(equity=equity):
            continue
        if risk_engine.should_turn_off_for_day(daily_pnl[day_key], equity=equity):
            continue

        t = candle.timestamp
        while (m15_ptr + 1) < len(candles_m15) and candles_m15[m15_ptr + 1].timestamp <= t:
            m15_ptr += 1
            slice_m15.append(candles_m15[m15_ptr])
        while (h1_ptr + 1) < len(candles_h1) and candles_h1[h1_ptr + 1].timestamp <= t:
            h1_ptr += 1
            slice_h1.append(candles_h1[h1_ptr])
        if m15_ptr <= 20 or h1_ptr <= 50:
            continue

        decision = strategy_engine.evaluate(
            epic=asset.epic,
            minimal_tick_buffer=asset.minimal_tick_buffer,
            candles_h1=slice_h1,
            candles_m15=slice_m15,
            candles_m5=slice_m5,
            current_spread=spread_now,
            spread_history=spread_history,
            news_blocked=False,
        )
        if decision.signal is None:
            continue
        signal = decision.signal
        if daily_gate is not None and daily_gate.enabled and gate_result is not None:
            gate_reasons: list[str] = list(gate_result.reasons)
            gate_bias = str(gate_result.bias).upper()
            if gate_bias == "FLAT":
                gate_reasons.append("DAILY_GATE_FLAT")
            elif gate_bias == "LONG" and str(signal.side).upper() != "LONG":
                gate_reasons.append("DAILY_GATE_LONG_ONLY")
            elif gate_bias == "SHORT" and str(signal.side).upper() != "SHORT":
                gate_reasons.append("DAILY_GATE_SHORT_ONLY")
            if gate_reasons:
                blocked_by_gate += 1
                for reason in list(dict.fromkeys(gate_reasons)):
                    blocked_by_gate_reasons[reason] += 1
                continue
        risk_dist = abs(float(signal.entry_price) - float(signal.stop_price))
        if risk_dist <= 0:
            continue

        expected_rr_lookback = max(10, int(config.backtest_tuning.expected_rr_lookback_bars))
        structure_target = _estimate_structure_target(
            side=signal.side,
            entry=float(signal.entry_price),
            candles=candles_m5[max(0, i - expected_rr_lookback + 1) : i + 1],
            lookback_bars=expected_rr_lookback,
        )
        requested_tp = float(signal.take_profit)
        if structure_target is None:
            rr_target = requested_tp
        elif signal.side == "LONG":
            rr_target = max(requested_tp, float(structure_target))
        else:
            rr_target = min(requested_tp, float(structure_target))
        expected_rr_value = _expected_rr(
            side=signal.side,
            entry=float(signal.entry_price),
            stop=float(signal.stop_price),
            target=rr_target,
        )
        try:
            expected_rr_value = max(expected_rr_value, float(signal.rr))
        except (TypeError, ValueError):
            pass
        if expected_rr_value < float(config.backtest_tuning.expected_rr_min):
            continue
        is_a_plus = bool(getattr(signal, "a_plus", False))
        if is_a_plus:
            target_total_r = float(config.backtest_tuning.tp_target_a_plus_r)
        else:
            target_total_r = float(config.backtest_tuning.tp_target_min_r)
        target_min_r = _tp2_r_for_target_total_r(
            target_total_r=target_total_r,
            tp1_trigger_r=float(config.backtest_tuning.tp1_trigger_r),
            tp1_fraction=float(config.backtest_tuning.tp1_fraction),
            mode=str(config.backtest_tuning.tp_profile_mode),
        )
        target_max_r = target_min_r
        tp_source = rr_target
        normalized_tp, _ = _normalize_tp_by_r(
            side=signal.side,
            entry=float(signal.entry_price),
            stop=float(signal.stop_price),
            requested_tp=tp_source,
            min_r=target_min_r,
            max_r=target_max_r,
        )
        signal.take_profit = normalized_tp

        effective_risk_per_trade = risk_engine.effective_risk_per_trade(
            risk_multiplier=1.0,
            equity=equity,
        )
        max_risk_cash = float(equity) * float(effective_risk_per_trade)
        raw_size = max_risk_cash / risk_dist if risk_dist > 0 else 0.0
        feasibility = validate_order(
            raw_size=raw_size,
            entry_price=float(signal.entry_price),
            stop_price=float(signal.stop_price),
            take_profit=float(signal.take_profit),
            min_size=float(asset.min_size),
            size_step=float(asset.size_step),
            max_risk_cash=max_risk_cash,
            equity=float(equity),
            open_positions_count=0,
            max_positions=int(config.risk.max_positions),
            spread=float(spread_now),
            spread_limit=(float(config.daily_gate.max_spread) if config.daily_gate.max_spread is not None else None),
            min_stop_distance=float(asset.minimal_tick_buffer),
            free_margin=float(equity),
            margin_requirement_pct=float(config.backtest_tuning.broker_margin_requirement_pct),
            max_leverage=float(config.backtest_tuning.broker_leverage),
            margin_safety_factor=1.0,
        )
        if not feasibility.ok:
            reject = feasibility.reason.value if feasibility.reason is not None else "UNKNOWN_REJECT"
            rejected_by_reason[reject] += 1
            continue
        size = float(feasibility.details.get("rounded_size", 0.0))
        if size <= 0:
            rejected_by_reason[RejectReason.SIZE_TOO_SMALL.value] += 1
            continue

        pending = _PendingOrder(
            side=signal.side,
            entry=signal.entry_price,
            stop=signal.stop_price,
            tp=signal.take_profit,
            size=size,
            expiry_index=i + config.execution.limit_ttl_bars,
            created_at=t,
            reason_open=",".join(decision.reason_codes) if decision.reason_codes else "SIGNAL",
            score=None,
            gate_bias=(str(gate_result.bias).upper() if gate_result is not None else None),
        )
        orders_submitted += 1
        daily_trades[day_key] += 1

    wins = sum(1 for trade in trades if trade.pnl > 0)
    losses = sum(1 for trade in trades if trade.pnl <= 0)
    total_pnl = sum(trade.pnl for trade in trades)
    trade_count = len(trades)
    expectancy = (total_pnl / trade_count) if trade_count else 0.0
    avg_r = (sum(trade.r_multiple for trade in trades) / trade_count) if trade_count else 0.0
    win_rate = (wins / trade_count) if trade_count else 0.0
    avg_win, avg_loss, payoff_ratio, profit_factor = _trade_quality_metrics(trades)
    avg_win_r, avg_loss_r, payoff_r = _trade_r_quality_metrics(trades)
    exit_reason_distribution = _exit_reason_distribution(trades)
    spread_cost_sum = sum(float(trade.spread_cost) for trade in trades)
    slippage_cost_sum = sum(float(trade.slippage_cost) for trade in trades)
    commission_cost_sum = sum(float(trade.commission_cost) for trade in trades)
    swap_cost_sum = sum(float(trade.swap_cost) for trade in trades)
    fx_cost_sum = sum(float(trade.fx_cost) for trade in trades)

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
        equity_end=config.risk.equity + total_pnl,
        trade_log=trades,
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
        spread_mode=spread_mode,
        assumed_spread_used=float(assumed_spread_used),
        fx_conversion_pct_used=float(config.backtest_tuning.fx_conversion_pct),
        daily_gate_mode=daily_gate_mode,
        daily_gate_bias_bars=dict(daily_gate_bias_bars),
        daily_gate_bias_days=dict(daily_gate_bias_days),
        blocked_by_gate=blocked_by_gate,
        blocked_by_gate_reasons=dict(blocked_by_gate_reasons),
        per_bias_trade_metrics=_per_bias_trade_metrics(trades),
        orders_submitted=orders_submitted,
        trades_filled=trades_filled,
        rejected_by_reason=dict(rejected_by_reason),
        spread_cost_sum=spread_cost_sum,
        slippage_cost_sum=slippage_cost_sum,
        commission_cost_sum=commission_cost_sum,
        swap_cost_sum=swap_cost_sum,
        fx_cost_sum=fx_cost_sum,
        total_pnl_net=total_pnl,
        expectancy_net=expectancy,
        profit_factor_net=profit_factor,
        max_drawdown_net=max_drawdown,
        account_currency=account_currency,
        instrument_currency=instrument_currency,
        fx_conversion_fee_rate_used=float(config.fx_conversion_fee_rate),
    )


def run_backtest_from_csv(
    *,
    config: AppConfig,
    asset: AssetConfig,
    csv_path: str | Path,
    assumed_spread: float = 0.2,
    slippage_points: float = 0.0,
    slippage_atr_multiplier: float = 0.0,
    daily_gate: DailyGateProvider | None = None,
) -> BacktestReport:
    candles = load_candles_csv(csv_path)
    return run_backtest(
        config=config,
        asset=asset,
        candles_m5=candles,
        assumed_spread=assumed_spread,
        slippage_points=slippage_points,
        slippage_atr_multiplier=slippage_atr_multiplier,
        daily_gate=daily_gate,
    )


def run_walk_forward_from_csv(
    *,
    config: AppConfig,
    asset: AssetConfig,
    csv_path: str | Path,
    wf_splits: int = 4,
    assumed_spread: float = 0.2,
    slippage_points: float = 0.0,
    slippage_atr_multiplier: float = 0.0,
    daily_gate: DailyGateProvider | None = None,
) -> WalkForwardReport:
    candles = load_candles_csv(csv_path)
    return run_walk_forward(
        config=config,
        asset=asset,
        candles_m5=candles,
        wf_splits=wf_splits,
        assumed_spread=assumed_spread,
        slippage_points=slippage_points,
        slippage_atr_multiplier=slippage_atr_multiplier,
        daily_gate=daily_gate,
    )


def run_walk_forward(
    *,
    config: AppConfig,
    asset: AssetConfig,
    candles_m5: list[Candle],
    wf_splits: int = 4,
    assumed_spread: float = 0.2,
    slippage_points: float = 0.0,
    slippage_atr_multiplier: float = 0.0,
    daily_gate: DailyGateProvider | None = None,
) -> WalkForwardReport:
    if wf_splits < 2:
        wf_splits = 2
    chunk = len(candles_m5) // wf_splits
    if chunk < 260:
        raise ValueError("Not enough candles for walk-forward splits")

    reports: list[BacktestReport] = []
    for split in range(wf_splits):
        start = split * chunk
        end = (split + 1) * chunk if split < (wf_splits - 1) else len(candles_m5)
        part = candles_m5[start:end]
        if len(part) < 260:
            continue
        reports.append(
            run_backtest(
                config=config,
                asset=asset,
                candles_m5=part,
                assumed_spread=assumed_spread,
                slippage_points=slippage_points,
                slippage_atr_multiplier=slippage_atr_multiplier,
                daily_gate=daily_gate,
            )
        )
    if not reports:
        raise ValueError("No valid walk-forward splits produced")

    aggregate = aggregate_backtest_reports(
        config=config,
        asset=asset,
        reports=reports,
    )
    return WalkForwardReport(epic=asset.epic, splits=reports, aggregate=aggregate)


def run_walk_forward_multi_strategy(
    *,
    config: AppConfig,
    asset: AssetConfig,
    candles_m5: list[Candle],
    wf_splits: int = 4,
    assumed_spread: float = 0.2,
    slippage_points: float = 0.0,
    slippage_atr_multiplier: float = 0.0,
    variant: BacktestVariant | None = None,
    execution_debug_path: str | Path | None = None,
    no_price_debug_path: str | Path | None = None,
    reaction_timeout_debug_path: str | Path | None = None,
    data_context: dict[str, Any] | None = None,
    daily_gate: DailyGateProvider | None = None,
) -> WalkForwardReport:
    if wf_splits < 2:
        wf_splits = 2
    chunk = len(candles_m5) // wf_splits
    if chunk < 260:
        raise ValueError("Not enough candles for walk-forward splits")

    reports: list[BacktestReport] = []
    for split in range(wf_splits):
        start = split * chunk
        end = (split + 1) * chunk if split < (wf_splits - 1) else len(candles_m5)
        part = candles_m5[start:end]
        if len(part) < 260:
            continue
        reports.append(
            run_backtest_multi_strategy(
                config=config,
                asset=asset,
                candles_m5=part,
                assumed_spread=assumed_spread,
                slippage_points=slippage_points,
                slippage_atr_multiplier=slippage_atr_multiplier,
                variant=variant,
                execution_debug_path=execution_debug_path,
                no_price_debug_path=no_price_debug_path,
                reaction_timeout_debug_path=reaction_timeout_debug_path,
                data_context=data_context,
                daily_gate=daily_gate,
            )
        )
    if not reports:
        raise ValueError("No valid walk-forward splits produced")

    aggregate = aggregate_backtest_reports(
        config=config,
        asset=asset,
        reports=reports,
    )
    return WalkForwardReport(epic=asset.epic, splits=reports, aggregate=aggregate)
