from __future__ import annotations

from collections import Counter
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bot.config import AppConfig, AssetConfig
from bot.data.candles import Candle
from bot.execution.sizing import position_size_from_risk
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
    trade_log: list[BacktestTrade]
    top_blockers: dict[str, int] = field(default_factory=dict)
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
            "signal_candidates": self.signal_candidates,
            "decision_counts": self.decision_counts,
            "top_blockers": self.top_blockers,
            "wait_timeout_resets": self.wait_timeout_resets,
            "wait_metrics": self.wait_metrics,
            "execution_fail_breakdown": self.execution_fail_breakdown,
            "avg_score": self.avg_score,
            "count_score_bins": self.score_bins,
            "spread_mode": self.spread_mode,
            "assumed_spread_used": self.assumed_spread_used,
            "spread_gate_adjustments": self.spread_gate_adjustments,
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


@dataclass(slots=True)
class _OpenPosition:
    side: str
    entry: float
    stop: float
    tp: float
    size: float
    opened_at: datetime
    be_moved: bool = False
    realized_partial: float = 0.0


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


@dataclass(slots=True)
class _ExecutionFailSample:
    ts_utc: str
    symbol: str
    strategy: str
    reason: str
    spread_ratio: float | None
    atr_m5: float | None


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
    if position.side == "LONG":
        base_exit = candle.bid if candle.bid is not None else (candle.close - (assumed_spread * 0.5))
        fill_price = base_exit - slippage
    else:
        base_exit = candle.ask if candle.ask is not None else (candle.close + (assumed_spread * 0.5))
        fill_price = base_exit + slippage
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


def _soft_reason_penalty_map(config: AppConfig) -> dict[str, float]:
    tuning = config.backtest_tuning
    return {
        "ORB_NO_RETEST": tuning.penalty_orb_no_retest,
        "ORB_CONFIRMATIONS_LOW": tuning.penalty_orb_confirm_low,
        "SCALP_NO_DISPLACEMENT": tuning.penalty_scalp_no_displacement,
        "SCALP_NO_MSS": tuning.penalty_scalp_no_mss,
        "SCALP_NO_FVG": tuning.penalty_scalp_no_fvg,
    }


def _apply_soft_reason_penalties(
    *,
    evaluation: StrategyEvaluation,
    config: AppConfig,
    enabled: bool,
) -> StrategyEvaluation:
    if not enabled:
        return evaluation
    penalties = _soft_reason_penalty_map(config)
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
    else:
        max_spread_ratio = 0.15
    spread_ratio = evaluation.metadata.get("spread_ratio")
    atr_value = evaluation.metadata.get("atr_m5", evaluation.snapshot.get("atr_m5"))
    spread_value = evaluation.snapshot.get("spread", evaluation.metadata.get("spread"))
    spread_mode = str(evaluation.metadata.get("spread_mode", "REAL_BIDASK")).upper()
    if spread_value is None:
        quote = evaluation.metadata.get("quote")
        if isinstance(quote, tuple) and len(quote) >= 3:
            spread_value = quote[2]
    close_value = evaluation.snapshot.get("close", evaluation.metadata.get("close"))
    if atr_value is None:
        gates["ExecutionGate"] = False
        reasons.append("EXEC_FAIL_MISSING_FEATURES")
    else:
        try:
            if float(atr_value) <= 0:
                gates["ExecutionGate"] = False
                reasons.append("EXEC_FAIL_INVALID_ATR")
        except (TypeError, ValueError):
            gates["ExecutionGate"] = False
            reasons.append("EXEC_FAIL_INVALID_ATR")
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
        reset_reason = "GATE_REACTION_TIMEOUT_RESET_REACTION"
    elif setup_state == "WAIT_MITIGATION":
        wait_type = "MITIGATION"
        timeout_bars = int(config.backtest_tuning.wait_mitigation_timeout_bars)
        base_reason = "GATE_REACTION_WAIT_MITIGATION"
        reset_reason = "GATE_REACTION_TIMEOUT_RESET_MITIGATION"
    else:
        state = wait_states.pop(strategy_key, None)
        if state is not None:
            wait_durations.setdefault(state.wait_type, []).append(max(0, bar_index - state.enter_bar_index))
        return []

    locked_bar = reset_block_bar.get(strategy_key)
    if locked_bar is not None and locked_bar < bar_index:
        reset_block_bar.pop(strategy_key, None)
        locked_bar = None
    if locked_bar is not None and locked_bar == bar_index:
        evaluation.metadata["wait_blocked_same_bar_after_reset"] = True
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
    timeouts_enabled = bool(config.backtest_tuning.reaction_timeout_force_enable or variant.reaction_timeout_reset)
    if timeouts_enabled and elapsed > timeout_bars:
        wait_states.pop(strategy_key, None)
        reset_block_bar[strategy_key] = bar_index
        wait_durations.setdefault(wait_type, []).append(elapsed)
        timeout_resets[wait_type] = int(timeout_resets.get(wait_type, 0)) + 1
        timeout_resets["REACTION_TIMEOUT_RESET"] = int(timeout_resets.get("REACTION_TIMEOUT_RESET", 0)) + 1
        evaluation.metadata["setup_state"] = "IDLE"
        evaluation.metadata["reaction_timeout_reset"] = True
        evaluation.metadata["reaction_timeout_bars"] = elapsed
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
        return [reset_reason]
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
    live_ts = datetime.fromtimestamp(last.timestamp.timestamp() + (timeframe_minutes * 60), tz=timezone.utc)
    return candles + [
        Candle(
            timestamp=live_ts,
            open=last.close,
            high=last.close,
            low=last.close,
            close=last.close,
            bid=last.bid,
            ask=last.ask,
            volume=0.0,
        )
    ]


def _action_priority(action: DecisionAction) -> int:
    if action == DecisionAction.TRADE:
        return 3
    if action == DecisionAction.SMALL:
        return 2
    if action == DecisionAction.MANAGE:
        return 1
    return 0


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
) -> BacktestReport:
    variant_cfg = variant or BacktestVariant()
    debug_path = Path(execution_debug_path) if execution_debug_path is not None else None
    no_price_path = Path(no_price_debug_path) if no_price_debug_path is not None else None
    reaction_timeout_path = Path(reaction_timeout_debug_path) if reaction_timeout_debug_path is not None else None
    backtest_context: dict[str, Any] = dict(data_context or {})
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

    candles_m15 = aggregate_candles(candles_m5, 15)
    candles_h1 = aggregate_candles(candles_m5, 60)
    atr_values = atr(candles_m5, config.indicators.atr_period)

    equity = config.risk.equity
    peak_equity = equity
    max_drawdown = 0.0
    trades: list[BacktestTrade] = []
    pending: _PendingOrder | None = None
    open_pos: _OpenPosition | None = None
    time_in_market_bars = 0
    decision_counts: Counter[str] = Counter()
    blockers: Counter[str] = Counter()
    signal_candidates = 0
    timeout_resets: Counter[str] = Counter()
    wait_states: dict[str, _WaitGateState] = {}
    wait_reset_block_bar: dict[str, int] = {}
    wait_durations: dict[str, list[int]] = {"REACTION": [], "MITIGATION": []}
    reaction_timeout_samples: list[_ReactionTimeoutSample] = []
    execution_fail_breakdown: Counter[str] = Counter()
    execution_fail_samples: list[_ExecutionFailSample] = []
    no_price_samples: list[_NoPriceSample] = []
    spread_gate_adjustments: Counter[str] = Counter()
    score_values: list[float] = []
    score_bins: Counter[str] = Counter()
    trade_thr_base, small_min_base, small_max_base = _thresholds_for_variant(config, variant_cfg)
    timeouts_enabled = bool(config.backtest_tuning.reaction_timeout_force_enable or variant_cfg.reaction_timeout_reset)

    daily_trades: dict[str, int] = {}
    daily_pnl: dict[str, float] = {}

    m15_ptr = -1
    h1_ptr = -1
    last_h1_closed_ts: datetime | None = None
    last_m15_closed_ts: datetime | None = None

    def _spread_for(index: int) -> float:
        candle = candles_m5[index]
        if candle.bid is not None and candle.ask is not None:
            return max(0.0, candle.ask - candle.bid)
        return max(0.0, assumed_spread)

    def _slippage_for(index: int) -> float:
        atr_term = 0.0
        if 0 <= index < len(atr_values):
            atr_val = atr_values[index]
            if atr_val is not None:
                atr_term = max(0.0, slippage_atr_multiplier * atr_val)
        return max(0.0, slippage_points) + atr_term

    start_idx = max(config.indicators.ema_period_h1 + 10, 250)
    for i in range(start_idx, len(candles_m5)):
        candle = candles_m5[i]
        day_key = candle.timestamp.date().isoformat()
        daily_trades.setdefault(day_key, 0)
        daily_pnl.setdefault(day_key, 0.0)

        if timeouts_enabled and wait_states:
            for key, state in list(wait_states.items()):
                timeout_bars = (
                    int(config.backtest_tuning.wait_reaction_timeout_bars)
                    if state.wait_type == "REACTION"
                    else int(config.backtest_tuning.wait_mitigation_timeout_bars)
                )
                elapsed = max(0, i - state.enter_bar_index)
                if elapsed <= timeout_bars:
                    continue
                wait_states.pop(key, None)
                wait_reset_block_bar[key] = i
                wait_durations.setdefault(state.wait_type, []).append(elapsed)
                timeout_resets[state.wait_type] = int(timeout_resets.get(state.wait_type, 0)) + 1
                timeout_resets["REACTION_TIMEOUT_RESET"] = int(timeout_resets.get("REACTION_TIMEOUT_RESET", 0)) + 1
                reason = (
                    "GATE_REACTION_TIMEOUT_RESET_REACTION"
                    if state.wait_type == "REACTION"
                    else "GATE_REACTION_TIMEOUT_RESET_MITIGATION"
                )
                blockers[reason] += 1
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

        if pending is not None and i > pending.expiry_index:
            pending = None

        if pending is not None:
            touched = pending.entry >= candle.low and pending.entry <= candle.high
            if touched:
                slippage = _slippage_for(i)
                if pending.side == "LONG":
                    base_entry = candle.ask if candle.ask is not None else (candle.close + (_spread_for(i) * 0.5))
                    entry_fill = base_entry + slippage
                else:
                    base_entry = candle.bid if candle.bid is not None else (candle.close - (_spread_for(i) * 0.5))
                    entry_fill = base_entry - slippage
                open_pos = _OpenPosition(
                    side=pending.side,
                    entry=entry_fill,
                    stop=pending.stop,
                    tp=pending.tp,
                    size=pending.size,
                    opened_at=candle.timestamp,
                )
                pending = None

        if open_pos is not None:
            time_in_market_bars += 1
            risk_dist = abs(open_pos.entry - open_pos.stop)
            if risk_dist > 0 and not open_pos.be_moved:
                one_r = open_pos.entry + risk_dist if open_pos.side == "LONG" else open_pos.entry - risk_dist
                reached_1r = candle.high >= one_r if open_pos.side == "LONG" else candle.low <= one_r
                if reached_1r:
                    half = open_pos.size * 0.5
                    open_pos.be_moved = True
                    open_pos.stop = open_pos.entry
                    open_pos.size = open_pos.size - half
                    open_pos.realized_partial += half * risk_dist

            should_close, exit_price, reason = _calc_exit(
                open_pos,
                candle,
                assumed_spread=_spread_for(i),
                slippage=_slippage_for(i),
            )
            if should_close:
                if open_pos.side == "LONG":
                    remaining_pnl = (exit_price - open_pos.entry) * open_pos.size
                else:
                    remaining_pnl = (open_pos.entry - exit_price) * open_pos.size
                total_pnl = open_pos.realized_partial + remaining_pnl
                equity += total_pnl
                daily_pnl[day_key] += total_pnl
                r_denom = risk_engine.per_trade_risk_amount(equity=config.risk.equity)
                r_mult = (total_pnl / r_denom) if r_denom > 0 else 0.0
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
                        r_multiple=r_mult,
                        reason=reason,
                    )
                )
                open_pos = None
                peak_equity = max(peak_equity, equity)
                drawdown = peak_equity - equity
                max_drawdown = max(max_drawdown, drawdown)

        if open_pos is not None or pending is not None:
            continue

        if daily_trades[day_key] >= config.risk.max_trades_per_day:
            blockers["RISK_MAX_TRADES_DAY"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue
        if risk_engine.should_turn_off_for_day(daily_pnl[day_key], equity=config.risk.equity):
            blockers["RISK_DAILY_STOP"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        t = candle.timestamp
        while (m15_ptr + 1) < len(candles_m15) and candles_m15[m15_ptr + 1].timestamp <= t:
            m15_ptr += 1
        while (h1_ptr + 1) < len(candles_h1) and candles_h1[h1_ptr + 1].timestamp <= t:
            h1_ptr += 1
        if m15_ptr < 20 or h1_ptr < 50:
            blockers["PIPELINE_WARMUP"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        m15_closed_ts = candles_m15[m15_ptr].timestamp if m15_ptr >= 0 else None
        h1_closed_ts = candles_h1[h1_ptr].timestamp if h1_ptr >= 0 else None
        m15_new_close = m15_closed_ts is not None and m15_closed_ts != last_m15_closed_ts
        h1_new_close = h1_closed_ts is not None and h1_closed_ts != last_h1_closed_ts
        if m15_new_close:
            last_m15_closed_ts = m15_closed_ts
        if h1_new_close:
            last_h1_closed_ts = h1_closed_ts

        slice_m5 = _append_live_placeholder(candles_m5[: i + 1], 5)
        slice_m15 = _append_live_placeholder(candles_m15[: m15_ptr + 1], 15)
        slice_h1 = _append_live_placeholder(candles_h1[: h1_ptr + 1], 60)
        spread_now = _spread_for(i)
        spread_history = [_spread_for(idx) for idx in range(max(0, i - config.spread_filter.window), i + 1)]
        quote = None
        if candle.bid is not None and candle.ask is not None:
            quote = (candle.bid, candle.ask, spread_now)

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
                },
            )
            strategy.preprocess(asset.epic, bundle)
            bias = strategy.compute_bias(asset.epic, bundle)
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
            if quote is not None:
                evaluation.metadata["quote"] = quote
                evaluation.metadata["bid"] = quote[0]
                evaluation.metadata["ask"] = quote[1]
            evaluation = _apply_soft_reason_penalties(
                evaluation=evaluation,
                config=config,
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
            for code in reaction_reasons:
                if code.startswith("GATE_REACTION_TIMEOUT_RESET_"):
                    blockers[code] += 1
            if gate_reasons:
                for code in gate_reasons:
                    if code not in evaluation.reasons_blocking:
                        evaluation.reasons_blocking.append(code)
                if not any(code.startswith("GATE_REACTION_TIMEOUT_RESET_") for code in gate_reasons):
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

        signal_candidates += 1
        risk_multiplier = _risk_multiplier_for(
            evaluation=best_outcome.evaluation,
            route_risk=best_route.risk,
            config=config,
        )
        size = position_size_from_risk(
            equity=config.risk.equity,
            risk_per_trade=config.risk.risk_per_trade * risk_multiplier,
            entry_price=best_outcome.order_request.entry_price,
            stop_price=best_outcome.order_request.stop_price,
            min_size=asset.min_size,
            size_step=asset.size_step,
        )
        if size <= 0:
            blockers["SIZE_INVALID"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        pending = _PendingOrder(
            side=best_outcome.order_request.side,
            entry=best_outcome.order_request.entry_price,
            stop=best_outcome.order_request.stop_price,
            tp=best_outcome.order_request.take_profit,
            size=size,
            expiry_index=i + config.execution.limit_ttl_bars,
            created_at=t,
        )
        daily_trades[day_key] += 1
        decision_counts[best_outcome.evaluation.action.value] += 1

    if candles_m5:
        last_index = len(candles_m5) - 1
        for state in wait_states.values():
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

    wins = sum(1 for trade in trades if trade.pnl > 0)
    losses = sum(1 for trade in trades if trade.pnl <= 0)
    total_pnl = sum(trade.pnl for trade in trades)
    trade_count = len(trades)
    expectancy = (total_pnl / trade_count) if trade_count else 0.0
    avg_r = (sum(trade.r_multiple for trade in trades) / trade_count) if trade_count else 0.0
    win_rate = (wins / trade_count) if trade_count else 0.0

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
        top_blockers=dict(blockers.most_common(10)),
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
    )


def run_backtest(
    *,
    config: AppConfig,
    asset: AssetConfig,
    candles_m5: list[Candle],
    assumed_spread: float = 0.2,
    slippage_points: float = 0.0,
    slippage_atr_multiplier: float = 0.0,
) -> BacktestReport:
    strategy_engine = StrategyEngine(config)
    risk_engine = RiskEngine(config.risk)
    candles_m15 = aggregate_candles(candles_m5, 15)
    candles_h1 = aggregate_candles(candles_m5, 60)
    atr_values = atr(candles_m5, config.indicators.atr_period)

    by_time_m15 = {c.timestamp: i for i, c in enumerate(candles_m15)}
    by_time_h1 = {c.timestamp: i for i, c in enumerate(candles_h1)}

    equity = config.risk.equity
    peak_equity = equity
    max_drawdown = 0.0
    trades: list[BacktestTrade] = []
    pending: _PendingOrder | None = None
    open_pos: _OpenPosition | None = None
    time_in_market_bars = 0

    daily_trades: dict[str, int] = {}
    daily_pnl: dict[str, float] = {}

    def _spread_for(index: int) -> float:
        candle = candles_m5[index]
        if candle.bid is not None and candle.ask is not None:
            return max(0.0, candle.ask - candle.bid)
        return max(0.0, assumed_spread)

    def _slippage_for(index: int) -> float:
        atr_term = 0.0
        if 0 <= index < len(atr_values):
            atr_val = atr_values[index]
            if atr_val is not None:
                atr_term = max(0.0, slippage_atr_multiplier * atr_val)
        return max(0.0, slippage_points) + atr_term

    start_idx = max(config.indicators.ema_period_h1 + 10, 250)
    for i in range(start_idx, len(candles_m5)):
        candle = candles_m5[i]
        day_key = candle.timestamp.date().isoformat()
        daily_trades.setdefault(day_key, 0)
        daily_pnl.setdefault(day_key, 0.0)

        if pending is not None and i > pending.expiry_index:
            pending = None

        if pending is not None:
            touched = pending.entry >= candle.low and pending.entry <= candle.high
            if touched:
                slippage = _slippage_for(i)
                if pending.side == "LONG":
                    base_entry = candle.ask if candle.ask is not None else (candle.close + (_spread_for(i) * 0.5))
                    entry_fill = base_entry + slippage
                else:
                    base_entry = candle.bid if candle.bid is not None else (candle.close - (_spread_for(i) * 0.5))
                    entry_fill = base_entry - slippage
                open_pos = _OpenPosition(
                    side=pending.side,
                    entry=entry_fill,
                    stop=pending.stop,
                    tp=pending.tp,
                    size=pending.size,
                    opened_at=candle.timestamp,
                )
                pending = None

        if open_pos is not None:
            time_in_market_bars += 1
            risk_dist = abs(open_pos.entry - open_pos.stop)
            if risk_dist > 0 and not open_pos.be_moved:
                one_r = open_pos.entry + risk_dist if open_pos.side == "LONG" else open_pos.entry - risk_dist
                reached_1r = candle.high >= one_r if open_pos.side == "LONG" else candle.low <= one_r
                if reached_1r:
                    half = open_pos.size * 0.5
                    open_pos.be_moved = True
                    open_pos.stop = open_pos.entry
                    open_pos.size = open_pos.size - half
                    open_pos.realized_partial += half * risk_dist

            should_close, exit_price, reason = _calc_exit(
                open_pos,
                candle,
                assumed_spread=_spread_for(i),
                slippage=_slippage_for(i),
            )
            if should_close:
                if open_pos.side == "LONG":
                    remaining_pnl = (exit_price - open_pos.entry) * open_pos.size
                else:
                    remaining_pnl = (open_pos.entry - exit_price) * open_pos.size
                total_pnl = open_pos.realized_partial + remaining_pnl
                equity += total_pnl
                daily_pnl[day_key] += total_pnl
                r_denom = risk_engine.per_trade_risk_amount(equity=config.risk.equity)
                r_mult = (total_pnl / r_denom) if r_denom > 0 else 0.0
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
                        r_multiple=r_mult,
                        reason=reason,
                    )
                )
                open_pos = None
                peak_equity = max(peak_equity, equity)
                drawdown = peak_equity - equity
                max_drawdown = max(max_drawdown, drawdown)

        if open_pos is not None or pending is not None:
            continue

        if daily_trades[day_key] >= config.risk.max_trades_per_day:
            continue
        if risk_engine.should_turn_off_for_day(daily_pnl[day_key], equity=config.risk.equity):
            continue

        t = candle.timestamp
        m15_idx = max([idx for ts, idx in by_time_m15.items() if ts <= t], default=-1)
        h1_idx = max([idx for ts, idx in by_time_h1.items() if ts <= t], default=-1)
        if m15_idx <= 20 or h1_idx <= 50:
            continue

        slice_m5 = candles_m5[: i + 1]
        slice_m15 = candles_m15[: m15_idx + 1]
        slice_h1 = candles_h1[: h1_idx + 1]
        decision = strategy_engine.evaluate(
            epic=asset.epic,
            minimal_tick_buffer=asset.minimal_tick_buffer,
            candles_h1=slice_h1,
            candles_m15=slice_m15,
            candles_m5=slice_m5,
            current_spread=_spread_for(i),
            spread_history=[_spread_for(idx) for idx in range(max(0, i - config.spread_filter.window), i + 1)],
            news_blocked=False,
        )
        if decision.signal is None:
            continue

        size = position_size_from_risk(
            equity=config.risk.equity,
            risk_per_trade=config.risk.risk_per_trade,
            entry_price=decision.signal.entry_price,
            stop_price=decision.signal.stop_price,
            min_size=asset.min_size,
            size_step=asset.size_step,
        )
        if size <= 0:
            continue

        pending = _PendingOrder(
            side=decision.signal.side,
            entry=decision.signal.entry_price,
            stop=decision.signal.stop_price,
            tp=decision.signal.take_profit,
            size=size,
            expiry_index=i + config.execution.limit_ttl_bars,
            created_at=t,
        )
        daily_trades[day_key] += 1

    wins = sum(1 for trade in trades if trade.pnl > 0)
    losses = sum(1 for trade in trades if trade.pnl <= 0)
    total_pnl = sum(trade.pnl for trade in trades)
    trade_count = len(trades)
    expectancy = (total_pnl / trade_count) if trade_count else 0.0
    avg_r = (sum(trade.r_multiple for trade in trades) / trade_count) if trade_count else 0.0
    win_rate = (wins / trade_count) if trade_count else 0.0

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
    )


def run_backtest_from_csv(
    *,
    config: AppConfig,
    asset: AssetConfig,
    csv_path: str | Path,
    assumed_spread: float = 0.2,
    slippage_points: float = 0.0,
    slippage_atr_multiplier: float = 0.0,
) -> BacktestReport:
    candles = load_candles_csv(csv_path)
    return run_backtest(
        config=config,
        asset=asset,
        candles_m5=candles,
        assumed_spread=assumed_spread,
        slippage_points=slippage_points,
        slippage_atr_multiplier=slippage_atr_multiplier,
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
            )
        )
    if not reports:
        raise ValueError("No valid walk-forward splits produced")

    total_trades = sum(report.trades for report in reports)
    total_wins = sum(report.wins for report in reports)
    total_losses = sum(report.losses for report in reports)
    total_pnl = sum(report.total_pnl for report in reports)
    total_time_in_market = sum(report.time_in_market_bars for report in reports)
    avg_r = (
        sum(report.avg_r * report.trades for report in reports) / total_trades
        if total_trades
        else 0.0
    )
    win_rate = (total_wins / total_trades) if total_trades else 0.0
    expectancy = (total_pnl / total_trades) if total_trades else 0.0
    aggregate = BacktestReport(
        epic=asset.epic,
        trades=total_trades,
        wins=total_wins,
        losses=total_losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        expectancy=expectancy,
        avg_r=avg_r,
        max_drawdown=max((report.max_drawdown for report in reports), default=0.0),
        time_in_market_bars=total_time_in_market,
        equity_end=config.risk.equity + total_pnl,
        trade_log=[],
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
            )
        )
    if not reports:
        raise ValueError("No valid walk-forward splits produced")

    total_trades = sum(report.trades for report in reports)
    total_wins = sum(report.wins for report in reports)
    total_losses = sum(report.losses for report in reports)
    total_pnl = sum(report.total_pnl for report in reports)
    total_time_in_market = sum(report.time_in_market_bars for report in reports)
    total_candidates = sum(report.signal_candidates for report in reports)
    decision_counts: Counter[str] = Counter()
    blockers: Counter[str] = Counter()
    execution_fail: Counter[str] = Counter()
    timeout_resets: Counter[str] = Counter()
    score_bins: Counter[str] = Counter()
    spread_adjustments: Counter[str] = Counter()
    spread_modes: set[str] = set()
    assumed_spread_values: list[float] = []
    score_values: list[float] = []
    for report in reports:
        decision_counts.update(report.decision_counts)
        blockers.update(report.top_blockers)
        execution_fail.update(report.execution_fail_breakdown)
        timeout_resets.update(report.wait_timeout_resets)
        score_bins.update(report.score_bins)
        spread_adjustments.update(report.spread_gate_adjustments)
        spread_modes.add(report.spread_mode)
        assumed_spread_values.append(float(report.assumed_spread_used))
        if report.avg_score is not None:
            score_values.append(float(report.avg_score))
    avg_r = (
        sum(report.avg_r * report.trades for report in reports) / total_trades
        if total_trades
        else 0.0
    )
    win_rate = (total_wins / total_trades) if total_trades else 0.0
    expectancy = (total_pnl / total_trades) if total_trades else 0.0
    aggregate = BacktestReport(
        epic=asset.epic,
        trades=total_trades,
        wins=total_wins,
        losses=total_losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        expectancy=expectancy,
        avg_r=avg_r,
        max_drawdown=max((report.max_drawdown for report in reports), default=0.0),
        time_in_market_bars=total_time_in_market,
        equity_end=config.risk.equity + total_pnl,
        trade_log=[],
        top_blockers=dict(blockers.most_common(10)),
        decision_counts=dict(decision_counts),
        signal_candidates=total_candidates,
        wait_timeout_resets=dict(timeout_resets),
        execution_fail_breakdown=dict(execution_fail),
        avg_score=round(sum(score_values) / len(score_values), 4) if score_values else None,
        score_bins=dict(score_bins),
        spread_mode=next(iter(spread_modes)) if len(spread_modes) == 1 else "MIXED",
        assumed_spread_used=round(sum(assumed_spread_values) / len(assumed_spread_values), 6) if assumed_spread_values else 0.0,
        spread_gate_adjustments=dict(spread_adjustments),
    )
    return WalkForwardReport(epic=asset.epic, splits=reports, aggregate=aggregate)
