"""Live/paper mode runtime helpers, dataclasses, and strategy wrappers — extracted from app_main.py."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from bot.clock import (
    should_poll_closed_candle,
)
from bot.config import AppConfig, AssetConfig
from bot.data.capital_client import CapitalClient
from bot.data.market_data import MarketDataService
from bot.gating.adaptive import (
    AdaptiveThresholdConfig,
    ReentryState,
    SoftGateResult,
    compute_adaptive_threshold,
    normalize_action_adaptive,
)
from bot.storage.models import DailyStats
from bot.strategy.contracts import (
    BiasState,
    DecisionAction,
    SetupCandidate,
    StrategyDataBundle,
    StrategyEvaluation,
    StrategyOutcome,
    StrategyPlugin,
)
from bot.strategy.decision_core import (
    MAIN_HARD_GATE_POLICY,
    MAIN_SCORE_POLICY,
    compute_v2_score_core,
    evaluate_hard_gates_core,
    normalize_action_fixed_threshold,
    quality_gate_reasons_core,
)
from bot.strategy.decision_core import (
    apply_orderflow_small_soft_gate as _apply_orderflow_small_soft_gate_core,
)
from bot.strategy.decision_core import (
    clamp_value as _clamp_core,
)
from bot.strategy.decision_core import (
    default_observe_evaluation as _default_observe_evaluation_core,
)
from bot.strategy.decision_core import (
    orderflow_param as _orderflow_param_core,
)
from bot.strategy.decision_core import (
    pick_best_candidate as _pick_best_candidate_core,
)
from bot.strategy.decision_core import (
    resolve_orderflow_mode as _resolve_orderflow_mode_core,
)
from bot.strategy.decision_core import (
    risk_multiplier_for as _risk_multiplier_for_core,
)
from bot.strategy.orderflow import OrderflowSnapshot
from bot.strategy.score_v3 import ScoreV3Config, ScoreV3Engine, apply_score_v3
from bot.strategy.state_machine import (
    H1Snapshot,
    M5Snapshot,
    M15Snapshot,
    StrategySignal,
)
from bot.strategy.trace import (
    DecisionTrace,
    is_new_closed_candle,
    map_reason_codes,
)

LOGGER = logging.getLogger("trading_bot")


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class AssetRuntimeState:
    asset: AssetConfig
    strategy_name: str = "UNKNOWN"
    cache: dict[str, list] = field(default_factory=dict)
    last_processed_closed_ts: dict[str, datetime | None] = field(default_factory=dict)
    last_poll_target_ts: dict[str, datetime | None] = field(default_factory=dict)
    last_poll_attempt_at: dict[str, datetime | None] = field(default_factory=dict)
    quote: tuple[float, float, float] | None = None
    quote_last_fetch_at: datetime | None = None
    last_reason_codes: list[str] = field(default_factory=list)
    stale_data: bool = False
    h1_snapshot: H1Snapshot | None = None
    m15_snapshot: M15Snapshot | None = None
    m5_snapshot: M5Snapshot | None = None
    bias_state: BiasState | None = None
    last_evaluation: StrategyEvaluation | None = None
    last_candidate: SetupCandidate | None = None
    pending_outcome: StrategyOutcome | None = None
    entry_state: str = "WAIT"
    last_trace_signature: str = ""
    reentry: ReentryState = field(default_factory=ReentryState)


@dataclass(slots=True)
class DailyRuntimeSummary:
    trading_day: str
    cycles: int = 0
    signal_candidates: int = 0
    blockers: Counter[str] = field(default_factory=Counter)
    api_requests_start: int = 0
    api_retries_start: int = 0
    api_429_start: int = 0

    def top_blockers(self, limit: int = 5) -> str:
        if not self.blockers:
            return "-"
        return ",".join(f"{key}:{value}" for key, value in self.blockers.most_common(limit))


@dataclass(slots=True)
class PendingOrderIntent:
    symbol: str
    state: AssetRuntimeState
    route_priority: int
    cooldown_seconds: int
    route_risk: dict[str, object]
    outcome: StrategyOutcome
    signal: StrategySignal
    risk_multiplier: float
    rank_score: float
    asset_stats_snapshot: DailyStats


# ── Functions ────────────────────────────────────────────────────────────────


def _timeframe_history(config: AppConfig) -> dict[str, int]:
    return {
        config.timeframes.h1: config.execution.history_bars.h1,
        config.timeframes.m15: config.execution.history_bars.m15,
        config.timeframes.m5: config.execution.history_bars.m5,
    }


def refresh_timeframe_cache(
    *,
    market_data: MarketDataService,
    state: AssetRuntimeState,
    now: datetime,
    timeframe: str,
    history_count: int,
    close_grace_seconds: int,
    retry_seconds: int,
) -> tuple[bool, datetime | None]:
    should_poll, target_closed_ts = should_poll_closed_candle(
        now_utc=now,
        timeframe=timeframe,
        last_processed_closed_ts=state.last_processed_closed_ts.get(timeframe),
        last_attempt_target_ts=state.last_poll_target_ts.get(timeframe),
        last_attempt_at=state.last_poll_attempt_at.get(timeframe),
        close_grace_seconds=close_grace_seconds,
        retry_seconds=retry_seconds,
    )
    state.last_poll_target_ts[timeframe] = target_closed_ts
    if not should_poll:
        return False, target_closed_ts

    state.last_poll_attempt_at[timeframe] = now
    full = market_data.fetch_candles(state.asset.epic, timeframe, max_points=history_count)
    if not full:
        return False, target_closed_ts
    is_new, closed_ts = is_new_closed_candle(
        full,
        state.last_processed_closed_ts.get(timeframe),
    )
    if not is_new:
        return False, closed_ts or target_closed_ts
    state.cache[timeframe] = full
    state.last_processed_closed_ts[timeframe] = closed_ts
    return True, closed_ts


def derive_entry_state(previous: str, *, has_open: bool, has_pending: bool) -> str:
    if has_open:
        return "FILLED"
    if has_pending:
        return "ORDER_PLACED"
    if previous == "ORDER_PLACED":
        return "EXPIRED"
    if previous == "EXPIRED":
        return "WAIT"
    return "WAIT"


def _bias_for_trace(h1: H1Snapshot | None) -> str:
    if h1 is None:
        return "NEUTRAL"
    if h1.side == "LONG":
        return "LONG"
    if h1.side == "SHORT":
        return "SHORT"
    return "NEUTRAL"


def _build_trace(
    *,
    state: AssetRuntimeState,
    now: datetime,
    h1_last_closed: datetime | None,
    h1_new_close: bool,
    m15_last_closed: datetime | None,
    m15_new_close: bool,
    m5_last_closed: datetime | None,
    m5_new_close: bool,
    strategy_name: str,
    evaluation: StrategyEvaluation | None,
    final_decision: str,
    reasons: list[str],
) -> DecisionTrace:
    trace = DecisionTrace(
        asset=state.asset.epic,
        created_at=now,
        strategy_name=strategy_name,
        score_total=evaluation.score_total if evaluation is not None else None,
        score_layers=dict(evaluation.score_layers) if evaluation is not None else {},
        score_breakdown=dict(evaluation.score_breakdown) if evaluation is not None else {},
        penalties=dict(evaluation.penalties) if evaluation is not None else {},
        gates=dict(evaluation.gates) if evaluation is not None else {},
        gate_blocked=evaluation.gate_blocked if evaluation is not None else None,
        reasons_blocking=list(evaluation.reasons_blocking) if evaluation is not None else [],
        would_enter_if=list(evaluation.would_enter_if) if evaluation is not None else [],
        snapshot=dict(evaluation.snapshot) if evaluation is not None else {},
        h1_last_closed_ts=h1_last_closed,
        h1_new_close=h1_new_close,
        m15_last_closed_ts=m15_last_closed,
        m15_new_close=m15_new_close,
        m5_last_closed_ts=m5_last_closed,
        m5_new_close=m5_new_close,
        final_decision=final_decision,
        reasons=map_reason_codes(reasons),
    )
    if state.h1_snapshot is not None:
        trace.h1.updated = h1_new_close
        trace.h1.bias_state = _bias_for_trace(state.h1_snapshot)
        trace.h1.safe_mode = state.h1_snapshot.safe_mode
        trace.h1.ema200_ready = state.h1_snapshot.ema200_ready
        trace.h1.ema200_value = state.h1_snapshot.ema200_value
        trace.h1.bos_state = state.h1_snapshot.bos_state
        trace.h1.bos_age = state.h1_snapshot.bos_age
        trace.h1.bars = state.h1_snapshot.bars
        trace.h1.required_bars = state.h1_snapshot.required_bars
        trace.h1.pd_state = state.h1_snapshot.pd_state
        trace.h1.close = state.h1_snapshot.last_close
        trace.h1.eq = state.h1_snapshot.eq
        trace.h1.dealing_low = state.h1_snapshot.dealing_low
        trace.h1.dealing_high = state.h1_snapshot.dealing_high
    if state.m15_snapshot is not None:
        trace.m15.updated = m15_new_close
        trace.m15.setup_state = state.m15_snapshot.setup_state
        trace.m15.sweep_dir = state.m15_snapshot.sweep_dir
        trace.m15.reject_ok = state.m15_snapshot.reject_ok
        trace.m15.sweep_level = state.m15_snapshot.sweep_level
        trace.m15.invalidation_level = state.m15_snapshot.invalidation_level
        trace.m15.setup_age_minutes = state.m15_snapshot.setup_age_minutes
    if state.m5_snapshot is not None:
        trace.m5.updated = m5_new_close
        trace.m5.mss_ok = state.m5_snapshot.mss_ok
        trace.m5.displacement_ok = state.m5_snapshot.displacement_ok
        trace.m5.fvg_ok = state.m5_snapshot.fvg_ok
        trace.m5.fvg_range = state.m5_snapshot.fvg_range
        trace.m5.fvg_mid = state.m5_snapshot.fvg_mid
        trace.m5.limit_price = state.m5_snapshot.limit_price
    trace.m5.entry_state = state.entry_state
    return trace


def _trace_signature(trace: DecisionTrace) -> str:
    payload = {
        "asset": trace.asset,
        "strategy": trace.strategy_name,
        "score_total": trace.score_total,
        "score_layers": trace.score_layers,
        "score_breakdown": trace.score_breakdown,
        "penalties": trace.penalties,
        "gates": trace.gates,
        "gate_blocked": trace.gate_blocked,
        "reasons_blocking": trace.reasons_blocking,
        "h1_new": trace.h1_new_close,
        "m15_new": trace.m15_new_close,
        "m5_new": trace.m5_new_close,
        "h1": {
            "bias": trace.h1.bias_state,
            "safe_mode": trace.h1.safe_mode,
            "ema_ready": trace.h1.ema200_ready,
            "bos": trace.h1.bos_state,
            "bos_age": trace.h1.bos_age,
            "bars": trace.h1.bars,
            "required": trace.h1.required_bars,
            "pd_state": trace.h1.pd_state,
        },
        "m15": {
            "setup": trace.m15.setup_state,
            "sweep": trace.m15.sweep_dir,
            "reject": trace.m15.reject_ok,
            "age": trace.m15.setup_age_minutes,
        },
        "m5": {
            "entry": trace.m5.entry_state,
            "mss": trace.m5.mss_ok,
            "disp": trace.m5.displacement_ok,
            "fvg": trace.m5.fvg_ok,
            "fvg_range": trace.m5.fvg_range,
            "fvg_mid": trace.m5.fvg_mid,
        },
        "final": trace.final_decision,
        "reasons": trace.reasons,
        "snapshot": trace.snapshot,
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def resolve_db_path(root: Path, *, paper_mode: bool) -> str:
    mode = "paper" if paper_mode else "dry"
    template = os.getenv("SQLITE_PATH_TEMPLATE")
    if template:
        db_path = template.replace("{mode}", mode)
    else:
        raw_path = os.getenv("SQLITE_PATH")
        if raw_path:
            raw_path = raw_path.strip()
            if "{mode}" in raw_path:
                db_path = raw_path.replace("{mode}", mode)
            else:
                base = Path(raw_path)
                suffix = base.suffix or ".db"
                db_path = str(base.with_name(f"{base.stem}_{mode}{suffix}"))
        else:
            db_path = "bot_state_paper.db" if paper_mode else "bot_state_dry.db"
    path = Path(db_path)
    if not path.is_absolute():
        path = root / path
    return str(path)


def should_refresh_quote(
    *,
    now: datetime,
    last_fetch_at: datetime | None,
    interval_seconds: int,
) -> bool:
    if last_fetch_at is None:
        return True
    return (now - last_fetch_at).total_seconds() >= max(1, interval_seconds)


def _quote_refresh_interval_seconds(
    *,
    config: AppConfig,
    trade_enabled: bool,
) -> int:
    default_value = (
        config.execution.quote_refresh_seconds_trade
        if trade_enabled
        else config.execution.quote_refresh_seconds_observe
    )
    env_name = "QUOTE_REFRESH_TRADE_SECONDS" if trade_enabled else "QUOTE_REFRESH_OBSERVE_SECONDS"
    return int(os.getenv(env_name, str(default_value)))


def _log_daily_summary(
    *,
    summary: DailyRuntimeSummary,
    client: CapitalClient | None,
) -> None:
    api_requests = 0
    api_retries = 0
    api_429 = 0
    if client is not None:
        metrics = client.metrics_snapshot()
        api_requests = metrics.get("total_requests", 0) - summary.api_requests_start
        api_retries = metrics.get("total_retries", 0) - summary.api_retries_start
        api_429 = metrics.get("http_429_count", 0) - summary.api_429_start
    LOGGER.info(
        "Daily summary day=%s cycles=%d signal_candidates=%d top_blockers=%s api_requests=%d retries=%d http429=%d",
        summary.trading_day,
        summary.cycles,
        summary.signal_candidates,
        summary.top_blockers(),
        api_requests,
        api_retries,
        api_429,
    )


# ── Thin strategy wrappers ──────────────────────────────────────────────────


def _default_observe_evaluation(*, symbol: str, reason: str) -> StrategyEvaluation:
    return _default_observe_evaluation_core(symbol=symbol, reason=reason)


def _pick_best_candidate(
    *,
    strategy: StrategyPlugin,
    symbol: str,
    candidates: list[SetupCandidate],
    data: StrategyDataBundle,
) -> tuple[SetupCandidate | None, StrategyEvaluation]:
    return _pick_best_candidate_core(
        strategy=strategy,
        symbol=symbol,
        candidates=candidates,
        data=data,
    )


def _normalize_action_for_score(
    *,
    evaluation: StrategyEvaluation,
    config: AppConfig,
    adaptive_cfg: AdaptiveThresholdConfig | None = None,
    soft_gate_result: SoftGateResult | None = None,
    session_threshold_adjust: float = 0.0,
) -> StrategyEvaluation:
    if evaluation.score_total is None:
        return evaluation
    if evaluation.reasons_blocking:
        evaluation.action = DecisionAction.OBSERVE
        return evaluation
    score = float(evaluation.score_total)

    # ── Adaptive threshold path ──────────────────────────
    if adaptive_cfg is not None and adaptive_cfg.enabled:
        # Apply soft-gate penalty to score (reversible metadata, not permanent)
        penalty = soft_gate_result.total_penalty if soft_gate_result else 0.0
        adjusted_score = score - penalty
        evaluation.metadata["adaptive_score_before_penalty"] = score
        evaluation.metadata["adaptive_soft_penalty"] = penalty
        evaluation.metadata["adaptive_score_adjusted"] = adjusted_score

        trend_regime = str(evaluation.metadata.get("trend_regime", "UNKNOWN"))
        vol_regime = str(evaluation.metadata.get("volatility_regime", "NORMAL"))
        threshold = compute_adaptive_threshold(
            config=adaptive_cfg,
            trend_regime=trend_regime,
            vol_regime=vol_regime,
        )
        # Apply session-based threshold adjustment
        threshold += session_threshold_adjust
        evaluation.metadata["adaptive_threshold"] = threshold
        evaluation.metadata["session_threshold_adjust"] = session_threshold_adjust

        action_str = normalize_action_adaptive(
            score=adjusted_score,
            threshold=threshold,
            small_band=5.0,
        )
        if action_str == "TRADE":
            evaluation.action = DecisionAction.TRADE
        elif action_str == "SMALL":
            evaluation.action = DecisionAction.SMALL
        else:
            evaluation.action = DecisionAction.OBSERVE
            if "SCORE_BELOW_MIN" not in evaluation.reasons_blocking:
                evaluation.reasons_blocking.append("SCORE_BELOW_MIN")
        return evaluation

    # ── Default fixed-threshold path ─────────────────────
    return normalize_action_fixed_threshold(
        evaluation=evaluation,
        config=config,
    )


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return _clamp_core(value, min_value, max_value)


def _resolve_orderflow_mode(
    *, symbol: str, route_params: dict[str, object], default_mode: str, full_symbols: set[str]
) -> str:
    return _resolve_orderflow_mode_core(
        symbol=symbol,
        route_params=route_params,
        default_mode=default_mode,
        full_symbols=full_symbols,
    )


def _orderflow_param(
    *,
    route_params: dict[str, object],
    settings: dict[str, float] | None,
    key: str,
    default: float,
) -> float:
    return _orderflow_param_core(
        route_params=route_params,
        settings=settings,
        key=key,
        default=default,
    )


def _compute_v2_score(
    *,
    symbol: str = "",
    strategy_name: str,
    bias: BiasState,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    news_blocked: bool,
    schedule_open: bool,
    orderflow_snapshot: OrderflowSnapshot | None = None,
    setup_side: str | None = None,
    orderflow_settings: dict[str, float] | None = None,
) -> StrategyEvaluation:
    del symbol
    return compute_v2_score_core(
        strategy_name=strategy_name,
        bias=bias,
        route_params=route_params,
        evaluation=evaluation,
        news_blocked=news_blocked,
        schedule_open=schedule_open,
        policy=MAIN_SCORE_POLICY,
        orderflow_snapshot=orderflow_snapshot,
        setup_side=setup_side,
        orderflow_settings=orderflow_settings,
    )


def _evaluate_hard_gates(
    *,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    now: datetime,
    timezone_name: str,
) -> tuple[dict[str, bool], list[str]]:
    result = evaluate_hard_gates_core(
        route_params=route_params,
        evaluation=evaluation,
        now=now,
        timezone_name=timezone_name,
        policy=MAIN_HARD_GATE_POLICY,
    )
    return result.gates, result.reasons


def _quality_gate_reasons(
    *,
    symbol: str,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    now: datetime,
    timezone_name: str,
) -> list[str]:
    del symbol
    return quality_gate_reasons_core(
        route_params=route_params,
        evaluation=evaluation,
        now=now,
        timezone_name=timezone_name,
        policy=MAIN_HARD_GATE_POLICY,
    )


def _apply_orderflow_small_soft_gate(
    *,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    orderflow_settings: dict[str, float] | None,
) -> StrategyEvaluation:
    return _apply_orderflow_small_soft_gate_core(
        route_params=route_params,
        evaluation=evaluation,
        orderflow_settings=orderflow_settings,
    )


def _risk_multiplier_for(
    *,
    evaluation: StrategyEvaluation,
    route_risk: dict[str, object],
    config: AppConfig,
) -> float:
    return _risk_multiplier_for_core(
        evaluation=evaluation,
        route_risk=route_risk,
        config=config,
    )


# ── ScoreV3 live integration ────────────────────────────────────────────────


def create_live_score_v3_engine(config: AppConfig) -> ScoreV3Engine | None:
    """Create a ScoreV3Engine from app config, or None if V3 is disabled."""
    v3cfg = config.score_v3
    if not v3cfg.enabled:
        return None
    return ScoreV3Engine(
        ScoreV3Config(
            enabled=True,
            mode=str(v3cfg.mode),
            model_path=str(v3cfg.model_path),
            trade_threshold=float(v3cfg.trade_threshold),
            small_min=float(v3cfg.small_min),
            small_max=float(v3cfg.small_max),
            tier_enabled=bool(v3cfg.tier_enabled),
            tier_a_plus_pct=float(v3cfg.tier_a_plus_pct),
            tier_a_pct=float(v3cfg.tier_a_pct),
            tier_b_pct=float(v3cfg.tier_b_pct),
            shadow_enabled=bool(v3cfg.shadow_enabled),
            shadow_output_path=str(v3cfg.shadow_output_path),
            shadow_simulate_outcomes=bool(v3cfg.shadow_simulate_outcomes),
            fill_prob_weight=float(v3cfg.fill_prob_weight),
        )
    )
