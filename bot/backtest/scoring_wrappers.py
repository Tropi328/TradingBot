"""Score computation wrappers and quality gates — extracted from engine.py."""

from __future__ import annotations

from datetime import datetime

from bot.backtest.models import BacktestVariant
from bot.config import AppConfig
from bot.strategy.contracts import (
    BiasState,
    SetupCandidate,
    StrategyDataBundle,
    StrategyEvaluation,
    StrategyPlugin,
)
from bot.strategy.decision_core import (
    BACKTEST_HARD_GATE_POLICY,
    BACKTEST_SCORE_POLICY,
    compute_v2_score_core,
    evaluate_hard_gates_core,
    normalize_action_fixed_threshold,
    quality_gate_reasons_core,
)
from bot.strategy.decision_core import (
    apply_orderflow_small_soft_gate as _apply_orderflow_small_soft_gate_core,
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
from bot.strategy.utils import as_int


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
    trade_threshold: float | None = None,
    small_min: float | None = None,
    small_max: float | None = None,
) -> StrategyEvaluation:
    return normalize_action_fixed_threshold(
        evaluation=evaluation,
        config=config,
        trade_threshold=trade_threshold,
        small_min=small_min,
        small_max=small_max,
    )


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
            spread_ratio_max = (
                float(gates_cfg.get("spread_ratio_max")) if gates_cfg.get("spread_ratio_max") is not None else None
            )
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
    return compute_v2_score_core(
        strategy_name=strategy_name,
        bias=bias,
        route_params=route_params,
        evaluation=evaluation,
        news_blocked=news_blocked,
        schedule_open=schedule_open,
        policy=BACKTEST_SCORE_POLICY,
        config=config,
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
        policy=BACKTEST_HARD_GATE_POLICY,
    )
    return result.gates, result.reasons


def _quality_gate_reasons(
    *,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    now: datetime,
    timezone_name: str,
) -> list[str]:
    return quality_gate_reasons_core(
        route_params=route_params,
        evaluation=evaluation,
        now=now,
        timezone_name=timezone_name,
        policy=BACKTEST_HARD_GATE_POLICY,
    )


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
        min_confirm = max(0, as_int(gates_cfg.get("min_confirm"), 0))
    if min_confirm > 0 and metadata.get("trigger_confirmations") is None:
        missing.append("trigger_confirmations")

    deduped = list(dict.fromkeys(missing))
    if deduped:
        evaluation.metadata["missing_features"] = deduped
        evaluation.metadata["is_ready"] = False
    else:
        evaluation.metadata["is_ready"] = True
    return deduped
