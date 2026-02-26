from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from bot.config import AppConfig

LOGGER = logging.getLogger(__name__)
from bot.strategy.contracts import (
    BiasState,
    DecisionAction,
    SetupCandidate,
    SetupState,
    StrategyDataBundle,
    StrategyEvaluation,
    StrategyPlugin,
)
from bot.strategy.orderflow import OrderflowSnapshot
from bot.strategy.schedule import is_schedule_open


@dataclass(frozen=True, slots=True)
class ScoreComputationPolicy:
    spread_source: Literal["snapshot_only", "snapshot_or_metadata"] = "snapshot_only"
    apply_ohlc_soft_spread_penalty: bool = False
    emit_orderflow_influence: bool = False
    include_component_breakdown: bool = False


@dataclass(frozen=True, slots=True)
class HardGatePolicy:
    allow_mid_close_fallback_without_spread: bool = False
    allow_ohlc_spread_soft_skip: bool = False
    enable_confirmation_rules: bool = False
    enable_trade_to_small_downgrade: bool = False
    enable_setup_state_reaction_gate: bool = False


@dataclass(slots=True)
class HardGateResult:
    gates: dict[str, bool]
    reasons: list[str]
    missing_features: list[str] = field(default_factory=list)
    metadata_updates: dict[str, Any] = field(default_factory=dict)


MAIN_SCORE_POLICY = ScoreComputationPolicy(
    spread_source="snapshot_only",
    apply_ohlc_soft_spread_penalty=False,
    emit_orderflow_influence=True,
    include_component_breakdown=True,
)

BACKTEST_SCORE_POLICY = ScoreComputationPolicy(
    spread_source="snapshot_or_metadata",
    apply_ohlc_soft_spread_penalty=True,
    emit_orderflow_influence=False,
    include_component_breakdown=False,
)

MAIN_HARD_GATE_POLICY = HardGatePolicy(
    allow_mid_close_fallback_without_spread=False,
    allow_ohlc_spread_soft_skip=False,
    enable_confirmation_rules=False,
    enable_trade_to_small_downgrade=False,
    enable_setup_state_reaction_gate=True,
)

BACKTEST_HARD_GATE_POLICY = HardGatePolicy(
    allow_mid_close_fallback_without_spread=True,
    allow_ohlc_spread_soft_skip=True,
    enable_confirmation_rules=True,
    enable_trade_to_small_downgrade=True,
    enable_setup_state_reaction_gate=False,
)


def clamp_value(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def resolve_spread(
    evaluation: "StrategyEvaluation",
    *,
    policy_source: str = "snapshot_only",
) -> float | None:
    """Single authoritative spread resolver used by both scoring and gates.

    policy_source:
      "snapshot_only"        – live mode; only trust real bid/ask snapshot
      "snapshot_or_metadata" – backtest mode; fall back to metadata spread
    """
    snap_val = evaluation.snapshot.get("spread")
    if snap_val is not None:
        try:
            return float(snap_val)
        except (TypeError, ValueError):
            pass
    if policy_source == "snapshot_or_metadata":
        meta_val = evaluation.metadata.get("spread")
        if meta_val is not None:
            try:
                return float(meta_val)
            except (TypeError, ValueError):
                pass
    return None


def default_observe_evaluation(*, symbol: str, reason: str) -> StrategyEvaluation:
    return StrategyEvaluation(
        action=DecisionAction.OBSERVE,
        score_total=0.0,
        reasons_blocking=[reason],
        would_enter_if=["VALID_CANDIDATE"],
        snapshot={"symbol": symbol},
    )


def pick_best_candidate(
    *,
    strategy: StrategyPlugin,
    symbol: str,
    candidates: list[SetupCandidate],
    data: StrategyDataBundle,
) -> tuple[SetupCandidate | None, StrategyEvaluation]:
    if not candidates:
        return None, default_observe_evaluation(symbol=symbol, reason="NO_CANDIDATE")
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


def resolve_orderflow_mode(*, symbol: str, route_params: dict[str, object], default_mode: str, full_symbols: set[str]) -> str:
    params = route_params.get("orderflow")
    if isinstance(params, dict):
        mode = str(params.get("mode", "")).strip().upper()
        if mode in {"LITE", "FULL"}:
            return mode
    if symbol.strip().upper() in full_symbols:
        return "FULL"
    mode = default_mode.strip().upper()
    return mode if mode in {"LITE", "FULL"} else "LITE"


def orderflow_param(
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


def normalize_action_fixed_threshold(
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


def compute_v2_score_core(
    *,
    strategy_name: str,
    bias: BiasState,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    news_blocked: bool,
    schedule_open: bool,
    policy: ScoreComputationPolicy,
    config: AppConfig | None = None,
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

    bias_regime = clamp_value((bias_raw / 20.0) * 15.0, 0.0, 15.0)
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
    liquidity_score = clamp_value((sweep_raw / 20.0) * 15.0, 0.0, 15.0)
    edge_score = clamp_value(bias_regime + location_score + liquidity_score, 0.0, 45.0)

    mitigation_quality = clamp_value((fvg_raw / 15.0) * 15.0, 0.0, 15.0)
    trigger_confirmations = int(evaluation.metadata.get("trigger_confirmations", 0))
    reaction_confirmed = clamp_value((trigger_confirmations / 3.0) * 15.0, 0.0, 15.0)
    trigger_clean = clamp_value(((mss_raw / 20.0) * 5.0) + ((displacement_raw / 20.0) * 5.0), 0.0, 10.0)
    trigger_score = clamp_value(mitigation_quality + reaction_confirmed + trigger_clean, 0.0, 40.0)

    atr_value = evaluation.metadata.get("atr_m5", evaluation.snapshot.get("atr_m5"))
    spread_value = resolve_spread(evaluation, policy_source=policy.spread_source)
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
    execution_score = clamp_value(spread_ratio_score + slippage_risk_score + market_state_score, 0.0, 15.0)

    of_trigger_bonus = 0.0
    of_execution_bonus = 0.0
    of_divergence_penalty = 0.0
    of_direction = ""
    of_pressure = 0.0
    if orderflow_snapshot is not None:
        of_dict = orderflow_snapshot.to_dict()
        evaluation.metadata["orderflow_snapshot"] = of_dict
        evaluation.snapshot["orderflow"] = of_dict

        # Minimum confidence gate: below this threshold the sample is too small
        # or too choppy to influence scoring. The confidence already encodes
        # data_quality (bar count / window) via the provider — see orderflow.py:186.
        _of_min_confidence = orderflow_param(
            route_params=route_params,
            settings=orderflow_settings,
            key="min_confidence_for_bonus",
            default=0.20,
        )
        confidence = float(orderflow_snapshot.confidence)
        chop_score = float(orderflow_snapshot.metrics.chop_score)
        of_spread_ratio = float(orderflow_snapshot.metrics.spread_ratio)
        of_pressure = float(orderflow_snapshot.pressure)
        of_direction = str(orderflow_snapshot.direction).upper()

        trigger_bonus_cap = orderflow_param(
            route_params=route_params,
            settings=orderflow_settings,
            key="trigger_bonus_max",
            default=10.0,
        )
        execution_bonus_cap = orderflow_param(
            route_params=route_params,
            settings=orderflow_settings,
            key="execution_bonus_max",
            default=5.0,
        )
        divergence_penalty_min = orderflow_param(
            route_params=route_params,
            settings=orderflow_settings,
            key="divergence_penalty_min",
            default=6.0,
        )
        divergence_penalty_max = orderflow_param(
            route_params=route_params,
            settings=orderflow_settings,
            key="divergence_penalty_max",
            default=10.0,
        )
        if divergence_penalty_min > divergence_penalty_max:
            divergence_penalty_min, divergence_penalty_max = divergence_penalty_max, divergence_penalty_min

        flow_alignment = clamp_value(abs(of_pressure), 0.0, 1.0)
        # Only apply bonuses when confidence exceeds the minimum threshold.
        # This prevents tiny samples (few bars) from inflating the score.
        if confidence >= _of_min_confidence:
            of_trigger_bonus = clamp_value(
                confidence * (1.0 - chop_score) * (0.5 + (0.5 * flow_alignment)) * trigger_bonus_cap,
                0.0,
                trigger_bonus_cap,
            )
            execution_quality = clamp_value(1.0 - (of_spread_ratio / max(max_spread_ratio, 1e-9)), 0.0, 1.0)
            of_execution_bonus = clamp_value(confidence * execution_quality * execution_bonus_cap, 0.0, execution_bonus_cap)
        else:
            of_trigger_bonus = 0.0
            of_execution_bonus = 0.0

        if side in {"LONG", "SHORT"} and of_direction in {"LONG", "SHORT"} and of_direction != side:
            of_divergence_penalty = clamp_value(
                divergence_penalty_min + ((divergence_penalty_max - divergence_penalty_min) * flow_alignment),
                divergence_penalty_min,
                divergence_penalty_max,
            )
        if policy.emit_orderflow_influence:
            evaluation.metadata["orderflow_influence"] = {
                "trigger_bonus": round(of_trigger_bonus, 4),
                "execution_bonus": round(of_execution_bonus, 4),
                "divergence_penalty": round(of_divergence_penalty, 4),
                "direction": of_direction,
                "pressure": round(of_pressure, 4),
            }

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

    if policy.apply_ohlc_soft_spread_penalty and spread_mode == "ASSUMED_OHLC" and spread_ratio is not None and spread_ratio > max_spread_ratio:
        if config is None:
            raise ValueError("config is required when apply_ohlc_soft_spread_penalty=True")
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
    score_total = clamp_value(score_pre_penalty - penalty_total, 0.0, 100.0)

    evaluation.score_layers = {
        "edge": round(edge_score, 2),
        "trigger": round(trigger_score, 2),
        "execution": round(execution_score, 2),
        "orderflow": round(of_trigger_bonus + of_execution_bonus, 2),
    }
    evaluation.penalties = {key: round(value, 2) for key, value in penalties.items()}
    evaluation.score_total = round(score_total, 2)

    score_breakdown = {
        "edge_total": round(edge_score, 2),
        "trigger_total": round(trigger_score, 2),
        "execution_total": round(execution_score, 2),
        "penalty_total": round(penalty_total, 2),
        "score_pre_penalty": round(score_pre_penalty, 2),
        "score_total": round(score_total, 2),
    }
    if policy.include_component_breakdown:
        score_breakdown = {
            "edge.bias_regime": round(bias_regime, 2),
            "edge.location": round(location_score, 2),
            "edge.liquidity_setup": round(liquidity_score, 2),
            "trigger.mitigation_quality": round(mitigation_quality, 2),
            "trigger.reaction_confirmed": round(reaction_confirmed, 2),
            "trigger.cleanliness": round(trigger_clean, 2),
            "execution.spread_ratio_score": round(spread_ratio_score, 2),
            "execution.slippage_risk_score": round(slippage_risk_score, 2),
            "execution.market_state_score": round(market_state_score, 2),
            "orderflow.trigger_bonus": round(of_trigger_bonus, 2),
            "orderflow.execution_bonus": round(of_execution_bonus, 2),
            "orderflow.divergence_penalty": round(of_divergence_penalty, 2),
            **score_breakdown,
        }
    evaluation.score_breakdown = score_breakdown

    if spread_ratio is not None:
        evaluation.metadata["spread_ratio"] = spread_ratio
    return evaluation


def evaluate_hard_gates_core(
    *,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    now: datetime,
    timezone_name: str,
    policy: HardGatePolicy,
) -> HardGateResult:
    gates = {
        "ExecutionGate": True,
        "ScheduleGate": True,
        "ReactionGate": True,
        "RiskGate": True,
    }
    reasons: list[str] = []
    missing_features: list[str] = []
    metadata_updates: dict[str, Any] = {}

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
    # Use resolve_spread() — same helper used by scoring — for consistent spread source.
    _gate_spread_source = (
        "snapshot_or_metadata"
        if (policy.allow_mid_close_fallback_without_spread or policy.allow_ohlc_spread_soft_skip or policy.enable_confirmation_rules)
        else "snapshot_only"
    )
    spread_value = resolve_spread(evaluation, policy_source=_gate_spread_source)
    if spread_value is None and _gate_spread_source == "snapshot_or_metadata":
        quote = evaluation.metadata.get("quote")
        if isinstance(quote, tuple) and len(quote) >= 3:
            spread_value = quote[2]
    close_value = (
        evaluation.snapshot.get("close", evaluation.metadata.get("close"))
        if _gate_spread_source == "snapshot_or_metadata"
        else evaluation.snapshot.get("close")
    )
    spread_mode = str(evaluation.metadata.get("spread_mode", "REAL_BIDASK")).upper()
    track_missing_features = bool(
        policy.allow_mid_close_fallback_without_spread
        or policy.allow_ohlc_spread_soft_skip
        or policy.enable_confirmation_rules
    )

    if atr_value is None:
        gates["ExecutionGate"] = False
        reasons.append("EXEC_FAIL_MISSING_FEATURES")
        if track_missing_features:
            missing_features.append("atr_m5")
    else:
        try:
            if float(atr_value) <= 0:
                gates["ExecutionGate"] = False
                reasons.append("EXEC_FAIL_INVALID_ATR")
                if track_missing_features:
                    missing_features.append("atr_m5")
        except (TypeError, ValueError):
            gates["ExecutionGate"] = False
            reasons.append("EXEC_FAIL_INVALID_ATR")
            if track_missing_features:
                missing_features.append("atr_m5")

    if spread_ratio is None:
        if atr_value is not None and spread_value is not None:
            try:
                atr_float = float(atr_value)
                spread_float = float(spread_value)
                if atr_float > 0:
                    spread_ratio = spread_float / atr_float
                    evaluation.metadata["spread_ratio"] = spread_ratio
                    metadata_updates["spread_ratio"] = spread_ratio
            except (TypeError, ValueError):
                spread_ratio = None

    if spread_ratio is None:
        if policy.allow_mid_close_fallback_without_spread and spread_value is None and close_value is not None:
            spread_ratio = 0.0
            evaluation.metadata["spread_ratio"] = spread_ratio
            metadata_updates["spread_ratio"] = spread_ratio
        elif spread_value is None:
            gates["ExecutionGate"] = False
            reasons.append("EXEC_FAIL_NO_PRICE")
        elif atr_value is not None and (isinstance(atr_value, (float, int)) and float(atr_value) > 0):
            gates["ExecutionGate"] = False
            reasons.append("EXEC_FAIL_MISSING_FEATURES")
            if track_missing_features:
                missing_features.append("spread_or_spread_ratio")
    elif float(spread_ratio) > max_spread_ratio:
        if policy.allow_ohlc_spread_soft_skip and spread_mode == "ASSUMED_OHLC":
            evaluation.metadata["spread_gate_ohlc_hard_skipped"] = True
            metadata_updates["spread_gate_ohlc_hard_skipped"] = True
            soft_reasons = evaluation.metadata.get("soft_reasons")
            if not isinstance(soft_reasons, list):
                soft_reasons = []
            if "ASSUMED_OHLC_SPREAD" not in soft_reasons:
                soft_reasons.append("ASSUMED_OHLC_SPREAD")
            evaluation.metadata["soft_reasons"] = soft_reasons
            metadata_updates["soft_reasons"] = list(soft_reasons)
        else:
            gates["ExecutionGate"] = False
            reasons.append("EXEC_FAIL_SPREAD_TOO_HIGH")

    if policy.enable_confirmation_rules:
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
                        policy.enable_trade_to_small_downgrade
                        and evaluation.action == DecisionAction.TRADE
                        and min_confirm_small > 0
                        and trigger_int >= min_confirm_small
                    )
                    if can_downgrade_to_small:
                        evaluation.action = DecisionAction.SMALL
                        downgrade = {
                            "from": "TRADE",
                            "to": "SMALL",
                            "trigger_confirmations": trigger_int,
                            "required_trade": min_confirm_trade,
                            "required_small": min_confirm_small,
                        }
                        evaluation.metadata["confirmations_downgrade"] = downgrade
                        metadata_updates["confirmations_downgrade"] = downgrade
                        soft_reasons = evaluation.metadata.get("soft_reasons")
                        if not isinstance(soft_reasons, list):
                            soft_reasons = []
                        if "CONFIRMATIONS_DOWNGRADE_SMALL" not in soft_reasons:
                            soft_reasons.append("CONFIRMATIONS_DOWNGRADE_SMALL")
                        evaluation.metadata["soft_reasons"] = soft_reasons
                        metadata_updates["soft_reasons"] = list(soft_reasons)
                    else:
                        gates["ExecutionGate"] = False
                        reasons.append("EXEC_FAIL_CONFIRMATIONS_LOW")

    if policy.enable_setup_state_reaction_gate:
        setup_state = str(evaluation.metadata.get("setup_state", SetupState.READY)).upper()
        if setup_state in SetupState.PENDING_STATES:
            gates["ReactionGate"] = False
            reasons.append(f"GATE_REACTION_{setup_state}")

    if track_missing_features:
        if close_value is None:
            missing_features.append("close")
        if missing_features:
            dedup_missing = list(dict.fromkeys(missing_features))
            evaluation.metadata["missing_features"] = dedup_missing
            metadata_updates["missing_features"] = list(dedup_missing)
            missing_features = dedup_missing

    evaluation.gates = gates
    blocked_gates = [key for key, value in gates.items() if not value]
    if blocked_gates:
        # Report the first blocked gate as primary (backwards-compatible field)
        # and store the full list for diagnostics.
        evaluation.gate_blocked = blocked_gates[0]
        metadata_updates["gate_blocked"] = blocked_gates[0]
        if len(blocked_gates) > 1:
            evaluation.metadata["all_blocked_gates"] = blocked_gates
            metadata_updates["all_blocked_gates"] = blocked_gates

    return HardGateResult(
        gates=gates,
        reasons=reasons,
        missing_features=missing_features,
        metadata_updates=metadata_updates,
    )


def quality_gate_reasons_core(
    *,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    now: datetime,
    timezone_name: str,
    policy: HardGatePolicy,
) -> list[str]:
    return evaluate_hard_gates_core(
        route_params=route_params,
        evaluation=evaluation,
        now=now,
        timezone_name=timezone_name,
        policy=policy,
    ).reasons


def apply_orderflow_small_soft_gate(
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

    conf_threshold = orderflow_param(
        route_params=route_params,
        settings=orderflow_settings,
        key="small_soft_gate_confidence",
        default=0.75,
    )
    chop_threshold = orderflow_param(
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


def risk_multiplier_for(
    *,
    evaluation: StrategyEvaluation,
    route_risk: dict[str, object],
    config: AppConfig,
) -> float:
    """Return the route-level risk multiplier (clamped to [0.01, 1.0]).

    NOTE: this is combined multiplicatively with RiskEngine.effective_risk_per_trade()
    which may apply an additional low_equity_risk_multiplier. The final effective
    risk fraction seen by the sizer is:
        risk_per_trade × risk_multiplier_for(…) × [low_equity_mult if applicable]
    """
    signal_override = evaluation.metadata.get("risk_multiplier_override")
    if signal_override is not None:
        mult = max(0.01, min(1.0, float(signal_override)))
        LOGGER.debug("risk_multiplier_for: signal override → %.4f", mult)
        return mult
    if evaluation.action == DecisionAction.SMALL:
        value = float(route_risk.get("small_risk_multiplier", config.decision_policy.small_risk_multiplier_default))
        mult = max(0.01, min(1.0, value))
        LOGGER.debug("risk_multiplier_for: SMALL action → %.4f", mult)
        return mult
    value = float(route_risk.get("trade_risk_multiplier", config.decision_policy.trade_risk_multiplier))
    mult = max(0.01, min(1.0, value))
    LOGGER.debug("risk_multiplier_for: TRADE action → %.4f", mult)
    return mult
