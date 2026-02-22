from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Protocol

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
from bot.strategy.decision_core import resolve_orderflow_mode
from bot.strategy.orderflow import OrderflowSnapshot
from bot.strategy.ranker import rank_score


class RoutePipelineProfile(str, Enum):
    MAIN = "MAIN"
    BACKTEST = "BACKTEST"


@dataclass(slots=True)
class RoutePipelineContext:
    profile: RoutePipelineProfile
    symbol: str
    now: datetime
    timezone_name: str
    timeframe: str
    strategy_name: str
    route_priority: int
    route_params: dict[str, object]
    route_risk: dict[str, object]
    strategy: StrategyPlugin | None
    bundle: StrategyDataBundle
    news_blocked: bool
    spread: float | None
    quote: tuple[float, float, float] | None
    orderflow_provider: Any
    orderflow_default_mode: str
    orderflow_default_window: int
    orderflow_full_symbols: set[str]
    orderflow_settings: dict[str, float] | None = None
    runtime: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RoutePipelineResult:
    context: RoutePipelineContext
    bias: BiasState
    candidate: SetupCandidate | None
    evaluation: StrategyEvaluation
    outcome: StrategyOutcome
    rank: float


@dataclass(slots=True)
class RoutePipelineStep:
    context: RoutePipelineContext
    bias: BiasState
    candidate: SetupCandidate | None
    evaluation: StrategyEvaluation
    orderflow_snapshot: OrderflowSnapshot | None
    schedule_open: bool
    prebuilt_result: RoutePipelineResult | None = None


class DefaultObserveFn(Protocol):
    def __call__(self, *, symbol: str, reason: str) -> StrategyEvaluation:
        ...


class PickBestCandidateFn(Protocol):
    def __call__(
        self,
        *,
        strategy: StrategyPlugin,
        symbol: str,
        candidates: list[SetupCandidate],
        data: StrategyDataBundle,
    ) -> tuple[SetupCandidate | None, StrategyEvaluation]:
        ...


class ComputeScoreFn(Protocol):
    def __call__(
        self,
        *,
        context: RoutePipelineContext,
        bias: BiasState,
        candidate: SetupCandidate | None,
        evaluation: StrategyEvaluation,
        schedule_open: bool,
        orderflow_snapshot: OrderflowSnapshot | None,
    ) -> StrategyEvaluation:
        ...


class EvalTransformFn(Protocol):
    def __call__(
        self,
        *,
        context: RoutePipelineContext,
        candidate: SetupCandidate | None,
        evaluation: StrategyEvaluation,
    ) -> StrategyEvaluation:
        ...


class BuildPayloadFn(Protocol):
    def __call__(
        self,
        *,
        context: RoutePipelineContext,
        candidate: SetupCandidate | None,
        evaluation: StrategyEvaluation,
    ) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class RoutePipelineHooks:
    default_observe_evaluation: DefaultObserveFn
    pick_best_candidate: PickBestCandidateFn
    compute_score: ComputeScoreFn
    normalize_and_gate: EvalTransformFn
    apply_orderflow_small_soft_gate: EvalTransformFn
    build_payload: BuildPayloadFn
    pre_score: EvalTransformFn | None = None
    build_reason_codes: Callable[[RoutePipelineContext, StrategyEvaluation], list[str]] | None = None
    on_outcome: Callable[[RoutePipelineContext, StrategyOutcome], None] | None = None
    unknown_rank: Callable[[RoutePipelineContext, StrategyEvaluation], float] | None = None
    rank: Callable[[RoutePipelineContext, StrategyEvaluation], float] | None = None
    build_unknown_payload: Callable[[RoutePipelineContext], dict[str, Any]] | None = None


def _resolve_orderflow_window(*, route_params: dict[str, object], default_window: int) -> int:
    route_of = route_params.get("orderflow")
    window = default_window
    if isinstance(route_of, dict):
        try:
            window = max(8, int(route_of.get("window", default_window)))
        except (TypeError, ValueError):
            window = default_window
    return window


def _default_rank(context: RoutePipelineContext, evaluation: StrategyEvaluation) -> float:
    return rank_score(evaluation) + (context.route_priority * 0.01)


def evaluate_route_pipeline_step(
    *,
    context: RoutePipelineContext,
    candidate_queue: CandidateQueue,
    hooks: RoutePipelineHooks,
) -> RoutePipelineStep:
    if context.strategy is None:
        reason = f"UNKNOWN_STRATEGY_{context.strategy_name}"
        evaluation = hooks.default_observe_evaluation(symbol=context.symbol, reason=reason)
        bias = BiasState(
            symbol=context.symbol,
            strategy_name=context.strategy_name,
            direction="NEUTRAL",
            timeframe=context.timeframe,
            updated_at=context.now,
            metadata={},
        )
        unknown_payload = hooks.build_unknown_payload(context) if hooks.build_unknown_payload is not None else {}
        outcome = StrategyOutcome(
            symbol=context.symbol,
            strategy_name=context.strategy_name,
            bias=bias,
            candidate=None,
            evaluation=evaluation,
            order_request=None,
            reason_codes=[reason],
            payload=unknown_payload,
        )
        unknown_rank_fn = hooks.unknown_rank or _default_rank
        result = RoutePipelineResult(
            context=context,
            bias=bias,
            candidate=None,
            evaluation=evaluation,
            outcome=outcome,
            rank=float(unknown_rank_fn(context, evaluation)),
        )
        return RoutePipelineStep(
            context=context,
            bias=bias,
            candidate=None,
            evaluation=evaluation,
            orderflow_snapshot=None,
            schedule_open=True,
            prebuilt_result=result,
        )

    strategy = context.strategy
    preprocessed = bool(context.runtime.get("preprocessed"))
    if not preprocessed:
        strategy.preprocess(context.symbol, context.bundle)
    runtime_bias = context.runtime.get("precomputed_bias")
    if isinstance(runtime_bias, BiasState):
        bias = runtime_bias
    else:
        bias = strategy.compute_bias(context.symbol, context.bundle)
    raw_candidates = strategy.detect_candidates(context.symbol, context.bundle)
    candidates = candidate_queue.put_many(
        symbol=context.symbol,
        strategy=context.strategy_name,
        candidates=raw_candidates,
        now=context.now,
    )
    candidate, evaluation = hooks.pick_best_candidate(
        strategy=strategy,
        symbol=context.symbol,
        candidates=candidates,
        data=context.bundle,
    )

    schedule_cfg = context.route_params.get("schedule")
    schedule_open = True
    if isinstance(schedule_cfg, dict):
        from bot.strategy.schedule import is_schedule_open

        schedule_open = is_schedule_open(context.now, schedule_cfg, context.timezone_name)

    mode_override = resolve_orderflow_mode(
        symbol=context.symbol,
        route_params=context.route_params,
        default_mode=context.orderflow_default_mode,
        full_symbols=context.orderflow_full_symbols,
    )
    window = _resolve_orderflow_window(
        route_params=context.route_params,
        default_window=context.orderflow_default_window,
    )

    atr_for_of = evaluation.metadata.get("atr_m5", evaluation.snapshot.get("atr_m5"))
    atr_for_of_float: float | None
    try:
        atr_for_of_float = float(atr_for_of) if atr_for_of is not None else None
    except (TypeError, ValueError):
        atr_for_of_float = None

    orderflow_snapshot = context.orderflow_provider.get_snapshot(
        symbol=context.symbol,
        tf=context.timeframe,
        window=window,
        candles=context.bundle.candles_m5,
        spread=context.spread,
        quote=context.quote,
        atr_value=atr_for_of_float,
        extra=context.bundle.extra,
        mode_override=mode_override,
    )

    if hooks.pre_score is not None:
        evaluation = hooks.pre_score(
            context=context,
            candidate=candidate,
            evaluation=evaluation,
        )
    evaluation = hooks.compute_score(
        context=context,
        bias=bias,
        candidate=candidate,
        evaluation=evaluation,
        schedule_open=schedule_open,
        orderflow_snapshot=orderflow_snapshot,
    )
    evaluation = hooks.normalize_and_gate(
        context=context,
        candidate=candidate,
        evaluation=evaluation,
    )
    evaluation = hooks.apply_orderflow_small_soft_gate(
        context=context,
        candidate=candidate,
        evaluation=evaluation,
    )

    return RoutePipelineStep(
        context=context,
        bias=bias,
        candidate=candidate,
        evaluation=evaluation,
        orderflow_snapshot=orderflow_snapshot,
        schedule_open=schedule_open,
    )


def finalize_route_outcome(
    *,
    step: RoutePipelineStep,
    hooks: RoutePipelineHooks,
) -> RoutePipelineResult:
    if step.prebuilt_result is not None:
        return step.prebuilt_result

    context = step.context
    strategy = context.strategy
    if strategy is None:
        raise ValueError("strategy is required to finalize non-prebuilt pipeline step")

    candidate = step.candidate
    evaluation = step.evaluation
    signal_request = (
        strategy.generate_order(context.symbol, evaluation, candidate, context.bundle)
        if candidate is not None and evaluation.action in {DecisionAction.TRADE, DecisionAction.SMALL}
        else None
    )
    if hooks.build_reason_codes is not None:
        reason_codes = list(hooks.build_reason_codes(context, evaluation))
    else:
        reason_codes = list(evaluation.reasons_blocking)
    payload = hooks.build_payload(
        context=context,
        candidate=candidate,
        evaluation=evaluation,
    )
    outcome = StrategyOutcome(
        symbol=context.symbol,
        strategy_name=context.strategy_name,
        bias=step.bias,
        candidate=candidate,
        evaluation=evaluation,
        order_request=signal_request,
        reason_codes=reason_codes,
        payload=payload,
    )
    if hooks.on_outcome is not None:
        hooks.on_outcome(context, outcome)
    rank_fn = hooks.rank or _default_rank
    return RoutePipelineResult(
        context=context,
        bias=step.bias,
        candidate=candidate,
        evaluation=evaluation,
        outcome=outcome,
        rank=float(rank_fn(context, evaluation)),
    )


def evaluate_and_finalize_route(
    *,
    context: RoutePipelineContext,
    candidate_queue: CandidateQueue,
    hooks: RoutePipelineHooks,
) -> RoutePipelineResult:
    step = evaluate_route_pipeline_step(
        context=context,
        candidate_queue=candidate_queue,
        hooks=hooks,
    )
    return finalize_route_outcome(
        step=step,
        hooks=hooks,
    )
