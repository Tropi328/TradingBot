from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

from bot.config import AppConfig
from bot.data.candles import Candle
from bot.strategy.candidate_queue import CandidateQueue
from bot.strategy.contracts import (
    BiasState,
    DecisionAction,
    SetupCandidate,
    StrategyDataBundle,
    StrategyEvaluation,
)
from bot.strategy.decision_core import (
    BACKTEST_SCORE_POLICY,
    MAIN_SCORE_POLICY,
    apply_orderflow_small_soft_gate,
    compute_v2_score_core,
    default_observe_evaluation,
    normalize_action_fixed_threshold,
    pick_best_candidate,
)
from bot.strategy.orderflow import OrderflowMetrics, OrderflowSnapshot
from bot.strategy.route_pipeline_core import (
    RoutePipelineContext,
    RoutePipelineHooks,
    RoutePipelineProfile,
    evaluate_and_finalize_route,
)
from bot.strategy.state_machine import StrategySignal


class _DummyOrderflowProvider:
    def __init__(self, snapshot: OrderflowSnapshot | None) -> None:
        self.snapshot = snapshot

    def get_snapshot(
        self,
        *,
        symbol: str,
        tf: str,
        window: int,
        candles: list[Candle],
        spread: float | None,
        quote: tuple[float, float, float] | None,
        atr_value: float | None,
        extra: dict[str, object],
        mode_override: str | None = None,
    ) -> OrderflowSnapshot | None:
        del symbol, tf, window, candles, spread, quote, atr_value, extra, mode_override
        return self.snapshot


class _DummyStrategy:
    name = "SCALP_ICT_PA"

    def __init__(self, evaluation: StrategyEvaluation) -> None:
        self._evaluation = evaluation

    def preprocess(self, symbol: str, data: StrategyDataBundle) -> None:
        del symbol, data

    def compute_bias(self, symbol: str, data: StrategyDataBundle) -> BiasState:
        return BiasState(
            symbol=symbol,
            strategy_name=self.name,
            direction="LONG",
            timeframe="M15",
            updated_at=data.now,
            metadata={},
        )

    def detect_candidates(self, symbol: str, data: StrategyDataBundle) -> list[SetupCandidate]:
        ts = data.now
        return [
            SetupCandidate(
                candidate_id=f"{symbol}-{int(ts.timestamp())}",
                symbol=symbol,
                strategy_name=self.name,
                side="LONG",
                created_at=ts,
                expires_at=ts + timedelta(minutes=20),
                source_timeframe="M5",
                setup_type="TEST",
                metadata={"setup_id": f"S-{int(ts.timestamp())}"},
            )
        ]

    def evaluate_candidate(self, symbol: str, candidate: SetupCandidate, data: StrategyDataBundle) -> StrategyEvaluation:
        del symbol, candidate, data
        return deepcopy(self._evaluation)

    def generate_order(
        self,
        symbol: str,
        evaluation: StrategyEvaluation,
        candidate: SetupCandidate,
        data: StrategyDataBundle,
    ) -> StrategySignal | None:
        del symbol, evaluation, candidate
        entry = float(data.candles_m5[-1].close)
        return StrategySignal(
            side="LONG",
            entry_price=entry,
            stop_price=entry - 1.0,
            take_profit=entry + 2.0,
            rr=2.0,
            a_plus=False,
            expires_at=data.now + timedelta(minutes=30),
            reason_codes=["TEST"],
            metadata={},
        )

    def manage_position(self, symbol: str, position, data: StrategyDataBundle) -> list[StrategySignal]:
        del symbol, position, data
        return []


def _build_bundle(*, now: datetime, spread: float, quote: tuple[float, float, float]) -> StrategyDataBundle:
    candles = [
        Candle(
            timestamp=now - timedelta(minutes=10),
            open=2499.5,
            high=2500.5,
            low=2499.0,
            close=2500.0,
            bid=2499.9,
            ask=2500.1,
            volume=1.0,
        ),
        Candle(
            timestamp=now - timedelta(minutes=5),
            open=2500.0,
            high=2501.0,
            low=2499.8,
            close=2500.6,
            bid=2500.5,
            ask=2500.7,
            volume=1.0,
        ),
    ]
    return StrategyDataBundle(
        symbol="XAUUSD",
        now=now,
        candles_h1=list(candles),
        candles_m15=list(candles),
        candles_m5=list(candles),
        spread=spread,
        spread_history=[spread],
        news_blocked=False,
        entry_state="WAIT",
        h1_new_close=True,
        m15_new_close=True,
        m5_new_close=True,
        quote=quote,
        extra={},
    )


def _base_evaluation(*, spread: float, atr: float, spread_mode: str = "REAL_BIDASK") -> StrategyEvaluation:
    return StrategyEvaluation(
        action=DecisionAction.TRADE,
        score_total=70.0,
        score_breakdown={
            "bias": 18.0,
            "sweep": 12.0,
            "mss": 14.0,
            "displacement": 12.0,
            "fvg": 10.0,
        },
        reasons_blocking=[],
        would_enter_if=[],
        snapshot={
            "spread": spread,
            "close": 2500.0,
            "h1_pd_eq": 2505.0,
            "h1_close": 2500.0,
        },
        metadata={
            "atr_m5": atr,
            "trigger_confirmations": 3,
            "side": "LONG",
            "spread_mode": spread_mode,
        },
    )


def test_route_pipeline_main_profile_preserves_orderflow_influence() -> None:
    now = datetime(2026, 2, 22, 12, 0, tzinfo=timezone.utc)
    quote = (2500.5, 2500.7, 0.2)
    strategy = _DummyStrategy(_base_evaluation(spread=0.2, atr=2.0))
    bundle = _build_bundle(now=now, spread=0.2, quote=quote)
    provider = _DummyOrderflowProvider(
        OrderflowSnapshot(
            confidence=0.9,
            mode="FULL",
            metrics=OrderflowMetrics(chop_score=0.2, spread_ratio=0.05),
            pressure=0.75,
            direction="LONG",
        )
    )

    config = AppConfig()
    context = RoutePipelineContext(
        profile=RoutePipelineProfile.MAIN,
        symbol="XAUUSD",
        now=now,
        timezone_name="Europe/Warsaw",
        timeframe="M5",
        strategy_name=strategy.name,
        route_priority=10,
        route_params={"quality_gates": {"spread_ratio_max": 0.15}},
        route_risk={},
        strategy=strategy,
        bundle=bundle,
        news_blocked=False,
        spread=0.2,
        quote=quote,
        orderflow_provider=provider,
        orderflow_default_mode="FULL",
        orderflow_default_window=16,
        orderflow_full_symbols={"XAUUSD"},
        orderflow_settings=None,
    )

    hooks = RoutePipelineHooks(
        default_observe_evaluation=default_observe_evaluation,
        pick_best_candidate=pick_best_candidate,
        compute_score=lambda **kwargs: compute_v2_score_core(
            strategy_name=kwargs["context"].strategy_name,
            bias=kwargs["bias"],
            route_params=kwargs["context"].route_params,
            evaluation=kwargs["evaluation"],
            news_blocked=kwargs["context"].news_blocked,
            schedule_open=kwargs["schedule_open"],
            policy=MAIN_SCORE_POLICY,
            orderflow_snapshot=kwargs["orderflow_snapshot"],
            setup_side=kwargs["candidate"].side if kwargs["candidate"] is not None else None,
            orderflow_settings=kwargs["context"].orderflow_settings,
        ),
        normalize_and_gate=lambda **kwargs: normalize_action_fixed_threshold(
            evaluation=kwargs["evaluation"],
            config=config,
        ),
        apply_orderflow_small_soft_gate=lambda **kwargs: apply_orderflow_small_soft_gate(
            route_params=kwargs["context"].route_params,
            evaluation=kwargs["evaluation"],
            orderflow_settings=kwargs["context"].orderflow_settings,
        ),
        build_payload=lambda **kwargs: {
            "strategy_name": kwargs["context"].strategy_name,
            "score_total": kwargs["evaluation"].score_total,
        },
    )

    result = evaluate_and_finalize_route(
        context=context,
        candidate_queue=CandidateQueue(),
        hooks=hooks,
    )

    assert result.outcome.order_request is not None
    assert "orderflow_influence" in result.evaluation.metadata
    assert result.outcome.payload["strategy_name"] == "SCALP_ICT_PA"


def test_route_pipeline_backtest_profile_keeps_assumed_ohlc_penalty_and_reason_dedup() -> None:
    now = datetime(2026, 2, 22, 12, 0, tzinfo=timezone.utc)
    quote = (2500.0, 2501.0, 1.0)
    strategy = _DummyStrategy(_base_evaluation(spread=1.0, atr=1.0, spread_mode="ASSUMED_OHLC"))
    bundle = _build_bundle(now=now, spread=1.0, quote=quote)
    provider = _DummyOrderflowProvider(None)
    config = AppConfig()

    context = RoutePipelineContext(
        profile=RoutePipelineProfile.BACKTEST,
        symbol="XAUUSD",
        now=now,
        timezone_name="Europe/Warsaw",
        timeframe="M5",
        strategy_name=strategy.name,
        route_priority=5,
        route_params={"quality_gates": {"spread_ratio_max": 0.15}},
        route_risk={},
        strategy=strategy,
        bundle=bundle,
        news_blocked=False,
        spread=1.0,
        quote=quote,
        orderflow_provider=provider,
        orderflow_default_mode="LITE",
        orderflow_default_window=12,
        orderflow_full_symbols=set(),
        orderflow_settings=None,
    )

    hooks = RoutePipelineHooks(
        default_observe_evaluation=default_observe_evaluation,
        pick_best_candidate=pick_best_candidate,
        compute_score=lambda **kwargs: compute_v2_score_core(
            strategy_name=kwargs["context"].strategy_name,
            bias=kwargs["bias"],
            route_params=kwargs["context"].route_params,
            evaluation=kwargs["evaluation"],
            news_blocked=kwargs["context"].news_blocked,
            schedule_open=kwargs["schedule_open"],
            policy=BACKTEST_SCORE_POLICY,
            config=config,
            orderflow_snapshot=kwargs["orderflow_snapshot"],
            setup_side=kwargs["candidate"].side if kwargs["candidate"] is not None else None,
            orderflow_settings=kwargs["context"].orderflow_settings,
        ),
        normalize_and_gate=lambda **kwargs: kwargs["evaluation"],
        apply_orderflow_small_soft_gate=lambda **kwargs: kwargs["evaluation"],
        build_payload=lambda **kwargs: {"score_total": kwargs["evaluation"].score_total},
        build_reason_codes=lambda _ctx, evaluation: list(dict.fromkeys(evaluation.reasons_blocking + ["REPEAT", "REPEAT"])),
        on_outcome=lambda _ctx, outcome: outcome.reason_codes.append("REPEAT"),
    )

    result = evaluate_and_finalize_route(
        context=context,
        candidate_queue=CandidateQueue(),
        hooks=hooks,
    )

    assert "ASSUMED_OHLC_SPREAD" in result.evaluation.penalties
    assert result.evaluation.metadata.get("spread_gate_soft_penalty_applied") is True
    assert result.outcome.reason_codes.count("REPEAT") == 2


def test_route_pipeline_unknown_strategy_uses_unknown_rank_and_payload() -> None:
    now = datetime(2026, 2, 22, 12, 0, tzinfo=timezone.utc)
    quote = (2500.5, 2500.7, 0.2)
    bundle = _build_bundle(now=now, spread=0.2, quote=quote)
    context = RoutePipelineContext(
        profile=RoutePipelineProfile.MAIN,
        symbol="XAUUSD",
        now=now,
        timezone_name="Europe/Warsaw",
        timeframe="M5",
        strategy_name="UNKNOWN_STRAT",
        route_priority=1,
        route_params={},
        route_risk={},
        strategy=None,
        bundle=bundle,
        news_blocked=False,
        spread=0.2,
        quote=quote,
        orderflow_provider=_DummyOrderflowProvider(None),
        orderflow_default_mode="LITE",
        orderflow_default_window=12,
        orderflow_full_symbols=set(),
    )
    hooks = RoutePipelineHooks(
        default_observe_evaluation=default_observe_evaluation,
        pick_best_candidate=pick_best_candidate,
        compute_score=lambda **kwargs: kwargs["evaluation"],
        normalize_and_gate=lambda **kwargs: kwargs["evaluation"],
        apply_orderflow_small_soft_gate=lambda **kwargs: kwargs["evaluation"],
        build_payload=lambda **kwargs: {},
        unknown_rank=lambda _ctx, _evaluation: -123.45,
        build_unknown_payload=lambda _ctx: {"kind": "unknown"},
    )

    result = evaluate_and_finalize_route(
        context=context,
        candidate_queue=CandidateQueue(),
        hooks=hooks,
    )

    assert result.rank == -123.45
    assert result.outcome.order_request is None
    assert result.outcome.reason_codes == ["UNKNOWN_STRATEGY_UNKNOWN_STRAT"]
    assert result.outcome.payload["kind"] == "unknown"
