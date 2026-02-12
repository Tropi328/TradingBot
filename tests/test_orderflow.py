from __future__ import annotations

from datetime import datetime, timedelta, timezone

import main as main_module
from bot.config import AppConfig
from bot.data.candles import Candle
from bot.strategy.contracts import BiasState, DecisionAction, StrategyEvaluation
from bot.strategy.orderflow import CompositeOrderflowProvider, OrderflowMetrics, OrderflowSnapshot


def _candles(start: datetime, count: int, step: float = 0.4) -> list[Candle]:
    out: list[Candle] = []
    price = 2500.0
    for i in range(count):
        ts = start + timedelta(minutes=5 * i)
        open_price = price
        close = open_price + (step if (i % 4 != 0) else (-step * 0.3))
        high = max(open_price, close) + 0.25
        low = min(open_price, close) - 0.2
        out.append(Candle(timestamp=ts, open=open_price, high=high, low=low, close=close))
        price = close
    # trailing live candle
    out.append(Candle(timestamp=start + timedelta(minutes=5 * count), open=price, high=price + 0.1, low=price - 0.1, close=price))
    return out


def _evaluation() -> StrategyEvaluation:
    return StrategyEvaluation(
        action=DecisionAction.OBSERVE,
        score_total=0.0,
        score_breakdown={
            "bias": 18.0,
            "sweep": 6.0,
            "mss": 8.0,
            "displacement": 8.0,
            "fvg": 6.0,
            "spread": 4.0,
        },
        reasons_blocking=[],
        would_enter_if=[],
        snapshot={"spread": 0.2},
        metadata={"atr_m5": 2.0, "trigger_confirmations": 2, "side": "LONG"},
    )


def test_of_lite_snapshot_is_deterministic() -> None:
    start = datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc)
    candles = _candles(start, 80, step=0.35)
    provider = CompositeOrderflowProvider(default_mode="LITE")

    first = provider.get_snapshot("GOLD", "M5", 64, candles=candles, spread=0.2, atr_value=1.5)
    second = provider.get_snapshot("GOLD", "M5", 64, candles=candles, spread=0.2, atr_value=1.5)

    assert first.to_dict() == second.to_dict()
    assert first.mode == "LITE"
    assert 0.0 <= first.confidence <= 1.0
    assert -1.0 <= first.metrics.delta_ratio <= 1.0
    assert 0.0 <= first.metrics.chop_score <= 1.0


def test_of_full_snapshot_uses_book_and_trades_payload() -> None:
    start = datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc)
    candles = _candles(start, 80, step=2.2)
    provider = CompositeOrderflowProvider(default_mode="FULL")

    snapshot = provider.get_snapshot(
        "BTCUSD",
        "M5",
        96,
        candles=candles,
        spread=12.0,
        atr_value=80.0,
        mode_override="FULL",
        extra={
            "orderflow_full": {
                "trades": [
                    {"side": "buy", "size": 8.0},
                    {"side": "buy", "size": 5.5},
                    {"side": "sell", "size": 2.0},
                ],
                "book": {
                    "bid": 67000.0,
                    "ask": 67010.0,
                    "bid_size": 220.0,
                    "ask_size": 140.0,
                },
            }
        },
    )

    assert snapshot.mode == "FULL"
    assert snapshot.confidence > 0.4
    assert snapshot.metrics.obi_k > 0.0
    assert snapshot.metrics.delta_ratio > 0.0
    assert snapshot.direction == "LONG"


def test_orderflow_divergence_drops_borderline_small_to_observe() -> None:
    config = AppConfig()
    bias = BiasState(
        symbol="XAUUSD",
        strategy_name="SCALP_ICT_PA",
        direction="LONG",
        timeframe="M15",
        updated_at=datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc),
        metadata={},
    )

    baseline_eval = main_module._compute_v2_score(
        symbol="XAUUSD",
        strategy_name="SCALP_ICT_PA",
        bias=bias,
        route_params={"quality_gates": {"spread_ratio_max": 0.15}},
        evaluation=_evaluation(),
        news_blocked=False,
        schedule_open=True,
    )
    baseline_eval = main_module._normalize_action_for_score(evaluation=baseline_eval, config=config)
    assert baseline_eval.action == DecisionAction.SMALL
    assert baseline_eval.score_total is not None
    assert 60.0 <= baseline_eval.score_total <= 64.99

    diverging_of = OrderflowSnapshot(
        confidence=0.95,
        mode="FULL",
        metrics=OrderflowMetrics(
            delta_ratio=-1.0,
            aggression=1.0,
            obi_k=-1.0,
            microprice_bias=-1.0,
            absorption_score=0.1,
            chop_score=0.95,
            spread_ratio=0.22,
            efficiency_ratio=0.05,
        ),
        pressure=-1.0,
        direction="SHORT",
    )
    diverged_eval = main_module._compute_v2_score(
        symbol="XAUUSD",
        strategy_name="SCALP_ICT_PA",
        bias=bias,
        route_params={"quality_gates": {"spread_ratio_max": 0.15}},
        evaluation=_evaluation(),
        news_blocked=False,
        schedule_open=True,
        orderflow_snapshot=diverging_of,
        setup_side="LONG",
        orderflow_settings={"divergence_penalty_min": 6.0, "divergence_penalty_max": 10.0},
    )
    diverged_eval = main_module._normalize_action_for_score(evaluation=diverged_eval, config=config)

    assert diverged_eval.penalties.get("OF_DIVERGENCE", 0.0) >= 6.0
    assert diverged_eval.score_total is not None
    assert diverged_eval.score_total < 60.0
    assert diverged_eval.action == DecisionAction.OBSERVE


def test_orderflow_small_soft_gate_blocks_high_confidence_chop() -> None:
    config = AppConfig()
    evaluation = StrategyEvaluation(
        action=DecisionAction.SMALL,
        score_total=62.5,
        score_breakdown={},
        reasons_blocking=[],
        would_enter_if=[],
        snapshot={},
        metadata={
            "orderflow_snapshot": {
                "mode": "FULL",
                "confidence": 0.92,
                "direction": "LONG",
                "pressure": 0.4,
                "metrics": {
                    "chop_score": 0.93,
                    "spread_ratio": 0.05,
                },
            }
        },
    )
    gated = main_module._apply_orderflow_small_soft_gate(
        route_params={"orderflow": {"small_soft_gate_confidence": 0.8, "small_soft_gate_chop": 0.8}},
        evaluation=evaluation,
        orderflow_settings={
            "small_soft_gate_confidence": float(config.orderflow.small_soft_gate_confidence),
            "small_soft_gate_chop": float(config.orderflow.small_soft_gate_chop),
        },
    )

    assert gated.action == DecisionAction.OBSERVE
    assert "OF_SOFT_GATE_CHOP" in gated.reasons_blocking
    assert gated.gates.get("OrderflowSoftGate") is False
