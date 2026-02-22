from __future__ import annotations

from datetime import datetime, timezone

from bot.config import AppConfig
from bot.strategy.contracts import BiasState, DecisionAction, StrategyEvaluation
from bot.strategy.decision_core import (
    BACKTEST_HARD_GATE_POLICY,
    BACKTEST_SCORE_POLICY,
    MAIN_HARD_GATE_POLICY,
    MAIN_SCORE_POLICY,
    compute_v2_score_core,
    evaluate_hard_gates_core,
    normalize_action_fixed_threshold,
)
from bot.strategy.orderflow import OrderflowMetrics, OrderflowSnapshot


def _bias(*, direction: str = "LONG") -> BiasState:
    return BiasState(
        symbol="XAUUSD",
        strategy_name="SCALP_ICT_PA",
        direction=direction,
        timeframe="M15",
        updated_at=datetime(2026, 2, 22, 12, 0, tzinfo=timezone.utc),
        metadata={},
    )


def _evaluation(
    *,
    action: DecisionAction = DecisionAction.TRADE,
    score_total: float = 70.0,
    spread: float = 0.2,
    atr: float = 2.0,
    setup_state: str = "READY",
) -> StrategyEvaluation:
    return StrategyEvaluation(
        action=action,
        score_total=score_total,
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
            "trigger_confirmations": 2,
            "side": "LONG",
            "setup_state": setup_state,
        },
    )


def test_main_score_policy_keeps_orderflow_influence_and_breakdown() -> None:
    evaluation = _evaluation()
    orderflow = OrderflowSnapshot(
        confidence=0.9,
        mode="FULL",
        metrics=OrderflowMetrics(chop_score=0.25, spread_ratio=0.04),
        pressure=0.8,
        direction="LONG",
    )
    out = compute_v2_score_core(
        strategy_name="SCALP_ICT_PA",
        bias=_bias(direction="LONG"),
        route_params={"quality_gates": {"spread_ratio_max": 0.15}},
        evaluation=evaluation,
        news_blocked=False,
        schedule_open=True,
        policy=MAIN_SCORE_POLICY,
        orderflow_snapshot=orderflow,
        setup_side="LONG",
    )
    assert "orderflow_influence" in out.metadata
    assert "edge.bias_regime" in out.score_breakdown
    assert "orderflow.trigger_bonus" in out.score_breakdown


def test_backtest_score_policy_applies_assumed_ohlc_soft_penalty() -> None:
    config = AppConfig()
    evaluation = _evaluation(spread=1.0, atr=1.0)
    evaluation.metadata["spread_mode"] = "ASSUMED_OHLC"
    out = compute_v2_score_core(
        strategy_name="SCALP_ICT_PA",
        bias=_bias(direction="LONG"),
        route_params={"quality_gates": {"spread_ratio_max": 0.15}},
        evaluation=evaluation,
        news_blocked=False,
        schedule_open=True,
        policy=BACKTEST_SCORE_POLICY,
        config=config,
        setup_side="LONG",
    )
    assert out.metadata.get("spread_gate_soft_penalty_applied") is True
    assert "ASSUMED_OHLC_SPREAD" in out.penalties
    assert "ASSUMED_OHLC_SPREAD" in out.metadata.get("soft_reasons", [])


def test_main_hard_gate_blocks_wait_reaction_state() -> None:
    evaluation = _evaluation(setup_state="WAIT_REACTION", spread=0.1, atr=2.0)
    result = evaluate_hard_gates_core(
        route_params={"quality_gates": {"spread_ratio_max": 0.2}},
        evaluation=evaluation,
        now=datetime(2026, 2, 22, 12, 0, tzinfo=timezone.utc),
        timezone_name="Europe/Warsaw",
        policy=MAIN_HARD_GATE_POLICY,
    )
    assert result.gates["ReactionGate"] is False
    assert "GATE_REACTION_WAIT_REACTION" in result.reasons


def test_backtest_hard_gate_can_downgrade_trade_to_small() -> None:
    evaluation = _evaluation(action=DecisionAction.TRADE, spread=0.1, atr=2.0)
    evaluation.metadata["candidate_id"] = "cid-1"
    evaluation.metadata["trigger_confirmations"] = 1
    result = evaluate_hard_gates_core(
        route_params={
            "quality_gates": {
                "spread_ratio_max": 0.2,
                "min_confirm_trade": 2,
                "min_confirm_small": 1,
            }
        },
        evaluation=evaluation,
        now=datetime(2026, 2, 22, 12, 0, tzinfo=timezone.utc),
        timezone_name="Europe/Warsaw",
        policy=BACKTEST_HARD_GATE_POLICY,
    )
    assert evaluation.action == DecisionAction.SMALL
    assert result.gates["ExecutionGate"] is True
    assert "EXEC_FAIL_CONFIRMATIONS_LOW" not in result.reasons
    assert "confirmations_downgrade" in evaluation.metadata


def test_fixed_threshold_normalization_matches_legacy_actions() -> None:
    config = AppConfig()
    trade_eval = normalize_action_fixed_threshold(evaluation=_evaluation(score_total=66.0), config=config)
    small_eval = normalize_action_fixed_threshold(evaluation=_evaluation(score_total=62.0), config=config)
    observe_eval = normalize_action_fixed_threshold(evaluation=_evaluation(score_total=59.0), config=config)

    assert trade_eval.action == DecisionAction.TRADE
    assert small_eval.action == DecisionAction.SMALL
    assert observe_eval.action == DecisionAction.OBSERVE
    assert "SCORE_BELOW_MIN" in observe_eval.reasons_blocking
