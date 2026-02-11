from __future__ import annotations

from datetime import datetime, timezone

import main as main_module
from bot.config import AppConfig
from bot.strategy.contracts import BiasState, DecisionAction, StrategyEvaluation


def _eval(score: float) -> StrategyEvaluation:
    return StrategyEvaluation(
        action=DecisionAction.OBSERVE,
        score_total=score,
        score_breakdown={},
        reasons_blocking=[],
        would_enter_if=[],
        snapshot={},
        metadata={},
    )


def test_score_thresholds_60_65() -> None:
    config = AppConfig()
    assert config.decision_policy.trade_score_threshold == 65.0
    assert config.decision_policy.small_score_min == 60.0
    assert config.decision_policy.small_score_max == 64.99

    trade_eval = main_module._normalize_action_for_score(evaluation=_eval(66.0), config=config)
    small_eval = main_module._normalize_action_for_score(evaluation=_eval(62.0), config=config)
    observe_eval = main_module._normalize_action_for_score(evaluation=_eval(59.0), config=config)

    assert trade_eval.action == DecisionAction.TRADE
    assert small_eval.action == DecisionAction.SMALL
    assert observe_eval.action == DecisionAction.OBSERVE
    assert "SCORE_BELOW_MIN" in observe_eval.reasons_blocking


def test_quality_gate_blocks_when_score_high_but_execution_bad() -> None:
    evaluation = StrategyEvaluation(
        action=DecisionAction.TRADE,
        score_total=72.0,
        score_breakdown={},
        reasons_blocking=[],
        would_enter_if=[],
        snapshot={"spread": 0.5},
        metadata={"atr_m5": 1.0, "trigger_confirmations": 3},
    )
    reasons = main_module._quality_gate_reasons(
        symbol="BTCUSD",
        route_params={"quality_gates": {"spread_ratio_max": 0.15, "min_confirm": 2, "min_atr_m5": 0.1}},
        evaluation=evaluation,
        now=datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc),
        timezone_name="Europe/Warsaw",
    )
    assert "GATE_EXECUTION_FAIL" in reasons


def test_schedule_blocks_index_outside_window() -> None:
    evaluation = StrategyEvaluation(
        action=DecisionAction.TRADE,
        score_total=80.0,
        score_breakdown={},
        reasons_blocking=[],
        would_enter_if=[],
        snapshot={"spread": 0.2},
        metadata={"atr_m5": 5.0, "trigger_confirmations": 3},
    )
    reasons = main_module._quality_gate_reasons(
        symbol="US100",
        route_params={
            "schedule": {
                "enabled": True,
                "timezone": "Europe/Warsaw",
                "weekdays": [0, 1, 2, 3, 4],
                "windows": ["08:00-22:00"],
            }
        },
        evaluation=evaluation,
        now=datetime(2026, 2, 15, 11, 0, tzinfo=timezone.utc),  # Sunday
        timezone_name="Europe/Warsaw",
    )
    assert "GATE_SCHEDULE_FAIL" in reasons


def test_scoring_v2_outputs_layers_penalties_and_total() -> None:
    config = AppConfig()
    evaluation = StrategyEvaluation(
        action=DecisionAction.TRADE,
        score_total=70.0,
        score_breakdown={"bias": 18.0, "sweep": 15.0, "mss": 20.0, "displacement": 12.0, "fvg": 10.0, "spread": 5.0},
        reasons_blocking=[],
        would_enter_if=[],
        snapshot={"spread": 0.2, "h1_pd_eq": 100.0, "h1_close": 95.0},
        metadata={"atr_m5": 2.0, "trigger_confirmations": 2, "side": "LONG"},
    )
    bias = BiasState(
        symbol="BTCUSD",
        strategy_name="SCALP_ICT_PA",
        direction="NEUTRAL",
        timeframe="M15",
        updated_at=datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc),
        metadata={},
    )

    scored = main_module._compute_v2_score(
        strategy_name="SCALP_ICT_PA",
        bias=bias,
        route_params={"quality_gates": {"spread_ratio_max": 0.2}},
        evaluation=evaluation,
        news_blocked=False,
        schedule_open=True,
    )

    assert "edge" in scored.score_layers
    assert "trigger" in scored.score_layers
    assert "execution" in scored.score_layers
    assert "NEUTRAL_BIAS" in scored.penalties
    assert scored.score_total is not None
    assert 0 <= scored.score_total <= 100
    assert "edge_total" in scored.score_breakdown
    assert "trigger_total" in scored.score_breakdown
    assert "execution_total" in scored.score_breakdown


def test_reaction_gate_blocks_wait_state() -> None:
    evaluation = StrategyEvaluation(
        action=DecisionAction.TRADE,
        score_total=80.0,
        score_breakdown={},
        reasons_blocking=[],
        would_enter_if=[],
        snapshot={"spread": 0.1},
        metadata={"atr_m5": 2.0, "setup_state": "WAIT_REACTION", "spread_ratio": 0.05},
    )
    reasons = main_module._quality_gate_reasons(
        symbol="BTCUSD",
        route_params={"quality_gates": {"spread_ratio_max": 0.15}},
        evaluation=evaluation,
        now=datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc),
        timezone_name="Europe/Warsaw",
    )

    assert "GATE_REACTION_WAIT_REACTION" in reasons
    assert evaluation.gates.get("ReactionGate") is False
