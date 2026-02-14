from __future__ import annotations

from datetime import datetime, timezone

from bot.backtest.engine import (
    BacktestVariant,
    _adjust_thresholds_dynamic,
    _apply_reaction_gate_with_timeout,
    _apply_soft_reason_penalties,
    _evaluate_hard_gates,
)
from bot.config import AppConfig
from bot.strategy.contracts import DecisionAction, StrategyEvaluation


def _evaluation(*, setup_state: str = "READY", atr_m5: float | None = 1.0, spread: float | None = 0.1, spread_ratio: float | None = None) -> StrategyEvaluation:
    metadata = {"setup_state": setup_state}
    if atr_m5 is not None:
        metadata["atr_m5"] = atr_m5
    if spread_ratio is not None:
        metadata["spread_ratio"] = spread_ratio
    return StrategyEvaluation(
        action=DecisionAction.TRADE,
        score_total=70.0,
        score_breakdown={},
        reasons_blocking=[],
        would_enter_if=[],
        snapshot={"spread": spread},
        metadata=metadata,
    )


def test_reaction_gate_timeout_reset_reaction() -> None:
    config = AppConfig()
    variant = BacktestVariant(code="W1", reaction_timeout_reset=True)
    wait_states: dict[str, object] = {}
    timeout_resets = {}
    wait_durations = {"REACTION": [], "MITIGATION": []}
    reset_block_bar: dict[str, int] = {}
    timeout_samples: list[object] = []

    eval_first = _evaluation(setup_state="WAIT_REACTION")
    reasons_first = _apply_reaction_gate_with_timeout(
        strategy_key="XAUUSD:SCALP_ICT_PA",
        bar_index=100,
        now=datetime(2026, 2, 12, 12, 0, tzinfo=timezone.utc),
        evaluation=eval_first,
        wait_states=wait_states,  # type: ignore[arg-type]
        variant=variant,
        config=config,
        timeout_resets=timeout_resets,  # type: ignore[arg-type]
        wait_durations=wait_durations,
        reset_block_bar=reset_block_bar,
        timeout_samples=timeout_samples,  # type: ignore[arg-type]
    )
    assert reasons_first == ["GATE_REACTION_WAIT_REACTION"]

    eval_late = _evaluation(setup_state="WAIT_REACTION")
    reasons_late = _apply_reaction_gate_with_timeout(
        strategy_key="XAUUSD:SCALP_ICT_PA",
        bar_index=130,
        now=datetime(2026, 2, 12, 12, 30, tzinfo=timezone.utc),
        evaluation=eval_late,
        wait_states=wait_states,  # type: ignore[arg-type]
        variant=variant,
        config=config,
        timeout_resets=timeout_resets,  # type: ignore[arg-type]
        wait_durations=wait_durations,
        reset_block_bar=reset_block_bar,
        timeout_samples=timeout_samples,  # type: ignore[arg-type]
    )
    assert reasons_late == []
    assert bool(eval_late.metadata.get("wait_timeout_soft_mode")) is True

    eval_same_bar = _evaluation(setup_state="WAIT_REACTION")
    reasons_same_bar = _apply_reaction_gate_with_timeout(
        strategy_key="XAUUSD:SCALP_ICT_PA",
        bar_index=130,
        now=datetime(2026, 2, 12, 12, 30, tzinfo=timezone.utc),
        evaluation=eval_same_bar,
        wait_states=wait_states,  # type: ignore[arg-type]
        variant=variant,
        config=config,
        timeout_resets=timeout_resets,  # type: ignore[arg-type]
        wait_durations=wait_durations,
        reset_block_bar=reset_block_bar,
        timeout_samples=timeout_samples,  # type: ignore[arg-type]
    )
    assert reasons_same_bar == []


def test_soft_reasons_move_to_penalties() -> None:
    config = AppConfig()
    evaluation = StrategyEvaluation(
        action=DecisionAction.OBSERVE,
        score_total=50.0,
        score_breakdown={"bias": 10.0},
        reasons_blocking=["SCALP_NO_MSS", "SCALP_NO_FVG", "HARD_OTHER"],
        would_enter_if=[],
        snapshot={},
        metadata={},
    )
    out = _apply_soft_reason_penalties(
        evaluation=evaluation,
        config=config,
        enabled=True,
    )
    assert "SCALP_NO_MSS" not in out.reasons_blocking
    assert "SCALP_NO_FVG" not in out.reasons_blocking
    assert "HARD_OTHER" in out.reasons_blocking
    assert "penalty_soft_scalp_no_mss" in out.score_breakdown
    assert "penalty_soft_scalp_no_fvg" in out.score_breakdown


def test_soft_penalty_route_override() -> None:
    config = AppConfig()
    evaluation = StrategyEvaluation(
        action=DecisionAction.OBSERVE,
        score_total=50.0,
        score_breakdown={"bias": 10.0},
        reasons_blocking=["TREND_CONFIRMATIONS_LOW"],
        would_enter_if=[],
        snapshot={},
        metadata={},
    )
    out = _apply_soft_reason_penalties(
        evaluation=evaluation,
        config=config,
        route_params={"soft_penalties": {"TREND_CONFIRMATIONS_LOW": -1.5}},
        enabled=True,
    )
    assert "TREND_CONFIRMATIONS_LOW" not in out.reasons_blocking
    assert out.score_breakdown["penalty_soft_trend_confirmations_low"] == -1.5


def test_execution_gate_breakdown_reasons() -> None:
    now = datetime(2026, 2, 12, 12, 0, tzinfo=timezone.utc)
    eval_spread = _evaluation(atr_m5=1.0, spread=1.0, spread_ratio=0.5)
    _, reasons_spread = _evaluate_hard_gates(
        route_params={"quality_gates": {"spread_ratio_max": 0.15}},
        evaluation=eval_spread,
        now=now,
        timezone_name="Europe/Warsaw",
    )
    assert "EXEC_FAIL_SPREAD_TOO_HIGH" in reasons_spread

    eval_invalid_atr = _evaluation(atr_m5=0.0, spread=0.2)
    _, reasons_atr = _evaluate_hard_gates(
        route_params={"quality_gates": {"spread_ratio_max": 0.15}},
        evaluation=eval_invalid_atr,
        now=now,
        timezone_name="Europe/Warsaw",
    )
    assert "EXEC_FAIL_INVALID_ATR" in reasons_atr

    eval_ohlc = _evaluation(atr_m5=1.0, spread=0.5, spread_ratio=0.5)
    eval_ohlc.metadata["spread_mode"] = "ASSUMED_OHLC"
    gates_ohlc, reasons_ohlc = _evaluate_hard_gates(
        route_params={"quality_gates": {"spread_ratio_max": 0.15}},
        evaluation=eval_ohlc,
        now=now,
        timezone_name="Europe/Warsaw",
    )
    assert gates_ohlc["ExecutionGate"] is True
    assert "EXEC_FAIL_SPREAD_TOO_HIGH" not in reasons_ohlc


def test_dynamic_threshold_bump() -> None:
    config = AppConfig()
    evaluation = _evaluation(atr_m5=0.1, spread=0.03, spread_ratio=0.16)
    trade, small_min, small_max, reasons = _adjust_thresholds_dynamic(
        trade_threshold=65.0,
        small_min=60.0,
        small_max=64.0,
        route_params={"quality_gates": {"spread_ratio_max": 0.17, "min_atr_m5": 0.1}},
        evaluation=evaluation,
        config=config,
        enabled=True,
    )
    assert trade > 65.0
    assert small_min > 60.0
    assert small_max > 64.0
    assert reasons


def test_execution_gate_allows_mid_close_fallback_without_spread() -> None:
    now = datetime(2026, 2, 12, 12, 0, tzinfo=timezone.utc)
    evaluation = StrategyEvaluation(
        action=DecisionAction.TRADE,
        score_total=70.0,
        score_breakdown={},
        reasons_blocking=[],
        would_enter_if=[],
        snapshot={"close": 2510.5, "spread": None},
        metadata={"atr_m5": 1.2},
    )
    gates, reasons = _evaluate_hard_gates(
        route_params={"quality_gates": {"spread_ratio_max": 0.2}},
        evaluation=evaluation,
        now=now,
        timezone_name="Europe/Warsaw",
    )
    assert gates["ExecutionGate"] is True
    assert "EXEC_FAIL_NO_PRICE" not in reasons


def test_reaction_gate_switches_to_soft_after_hard_block_window() -> None:
    config = AppConfig()
    variant = BacktestVariant(code="W1", reaction_timeout_reset=True)
    wait_states: dict[str, object] = {}
    timeout_resets = {}
    wait_durations = {"REACTION": [], "MITIGATION": []}
    reset_block_bar: dict[str, int] = {}
    timeout_samples: list[object] = []

    eval_first = _evaluation(setup_state="WAIT_REACTION")
    reasons_first = _apply_reaction_gate_with_timeout(
        strategy_key="XAUUSD:SCALP_ICT_PA",
        bar_index=100,
        now=datetime(2026, 2, 12, 12, 0, tzinfo=timezone.utc),
        evaluation=eval_first,
        wait_states=wait_states,  # type: ignore[arg-type]
        variant=variant,
        config=config,
        timeout_resets=timeout_resets,  # type: ignore[arg-type]
        wait_durations=wait_durations,
        reset_block_bar=reset_block_bar,
        timeout_samples=timeout_samples,  # type: ignore[arg-type]
    )
    assert reasons_first == ["GATE_REACTION_WAIT_REACTION"]

    eval_soft = _evaluation(setup_state="WAIT_REACTION")
    reasons_soft = _apply_reaction_gate_with_timeout(
        strategy_key="XAUUSD:SCALP_ICT_PA",
        bar_index=103,
        now=datetime(2026, 2, 12, 12, 15, tzinfo=timezone.utc),
        evaluation=eval_soft,
        wait_states=wait_states,  # type: ignore[arg-type]
        variant=variant,
        config=config,
        timeout_resets=timeout_resets,  # type: ignore[arg-type]
        wait_durations=wait_durations,
        reset_block_bar=reset_block_bar,
        timeout_samples=timeout_samples,  # type: ignore[arg-type]
    )
    assert reasons_soft == []
    assert bool(eval_soft.metadata.get("wait_timeout_soft_mode")) is True


def test_reaction_soft_wait_decays_and_clears_state() -> None:
    config = AppConfig()
    variant = BacktestVariant(code="W1", reaction_timeout_reset=True)
    wait_states: dict[str, object] = {}
    timeout_resets = {}
    wait_durations = {"REACTION": [], "MITIGATION": []}
    reset_block_bar: dict[str, int] = {}
    timeout_samples: list[object] = []
    key = "XAUUSD:SCALP_ICT_PA"

    eval_enter = _evaluation(setup_state="WAIT_REACTION")
    reasons_enter = _apply_reaction_gate_with_timeout(
        strategy_key=key,
        bar_index=100,
        now=datetime(2026, 2, 12, 12, 0, tzinfo=timezone.utc),
        evaluation=eval_enter,
        wait_states=wait_states,  # type: ignore[arg-type]
        variant=variant,
        config=config,
        timeout_resets=timeout_resets,  # type: ignore[arg-type]
        wait_durations=wait_durations,
        reset_block_bar=reset_block_bar,
        timeout_samples=timeout_samples,  # type: ignore[arg-type]
    )
    assert reasons_enter == ["GATE_REACTION_WAIT_REACTION"]

    eval_timeout = _evaluation(setup_state="WAIT_REACTION")
    _apply_reaction_gate_with_timeout(
        strategy_key=key,
        bar_index=109,
        now=datetime(2026, 2, 12, 12, 40, tzinfo=timezone.utc),
        evaluation=eval_timeout,
        wait_states=wait_states,  # type: ignore[arg-type]
        variant=variant,
        config=config,
        timeout_resets=timeout_resets,  # type: ignore[arg-type]
        wait_durations=wait_durations,
        reset_block_bar=reset_block_bar,
        timeout_samples=timeout_samples,  # type: ignore[arg-type]
    )

    eval_decay_clear = _evaluation(setup_state="WAIT_REACTION")
    reasons_clear = _apply_reaction_gate_with_timeout(
        strategy_key=key,
        bar_index=112,
        now=datetime(2026, 2, 12, 13, 0, tzinfo=timezone.utc),
        evaluation=eval_decay_clear,
        wait_states=wait_states,  # type: ignore[arg-type]
        variant=variant,
        config=config,
        timeout_resets=timeout_resets,  # type: ignore[arg-type]
        wait_durations=wait_durations,
        reset_block_bar=reset_block_bar,
        timeout_samples=timeout_samples,  # type: ignore[arg-type]
    )
    assert reasons_clear == []
    assert key not in wait_states
    assert bool(eval_decay_clear.metadata.get("wait_soft_decay_cleared")) is True


def test_execution_gate_blocks_when_confirmations_below_min_for_candidate() -> None:
    now = datetime(2026, 2, 12, 12, 0, tzinfo=timezone.utc)
    evaluation = StrategyEvaluation(
        action=DecisionAction.TRADE,
        score_total=70.0,
        score_breakdown={},
        reasons_blocking=[],
        would_enter_if=[],
        snapshot={"spread": 0.1, "close": 2500.0},
        metadata={"atr_m5": 1.0, "trigger_confirmations": 2, "candidate_id": "cand-1"},
    )
    _, reasons = _evaluate_hard_gates(
        route_params={"quality_gates": {"spread_ratio_max": 0.2, "min_confirm": 3, "min_confirm_small": 3}},
        evaluation=evaluation,
        now=now,
        timezone_name="Europe/Warsaw",
    )
    assert "EXEC_FAIL_CONFIRMATIONS_LOW" in reasons


def test_execution_gate_downgrades_trade_to_small_on_partial_confirmations() -> None:
    now = datetime(2026, 2, 12, 12, 0, tzinfo=timezone.utc)
    evaluation = StrategyEvaluation(
        action=DecisionAction.TRADE,
        score_total=71.0,
        score_breakdown={},
        reasons_blocking=[],
        would_enter_if=[],
        snapshot={"spread": 0.1, "close": 2500.0},
        metadata={"atr_m5": 1.0, "trigger_confirmations": 2, "candidate_id": "cand-2"},
    )
    gates, reasons = _evaluate_hard_gates(
        route_params={
            "quality_gates": {
                "spread_ratio_max": 0.2,
                "min_confirm_trade": 3,
                "min_confirm_small": 2,
            }
        },
        evaluation=evaluation,
        now=now,
        timezone_name="Europe/Warsaw",
    )
    assert gates["ExecutionGate"] is True
    assert "EXEC_FAIL_CONFIRMATIONS_LOW" not in reasons
    assert evaluation.action == DecisionAction.SMALL
    soft_reasons = evaluation.metadata.get("soft_reasons")
    assert isinstance(soft_reasons, list)
    assert "CONFIRMATIONS_DOWNGRADE_SMALL" in soft_reasons
