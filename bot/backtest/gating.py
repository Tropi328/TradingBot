"""Reaction/wait gate logic — extracted from engine.py."""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from bot.backtest.models import BacktestVariant, _ReactionTimeoutSample, _WaitGateState
from bot.config import AppConfig
from bot.strategy.contracts import DecisionAction, StrategyEvaluation
from bot.strategy.utils import as_float


def _apply_wait_timeout_soft_mode(
    *,
    evaluation: StrategyEvaluation,
    config: AppConfig,
) -> StrategyEvaluation:
    if not bool(evaluation.metadata.get("wait_timeout_soft_mode")):
        return evaluation

    soft_penalty = max(0.0, float(config.backtest_tuning.wait_timeout_soft_penalty))
    if evaluation.score_total is not None and soft_penalty > 0:
        score_now = max(0.0, float(evaluation.score_total) - soft_penalty)
        evaluation.score_total = round(score_now, 2)
        evaluation.score_breakdown["penalty_wait_timeout_soft"] = -round(soft_penalty, 4)

    if evaluation.action == DecisionAction.TRADE:
        evaluation.action = DecisionAction.SMALL

    current_override = evaluation.metadata.get("risk_multiplier_override")
    current_value = as_float(current_override, 1.0)
    timeout_small = max(0.01, min(1.0, float(config.backtest_tuning.wait_timeout_small_risk_multiplier)))
    evaluation.metadata["risk_multiplier_override"] = min(current_value, timeout_small)

    soft_reasons = evaluation.metadata.get("soft_reasons")
    if not isinstance(soft_reasons, list):
        soft_reasons = []
    if "WAIT_TIMEOUT_SOFT_MODE" not in soft_reasons:
        soft_reasons.append("WAIT_TIMEOUT_SOFT_MODE")
    evaluation.metadata["soft_reasons"] = soft_reasons
    return evaluation


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
        reset_reason = "REACTION_TIMEOUT_SOFT_REACTION"
    elif setup_state == "WAIT_MITIGATION":
        wait_type = "MITIGATION"
        timeout_bars = int(config.backtest_tuning.wait_mitigation_timeout_bars)
        base_reason = "GATE_REACTION_WAIT_MITIGATION"
        reset_reason = "REACTION_TIMEOUT_SOFT_MITIGATION"
    else:
        state = wait_states.pop(strategy_key, None)
        if state is not None and not state.timed_out_soft:
            wait_durations.setdefault(state.wait_type, []).append(max(0, bar_index - state.enter_bar_index))
        return []

    locked_bar = reset_block_bar.get(strategy_key)
    if locked_bar is not None and locked_bar < bar_index:
        reset_block_bar.pop(strategy_key, None)
        locked_bar = None
    if locked_bar is not None and bar_index <= locked_bar:
        evaluation.metadata["wait_soft_grace_active"] = True
        evaluation.metadata["wait_timeout_soft_mode"] = True
        evaluation.metadata["setup_state"] = "SOFT_READY"
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
    if state.timed_out_soft:
        soft_decay_bars = max(0, int(config.backtest_tuning.wait_timeout_soft_grace_bars))
        if elapsed > (timeout_bars + soft_decay_bars):
            wait_states.pop(strategy_key, None)
            if soft_decay_bars > 0:
                reset_block_bar[strategy_key] = bar_index + soft_decay_bars
            evaluation.metadata["wait_soft_decay_cleared"] = True
            evaluation.metadata["setup_state"] = "SOFT_READY"
            soft_reasons = evaluation.metadata.get("soft_reasons")
            if not isinstance(soft_reasons, list):
                soft_reasons = []
            clear_code = f"REACTION_SOFT_DECAY_CLEAR_{wait_type}"
            if clear_code not in soft_reasons:
                soft_reasons.append(clear_code)
            evaluation.metadata["soft_reasons"] = soft_reasons
            return []
        evaluation.metadata["wait_timeout_soft_mode"] = True
        evaluation.metadata["wait_timeout_type"] = wait_type
        soft_reasons = evaluation.metadata.get("soft_reasons")
        if not isinstance(soft_reasons, list):
            soft_reasons = []
        soft_code = f"REACTION_TIMEOUT_SOFT_{wait_type}"
        if soft_code not in soft_reasons:
            soft_reasons.append(soft_code)
        evaluation.metadata["soft_reasons"] = soft_reasons
        return []
    timeouts_enabled = bool(config.backtest_tuning.reaction_timeout_force_enable or variant.reaction_timeout_reset)
    if timeouts_enabled and elapsed > timeout_bars:
        state.timed_out_soft = True
        wait_states[strategy_key] = state
        wait_durations.setdefault(wait_type, []).append(elapsed)
        timeout_resets[wait_type] = int(timeout_resets.get(wait_type, 0)) + 1
        timeout_resets["REACTION_TIMEOUT_RESET"] = int(timeout_resets.get("REACTION_TIMEOUT_RESET", 0)) + 1
        evaluation.metadata["reaction_timeout_reset"] = False
        evaluation.metadata["reaction_timeout_bars"] = elapsed
        evaluation.metadata["wait_timeout_soft_mode"] = True
        evaluation.metadata["wait_timeout_type"] = wait_type
        soft_reasons = evaluation.metadata.get("soft_reasons")
        if not isinstance(soft_reasons, list):
            soft_reasons = []
        soft_code = f"REACTION_TIMEOUT_SOFT_{wait_type}"
        if soft_code not in soft_reasons:
            soft_reasons.append(soft_code)
        evaluation.metadata["soft_reasons"] = soft_reasons
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
        return []
    hard_block_bars = max(0, int(config.backtest_tuning.wait_hard_block_bars))
    if elapsed > hard_block_bars:
        evaluation.metadata["wait_timeout_soft_mode"] = True
        evaluation.metadata["wait_timeout_type"] = wait_type
        evaluation.metadata["setup_state"] = "SOFT_READY"
        soft_reasons = evaluation.metadata.get("soft_reasons")
        if not isinstance(soft_reasons, list):
            soft_reasons = []
        progress_code = f"REACTION_WAIT_SOFT_{wait_type}"
        if progress_code not in soft_reasons:
            soft_reasons.append(progress_code)
        evaluation.metadata["soft_reasons"] = soft_reasons
        return []
    return [base_reason]
