from __future__ import annotations

from bot.strategy.contracts import StrategyEvaluation


def rank_score(evaluation: StrategyEvaluation) -> float:
    base = float(evaluation.score_total or 0.0)
    confirmations = float(evaluation.metadata.get("trigger_confirmations", 0.0))
    execution_penalty = float(evaluation.metadata.get("execution_penalty", 0.0))
    execution_bonus = float(evaluation.metadata.get("execution_bonus", 0.0))
    return base + (2.0 * confirmations) + execution_bonus - execution_penalty
