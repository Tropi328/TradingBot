from __future__ import annotations

from bot.research.optimizer import optimizer_rank_key


def test_optimizer_rank_key_tie_break_policy() -> None:
    best = {
        "objective_value": 10.0,
        "oos_total_pnl_net": 100.0,
        "oos_dd_ref_pct": 5.0,
        "oos_expectancy_net": 2.0,
    }
    lower_pnl = {
        "objective_value": 10.0,
        "oos_total_pnl_net": 90.0,
        "oos_dd_ref_pct": 1.0,
        "oos_expectancy_net": 5.0,
    }
    higher_dd = {
        "objective_value": 10.0,
        "oos_total_pnl_net": 100.0,
        "oos_dd_ref_pct": 6.0,
        "oos_expectancy_net": 3.0,
    }
    lower_expectancy = {
        "objective_value": 10.0,
        "oos_total_pnl_net": 100.0,
        "oos_dd_ref_pct": 5.0,
        "oos_expectancy_net": 1.0,
    }

    assert optimizer_rank_key(best) < optimizer_rank_key(lower_pnl)
    assert optimizer_rank_key(best) < optimizer_rank_key(higher_dd)
    assert optimizer_rank_key(best) < optimizer_rank_key(lower_expectancy)
