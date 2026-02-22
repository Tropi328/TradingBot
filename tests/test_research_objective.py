from __future__ import annotations

from bot.research.objective import (
    OBJECTIVE_FAIL_VALUE,
    aggregate_reports,
    augment_report,
    compute_objective_value,
    objective_rank_key,
)


def test_augment_report_adds_required_research_fields() -> None:
    report = {
        "trades": 140,
        "total_pnl_net": 123.4,
        "max_drawdown": 10.0,
        "rejected_by_reason": {"SIZE_TOO_SMALL": 4},
        "blocked_by_gate_reasons": {"NEWS_WINDOW": 2},
        "spread_cost_sum": 1.0,
        "slippage_cost_sum": 2.0,
        "commission_cost_sum": 3.0,
        "swap_cost_sum": 4.0,
        "fx_cost_sum": 5.0,
    }
    enriched = augment_report(
        report,
        initial_equity=100.0,
        dd_cap_pct=25.0,
        dd_cap_basis="both",
        min_trades_oos=120,
        objective_mode="pnl_dd_cap",
    )

    assert enriched["oos_pass"] is True
    assert enriched["constraint_dd_cap_pass"] is True
    assert enriched["constraint_dd_cap_pass_peak"] is True
    assert enriched["constraint_dd_cap_pass_initial"] is True
    assert enriched["objective_value"] == 123.4
    assert enriched["blocked_by_reason"]["SIZE_TOO_SMALL"] == 4
    assert enriched["blocked_by_reason"]["NEWS_WINDOW"] == 2
    assert enriched["cost_breakdown_net"]["fx_cost_sum"] == 5.0
    assert enriched["max_drawdown_pct_peak"] == 10.0
    assert enriched["max_drawdown_pct_initial"] == 10.0
    assert enriched["max_drawdown_pct"] == 10.0


def test_aggregate_reports_applies_dd_cap_objective_filter() -> None:
    reports = [
        {
            "trades": 80,
            "total_pnl_net": 50.0,
            "max_drawdown_pct_peak": 30.0,
            "max_drawdown_pct_initial": 30.0,
            "expectancy": 1.0,
        },
        {
            "trades": 60,
            "total_pnl_net": 30.0,
            "max_drawdown_pct_peak": 12.0,
            "max_drawdown_pct_initial": 12.0,
            "expectancy": 2.0,
        },
    ]
    summary = aggregate_reports(
        reports,
        initial_equity=100.0,
        dd_cap_pct=25.0,
        dd_cap_basis="both",
        min_trades_oos=120,
        objective_mode="pnl_dd_cap",
    )

    assert summary["trades"] == 140
    assert summary["max_drawdown_pct_peak"] == 30.0
    assert summary["max_drawdown_pct_initial"] == 30.0
    assert summary["constraint_dd_cap_pass"] is False
    assert summary["objective_value"] == OBJECTIVE_FAIL_VALUE


def test_dd_cap_basis_initial_peak_both() -> None:
    report = {
        "trades": 200,
        "total_pnl_net": 10.0,
        "max_drawdown": 40.0,
        "max_drawdown_pct_peak": 10.0,
    }
    initial_only = augment_report(
        report,
        initial_equity=100.0,
        dd_cap_pct=25.0,
        dd_cap_basis="initial",
        min_trades_oos=1,
        objective_mode="pnl_dd_cap",
    )
    peak_only = augment_report(
        report,
        initial_equity=100.0,
        dd_cap_pct=25.0,
        dd_cap_basis="peak",
        min_trades_oos=1,
        objective_mode="pnl_dd_cap",
    )
    both = augment_report(
        report,
        initial_equity=100.0,
        dd_cap_pct=25.0,
        dd_cap_basis="both",
        min_trades_oos=1,
        objective_mode="pnl_dd_cap",
    )

    assert initial_only["max_drawdown_pct_initial"] == 40.0
    assert initial_only["constraint_dd_cap_pass_peak"] is True
    assert initial_only["constraint_dd_cap_pass_initial"] is False
    assert initial_only["constraint_dd_cap_pass"] is False

    assert peak_only["constraint_dd_cap_pass"] is True
    assert peak_only["objective_value"] == 10.0

    assert both["constraint_dd_cap_pass"] is False
    assert both["objective_value"] == OBJECTIVE_FAIL_VALUE


def test_objective_rank_key_prefers_higher_objective_then_lower_dd() -> None:
    better = {"objective_value": 10.0, "max_drawdown_pct_peak": 12.0, "expectancy_net": 1.0}
    worse = {"objective_value": 5.0, "max_drawdown_pct_peak": 1.0, "expectancy_net": 9.0}
    assert objective_rank_key(better) < objective_rank_key(worse)


def test_compute_objective_value_risk_adjusted_uses_dd_floor() -> None:
    value = compute_objective_value(
        objective_mode="risk_adjusted_pnl_dd",
        total_pnl_net=0.0,
        oos_pass=True,
        constraint_dd_cap_pass=True,
        oos_total_pnl_net=100.0,
        oos_dd_ref_pct=0.0,
    )
    assert value == 400.0


def test_aggregate_reports_risk_adjusted_fails_when_dd_cap_broken() -> None:
    reports = [
        {
            "trades": 150,
            "total_pnl_net": 100.0,
            "max_drawdown_pct_peak": 30.0,
            "max_drawdown_pct_initial": 30.0,
            "expectancy_net": 1.0,
        }
    ]
    summary = aggregate_reports(
        reports,
        initial_equity=100.0,
        dd_cap_pct=25.0,
        dd_cap_basis="both",
        min_trades_oos=120,
        objective_mode="risk_adjusted_pnl_dd",
    )
    assert summary["constraint_dd_cap_pass"] is False
    assert summary["objective_value"] == OBJECTIVE_FAIL_VALUE
