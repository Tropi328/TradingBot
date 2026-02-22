from __future__ import annotations

import pytest

from bot.config import BacktestRuntimeConfig, ResearchConfig


def test_research_config_defaults() -> None:
    cfg = ResearchConfig()
    assert cfg.objective_mode == "pnl_dd_cap"
    assert cfg.dd_cap_pct == 25.0
    assert cfg.dd_cap_basis == "both"
    assert cfg.max_workers == 3
    assert cfg.symbols == ["XAUUSD", "BTCUSD"]
    assert cfg.optimize.runtime_budget == "deep"
    assert cfg.optimize.objective_mode == "risk_adjusted_pnl_dd"
    assert cfg.optimize.top_gate_keep == 10
    assert len(cfg.search_space.risk_profiles) == 12


def test_research_config_rejects_invalid_dd_cap() -> None:
    with pytest.raises(ValueError, match="dd_cap_pct"):
        ResearchConfig(dd_cap_pct=0.0)


def test_research_config_rejects_invalid_dd_cap_basis() -> None:
    with pytest.raises(ValueError, match="dd_cap_basis"):
        ResearchConfig(dd_cap_basis="invalid")


def test_research_config_accepts_risk_adjusted_mode() -> None:
    cfg = ResearchConfig(objective_mode="risk_adjusted_pnl_dd")
    assert cfg.objective_mode == "risk_adjusted_pnl_dd"


def test_backtest_runtime_parallel_workers_are_capped() -> None:
    cfg = BacktestRuntimeConfig(parallel_workers=10)
    assert cfg.parallel_workers == 3
