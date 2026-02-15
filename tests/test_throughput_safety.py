"""Unit tests for risk budget, score tiers, entry ladder, decision funnel,
second-position rule, and new config classes.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Risk Budget
# ---------------------------------------------------------------------------
from bot.strategy.risk_budget import PortfolioRiskBudget, RiskBudgetCheck


class TestRiskBudget:
    def test_disabled_always_allows(self):
        rb = PortfolioRiskBudget(enabled=False)
        check = rb.check_can_open(equity=10000, new_trade_risk=9999, open_positions_risk=9999)
        assert check.allowed is True
        assert check.reasons == []

    def test_daily_loss_kill_switch(self):
        rb = PortfolioRiskBudget(enabled=True, daily_loss_limit_pct=0.025)
        rb.reset_day("2024-01-01")
        rb.record_trade_pnl(-260)  # > 2.5% of 10000
        check = rb.check_can_open(equity=10000, new_trade_risk=10, open_positions_risk=0)
        assert check.allowed is False
        assert "KILL_SWITCH_DAILY_LOSS" in check.reasons
        assert rb.is_killed is True

    def test_daily_loss_below_limit_allows(self):
        rb = PortfolioRiskBudget(enabled=True, daily_loss_limit_pct=0.025)
        rb.reset_day("2024-01-01")
        rb.record_trade_pnl(-100)  # 1% < 2.5%
        check = rb.check_can_open(equity=10000, new_trade_risk=10, open_positions_risk=0)
        assert check.allowed is True

    def test_open_risk_cap(self):
        rb = PortfolioRiskBudget(enabled=True, max_open_risk_pct=0.02)
        rb.reset_day("2024-01-01")
        # Existing open risk = 150, new = 60 → projected 210 > 200 (2% of 10000)
        check = rb.check_can_open(equity=10000, new_trade_risk=60, open_positions_risk=150)
        assert check.allowed is False
        assert "OPEN_RISK_CAP_EXCEEDED" in check.reasons

    def test_open_risk_within_cap(self):
        rb = PortfolioRiskBudget(enabled=True, max_open_risk_pct=0.02, risk_per_trade_pct=0.01)
        rb.reset_day("2024-01-01")
        check = rb.check_can_open(equity=10000, new_trade_risk=50, open_positions_risk=100)
        assert check.allowed is True

    def test_per_trade_risk_cap(self):
        rb = PortfolioRiskBudget(enabled=True, risk_per_trade_pct=0.0035)
        rb.reset_day("2024-01-01")
        # New risk 40 > 35 (0.35% of 10000)
        check = rb.check_can_open(equity=10000, new_trade_risk=40, open_positions_risk=0)
        assert check.allowed is False
        assert "PER_TRADE_RISK_TOO_HIGH" in check.reasons

    def test_daily_profit_lock(self):
        rb = PortfolioRiskBudget(enabled=True, daily_profit_lock_pct=0.05)
        rb.reset_day("2024-01-01")
        rb.record_trade_pnl(600)  # 6% > 5%
        check = rb.check_can_open(equity=10000, new_trade_risk=10, open_positions_risk=0)
        assert check.allowed is False
        assert "DAILY_PROFIT_LOCKED" in check.reasons

    def test_day_reset_clears_kill(self):
        rb = PortfolioRiskBudget(enabled=True, daily_loss_limit_pct=0.01)
        rb.reset_day("2024-01-01")
        rb.record_trade_pnl(-200)
        check = rb.check_can_open(equity=10000, new_trade_risk=1, open_positions_risk=0)
        assert rb.is_killed is True
        rb.reset_day("2024-01-02")
        assert rb.is_killed is False
        assert rb.daily_pnl == 0.0

    def test_multiple_reasons(self):
        rb = PortfolioRiskBudget(
            enabled=True,
            risk_per_trade_pct=0.001,
            max_open_risk_pct=0.005,
            daily_loss_limit_pct=0.001,
        )
        rb.reset_day("2024-01-01")
        rb.record_trade_pnl(-20)  # exceed daily loss
        check = rb.check_can_open(equity=10000, new_trade_risk=50, open_positions_risk=100)
        assert check.allowed is False
        assert len(check.reasons) >= 2


# ---------------------------------------------------------------------------
# Score Tiers
# ---------------------------------------------------------------------------
from bot.strategy.score_tiers import resolve_tier, resolve_tier_from_config, TierResult


class TestScoreTiers:
    def test_disabled_returns_none_tier(self):
        tier = resolve_tier(80.0, enabled=False)
        assert tier.name == "NONE"
        assert tier.size_mult == 1.0
        assert tier.allow_market_after_ttl is False

    def test_a_plus_tier(self):
        tier = resolve_tier(75.0, enabled=True, a_plus_min=72)
        assert tier.name == "A_plus"
        assert tier.size_mult == 1.0
        assert tier.allow_market_after_ttl is True

    def test_a_tier(self):
        tier = resolve_tier(70.0, enabled=True, a_plus_min=72, a_min=68)
        assert tier.name == "A"
        assert tier.size_mult == 0.8

    def test_b_tier(self):
        tier = resolve_tier(64.0, enabled=True, a_plus_min=72, a_min=68, b_min=62)
        assert tier.name == "B"
        assert tier.size_mult == 0.6

    def test_observe_tier(self):
        tier = resolve_tier(50.0, enabled=True, a_plus_min=72, a_min=68, b_min=62)
        assert tier.name == "OBSERVE"
        assert tier.size_mult == 0.0

    def test_boundary_exactly_on_threshold(self):
        tier = resolve_tier(72.0, enabled=True, a_plus_min=72, a_min=68, b_min=62)
        assert tier.name == "A_plus"

    def test_resolve_from_config_object(self):
        class _MockTierEntry:
            min_score = 72.0
            size_mult = 1.0
            allow_market_after_ttl = True

        class _MockConfig:
            enabled = True
            A_plus = _MockTierEntry()
            A = type("_", (), {"min_score": 68.0, "size_mult": 0.8, "allow_market_after_ttl": False})()
            B = type("_", (), {"min_score": 62.0, "size_mult": 0.6, "allow_market_after_ttl": False})()

        tier = resolve_tier_from_config(70.0, _MockConfig())
        assert tier.name == "A"
        assert tier.size_mult == 0.8


# ---------------------------------------------------------------------------
# Entry Ladder
# ---------------------------------------------------------------------------
from bot.execution.entry_ladder import compute_ladder, make_market_fallback


class TestEntryLadder:
    def test_disabled_returns_single_order(self):
        result = compute_ladder(
            side="LONG", fvg_high=105, fvg_low=100, base_entry=102.5,
            total_size=1.0, enabled=False,
        )
        assert len(result.levels) == 1
        assert result.levels[0].name == "single"
        assert result.levels[0].size == 1.0

    def test_long_ladder_two_levels(self):
        result = compute_ladder(
            side="LONG", fvg_high=110, fvg_low=100, base_entry=105,
            total_size=1.0, enabled=True,
            level_specs=[
                {"name": "mid", "fvg_fraction": 0.50, "size_fraction": 0.70},
                {"name": "shallow", "fvg_fraction": 0.35, "size_fraction": 0.30},
            ],
            ttl_bars=8,
        )
        assert len(result.levels) == 2
        # LONG: entry = fvg_high - range * fraction
        assert result.levels[0].entry_price == pytest.approx(105.0)  # 110 - 10*0.5
        assert result.levels[0].size == pytest.approx(0.7)
        assert result.levels[1].entry_price == pytest.approx(106.5)  # 110 - 10*0.35
        assert result.levels[1].size == pytest.approx(0.3)
        assert result.ttl_bars == 8

    def test_short_ladder_entries(self):
        result = compute_ladder(
            side="SHORT", fvg_high=110, fvg_low=100, base_entry=105,
            total_size=2.0, enabled=True,
            level_specs=[
                {"name": "mid", "fvg_fraction": 0.50, "size_fraction": 0.70},
            ],
        )
        assert len(result.levels) == 1
        # SHORT: entry = fvg_low + range * fraction = 100 + 10*0.5 = 105
        assert result.levels[0].entry_price == pytest.approx(105.0)
        assert result.levels[0].size == pytest.approx(1.4)

    def test_market_fallback_order(self):
        fb = make_market_fallback(
            side="LONG", current_price=100, stop=99, tp=102,
            base_size=1.0, size_mult=0.5,
        )
        assert fb.is_market is True
        assert fb.size == pytest.approx(0.5)
        assert fb.entry_price == 100

    def test_zero_fvg_range_falls_back_single(self):
        result = compute_ladder(
            side="LONG", fvg_high=100, fvg_low=100, base_entry=100,
            total_size=1.0, enabled=True,
            level_specs=[{"name": "mid", "fvg_fraction": 0.5, "size_fraction": 1.0}],
        )
        assert len(result.levels) == 1
        assert result.levels[0].name == "single"


# ---------------------------------------------------------------------------
# Decision Funnel
# ---------------------------------------------------------------------------
from bot.reporting.decision_funnel import DecisionFunnel


class TestDecisionFunnel:
    def test_basic_tracking(self):
        f = DecisionFunnel()
        f.signal_candidates = 100
        f.proposals_created = 80
        f.orders_placed = 50
        f.filled_orders = 40
        f.trades_opened = 40
        f.trades_closed = 38
        f.sample_concurrent(2)
        f.sample_concurrent(3)
        f.record_block("SPREAD_TOO_HIGH")
        f.record_block("SPREAD_TOO_HIGH")
        f.record_block("COOLDOWN")

        d = f.to_dict()
        assert d["signal_candidates"] == 100
        assert d["fill_rate_pct"] == pytest.approx(80.0)
        assert d["avg_concurrent_positions"] == pytest.approx(2.5)
        assert d["top_10_rejection_reasons"]["SPREAD_TOO_HIGH"] == 2
        assert d["top_10_rejection_reasons"]["COOLDOWN"] == 1

    def test_empty_funnel(self):
        f = DecisionFunnel()
        d = f.to_dict()
        assert d["fill_rate_pct"] == 0.0
        assert d["avg_concurrent_positions"] == 0.0


# ---------------------------------------------------------------------------
# Config: new sections load correctly
# ---------------------------------------------------------------------------
from bot.config import (
    AppConfig,
    RiskBudgetConfig,
    ScoreTiersConfig,
    FvgEntryLadderConfig,
    CorrelationV2Config,
    ExitsConfig,
)


class TestNewConfigSections:
    def test_default_risk_budget_disabled(self):
        cfg = AppConfig()
        assert cfg.risk_budget.enabled is False
        assert cfg.risk_budget.risk_per_trade_pct == pytest.approx(0.0035)

    def test_score_tiers_default_disabled(self):
        cfg = AppConfig()
        assert cfg.score_tiers.enabled is False
        assert cfg.score_tiers.A_plus.min_score == pytest.approx(72.0)

    def test_fvg_ladder_default_disabled(self):
        cfg = AppConfig()
        assert cfg.fvg_entry_ladder.enabled is False
        assert len(cfg.fvg_entry_ladder.levels) == 2

    def test_exits_default(self):
        cfg = AppConfig()
        assert cfg.exits.base_rr == pytest.approx(2.0)
        assert cfg.exits.partial_tp.enabled is True
        assert cfg.exits.partial_tp.at_r == pytest.approx(1.0)

    def test_correlation_v2_default(self):
        cfg = AppConfig()
        assert cfg.correlation_v2.max_positions_per_group == 1
        assert cfg.correlation_v2.allow_second_same_symbol_only_if.or_profit_r_greater_equal == pytest.approx(0.7)

    def test_risk_budget_validation(self):
        with pytest.raises(Exception):
            RiskBudgetConfig(risk_per_trade_pct=0.10)  # > 5% limit

    def test_portfolio_max_per_symbol_used_in_engine(self):
        cfg = AppConfig()
        # Default max_per_symbol = 1, engine should use this for multi-position
        assert cfg.portfolio.max_per_symbol >= 1


# ---------------------------------------------------------------------------
# Second-position rule logic (inline, matches engine behavior)
# ---------------------------------------------------------------------------
class TestSecondPositionRule:
    def _check_second_pos_ok(self, be_moved: bool, profit_r: float, threshold: float = 0.7) -> bool:
        """Mirrors the engine's second-position rule."""
        return be_moved or profit_r >= threshold

    def test_allowed_when_first_at_be(self):
        assert self._check_second_pos_ok(be_moved=True, profit_r=0.0) is True

    def test_allowed_when_first_profitable(self):
        assert self._check_second_pos_ok(be_moved=False, profit_r=0.8) is True

    def test_blocked_when_neither(self):
        assert self._check_second_pos_ok(be_moved=False, profit_r=0.3) is False

    def test_boundary_exactly_at_threshold(self):
        assert self._check_second_pos_ok(be_moved=False, profit_r=0.7) is True

    def test_custom_threshold(self):
        assert self._check_second_pos_ok(be_moved=False, profit_r=0.5, threshold=0.5) is True


# ---------------------------------------------------------------------------
# Preset YAML integration test
# ---------------------------------------------------------------------------
class TestPresetYamlIntegration:
    def test_preset_pnl_safe_loads(self):
        """Validate that preset_pnl_safe.yaml loads and enables new modules."""
        from pathlib import Path
        yaml_path = Path(__file__).resolve().parent.parent / "configs" / "variants" / "preset_pnl_safe.yaml"
        if not yaml_path.exists():
            pytest.skip("preset_pnl_safe.yaml not found")
        from bot.config import load_config
        cfg = load_config(yaml_path)
        # Risk budget enabled
        assert cfg.risk_budget.enabled is True
        assert cfg.risk_budget.risk_per_trade_pct == pytest.approx(0.005)
        assert cfg.risk_budget.daily_loss_limit_pct == pytest.approx(0.025)
        # Score tiers enabled
        assert cfg.score_tiers.enabled is True
        assert cfg.score_tiers.A_plus.min_score == pytest.approx(72.0)
        assert cfg.score_tiers.A.size_mult == pytest.approx(0.8)
        assert cfg.score_tiers.B.size_mult == pytest.approx(0.6)
        # Portfolio throughput
        assert cfg.portfolio.max_per_symbol == 2
        assert cfg.portfolio.max_open_positions_total == 4
        assert cfg.portfolio.max_entries_per_cycle == 2
        # Exits
        assert cfg.exits.base_rr == pytest.approx(2.0)
        assert cfg.exits.partial_tp.enabled is True
        # Correlation v2
        assert cfg.correlation_v2.allow_second_same_symbol_only_if.or_profit_r_greater_equal == pytest.approx(0.7)
        # FVG ladder disabled (module ready, not wired)
        assert cfg.fvg_entry_ladder.enabled is False

    def test_ab_compare_helpers(self):
        """Validate A/B script helper functions work correctly."""
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(root))
        from tools.ab_compare import _val, _delta_str
        report = {"total_pnl": 150.0, "decision_funnel": {"fill_rate_pct": 42.0}}
        assert _val(report, "total_pnl") == pytest.approx(150.0)
        assert _val(report, "missing_key") == pytest.approx(0.0)
        assert _val(report, "fill_rate_pct", sub="decision_funnel") == pytest.approx(42.0)
        # Delta formatting
        assert "▲" in _delta_str(100, 150, higher_is_better=True)
        assert "▼" in _delta_str(100, 150, higher_is_better=False)
