"""Tests for tier-based sizing, compound equity, and multi-TP profile."""

from __future__ import annotations

import pytest

from bot.execution.sizing import (
    compute_compound_equity,
    position_size_from_risk,
    position_size_tiered,
    resolve_score_tier,
    tier_risk_multiplier,
)
from bot.execution.position_manager import (
    MultiTPProfile,
    TPLevel,
    build_multi_tp_profile,
)


# =========================================================================
# resolve_score_tier
# =========================================================================
class TestResolveScoreTier:
    def test_none_score(self):
        assert resolve_score_tier(None) == "OBSERVE"

    def test_default_thresholds(self):
        assert resolve_score_tier(85.0) == "A_PLUS"
        assert resolve_score_tier(80.0) == "A_PLUS"
        assert resolve_score_tier(70.0) == "A"
        assert resolve_score_tier(65.0) == "A"
        assert resolve_score_tier(60.0) == "B"
        assert resolve_score_tier(55.0) == "B"
        assert resolve_score_tier(50.0) == "OBSERVE"
        assert resolve_score_tier(0.0) == "OBSERVE"

    def test_custom_thresholds(self):
        class Cfg:
            a_plus_min_score = 90
            a_min_score = 70
            b_min_score = 50
        assert resolve_score_tier(91.0, tier_cfg=Cfg()) == "A_PLUS"
        assert resolve_score_tier(80.0, tier_cfg=Cfg()) == "A"
        assert resolve_score_tier(55.0, tier_cfg=Cfg()) == "B"
        assert resolve_score_tier(40.0, tier_cfg=Cfg()) == "OBSERVE"


# =========================================================================
# tier_risk_multiplier
# =========================================================================
class TestTierRiskMultiplier:
    def test_defaults(self):
        assert tier_risk_multiplier("A_PLUS") == 1.5
        assert tier_risk_multiplier("A") == 1.0
        assert tier_risk_multiplier("B") == 0.6
        assert tier_risk_multiplier("OBSERVE") == 0.0

    def test_unknown_tier_returns_1(self):
        assert tier_risk_multiplier("UNKNOWN") == 1.0

    def test_custom_config(self):
        class Cfg:
            a_plus_mult = 2.0
            a_mult = 1.2
            b_mult = 0.5
            observe_mult = 0.1
        assert tier_risk_multiplier("A_PLUS", tier_cfg=Cfg()) == 2.0
        assert tier_risk_multiplier("OBSERVE", tier_cfg=Cfg()) == 0.1


# =========================================================================
# compute_compound_equity
# =========================================================================
class TestComputeCompoundEquity:
    def test_normal(self):
        assert compute_compound_equity(150.0) == 150.0

    def test_floor(self):
        assert compute_compound_equity(30.0, floor_equity=50.0) == 50.0

    def test_cap(self):
        assert compute_compound_equity(200.0, cap_equity=150.0) == 150.0

    def test_floor_higher_than_raw(self):
        assert compute_compound_equity(40.0, floor_equity=100.0) == 100.0

    def test_cap_zero_ignored(self):
        assert compute_compound_equity(200.0, cap_equity=0.0) == 200.0

    def test_both_floor_and_cap(self):
        assert compute_compound_equity(120.0, floor_equity=50.0, cap_equity=100.0) == 100.0
        assert compute_compound_equity(30.0, floor_equity=50.0, cap_equity=100.0) == 50.0


# =========================================================================
# position_size_tiered
# =========================================================================
class TestPositionSizeTiered:
    def test_no_tier_no_compound(self):
        """Without tier/compound it should behave like plain sizing."""
        size, tier, meta = position_size_tiered(
            equity=10000.0,
            risk_per_trade=0.01,
            entry_price=100.0,
            stop_price=99.0,
            min_size=0.01,
            size_step=0.01,
        )
        expected = position_size_from_risk(
            equity=10000.0,
            risk_per_trade=0.01,
            entry_price=100.0,
            stop_price=99.0,
            min_size=0.01,
            size_step=0.01,
        )
        assert size == expected
        assert tier == "A"

    def test_tier_observe_returns_zero(self):
        class TierCfg:
            enabled = True
            a_plus_min_score = 80
            a_min_score = 65
            b_min_score = 55
            a_plus_mult = 1.5
            a_mult = 1.0
            b_mult = 0.6
            observe_mult = 0.0
        size, tier, meta = position_size_tiered(
            equity=10000.0,
            risk_per_trade=0.01,
            entry_price=100.0,
            stop_price=99.0,
            min_size=0.01,
            size_step=0.01,
            score=40.0,
            tier_cfg=TierCfg(),
        )
        assert size == 0.0
        assert tier == "OBSERVE"

    def test_session_risk_mult(self):
        size_full, _, _ = position_size_tiered(
            equity=10000.0,
            risk_per_trade=0.01,
            entry_price=100.0,
            stop_price=99.0,
            min_size=0.01,
            size_step=0.01,
            session_risk_mult=1.0,
        )
        size_half, _, meta = position_size_tiered(
            equity=10000.0,
            risk_per_trade=0.01,
            entry_price=100.0,
            stop_price=99.0,
            min_size=0.01,
            size_step=0.01,
            session_risk_mult=0.5,
        )
        # Half session risk → half size (approximately, due to rounding)
        assert size_half <= size_full
        assert meta["session_mult"] == 0.5

    def test_compound_equity(self):
        class CompCfg:
            enabled = True
            floor_equity = 50
            cap_equity = 0
        size_compound, _, meta = position_size_tiered(
            equity=200.0,
            risk_per_trade=0.01,
            entry_price=100.0,
            stop_price=99.0,
            min_size=0.01,
            size_step=0.01,
            compound_cfg=CompCfg(),
        )
        assert meta["sizing_equity"] == 200.0  # equity > floor → used as-is

    def test_tier_a_plus_mult(self):
        class TierCfg:
            enabled = True
            a_plus_min_score = 80
            a_min_score = 65
            b_min_score = 55
            a_plus_mult = 1.5
            a_mult = 1.0
            b_mult = 0.6
            observe_mult = 0.0
        size_a, _, _ = position_size_tiered(
            equity=10000.0,
            risk_per_trade=0.01,
            entry_price=100.0,
            stop_price=99.0,
            min_size=0.01,
            size_step=0.01,
            score=70.0,
            tier_cfg=TierCfg(),
        )
        size_aplus, _, _ = position_size_tiered(
            equity=10000.0,
            risk_per_trade=0.01,
            entry_price=100.0,
            stop_price=99.0,
            min_size=0.01,
            size_step=0.01,
            score=85.0,
            tier_cfg=TierCfg(),
        )
        assert size_aplus >= size_a  # A+ gets bigger position


# =========================================================================
# build_multi_tp_profile
# =========================================================================
class TestBuildMultiTPProfile:
    def test_disabled(self):
        profile = build_multi_tp_profile(None)
        assert profile.enabled is False
        assert profile.levels == ()

    def test_config_disabled(self):
        class Cfg:
            enabled = False
        profile = build_multi_tp_profile(Cfg())
        assert profile.enabled is False

    def test_enabled(self):
        class Level:
            def __init__(self, name, trigger_r, close_fraction, move_sl_to_be):
                self.name = name
                self.trigger_r = trigger_r
                self.close_fraction = close_fraction
                self.move_sl_to_be = move_sl_to_be

        class Cfg:
            enabled = True
            levels = [
                Level("TP3", 3.0, 1.0, False),
                Level("TP1", 1.0, 0.2, True),
                Level("TP2", 2.0, 0.5, False),
            ]
            be_offset_r = 0.05
            be_delay_bars = 4
            trailing_enabled = True
            trailing_swing_window = 12
            trailing_buffer_r = 0.12

        profile = build_multi_tp_profile(Cfg())
        assert profile.enabled is True
        assert len(profile.levels) == 3
        # Should be sorted by trigger_r ascending
        assert profile.levels[0].name == "TP1"
        assert profile.levels[1].name == "TP2"
        assert profile.levels[2].name == "TP3"
        assert profile.be_delay_bars == 4
        assert profile.trailing_buffer_r == 0.12


# =========================================================================
# MultiTPProfile / TPLevel dataclasses
# =========================================================================
class TestMultiTPProfile:
    def test_frozen(self):
        profile = MultiTPProfile()
        with pytest.raises(AttributeError):
            profile.enabled = True  # type: ignore[misc]

    def test_tp_level_frozen(self):
        level = TPLevel(name="TP1", trigger_r=1.0, close_fraction=0.2, move_sl_to_be=True)
        with pytest.raises(AttributeError):
            level.trigger_r = 2.0  # type: ignore[misc]

    def test_default_profile(self):
        profile = MultiTPProfile()
        assert profile.enabled is False
        assert profile.levels == ()
        assert profile.be_offset_r == 0.05
        assert profile.trailing_enabled is True


# =========================================================================
# EDGE_TOO_SMALL in soft gates
# =========================================================================
class TestEdgeTooSmallSoftGate:
    def test_edge_too_small_is_soft_eligible(self):
        from bot.gating.adaptive import SOFT_GATE_ELIGIBLE, apply_soft_gates

        assert "EDGE_TOO_SMALL" in SOFT_GATE_ELIGIBLE

    def test_edge_too_small_becomes_penalty(self):
        from bot.gating.adaptive import apply_soft_gates

        result = apply_soft_gates(
            ["EDGE_TOO_SMALL"],
            soft_gates_enabled=True,
            soft_gate_penalty=4.0,
        )
        assert not result.blocked
        assert result.hard_reasons == []
        assert "EDGE_TOO_SMALL" in result.soft_reasons
        assert result.total_penalty == 4.0

    def test_edge_too_small_mixed_with_hard(self):
        from bot.gating.adaptive import apply_soft_gates

        result = apply_soft_gates(
            ["EDGE_TOO_SMALL", "EXEC_FAIL_MISSING_FEATURES"],
            soft_gates_enabled=True,
            soft_gate_penalty=4.0,
        )
        assert result.blocked
        assert "EXEC_FAIL_MISSING_FEATURES" in result.hard_reasons
        assert "EDGE_TOO_SMALL" in result.soft_reasons
