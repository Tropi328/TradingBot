"""Tests for new PnL-upgrade config models (MultiTP, TierSizing, Session, Compound)."""

from __future__ import annotations

import pytest

from bot.config import (
    AppConfig,
    CompoundEquityConfig,
    MultiTPConfig,
    SessionFilterConfig,
    TPLevelConfig,
    TierSizingConfig,
)


class TestMultiTPConfig:
    def test_defaults(self):
        cfg = MultiTPConfig()
        assert cfg.enabled is False
        assert len(cfg.levels) == 3
        assert cfg.levels[0].name == "TP1"
        assert cfg.levels[0].trigger_r == 1.0
        assert cfg.levels[0].close_fraction == 0.2
        assert cfg.levels[0].move_sl_to_be is True
        assert cfg.be_delay_bars == 4
        assert cfg.trailing_enabled is True

    def test_custom(self):
        cfg = MultiTPConfig(
            enabled=True,
            levels=[
                TPLevelConfig(name="X1", trigger_r=0.5, close_fraction=0.5, move_sl_to_be=False),
            ],
            be_offset_r=0.1,
        )
        assert cfg.enabled is True
        assert len(cfg.levels) == 1
        assert cfg.be_offset_r == 0.1


class TestTierSizingConfig:
    def test_defaults(self):
        cfg = TierSizingConfig()
        assert cfg.enabled is False
        assert cfg.a_plus_mult == 1.5
        assert cfg.a_mult == 1.0
        assert cfg.b_mult == 0.6
        assert cfg.observe_mult == 0.0
        assert cfg.a_plus_min_score == 80
        assert cfg.a_min_score == 65
        assert cfg.b_min_score == 55

    def test_custom(self):
        cfg = TierSizingConfig(enabled=True, a_plus_mult=2.0)
        assert cfg.a_plus_mult == 2.0


class TestSessionFilterConfig:
    def test_defaults(self):
        cfg = SessionFilterConfig()
        assert cfg.enabled is False
        assert len(cfg.sessions) == 4
        assert cfg.block_outside_sessions is False

    def test_session_names(self):
        cfg = SessionFilterConfig()
        names = [s.name for s in cfg.sessions]
        assert "LONDON" in names
        assert "NY_OVERLAP" in names


class TestCompoundEquityConfig:
    def test_defaults(self):
        cfg = CompoundEquityConfig()
        assert cfg.enabled is False
        assert cfg.floor_equity == 50.0
        assert cfg.cap_equity == 0.0
        assert cfg.smooth_window == 0


class TestAppConfigPnlUpgradeFields:
    def test_fields_exist_with_defaults(self):
        """AppConfig should have all four new fields with sane defaults."""
        # Load minimal config — just check the fields exist
        cfg = AppConfig()
        assert hasattr(cfg, "multi_tp")
        assert hasattr(cfg, "tier_sizing")
        assert hasattr(cfg, "session_filter")
        assert hasattr(cfg, "compound_equity")
        assert cfg.multi_tp.enabled is False
        assert cfg.tier_sizing.enabled is False
        assert cfg.session_filter.enabled is False
        assert cfg.compound_equity.enabled is False
