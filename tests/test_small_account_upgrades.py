"""Tests for small-account upgrades: relative threshold, auto min_risk_cash, TP gradient."""
from __future__ import annotations

import pytest

from bot.config import RiskConfig, BacktestTuningConfig
from bot.strategy.risk import RiskEngine
from bot.backtest.engine import _resolve_tp_target_r


# =====================================================================
# 1. Relative low-equity threshold (low_equity_threshold_pct)
# =====================================================================

class TestRelativeLowEquityThreshold:
    """When threshold_pct > 0, low-equity triggers at equity × pct."""

    def test_pct_threshold_triggers_when_equity_below(self) -> None:
        """Equity $25, pct=0.5 → threshold=$12.50. Current $10 → low equity."""
        engine = RiskEngine(
            RiskConfig(
                equity=25,
                risk_per_trade=0.02,
                max_trades_per_day=8,
                daily_stop_pct=0.04,
                max_positions=2,
                low_equity_mode_enabled=True,
                low_equity_threshold=250.0,  # ignored when pct > 0
                low_equity_threshold_pct=0.5,
            )
        )
        assert engine.is_low_equity_mode(equity=10.0) is True

    def test_pct_threshold_not_triggered_when_equity_above(self) -> None:
        """Equity $25, pct=0.5 → threshold=$12.50. Current $20 → NOT low equity."""
        engine = RiskEngine(
            RiskConfig(
                equity=25,
                risk_per_trade=0.02,
                max_trades_per_day=8,
                daily_stop_pct=0.04,
                max_positions=2,
                low_equity_mode_enabled=True,
                low_equity_threshold=250.0,
                low_equity_threshold_pct=0.5,
            )
        )
        assert engine.is_low_equity_mode(equity=20.0) is False

    def test_pct_threshold_at_boundary(self) -> None:
        """Exactly at threshold → should be low equity (<=)."""
        engine = RiskEngine(
            RiskConfig(
                equity=100,
                risk_per_trade=0.01,
                max_trades_per_day=8,
                daily_stop_pct=0.04,
                max_positions=2,
                low_equity_mode_enabled=True,
                low_equity_threshold_pct=0.5,
            )
        )
        # threshold = 100 × 0.5 = 50.0
        assert engine.is_low_equity_mode(equity=50.0) is True
        assert engine.is_low_equity_mode(equity=50.01) is False

    def test_pct_zero_falls_back_to_absolute(self) -> None:
        """pct=0 (default) → use absolute threshold as before."""
        engine = RiskEngine(
            RiskConfig(
                equity=10000,
                risk_per_trade=0.005,
                max_trades_per_day=8,
                daily_stop_pct=0.015,
                max_positions=2,
                low_equity_mode_enabled=True,
                low_equity_threshold=250.0,
                low_equity_threshold_pct=0.0,
            )
        )
        assert engine.is_low_equity_mode(equity=200.0) is True
        assert engine.is_low_equity_mode(equity=300.0) is False

    def test_disabled_mode_ignores_pct(self) -> None:
        """low_equity_mode_enabled=False → never in low equity."""
        engine = RiskEngine(
            RiskConfig(
                equity=25,
                risk_per_trade=0.02,
                max_trades_per_day=8,
                daily_stop_pct=0.04,
                max_positions=2,
                low_equity_mode_enabled=False,
                low_equity_threshold_pct=0.5,
            )
        )
        assert engine.is_low_equity_mode(equity=1.0) is False


# =====================================================================
# 2. Auto min_risk_cash scaling (min_risk_cash_auto)
# =====================================================================

class TestMinRiskCashAuto:
    """When min_risk_cash_auto=True, validator computes min_risk_cash from equity."""

    def test_auto_computes_min_risk_cash(self) -> None:
        """equity=25, pct=0.004 → min_risk_cash = $0.10."""
        cfg = RiskConfig(
            equity=25,
            risk_per_trade=0.02,
            max_trades_per_day=8,
            daily_stop_pct=0.04,
            max_positions=2,
            min_risk_cash_auto=True,
            min_risk_cash_auto_pct=0.004,
        )
        assert cfg.min_risk_cash_per_trade == pytest.approx(0.10, abs=0.001)

    def test_auto_disabled_keeps_original(self) -> None:
        """min_risk_cash_auto=False → original value untouched."""
        cfg = RiskConfig(
            equity=25,
            risk_per_trade=0.02,
            max_trades_per_day=8,
            daily_stop_pct=0.04,
            max_positions=2,
            min_risk_cash_per_trade=0.50,
            min_risk_cash_auto=False,
        )
        assert cfg.min_risk_cash_per_trade == 0.50

    def test_auto_with_large_equity(self) -> None:
        """equity=10000, pct=0.003 → min_risk_cash = $30."""
        cfg = RiskConfig(
            equity=10000,
            risk_per_trade=0.005,
            max_trades_per_day=3,
            daily_stop_pct=0.015,
            max_positions=1,
            min_risk_cash_auto=True,
            min_risk_cash_auto_pct=0.003,
        )
        assert cfg.min_risk_cash_per_trade == pytest.approx(30.0, abs=0.01)

    def test_auto_with_zero_equity_rejected(self) -> None:
        """equity=0 → rejected by validator (equity must be > 0)."""
        with pytest.raises(Exception, match="equity"):
            RiskConfig(
                equity=0,
                risk_per_trade=0.01,
                max_trades_per_day=3,
                daily_stop_pct=0.015,
                max_positions=1,
                min_risk_cash_per_trade=0.5,
                min_risk_cash_auto=True,
                min_risk_cash_auto_pct=0.004,
            )


# =====================================================================
# 3. TP gradient by score (_resolve_tp_target_r)
# =====================================================================

class _FakeTuning:
    """Lightweight mock for BacktestTuningConfig fields used by _resolve_tp_target_r."""
    def __init__(
        self,
        *,
        tp_gradient_enabled: bool = False,
        tp_gradient_tier_high_score: float = 75.0,
        tp_gradient_tier_high_r: float = 3.0,
        tp_gradient_tier_mid_score: float = 62.0,
        tp_gradient_tier_mid_r: float = 2.5,
        tp_gradient_tier_low_r: float = 2.0,
        tp_target_a_plus_r: float = 3.0,
        tp_target_min_r: float = 2.0,
    ):
        self.tp_gradient_enabled = tp_gradient_enabled
        self.tp_gradient_tier_high_score = tp_gradient_tier_high_score
        self.tp_gradient_tier_high_r = tp_gradient_tier_high_r
        self.tp_gradient_tier_mid_score = tp_gradient_tier_mid_score
        self.tp_gradient_tier_mid_r = tp_gradient_tier_mid_r
        self.tp_gradient_tier_low_r = tp_gradient_tier_low_r
        self.tp_target_a_plus_r = tp_target_a_plus_r
        self.tp_target_min_r = tp_target_min_r


class TestTPGradient:
    """Score-based TP gradient selects R target by tier."""

    # -- Gradient enabled, score provided --

    def test_high_score_returns_high_r(self) -> None:
        tuning = _FakeTuning(tp_gradient_enabled=True)
        r, label = _resolve_tp_target_r(is_a_plus=False, score=80.0, tuning=tuning)
        assert r == 3.0
        assert label == "GRADIENT_HIGH"

    def test_mid_score_returns_mid_r(self) -> None:
        tuning = _FakeTuning(tp_gradient_enabled=True)
        r, label = _resolve_tp_target_r(is_a_plus=False, score=68.0, tuning=tuning)
        assert r == 2.5
        assert label == "GRADIENT_MID"

    def test_low_score_returns_low_r(self) -> None:
        tuning = _FakeTuning(tp_gradient_enabled=True)
        r, label = _resolve_tp_target_r(is_a_plus=False, score=55.0, tuning=tuning)
        assert r == 2.0
        assert label == "GRADIENT_LOW"

    def test_boundary_high_exactly(self) -> None:
        tuning = _FakeTuning(tp_gradient_enabled=True)
        r, label = _resolve_tp_target_r(is_a_plus=False, score=75.0, tuning=tuning)
        assert r == 3.0
        assert label == "GRADIENT_HIGH"

    def test_boundary_mid_exactly(self) -> None:
        tuning = _FakeTuning(tp_gradient_enabled=True)
        r, label = _resolve_tp_target_r(is_a_plus=False, score=62.0, tuning=tuning)
        assert r == 2.5
        assert label == "GRADIENT_MID"

    def test_boundary_below_mid(self) -> None:
        tuning = _FakeTuning(tp_gradient_enabled=True)
        r, label = _resolve_tp_target_r(is_a_plus=False, score=61.9, tuning=tuning)
        assert r == 2.0
        assert label == "GRADIENT_LOW"

    # -- Gradient enabled but score=None → fallback --

    def test_gradient_no_score_a_plus_fallback(self) -> None:
        tuning = _FakeTuning(tp_gradient_enabled=True)
        r, label = _resolve_tp_target_r(is_a_plus=True, score=None, tuning=tuning)
        assert r == 3.0
        assert label == "A_PLUS_3R"

    def test_gradient_no_score_standard_fallback(self) -> None:
        tuning = _FakeTuning(tp_gradient_enabled=True)
        r, label = _resolve_tp_target_r(is_a_plus=False, score=None, tuning=tuning)
        assert r == 2.0
        assert label == "STANDARD_2R"

    # -- Gradient disabled → classic behaviour --

    def test_disabled_a_plus(self) -> None:
        tuning = _FakeTuning(tp_gradient_enabled=False)
        r, label = _resolve_tp_target_r(is_a_plus=True, score=90.0, tuning=tuning)
        assert r == 3.0
        assert label == "A_PLUS_3R"

    def test_disabled_standard(self) -> None:
        tuning = _FakeTuning(tp_gradient_enabled=False)
        r, label = _resolve_tp_target_r(is_a_plus=False, score=90.0, tuning=tuning)
        assert r == 2.0
        assert label == "STANDARD_2R"

    # -- Custom tier values --

    def test_custom_tier_values(self) -> None:
        tuning = _FakeTuning(
            tp_gradient_enabled=True,
            tp_gradient_tier_high_score=80.0,
            tp_gradient_tier_high_r=4.0,
            tp_gradient_tier_mid_score=65.0,
            tp_gradient_tier_mid_r=3.0,
            tp_gradient_tier_low_r=1.5,
        )
        r_hi, _ = _resolve_tp_target_r(is_a_plus=False, score=85.0, tuning=tuning)
        r_mid, _ = _resolve_tp_target_r(is_a_plus=False, score=70.0, tuning=tuning)
        r_lo, _ = _resolve_tp_target_r(is_a_plus=False, score=50.0, tuning=tuning)
        assert r_hi == 4.0
        assert r_mid == 3.0
        assert r_lo == 1.5

    # -- a_plus is ignored when gradient picks tier (score takes priority) --

    def test_gradient_ignores_a_plus_when_score_available(self) -> None:
        """Even with a_plus=True, gradient by score takes priority."""
        tuning = _FakeTuning(tp_gradient_enabled=True)
        r, label = _resolve_tp_target_r(is_a_plus=True, score=55.0, tuning=tuning)
        assert r == 2.0  # low tier by score, not 3.0R from a_plus
        assert label == "GRADIENT_LOW"


# =====================================================================
# 4. BacktestTuningConfig TP gradient fields default correctly
# =====================================================================

class TestBacktestTuningTPGradientDefaults:
    """New TP gradient fields exist with safe defaults (disabled)."""

    def test_gradient_disabled_by_default(self) -> None:
        cfg = BacktestTuningConfig()
        assert cfg.tp_gradient_enabled is False
        assert cfg.tp_gradient_tier_high_score == 75.0
        assert cfg.tp_gradient_tier_high_r == 3.0
        assert cfg.tp_gradient_tier_mid_score == 62.0
        assert cfg.tp_gradient_tier_mid_r == 2.5
        assert cfg.tp_gradient_tier_low_r == 2.0


# =====================================================================
# 5. RiskConfig new fields default correctly (backward compat)
# =====================================================================

class TestRiskConfigNewFieldDefaults:
    """New RiskConfig fields have safe defaults that don't change behaviour."""

    def test_threshold_pct_default_zero(self) -> None:
        cfg = RiskConfig(
            equity=10000,
            risk_per_trade=0.005,
            max_trades_per_day=3,
            daily_stop_pct=0.015,
            max_positions=1,
        )
        assert cfg.low_equity_threshold_pct == 0.0

    def test_min_risk_auto_default_false(self) -> None:
        cfg = RiskConfig(
            equity=10000,
            risk_per_trade=0.005,
            max_trades_per_day=3,
            daily_stop_pct=0.015,
            max_positions=1,
        )
        assert cfg.min_risk_cash_auto is False
        assert cfg.min_risk_cash_auto_pct == 0.003


# =====================================================================
# 6. Variant config loads without errors
# =====================================================================

class TestVariantConfigLoads:
    """100PLN_V2 variant YAML loads and produces a valid AppConfig."""

    def test_100pln_v2_loads(self) -> None:
        from bot.config import load_config
        import os

        variant_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "variants",
            "config.variant_100PLN_V2.yaml",
        )
        if not os.path.exists(variant_path):
            pytest.skip("100PLN_V2 variant file not found")

        cfg = load_config(variant_path)
        # Verify key small-account settings
        assert cfg.risk.equity == 25
        assert cfg.risk.low_equity_threshold_pct == 0.5
        assert cfg.risk.min_risk_cash_auto is True
        assert cfg.backtest_tuning.tp_gradient_enabled is True
        assert cfg.backtest_tuning.thresholds_v2_trade == 37.0
        assert cfg.compound_equity.enabled is True
