"""Tests for adaptive threshold, soft gates, and re-entry logic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bot.gating.adaptive import (
    AdaptiveThresholdConfig,
    ReentryState,
    SoftGateResult,
    apply_soft_gates,
    build_adaptive_config,
    compute_adaptive_threshold,
    normalize_action_adaptive,
)


# =========================================================================
# Soft gates
# =========================================================================
class TestApplySoftGates:
    def test_disabled(self):
        reasons = ["GATE_REACTION_WAIT_REACTION", "EXEC_FAIL_SPREAD_TOO_HIGH"]
        result = apply_soft_gates(reasons, soft_gates_enabled=False, soft_gate_penalty=4.0)
        assert result.blocked
        assert result.hard_reasons == reasons
        assert result.soft_reasons == []
        assert result.total_penalty == 0.0

    def test_all_soft(self):
        reasons = ["GATE_REACTION_WAIT_REACTION", "EXEC_FAIL_SPREAD_TOO_HIGH"]
        result = apply_soft_gates(reasons, soft_gates_enabled=True, soft_gate_penalty=4.0)
        assert not result.blocked
        assert result.hard_reasons == []
        assert len(result.soft_reasons) == 2
        assert result.total_penalty == 8.0

    def test_mixed(self):
        reasons = ["GATE_REACTION_WAIT_MITIGATION", "EXEC_FAIL_MISSING_FEATURES"]
        result = apply_soft_gates(reasons, soft_gates_enabled=True, soft_gate_penalty=4.0)
        assert result.blocked
        assert result.hard_reasons == ["EXEC_FAIL_MISSING_FEATURES"]
        assert result.soft_reasons == ["GATE_REACTION_WAIT_MITIGATION"]
        assert result.total_penalty == 4.0

    def test_empty_reasons(self):
        result = apply_soft_gates([], soft_gates_enabled=True, soft_gate_penalty=4.0)
        assert not result.blocked
        assert result.total_penalty == 0.0

    def test_hard_only(self):
        reasons = ["EXEC_FAIL_MARKET_CLOSED", "EXEC_FAIL_NO_PRICE"]
        result = apply_soft_gates(reasons, soft_gates_enabled=True, soft_gate_penalty=4.0)
        assert result.blocked
        assert result.total_penalty == 0.0
        assert result.hard_reasons == reasons

    def test_converted_gates_listed(self):
        reasons = ["GATE_REACTION_WAIT_REACTION"]
        result = apply_soft_gates(reasons, soft_gates_enabled=True, soft_gate_penalty=5.0)
        assert result.converted_gates == ["GATE_REACTION_WAIT_REACTION"]


# =========================================================================
# Adaptive threshold
# =========================================================================
class TestComputeAdaptiveThreshold:
    def test_disabled_returns_base(self):
        cfg = AdaptiveThresholdConfig(enabled=False, base_threshold=62.0)
        assert compute_adaptive_threshold(config=cfg) == 62.0

    def test_range_lowers_threshold(self):
        cfg = AdaptiveThresholdConfig(enabled=True, base_threshold=62.0, range_adjust=-6.0)
        th = compute_adaptive_threshold(config=cfg, trend_regime="RANGING")
        assert th == 56.0

    def test_trend_lowers_slightly(self):
        cfg = AdaptiveThresholdConfig(enabled=True, base_threshold=62.0, trend_adjust=-2.0)
        th = compute_adaptive_threshold(config=cfg, trend_regime="TRENDING")
        assert th == 60.0

    def test_high_vol_raises(self):
        cfg = AdaptiveThresholdConfig(enabled=True, base_threshold=62.0, high_vol_adjust=2.0)
        th = compute_adaptive_threshold(config=cfg, vol_regime="HIGH")
        assert th == 64.0

    def test_low_vol_lowers(self):
        cfg = AdaptiveThresholdConfig(enabled=True, base_threshold=62.0, low_vol_adjust=-3.0)
        th = compute_adaptive_threshold(config=cfg, vol_regime="LOW")
        assert th == 59.0

    def test_combined(self):
        cfg = AdaptiveThresholdConfig(
            enabled=True, base_threshold=62.0,
            range_adjust=-6.0, low_vol_adjust=-3.0,
        )
        th = compute_adaptive_threshold(config=cfg, trend_regime="RANGING", vol_regime="LOW")
        assert th == 53.0

    def test_clamp_min(self):
        cfg = AdaptiveThresholdConfig(
            enabled=True, base_threshold=50.0,
            range_adjust=-6.0, low_vol_adjust=-3.0,
            min_threshold=48.0,
        )
        th = compute_adaptive_threshold(config=cfg, trend_regime="RANGING", vol_regime="LOW")
        assert th == 48.0

    def test_clamp_max(self):
        cfg = AdaptiveThresholdConfig(
            enabled=True, base_threshold=72.0,
            high_vol_adjust=5.0,
            max_threshold=75.0,
        )
        th = compute_adaptive_threshold(config=cfg, vol_regime="HIGH")
        assert th == 75.0

    def test_unknown_regime_no_adjust(self):
        cfg = AdaptiveThresholdConfig(enabled=True, base_threshold=62.0)
        th = compute_adaptive_threshold(config=cfg, trend_regime="UNKNOWN", vol_regime="NORMAL")
        assert th == 62.0


class TestNormalizeActionAdaptive:
    def test_trade(self):
        assert normalize_action_adaptive(score=65.0, threshold=62.0) == "TRADE"

    def test_small(self):
        assert normalize_action_adaptive(score=59.0, threshold=62.0) == "SMALL"

    def test_observe(self):
        assert normalize_action_adaptive(score=50.0, threshold=62.0) == "OBSERVE"

    def test_exact_threshold_is_trade(self):
        assert normalize_action_adaptive(score=62.0, threshold=62.0) == "TRADE"

    def test_exact_small_band_boundary(self):
        assert normalize_action_adaptive(score=57.0, threshold=62.0, small_band=5.0) == "SMALL"

    def test_just_below_small(self):
        assert normalize_action_adaptive(score=56.9, threshold=62.0, small_band=5.0) == "OBSERVE"


# =========================================================================
# Re-entry tracking
# =========================================================================
NOW = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)


class TestReentryState:
    def test_fresh_state_allows_entry(self):
        r = ReentryState()
        ok, reason = r.can_reenter("LONG", NOW)
        assert ok
        assert reason == ""

    def test_after_profitable_close_allows_reentry(self):
        r = ReentryState()
        r.record_close("LONG", "TP", pnl=5.0, closed_at=NOW - timedelta(minutes=5))
        ok, reason = r.can_reenter("LONG", NOW)
        assert ok

    def test_after_stop_loss_blocks(self):
        r = ReentryState()
        r.record_close("LONG", "SL", pnl=-2.0, closed_at=NOW - timedelta(minutes=5))
        ok, reason = r.can_reenter("LONG", NOW)
        assert not ok
        assert reason == "REENTRY_AFTER_STOP"

    def test_after_loss_blocks(self):
        r = ReentryState()
        r.record_close("LONG", "BE", pnl=-0.1, closed_at=NOW - timedelta(minutes=5))
        ok, reason = r.can_reenter("LONG", NOW)
        assert not ok
        assert reason == "REENTRY_AFTER_LOSS"

    def test_max_reentries(self):
        r = ReentryState(max_reentries_per_leg=1)
        r.record_close("LONG", "TP", pnl=5.0, closed_at=NOW - timedelta(minutes=10))
        ok, _ = r.can_reenter("LONG", NOW)
        assert ok
        r.mark_reentry(NOW)
        ok, reason = r.can_reenter("LONG", NOW)
        assert not ok
        assert reason == "REENTRY_MAX_REACHED"

    def test_cooldown(self):
        r = ReentryState(reentry_cooldown_seconds=120)
        r.record_close("LONG", "TP", pnl=3.0, closed_at=NOW)
        ok, reason = r.can_reenter("LONG", NOW + timedelta(seconds=60))
        assert not ok
        assert reason == "REENTRY_COOLDOWN"
        ok, _ = r.can_reenter("LONG", NOW + timedelta(seconds=121))
        assert ok

    def test_different_direction_always_ok(self):
        r = ReentryState()
        r.record_close("LONG", "SL", pnl=-2.0, closed_at=NOW)
        ok, reason = r.can_reenter("SHORT", NOW)
        assert ok

    def test_direction_flip_resets_counter(self):
        r = ReentryState(max_reentries_per_leg=1)
        r.record_close("LONG", "TP", pnl=5.0, closed_at=NOW - timedelta(minutes=10))
        r.mark_reentry(NOW)
        assert r.reentries_this_leg == 1
        # Flip direction
        r.record_close("SHORT", "TP", pnl=3.0, closed_at=NOW)
        assert r.reentries_this_leg == 0

    def test_reset_leg(self):
        r = ReentryState()
        r.record_close("LONG", "TP", pnl=5.0, closed_at=NOW)
        r.mark_reentry(NOW)
        r.reset_leg()
        assert r.last_close_side is None
        assert r.reentries_this_leg == 0

    def test_to_dict(self):
        r = ReentryState()
        r.record_close("SHORT", "TP", pnl=2.0, closed_at=NOW)
        d = r.to_dict()
        assert d["last_close_side"] == "SHORT"
        assert d["last_close_exit_type"] == "TP"
        assert d["reentries_this_leg"] == 0


# =========================================================================
# Build config helper
# =========================================================================
class TestBuildAdaptiveConfig:
    def test_from_pydantic(self):
        from bot.config import AppConfig
        app = AppConfig()
        cfg = build_adaptive_config(app.adaptive_threshold)
        assert cfg.base_threshold == 62.0
        assert cfg.soft_gates_enabled is True
        assert cfg.soft_gate_penalty == 4.0

    def test_from_dict_like(self):
        class FakeCfg:
            enabled = True
            base_threshold = 60.0
            range_adjust = -5.0
            trend_adjust = -1.0
            high_vol_adjust = 3.0
            low_vol_adjust = -2.0
            soft_gates_enabled = False
            soft_gate_penalty = 6.0
        cfg = build_adaptive_config(FakeCfg())
        assert cfg.enabled is True
        assert cfg.soft_gates_enabled is False
        assert cfg.soft_gate_penalty == 6.0


# =========================================================================
# ClosedPositionEvent extended fields
# =========================================================================
class TestClosedPositionEventExtended:
    def test_new_fields_have_defaults(self):
        from bot.storage.models import ClosedPositionEvent
        e = ClosedPositionEvent(deal_id="x", epic="XAUUSD", pnl=1.0, closed_at=NOW)
        assert e.side == ""
        assert e.exit_type == ""

    def test_new_fields_set(self):
        from bot.storage.models import ClosedPositionEvent
        e = ClosedPositionEvent(deal_id="x", epic="XAUUSD", pnl=-2.0, closed_at=NOW, side="LONG", exit_type="SL")
        assert e.side == "LONG"
        assert e.exit_type == "SL"
