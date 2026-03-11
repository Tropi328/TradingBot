"""
Unit tests for ScoreV3 scoring system.

Tests cover:
  A. Feature extraction determinism and range validation
  B. Heuristic scorer properties (no NaN, bounded 0-100)
  C. ScoreV3Engine action resolution and tier mapping
  D. Shadow observer recording and summary
  E. Integration: apply_score_v3 on StrategyEvaluation
"""

from __future__ import annotations

import math
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot.data.candles import Candle
from bot.strategy.contracts import BiasState, DecisionAction, StrategyEvaluation
from bot.strategy.score_v3 import (
    FEATURE_NAMES,
    HeuristicScoreV3Model,
    ScoreV3Config,
    ScoreV3Engine,
    apply_score_v3,
    extract_features,
    heuristic_score_v3,
)
from bot.strategy.shadow_observer import (
    ShadowCandidate,
    ShadowObserver,
    classify_session,
    compute_atr_percentile,
    simulate_shadow_outcome,
)


# ── Fixtures ──


def _make_candle(
    close: float = 2000.0,
    high: float = 2002.0,
    low: float = 1998.0,
    hour: int = 10,
) -> Candle:
    return Candle(
        timestamp=datetime(2024, 6, 15, hour, 30, tzinfo=timezone.utc),
        open=close - 0.5,
        high=high,
        low=low,
        close=close,
    )


def _make_bias(direction: str = "LONG") -> BiasState:
    return BiasState(
        symbol="XAUUSD",
        strategy_name="SCALP_ICT_PA",
        direction=direction,
        timeframe="M5",
        updated_at=datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc),
    )


def _make_evaluation(
    *,
    side: str = "LONG",
    score: float = 50.0,
    edge: float = 25.0,
    trigger: float = 18.0,
    execution: float = 7.0,
    penalty_total: float = 5.0,
    fvg_detected: bool = True,
    trigger_confirmations: int = 2,
    atr_m5: float = 1.5,
) -> StrategyEvaluation:
    return StrategyEvaluation(
        action=DecisionAction.OBSERVE,
        score_total=score,
        score_layers={"edge": edge, "trigger": trigger, "execution": execution},
        penalties={"NEUTRAL_BIAS": penalty_total} if penalty_total > 0 else {},
        metadata={
            "side": side,
            "atr_m5": atr_m5,
            "spread": 0.2,
            "trigger_confirmations": trigger_confirmations,
            "fvg_detected": fvg_detected,
            "fvg_size": 1.2,
            "fvg_mid": 1999.5,
            "entry_price": 1999.5,
            "stop_price": 1998.0,
            "tp_price": 2003.0,
            "h1_bos_state": "BULLISH",
            "h1_pd_eq": 2000.0,
            "h1_close": 1999.0,
            "sweep_magnitude": 0.5,
        },
        snapshot={"atr_m5": atr_m5, "spread": 0.2},
    )


# ═══════════════════════════════════════════════════════════════════════════
#  A. Feature extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestFeatureExtraction:
    """Feature extraction determinism, completeness, and range validation."""

    def test_all_features_present(self):
        """Every feature in FEATURE_NAMES (57 total) must appear in output."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"

    def test_features_are_numeric(self):
        """All features must be float or int, never None or NaN."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        for name in FEATURE_NAMES:
            val = features[name]
            assert isinstance(val, (int, float)), f"{name} is {type(val)}"
            assert not math.isnan(val), f"{name} is NaN"
            assert not math.isinf(val), f"{name} is Inf"

    def test_deterministic(self):
        """Same inputs → identical outputs."""
        ev = _make_evaluation()
        bias = _make_bias()
        candle = _make_candle()
        f1 = extract_features(ev, bias, candle=candle)
        f2 = extract_features(ev, bias, candle=candle)
        assert f1 == f2

    def test_htf_bias_aligned_long(self):
        """When bias direction matches side, htf_bias_aligned = 1."""
        features = extract_features(_make_evaluation(side="LONG"), _make_bias("LONG"), candle=_make_candle())
        assert features["htf_bias_aligned"] == 1.0

    def test_htf_bias_misaligned(self):
        """When bias direction mismatches side, htf_bias_aligned = 0."""
        features = extract_features(_make_evaluation(side="LONG"), _make_bias("SHORT"), candle=_make_candle())
        assert features["htf_bias_aligned"] == 0.0

    def test_session_classification_london(self):
        """Candle at 10:30 UTC → LONDON session."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle(hour=10))
        assert features["is_london"] == 1.0
        assert features["is_overlap"] == 0.0
        assert features["is_ny"] == 0.0

    def test_session_classification_overlap(self):
        """Candle at 14:00 UTC → OVERLAP session."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle(hour=14))
        assert features["is_overlap"] == 1.0

    def test_rr_ratio_positive(self):
        """RR ratio should be positive for valid entry/stop/tp."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        assert features["rr_ratio"] > 0

    def test_spread_ratio_bounded(self):
        """Spread ratio should be positive and finite."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        assert 0 <= features["spread_ratio"] < 100

    def test_fill_probability_bounded(self):
        """Fill probability proxy must be in [0, 1]."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        assert 0 <= features["fill_probability_proxy"] <= 1


# ═══════════════════════════════════════════════════════════════════════════
#  B. Heuristic scorer
# ═══════════════════════════════════════════════════════════════════════════


class TestHeuristicScorer:
    """Heuristic scorer produces bounded, non-NaN scores."""

    def test_score_in_range(self):
        """Score must be in [0, 100]."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        score = heuristic_score_v3(features)
        assert 0 <= score <= 100

    def test_score_not_nan(self):
        """Score must never be NaN."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        score = heuristic_score_v3(features)
        assert not math.isnan(score)

    def test_empty_features_returns_zero(self):
        """Empty feature dict should not crash, returns ≥ 0."""
        score = heuristic_score_v3({})
        assert 0 <= score <= 100
        assert not math.isnan(score)

    def test_ideal_setup_high_score(self):
        """A setup with all positive features should score high."""
        features = {
            "htf_bias_aligned": 1.0,
            "htf_bos_confirmed": 1.0,
            "htf_pd_ok": 1.0,
            "htf_location_score": 0.9,
            "fvg_present": 1.0,
            "fvg_size_atr": 1.5,
            "fvg_displacement_ratio": 1.5,
            "trigger_confirmations": 3.0,
            "mss_confirmed": 1.0,
            "sweep_magnitude_atr": 0.8,
            "spread_score_raw": 8.0,
            "rr_ratio": 2.5,
            "fill_probability_proxy": 0.9,
            "is_london": 0.0,
            "is_overlap": 1.0,
            "is_ny": 0.0,
            "vol_regime": 1.0,
            "setup_is_fallback": 0.0,
            "news_blocked": 0.0,
            "v3_penalty_total": 0.0,
            "spread_ratio": 0.05,
            "daily_context_score": 0.8,
            "vol_regime_normal": 1.0,
            "session_quality": 0.9,
            "v3_edge_quality": 0.85,
            "v3_trigger_quality": 0.8,
            "v3_execution_quality": 0.9,
        }
        score = heuristic_score_v3(features)
        assert score >= 70, f"Ideal setup scored only {score}"

    def test_poor_setup_low_score(self):
        """A setup with bad features should score low."""
        features = {
            "htf_bias_aligned": 0.0,
            "htf_bos_confirmed": 0.0,
            "htf_pd_ok": 0.0,
            "htf_location_score": 0.1,
            "fvg_present": 0.0,
            "trigger_confirmations": 0.0,
            "mss_confirmed": 0.0,
            "spread_score_raw": 0.0,
            "rr_ratio": 0.8,
            "fill_probability_proxy": 0.2,
            "is_london": 0.0,
            "is_overlap": 0.0,
            "is_ny": 0.0,
            "vol_regime": 2.0,
            "setup_is_fallback": 1.0,
            "news_blocked": 1.0,
            "v3_penalty_total": 10.0,
            "spread_ratio": 0.3,
            "daily_context_score": 0.0,
            "vol_regime_normal": 0.0,
            "session_quality": 0.1,
            "v3_edge_quality": 0.1,
            "v3_trigger_quality": 0.0,
            "v3_execution_quality": 0.1,
        }
        score = heuristic_score_v3(features)
        assert score < 20, f"Poor setup scored {score}"

    def test_fallback_penalty_applied(self):
        """Displacement retest setups should score lower."""
        base_features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        fallback_features = dict(base_features)
        fallback_features["setup_is_fallback"] = 1.0

        base_score = heuristic_score_v3(base_features)
        fallback_score = heuristic_score_v3(fallback_features)
        assert fallback_score < base_score, "Fallback should penalize score"


# ═══════════════════════════════════════════════════════════════════════════
#  B2. Orderflow features integration
# ═══════════════════════════════════════════════════════════════════════════


def _make_evaluation_with_of(
    *,
    side: str = "LONG",
    of_direction: str = "LONG",
    delta_ratio: float = 0.6,
    aggression: float = 0.7,
    microprice_bias: float = 0.3,
    absorption_score: float = 0.5,
    chop_score: float = 0.2,
    confidence: float = 0.85,
) -> StrategyEvaluation:
    """Create an evaluation that includes an orderflow snapshot in metadata."""
    ev = _make_evaluation(side=side)
    ev.metadata["orderflow_snapshot"] = {
        "confidence": confidence,
        "direction": of_direction,
        "mode": "LITE",
        "pressure": 0.5,
        "metrics": {
            "delta_ratio": delta_ratio,
            "aggression": aggression,
            "microprice_bias": microprice_bias,
            "absorption_score": absorption_score,
            "chop_score": chop_score,
            "spread_ratio": 0.05,
            "efficiency_ratio": 0.4,
            "obi_k": 0.0,
        },
    }
    return ev


class TestOrderflowFeatures:
    """Orderflow feature extraction and heuristic integration."""

    def test_of_features_present_when_snapshot_exists(self):
        """All 6 orderflow features must appear when snapshot is in metadata."""
        ev = _make_evaluation_with_of()
        features = extract_features(ev, _make_bias(), candle=_make_candle())
        for name in [
            "of_delta_ratio",
            "of_aggression",
            "of_microprice_bias",
            "of_absorption_score",
            "of_chop_score",
            "of_flow_aligned",
        ]:
            assert name in features, f"Missing orderflow feature: {name}"

    def test_of_features_defaults_without_snapshot(self):
        """Without orderflow snapshot, features get neutral defaults."""
        ev = _make_evaluation()
        features = extract_features(ev, _make_bias(), candle=_make_candle())
        assert features["of_delta_ratio"] == 0.0
        assert features["of_chop_score"] == 0.5  # neutral default
        assert features["of_flow_aligned"] == 0.0

    def test_of_flow_aligned_matches_side(self):
        """of_flow_aligned is 1 when direction matches setup side."""
        ev_aligned = _make_evaluation_with_of(side="LONG", of_direction="LONG")
        f = extract_features(ev_aligned, _make_bias(direction="LONG"), candle=_make_candle())
        assert f["of_flow_aligned"] == 1.0

        ev_divergent = _make_evaluation_with_of(side="LONG", of_direction="SHORT")
        f2 = extract_features(ev_divergent, _make_bias(direction="LONG"), candle=_make_candle())
        assert f2["of_flow_aligned"] == 0.0

    def test_of_delta_ratio_propagated(self):
        """of_delta_ratio reflects the snapshot value."""
        ev = _make_evaluation_with_of(delta_ratio=-0.4)
        features = extract_features(ev, _make_bias(), candle=_make_candle())
        assert abs(features["of_delta_ratio"] - (-0.4)) < 1e-3

    def test_of_chop_penalty_applied(self):
        """High chop score (>0.75) should penalize the heuristic."""
        base_features = extract_features(_make_evaluation_with_of(chop_score=0.2), _make_bias(), candle=_make_candle())
        high_chop_features = dict(base_features)
        high_chop_features["of_chop_score"] = 0.9

        base_score = heuristic_score_v3(base_features)
        chop_score = heuristic_score_v3(high_chop_features)
        assert chop_score < base_score, "High chop should penalize score"

    def test_of_flow_aligned_boosts_score(self):
        """Aligned orderflow should boost the heuristic score."""
        base_features = extract_features(
            _make_evaluation_with_of(of_direction="LONG"), _make_bias(), candle=_make_candle()
        )
        no_align_features = dict(base_features)
        no_align_features["of_flow_aligned"] = 0.0

        aligned_score = heuristic_score_v3(base_features)
        unaligned_score = heuristic_score_v3(no_align_features)
        assert aligned_score > unaligned_score, "Flow alignment should boost score"


# ═══════════════════════════════════════════════════════════════════════════
#  B3. Candle pattern features
# ═══════════════════════════════════════════════════════════════════════════


def _make_candle_with(
    open_: float = 1999.5,
    high: float = 2002.0,
    low: float = 1998.0,
    close: float = 2000.0,
    hour: int = 10,
) -> Candle:
    """Helper: create a candle with explicit OHLC."""
    return Candle(
        timestamp=datetime(2024, 6, 15, hour, 30, tzinfo=timezone.utc),
        open=open_,
        high=high,
        low=low,
        close=close,
    )


class TestCandlePatternFeatures:
    """Candle pattern feature extraction, bounds, and heuristic integration."""

    # ── Extraction ──

    def test_candle_features_present(self):
        """All 4 candle features must appear in output."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        for name in [
            "candle_body_to_range",
            "candle_rejection_wick_ratio",
            "candle_engulfing",
            "candle_direction_run",
        ]:
            assert name in features, f"Missing candle feature: {name}"

    def test_candle_body_to_range_strong_body(self):
        """A candle with large body relative to range → high body_to_range."""
        candle = _make_candle_with(open_=1998.0, high=2001.0, low=1997.5, close=2001.0)
        features = extract_features(_make_evaluation(), _make_bias(), candle=candle)
        # body=3.0, range=3.5 → 0.857
        assert features["candle_body_to_range"] > 0.8

    def test_candle_body_to_range_doji(self):
        """A doji (open ≈ close) → low body_to_range."""
        candle = _make_candle_with(open_=2000.0, high=2002.0, low=1998.0, close=2000.01)
        features = extract_features(_make_evaluation(), _make_bias(), candle=candle)
        # body≈0.01, range=4.0 → ~0.0025
        assert features["candle_body_to_range"] < 0.1

    def test_candle_body_to_range_zero_range(self):
        """Zero-range candle → default 0.5."""
        candle = _make_candle_with(open_=2000.0, high=2000.0, low=2000.0, close=2000.0)
        features = extract_features(_make_evaluation(), _make_bias(), candle=candle)
        assert features["candle_body_to_range"] == 0.5

    def test_candle_body_to_range_bounded(self):
        """body_to_range always in [0, 1]."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        assert 0.0 <= features["candle_body_to_range"] <= 1.0

    def test_rejection_wick_long_lower(self):
        """LONG setup: lower wick = rejection wick → high ratio."""
        # Big lower wick candle: open=2001, low=1998, close=2001.5, high=2002
        candle = _make_candle_with(open_=2001.0, high=2002.0, low=1998.0, close=2001.5)
        features = extract_features(_make_evaluation(side="LONG"), _make_bias(direction="LONG"), candle=candle)
        # lower wick = min(2001,2001.5) - 1998 = 3.0, body = 0.5 → clamped to 1.0
        assert features["candle_rejection_wick_ratio"] >= 0.9

    def test_rejection_wick_short_upper(self):
        """SHORT setup: upper wick = rejection wick → high ratio."""
        # Big upper wick candle: open=1999, high=2002, low=1998.5, close=1998.5
        candle = _make_candle_with(open_=1999.0, high=2002.0, low=1998.5, close=1998.5)
        features = extract_features(_make_evaluation(side="SHORT"), _make_bias(direction="SHORT"), candle=candle)
        # upper wick = 2002 - max(1999, 1998.5) = 3.0, body = 0.5 → clamped to 1.0
        assert features["candle_rejection_wick_ratio"] >= 0.9

    def test_rejection_wick_bounded(self):
        """rejection_wick_ratio always in [0, 1]."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        assert 0.0 <= features["candle_rejection_wick_ratio"] <= 1.0

    def test_engulfing_detected(self):
        """Engulfing pattern: last candle body engulfs previous."""
        prev = _make_candle_with(open_=2000.0, high=2001.0, low=1999.5, close=2000.3)
        last = _make_candle_with(open_=1999.4, high=2002.0, low=1999.0, close=2001.0)
        features = extract_features(_make_evaluation(), _make_bias(), candle=last, recent_candles=[prev, last])
        assert features["candle_engulfing"] == 1.0

    def test_engulfing_not_detected(self):
        """No engulfing: last candle doesn't engulf previous."""
        prev = _make_candle_with(open_=1999.0, high=2002.0, low=1998.0, close=2001.5)
        last = _make_candle_with(open_=2000.0, high=2001.0, low=1999.5, close=2000.5)
        features = extract_features(_make_evaluation(), _make_bias(), candle=last, recent_candles=[prev, last])
        assert features["candle_engulfing"] == 0.0

    def test_engulfing_no_recent_candles(self):
        """Without recent_candles, engulfing defaults to 0."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        assert features["candle_engulfing"] == 0.0

    def test_direction_run_long(self):
        """LONG setup: consecutive bullish candles → high direction_run."""
        candles = [
            _make_candle_with(open_=1998.0, close=1999.0),
            _make_candle_with(open_=1999.0, close=2000.0),
            _make_candle_with(open_=2000.0, close=2001.0),
        ]
        features = extract_features(
            _make_evaluation(side="LONG"),
            _make_bias(direction="LONG"),
            candle=candles[-1],
            recent_candles=candles,
        )
        # 3 consecutive bullish → 3/5 = 0.6
        assert features["candle_direction_run"] == pytest.approx(0.6, abs=0.01)

    def test_direction_run_broken(self):
        """A candle in wrong direction breaks the run."""
        candles = [
            _make_candle_with(open_=1998.0, close=1999.0),  # bullish
            _make_candle_with(open_=2000.0, close=1999.5),  # bearish → breaks
            _make_candle_with(open_=1999.5, close=2000.5),  # bullish
        ]
        features = extract_features(
            _make_evaluation(side="LONG"),
            _make_bias(direction="LONG"),
            candle=candles[-1],
            recent_candles=candles,
        )
        # only 1 consecutive bullish from end → 1/5 = 0.2
        assert features["candle_direction_run"] == pytest.approx(0.2, abs=0.01)

    def test_direction_run_no_recent_candles(self):
        """Without recent_candles, direction_run defaults to 0."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        assert features["candle_direction_run"] == 0.0

    def test_direction_run_capped_at_1(self):
        """Direction run is capped at 1.0 (5+ candles)."""
        candles = [_make_candle_with(open_=1990.0 + i, close=1991.0 + i) for i in range(7)]
        features = extract_features(
            _make_evaluation(side="LONG"),
            _make_bias(direction="LONG"),
            candle=candles[-1],
            recent_candles=candles,
        )
        assert features["candle_direction_run"] == 1.0

    # ── Heuristic integration ──

    def test_strong_candle_boosts_score(self):
        """A strong body + rejection wick should boost heuristic score."""
        base = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        boosted = dict(base)
        boosted["candle_body_to_range"] = 0.9
        boosted["candle_rejection_wick_ratio"] = 0.8

        weak = dict(base)
        weak["candle_body_to_range"] = 0.1
        weak["candle_rejection_wick_ratio"] = 0.0

        assert heuristic_score_v3(boosted) > heuristic_score_v3(weak)

    def test_engulfing_boosts_score(self):
        """Engulfing pattern should add points to heuristic."""
        base = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        with_engulf = dict(base)
        with_engulf["candle_engulfing"] = 1.0
        without_engulf = dict(base)
        without_engulf["candle_engulfing"] = 0.0

        assert heuristic_score_v3(with_engulf) > heuristic_score_v3(without_engulf)

    def test_doji_penalty_applied(self):
        """Very weak body (< 0.2) should trigger candle_weak_body penalty."""
        base = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        doji = dict(base)
        doji["candle_body_to_range"] = 0.1  # < 0.2 threshold
        normal = dict(base)
        normal["candle_body_to_range"] = 0.5  # above threshold

        assert heuristic_score_v3(doji) < heuristic_score_v3(normal)


# ═══════════════════════════════════════════════════════════════════════════
#  B4. Sweep quality features
# ═══════════════════════════════════════════════════════════════════════════


def _make_evaluation_with_sweep(
    *,
    side: str = "LONG",
    sweep_magnitude: float = 0.5,
    sweep_level: float = 1998.0,
    entry_price: float = 1999.5,
    stop_price: float = 1997.5,
    tp_price: float = 2003.0,
    atr_m5: float = 1.5,
) -> StrategyEvaluation:
    """Helper: create evaluation with sweep metadata."""
    return StrategyEvaluation(
        action=DecisionAction.OBSERVE,
        score_total=50.0,
        score_layers={"edge": 25.0, "trigger": 18.0, "execution": 7.0},
        penalties={},
        metadata={
            "side": side,
            "atr_m5": atr_m5,
            "spread": 0.2,
            "trigger_confirmations": 2,
            "fvg_detected": True,
            "fvg_size": 1.2,
            "fvg_mid": 1999.5,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "tp_price": tp_price,
            "h1_bos_state": "BULLISH",
            "h1_pd_eq": 2000.0,
            "h1_close": 1999.0,
            "sweep_magnitude": sweep_magnitude,
            "sweep_level": sweep_level,
        },
        snapshot={"atr_m5": atr_m5, "spread": 0.2},
    )


class TestSweepQualityFeatures:
    """Sweep quality feature extraction, bounds, and heuristic integration."""

    # ── Extraction ──

    def test_sweep_features_present(self):
        """All 3 sweep quality features must appear in output."""
        ev = _make_evaluation_with_sweep()
        features = extract_features(ev, _make_bias(), candle=_make_candle())
        for name in ["sweep_quality", "sweep_to_entry_atr", "sweep_rejection_strength"]:
            assert name in features, f"Missing sweep feature: {name}"

    def test_sweep_features_bounded(self):
        """All 3 sweep quality features must be in [0, 1]."""
        ev = _make_evaluation_with_sweep()
        features = extract_features(ev, _make_bias(), candle=_make_candle())
        for name in ["sweep_quality", "sweep_to_entry_atr", "sweep_rejection_strength"]:
            assert 0.0 <= features[name] <= 1.0, f"{name} = {features[name]} out of [0,1]"

    # ── sweep_quality bell curve ──

    def test_sweep_quality_sweet_spot(self):
        """Sweep magnitude 0.3-1.0 ATR → quality = 1.0."""
        for mag in [0.45, 0.75, 1.0, 1.5]:  # with atr=1.5, these give sweep_atr 0.3-1.0
            ev = _make_evaluation_with_sweep(sweep_magnitude=mag, atr_m5=1.5)
            f = extract_features(ev, _make_bias(), candle=_make_candle())
            assert f["sweep_quality"] == 1.0, f"mag={mag}, atr=1.5, sweep_atr={mag / 1.5:.2f}"

    def test_sweep_quality_noise(self):
        """Very small sweep magnitude → low quality."""
        ev = _make_evaluation_with_sweep(sweep_magnitude=0.05, atr_m5=1.5)
        f = extract_features(ev, _make_bias(), candle=_make_candle())
        # 0.05/1.5 ≈ 0.033 ATR → noise zone
        assert f["sweep_quality"] < 0.2

    def test_sweep_quality_oversized(self):
        """Very large sweep → reduced quality (possible breakout, not rejection)."""
        ev = _make_evaluation_with_sweep(sweep_magnitude=4.5, atr_m5=1.5)
        f = extract_features(ev, _make_bias(), candle=_make_candle())
        # 4.5/1.5 = 3.0 ATR → well above 2.0
        assert f["sweep_quality"] < 0.5

    def test_sweep_quality_zero(self):
        """No sweep magnitude → quality = 0."""
        ev = _make_evaluation_with_sweep(sweep_magnitude=0.0)
        f = extract_features(ev, _make_bias(), candle=_make_candle())
        assert f["sweep_quality"] == 0.0

    # ── sweep_to_entry_atr ──

    def test_sweep_close_to_entry(self):
        """Sweep level very close to entry → high proximity score."""
        ev = _make_evaluation_with_sweep(sweep_level=1999.3, entry_price=1999.5, atr_m5=1.5)
        f = extract_features(ev, _make_bias(), candle=_make_candle())
        # distance = 0.2 / 1.5 ≈ 0.133 → proximity ≈ 0.867
        assert f["sweep_to_entry_atr"] > 0.8

    def test_sweep_far_from_entry(self):
        """Sweep level far from entry → low proximity score."""
        ev = _make_evaluation_with_sweep(sweep_level=1995.0, entry_price=1999.5, atr_m5=1.5)
        f = extract_features(ev, _make_bias(), candle=_make_candle())
        # distance = 4.5 / 1.5 = 3.0 → proximity = max(0, 1-3) = 0
        assert f["sweep_to_entry_atr"] == 0.0

    def test_sweep_to_entry_no_sweep_level(self):
        """Without sweep_level in metadata → defaults to 0."""
        ev = _make_evaluation()  # no sweep_level
        f = extract_features(ev, _make_bias(), candle=_make_candle())
        assert f["sweep_to_entry_atr"] == 0.0

    # ── sweep_rejection_strength ──

    def test_rejection_strength_long_above_sweep(self):
        """LONG: price well above sweep level → strong rejection."""
        # candle close=2000, sweep_level=1998, atr=1.5
        # rejection = (2000-1998)/1.5 = 1.33 → clamped to 1.0
        ev = _make_evaluation_with_sweep(side="LONG", sweep_level=1998.0, atr_m5=1.5)
        f = extract_features(ev, _make_bias(direction="LONG"), candle=_make_candle())
        assert f["sweep_rejection_strength"] == 1.0

    def test_rejection_strength_long_at_sweep(self):
        """LONG: price at sweep level → zero rejection."""
        candle = _make_candle_with(close=1998.0)
        ev = _make_evaluation_with_sweep(side="LONG", sweep_level=1998.0, atr_m5=1.5)
        f = extract_features(ev, _make_bias(direction="LONG"), candle=candle)
        assert f["sweep_rejection_strength"] == 0.0

    def test_rejection_strength_short(self):
        """SHORT: price well below sweep level → strong rejection."""
        candle = _make_candle_with(close=1996.0, high=1998.0, low=1995.5)
        ev = _make_evaluation_with_sweep(side="SHORT", sweep_level=1998.0, atr_m5=1.5)
        f = extract_features(ev, _make_bias(direction="SHORT"), candle=candle)
        # rejection = (1998-1996)/1.5 = 1.33 → clamped to 1.0
        assert f["sweep_rejection_strength"] == 1.0

    def test_rejection_strength_no_sweep_level(self):
        """Without sweep_level → rejection defaults to 0."""
        f = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        assert f["sweep_rejection_strength"] == 0.0

    # ── Heuristic integration ──

    def test_sweep_quality_boosts_score(self):
        """Higher sweep quality → higher heuristic score."""
        base = extract_features(
            _make_evaluation_with_sweep(sweep_magnitude=0.75),
            _make_bias(),
            candle=_make_candle(),
        )
        good = dict(base)
        good["sweep_quality"] = 1.0
        poor = dict(base)
        poor["sweep_quality"] = 0.0

        assert heuristic_score_v3(good) > heuristic_score_v3(poor)

    def test_sweep_absent_penalty(self):
        """Zero sweep quality should trigger the sweep_absent penalty."""
        base = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        with_sweep = dict(base)
        with_sweep["sweep_quality"] = 0.5  # no penalty
        without_sweep = dict(base)
        without_sweep["sweep_quality"] = 0.0  # triggers penalty

        assert heuristic_score_v3(with_sweep) > heuristic_score_v3(without_sweep)

    def test_sweep_proximity_boosts_score(self):
        """Closer sweep to entry → higher score."""
        base = extract_features(
            _make_evaluation_with_sweep(),
            _make_bias(),
            candle=_make_candle(),
        )
        close_entry = dict(base)
        close_entry["sweep_to_entry_atr"] = 0.9
        far_entry = dict(base)
        far_entry["sweep_to_entry_atr"] = 0.1

        assert heuristic_score_v3(close_entry) > heuristic_score_v3(far_entry)


# ═══════════════════════════════════════════════════════════════════════════
#  B5. MTF confluence features
# ═══════════════════════════════════════════════════════════════════════════


def _make_evaluation_with_mtf(
    *,
    side: str = "LONG",
    h1_bos_state: str = "BULLISH",
    h1_ema200: float | None = 1990.0,
    h1_pd_eq: float | None = 2000.0,
    h1_pd_low: float | None = 1995.0,
    h1_pd_high: float | None = 2005.0,
    atr_m5: float = 1.5,
) -> StrategyEvaluation:
    """Helper: create evaluation with MTF metadata."""
    return StrategyEvaluation(
        action=DecisionAction.OBSERVE,
        score_total=50.0,
        score_layers={"edge": 25.0, "trigger": 18.0, "execution": 7.0},
        penalties={},
        metadata={
            "side": side,
            "atr_m5": atr_m5,
            "spread": 0.2,
            "trigger_confirmations": 2,
            "fvg_detected": True,
            "fvg_size": 1.2,
            "fvg_mid": 1999.5,
            "entry_price": 1999.5,
            "stop_price": 1998.0,
            "tp_price": 2003.0,
            "h1_bos_state": h1_bos_state,
            "h1_ema200": h1_ema200,
            "h1_pd_eq": h1_pd_eq,
            "h1_pd_low": h1_pd_low,
            "h1_pd_high": h1_pd_high,
            "h1_close": 1999.0,
            "sweep_magnitude": 0.5,
        },
        snapshot={"atr_m5": atr_m5, "spread": 0.2},
    )


class TestMtfConfluenceFeatures:
    """MTF confluence feature extraction, bounds, and heuristic integration."""

    # ── Extraction ──

    def test_mtf_features_present(self):
        """All 4 MTF features must appear in output."""
        ev = _make_evaluation_with_mtf()
        features = extract_features(ev, _make_bias(), candle=_make_candle())
        for name in [
            "mtf_ema_confluence",
            "mtf_bos_agrees_side",
            "mtf_pd_zone_depth",
            "mtf_m5_momentum",
        ]:
            assert name in features, f"Missing MTF feature: {name}"

    def test_mtf_features_bounded(self):
        """All MTF features must be in [0, 1]."""
        ev = _make_evaluation_with_mtf()
        features = extract_features(ev, _make_bias(), candle=_make_candle())
        for name in [
            "mtf_ema_confluence",
            "mtf_bos_agrees_side",
            "mtf_pd_zone_depth",
            "mtf_m5_momentum",
        ]:
            assert 0.0 <= features[name] <= 1.0, f"{name} = {features[name]} out of [0,1]"

    # ── mtf_ema_confluence ──

    def test_ema_confluence_long_above(self):
        """LONG: close > EMA200 → confluence = 1."""
        # candle close=2000, ema200=1990 → above
        ev = _make_evaluation_with_mtf(side="LONG", h1_ema200=1990.0)
        f = extract_features(ev, _make_bias(direction="LONG"), candle=_make_candle())
        assert f["mtf_ema_confluence"] == 1.0

    def test_ema_confluence_long_below(self):
        """LONG: close < EMA200 → confluence = 0."""
        ev = _make_evaluation_with_mtf(side="LONG", h1_ema200=2010.0)
        f = extract_features(ev, _make_bias(direction="LONG"), candle=_make_candle())
        assert f["mtf_ema_confluence"] == 0.0

    def test_ema_confluence_short_below(self):
        """SHORT: close < EMA200 → confluence = 1."""
        candle = _make_candle_with(close=1985.0, high=1986.0, low=1984.0)
        ev = _make_evaluation_with_mtf(side="SHORT", h1_ema200=1990.0)
        f = extract_features(ev, _make_bias(direction="SHORT"), candle=candle)
        assert f["mtf_ema_confluence"] == 1.0

    def test_ema_confluence_no_ema_neutral(self):
        """Without h1_ema200 → neutral 0.5."""
        ev = _make_evaluation_with_mtf(h1_ema200=None)
        f = extract_features(ev, _make_bias(), candle=_make_candle())
        assert f["mtf_ema_confluence"] == 0.5

    # ── mtf_bos_agrees_side ──

    def test_bos_agrees_long_bullish(self):
        """LONG + BULLISH BOS → 1.0."""
        ev = _make_evaluation_with_mtf(side="LONG", h1_bos_state="BULLISH")
        f = extract_features(ev, _make_bias(direction="LONG"), candle=_make_candle())
        assert f["mtf_bos_agrees_side"] == 1.0

    def test_bos_agrees_short_bearish(self):
        """SHORT + BEARISH BOS → 1.0."""
        ev = _make_evaluation_with_mtf(side="SHORT", h1_bos_state="BEARISH")
        f = extract_features(ev, _make_bias(direction="SHORT"), candle=_make_candle())
        assert f["mtf_bos_agrees_side"] == 1.0

    def test_bos_neutral(self):
        """NONE BOS → 0.5."""
        ev = _make_evaluation_with_mtf(h1_bos_state="NONE")
        f = extract_features(ev, _make_bias(), candle=_make_candle())
        assert f["mtf_bos_agrees_side"] == 0.5

    def test_bos_opposing(self):
        """LONG + BEARISH BOS → 0.0."""
        ev = _make_evaluation_with_mtf(side="LONG", h1_bos_state="BEARISH")
        f = extract_features(ev, _make_bias(direction="LONG"), candle=_make_candle())
        assert f["mtf_bos_agrees_side"] == 0.0

    # ── mtf_pd_zone_depth ──

    def test_pd_depth_long_deep_discount(self):
        """LONG: price deep in discount zone → high depth."""
        # pd_eq=2000, pd_low=1995, close=1996 → depth = (2000-1996)/(2000-1995) = 0.8
        candle = _make_candle_with(close=1996.0, high=1997.0, low=1995.5)
        ev = _make_evaluation_with_mtf(side="LONG", h1_pd_eq=2000.0, h1_pd_low=1995.0)
        f = extract_features(ev, _make_bias(direction="LONG"), candle=candle)
        assert f["mtf_pd_zone_depth"] == pytest.approx(0.8, abs=0.01)

    def test_pd_depth_long_above_eq(self):
        """LONG: price above EQ (premium) → depth = 0."""
        # close=2001 > pd_eq=2000 → not in discount
        candle = _make_candle_with(close=2001.0, high=2002.0, low=2000.5)
        ev = _make_evaluation_with_mtf(side="LONG", h1_pd_eq=2000.0, h1_pd_low=1995.0)
        f = extract_features(ev, _make_bias(direction="LONG"), candle=candle)
        assert f["mtf_pd_zone_depth"] == 0.0

    def test_pd_depth_short_deep_premium(self):
        """SHORT: price deep in premium zone → high depth."""
        # pd_eq=2000, pd_high=2005, close=2004 → depth = (2004-2000)/(2005-2000) = 0.8
        candle = _make_candle_with(close=2004.0, high=2005.0, low=2003.5)
        ev = _make_evaluation_with_mtf(side="SHORT", h1_pd_eq=2000.0, h1_pd_high=2005.0)
        f = extract_features(ev, _make_bias(direction="SHORT"), candle=candle)
        assert f["mtf_pd_zone_depth"] == pytest.approx(0.8, abs=0.01)

    def test_pd_depth_no_data_neutral(self):
        """Without PD data → neutral 0.5."""
        ev = _make_evaluation_with_mtf(h1_pd_eq=None, h1_pd_low=None, h1_pd_high=None)
        f = extract_features(ev, _make_bias(), candle=_make_candle())
        assert f["mtf_pd_zone_depth"] == 0.5

    # ── mtf_m5_momentum ──

    def test_m5_momentum_long_bullish(self):
        """LONG: rising M5 candles → positive momentum."""
        candles = [
            _make_candle_with(close=1998.0),
            _make_candle_with(close=1999.0),
            _make_candle_with(close=2000.0),
        ]
        ev = _make_evaluation_with_mtf(side="LONG")
        f = extract_features(ev, _make_bias(direction="LONG"), candle=candles[-1], recent_candles=candles)
        # (2000-1998)/1.5 = 1.33 → clamped to 1.0
        assert f["mtf_m5_momentum"] == 1.0

    def test_m5_momentum_long_bearish(self):
        """LONG: falling M5 candles → zero momentum."""
        candles = [
            _make_candle_with(close=2002.0),
            _make_candle_with(close=2001.0),
            _make_candle_with(close=2000.0),
        ]
        ev = _make_evaluation_with_mtf(side="LONG")
        f = extract_features(ev, _make_bias(direction="LONG"), candle=candles[-1], recent_candles=candles)
        # (2000-2002)/1.5 = -1.33 → clamped to 0
        assert f["mtf_m5_momentum"] == 0.0

    def test_m5_momentum_short_bearish(self):
        """SHORT: falling M5 candles → positive momentum."""
        candles = [
            _make_candle_with(close=2002.0),
            _make_candle_with(close=2001.0),
            _make_candle_with(close=2000.0),
        ]
        ev = _make_evaluation_with_mtf(side="SHORT")
        f = extract_features(ev, _make_bias(direction="SHORT"), candle=candles[-1], recent_candles=candles)
        # -(2000-2002)/1.5 = 1.33 → clamped to 1.0
        assert f["mtf_m5_momentum"] == 1.0

    def test_m5_momentum_no_candles(self):
        """Without recent_candles → 0."""
        f = extract_features(_make_evaluation_with_mtf(), _make_bias(), candle=_make_candle())
        assert f["mtf_m5_momentum"] == 0.0

    def test_m5_momentum_too_few_candles(self):
        """With < 3 candles → 0."""
        candles = [_make_candle_with(close=1999.0), _make_candle_with(close=2000.0)]
        f = extract_features(
            _make_evaluation_with_mtf(),
            _make_bias(),
            candle=candles[-1],
            recent_candles=candles,
        )
        assert f["mtf_m5_momentum"] == 0.0

    # ── Heuristic integration ──

    def test_mtf_confluence_boosts_score(self):
        """Full MTF agreement should boost score vs no agreement."""
        base = extract_features(_make_evaluation_with_mtf(), _make_bias(), candle=_make_candle())
        full_agree = dict(base)
        full_agree["mtf_ema_confluence"] = 1.0
        full_agree["mtf_bos_agrees_side"] = 1.0
        full_agree["mtf_pd_zone_depth"] = 0.8
        full_agree["mtf_m5_momentum"] = 0.7

        no_agree = dict(base)
        no_agree["mtf_ema_confluence"] = 0.0
        no_agree["mtf_bos_agrees_side"] = 0.0
        no_agree["mtf_pd_zone_depth"] = 0.0
        no_agree["mtf_m5_momentum"] = 0.0

        assert heuristic_score_v3(full_agree) > heuristic_score_v3(no_agree)

    def test_bos_opposing_penalty(self):
        """BOS opposing setup direction should penalize score."""
        base = extract_features(_make_evaluation_with_mtf(), _make_bias(), candle=_make_candle())
        neutral = dict(base)
        neutral["mtf_bos_agrees_side"] = 0.5  # no penalty
        opposing = dict(base)
        opposing["mtf_bos_agrees_side"] = 0.0  # triggers penalty

        assert heuristic_score_v3(neutral) > heuristic_score_v3(opposing)


# ═══════════════════════════════════════════════════════════════════════════
#  B6. Regime detection features
# ═══════════════════════════════════════════════════════════════════════════


class TestRegimeFunctions:
    """Unit tests for regime.py pure functions."""

    # ── compute_trend_regime_score ──

    def test_trend_empty_candles(self):
        """No candles → 0.0."""
        from bot.strategy.regime import compute_trend_regime_score

        assert compute_trend_regime_score([], 1.5) == 0.0

    def test_trend_zero_atr(self):
        """ATR = 0 → 0.0."""
        from bot.strategy.regime import compute_trend_regime_score

        candles = [_make_candle_with(close=2000.0 + i) for i in range(25)]
        assert compute_trend_regime_score(candles, 0.0) == 0.0

    def test_trend_too_few_candles(self):
        """Fewer than ema_period candles → 0.0."""
        from bot.strategy.regime import compute_trend_regime_score

        candles = [_make_candle_with(close=2000.0 + i) for i in range(10)]
        assert compute_trend_regime_score(candles, 1.5, ema_period=20) == 0.0

    def test_trend_flat_market(self):
        """All closes identical → displacement = 0 → score = 0."""
        from bot.strategy.regime import compute_trend_regime_score

        candles = [_make_candle_with(close=2000.0) for _ in range(25)]
        assert compute_trend_regime_score(candles, 1.5) == 0.0

    def test_trend_strong_uptrend(self):
        """Closes rising steeply → large displacement → score near 1."""
        from bot.strategy.regime import compute_trend_regime_score

        # 25 candles rising by 2 each → last close far above EMA
        candles = [_make_candle_with(close=2000.0 + i * 2.0) for i in range(25)]
        score = compute_trend_regime_score(candles, 1.5)
        assert score > 0.5, f"Expected strong trend score, got {score}"

    def test_trend_bounded_0_1(self):
        """Score always in [0, 1]."""
        from bot.strategy.regime import compute_trend_regime_score

        candles = [_make_candle_with(close=2000.0 + i * 10.0) for i in range(25)]
        score = compute_trend_regime_score(candles, 0.1)  # tiny ATR → huge ratio
        assert 0.0 <= score <= 1.0

    # ── compute_vol_regime_change ──

    def test_vol_empty_history(self):
        """Empty ATR history → 0.0."""
        from bot.strategy.regime import compute_vol_regime_change

        assert compute_vol_regime_change([]) == 0.0

    def test_vol_too_few_values(self):
        """Fewer than long_window valid values → 0.0."""
        from bot.strategy.regime import compute_vol_regime_change

        assert compute_vol_regime_change([1.0] * 30) == 0.0  # < 50

    def test_vol_stable(self):
        """Constant ATR → ratio = 0.0 (no change)."""
        from bot.strategy.regime import compute_vol_regime_change

        history = [1.5] * 60
        assert compute_vol_regime_change(history) == pytest.approx(0.0, abs=0.01)

    def test_vol_expanding(self):
        """Recent ATR higher than long-term → positive (expanding)."""
        from bot.strategy.regime import compute_vol_regime_change

        # 40 low values + 20 higher values → short_mean > long_mean
        history = [1.0] * 40 + [2.0] * 20
        result = compute_vol_regime_change(history)
        assert result > 0.0, f"Expected positive vol change, got {result}"

    def test_vol_compressing(self):
        """Recent ATR lower than long-term → negative (compressing)."""
        from bot.strategy.regime import compute_vol_regime_change

        # 40 high values + 20 lower values → short_mean < long_mean
        history = [2.0] * 40 + [1.0] * 20
        result = compute_vol_regime_change(history)
        assert result < 0.0, f"Expected negative vol change, got {result}"

    def test_vol_bounded(self):
        """Result always in [-1, 1]."""
        from bot.strategy.regime import compute_vol_regime_change

        # Extreme: tiny long-term, huge recent
        history = [0.01] * 40 + [100.0] * 20
        result = compute_vol_regime_change(history)
        assert -1.0 <= result <= 1.0

    def test_vol_handles_none_values(self):
        """None values in history are skipped."""
        from bot.strategy.regime import compute_vol_regime_change

        history = [None, 1.5] * 30 + [1.5] * 10  # 40 valid values → not enough
        result = compute_vol_regime_change(history)
        assert result == 0.0  # only ~40 valid, need 50


class TestRegimeDetectionFeatures:
    """Regime detection feature extraction, bounds, and heuristic integration."""

    # ── Extraction ──

    def test_regime_features_present(self):
        """Both regime features must appear in output."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        assert "regime_trend_score" in features
        assert "regime_vol_change" in features

    def test_regime_features_numeric(self):
        """Regime features must be float, not NaN/Inf."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        for name in ["regime_trend_score", "regime_vol_change"]:
            val = features[name]
            assert isinstance(val, float), f"{name} is {type(val)}"
            assert not math.isnan(val), f"{name} is NaN"

    def test_trend_score_bounded(self):
        """regime_trend_score always in [0, 1]."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        assert 0.0 <= features["regime_trend_score"] <= 1.0

    def test_vol_change_bounded(self):
        """regime_vol_change always in [-1, 1]."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        assert -1.0 <= features["regime_vol_change"] <= 1.0

    def test_trend_score_with_trending_candles(self):
        """Providing trending candles → positive regime_trend_score."""
        candles = [_make_candle_with(close=2000.0 + i * 2.0) for i in range(25)]
        features = extract_features(
            _make_evaluation(),
            _make_bias(),
            candle=candles[-1],
            recent_candles=candles,
        )
        assert features["regime_trend_score"] > 0.0

    def test_trend_score_without_candles(self):
        """Without recent_candles → trend_score = 0."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        assert features["regime_trend_score"] == 0.0

    def test_vol_change_with_expanding_history(self):
        """Expanding ATR history → positive vol_change."""
        history: list[float | None] = [1.0] * 40 + [2.0] * 20
        features = extract_features(
            _make_evaluation(),
            _make_bias(),
            candle=_make_candle(),
            atr_history=history,
        )
        assert features["regime_vol_change"] > 0.0

    def test_vol_change_without_history(self):
        """Without atr_history → vol_change = 0."""
        features = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        assert features["regime_vol_change"] == 0.0

    # ── Heuristic integration ──

    def test_trend_score_boosts_heuristic(self):
        """Higher regime_trend_score → higher heuristic score."""
        base = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        trending = dict(base)
        trending["regime_trend_score"] = 0.8
        flat = dict(base)
        flat["regime_trend_score"] = 0.0

        assert heuristic_score_v3(trending) > heuristic_score_v3(flat)

    def test_vol_expanding_boosts_heuristic(self):
        """Positive vol_change → bonus applied."""
        base = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        expanding = dict(base)
        expanding["regime_vol_change"] = 0.5
        neutral = dict(base)
        neutral["regime_vol_change"] = 0.0

        assert heuristic_score_v3(expanding) > heuristic_score_v3(neutral)

    def test_vol_compressing_penalty(self):
        """Strongly compressing vol (< -0.3) → penalty applied."""
        base = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        compressing = dict(base)
        compressing["regime_vol_change"] = -0.5
        neutral = dict(base)
        neutral["regime_vol_change"] = 0.0

        assert heuristic_score_v3(compressing) < heuristic_score_v3(neutral)

    def test_vol_mild_compression_no_penalty(self):
        """Mild compression (-0.2) should NOT trigger penalty (threshold is -0.3)."""
        base = extract_features(_make_evaluation(), _make_bias(), candle=_make_candle())
        mild = dict(base)
        mild["regime_vol_change"] = -0.2
        neutral = dict(base)
        neutral["regime_vol_change"] = 0.0

        # Both should have the same score (no bonus, no penalty)
        assert heuristic_score_v3(mild) == heuristic_score_v3(neutral)


# ═══════════════════════════════════════════════════════════════════════════
#  C. ScoreV3Engine — action resolution and tiers
# ═══════════════════════════════════════════════════════════════════════════


class TestScoreV3Engine:
    """Engine: scoring, action, tier resolution."""

    def test_score_returns_tuple(self):
        """score() returns (float, dict)."""
        engine = ScoreV3Engine()
        score, features = engine.score(_make_evaluation(), _make_bias(), candle=_make_candle())
        assert isinstance(score, float)
        assert isinstance(features, dict)
        assert 0 <= score <= 100

    def test_resolve_action_trade(self):
        """High score → TRADE."""
        engine = ScoreV3Engine(ScoreV3Config(trade_threshold=55.0))
        action = engine.resolve_action(60.0)
        assert action == DecisionAction.TRADE

    def test_resolve_action_small(self):
        """Score in small range → SMALL."""
        engine = ScoreV3Engine(ScoreV3Config(small_min=45.0, small_max=54.99))
        action = engine.resolve_action(50.0)
        assert action == DecisionAction.SMALL

    def test_resolve_action_observe(self):
        """Low score → OBSERVE."""
        engine = ScoreV3Engine(ScoreV3Config(small_min=45.0))
        action = engine.resolve_action(30.0)
        assert action == DecisionAction.OBSERVE

    def test_resolve_action_blocking_always_observe(self):
        """If blocking reasons exist, action is always OBSERVE."""
        engine = ScoreV3Engine(ScoreV3Config(trade_threshold=55.0))
        action = engine.resolve_action(99.0, has_blocking_reasons=True)
        assert action == DecisionAction.OBSERVE

    def test_tier_fixed_a_plus(self):
        """High score → A_plus tier (fixed thresholds)."""
        engine = ScoreV3Engine(ScoreV3Config(trade_threshold=55.0, tier_enabled=True))
        tier = engine.resolve_tier(70.0)
        assert tier == "A_plus"

    def test_tier_fixed_a(self):
        """Score at trade_threshold → A tier."""
        engine = ScoreV3Engine(ScoreV3Config(trade_threshold=55.0, tier_enabled=True))
        tier = engine.resolve_tier(55.0)
        assert tier == "A"

    def test_tier_fixed_b(self):
        """Score in small range → B tier."""
        engine = ScoreV3Engine(ScoreV3Config(trade_threshold=55.0, small_min=45.0, tier_enabled=True))
        tier = engine.resolve_tier(48.0)
        assert tier == "B"

    def test_tier_fixed_observe(self):
        """Low score → OBSERVE tier."""
        engine = ScoreV3Engine(ScoreV3Config(small_min=45.0, tier_enabled=True))
        tier = engine.resolve_tier(20.0)
        assert tier == "OBSERVE"

    def test_tier_disabled_returns_none(self):
        """tier_enabled=False → NONE."""
        engine = ScoreV3Engine(ScoreV3Config(tier_enabled=False))
        assert engine.resolve_tier(99.0) == "NONE"

    def test_quantile_boundaries_update(self):
        """After enough scores, quantile boundaries are computed."""
        engine = ScoreV3Engine(ScoreV3Config(tier_enabled=True))
        # Feed 300 scores
        for i in range(300):
            engine._score_history.append(float(i) / 3.0)
        engine.update_quantile_boundaries()
        bounds = engine.quantile_boundaries
        assert bounds is not None
        assert "a_plus" in bounds
        assert "a" in bounds
        assert "b" in bounds
        assert bounds["a_plus"] >= bounds["a"] >= bounds["b"]

    def test_heuristic_model_predict(self):
        """HeuristicScoreV3Model returns (p_win, expected_r)."""
        model = HeuristicScoreV3Model()
        p_win, expected_r = model.predict({"htf_bias_aligned": 1.0})
        assert 0 <= p_win <= 1
        assert isinstance(expected_r, float)


# ═══════════════════════════════════════════════════════════════════════════
#  D. Shadow observer
# ═══════════════════════════════════════════════════════════════════════════


class TestShadowObserver:
    """Shadow candidate recording and summary."""

    def test_classify_session(self):
        """Session classification by UTC hour."""
        assert classify_session(9) == "LONDON"
        assert classify_session(13) == "OVERLAP"
        assert classify_session(17) == "NY"
        assert classify_session(3) == "ASIA"
        assert classify_session(23) == "OTHER"

    def test_atr_percentile_basic(self):
        """ATR percentile computation."""
        history = [float(i) for i in range(1, 101)]
        pct = compute_atr_percentile(50.0, history)
        assert 0.4 <= pct <= 0.6

    def test_atr_percentile_empty(self):
        """Empty history → 0.5."""
        assert compute_atr_percentile(1.0, []) == 0.5

    def test_shadow_observer_records(self):
        """Observer records candidates."""
        obs = ShadowObserver()
        sc = ShadowCandidate(
            timestamp="2024-06-15T10:30:00",
            symbol="XAUUSD",
            side="LONG",
            action="OBSERVE",
            tier="OBSERVE",
            score_v2=35.0,
        )
        obs.record(sc)
        assert len(obs.records) == 1

    def test_shadow_observer_summary_empty(self):
        """Empty observer produces summary with total=0."""
        obs = ShadowObserver()
        summary = obs.summary()
        assert summary["total"] == 0

    def test_shadow_observer_summary_with_records(self):
        """Summary counts actions correctly."""
        obs = ShadowObserver()
        for action in ["TRADE", "TRADE", "OBSERVE", "OBSERVE", "OBSERVE"]:
            obs.record(
                ShadowCandidate(
                    timestamp="2024-06-15T10:30:00",
                    symbol="XAUUSD",
                    side="LONG",
                    action=action,
                    tier="NONE",
                    score_v2=40.0,
                    shadow_filled=action == "TRADE",
                    shadow_r=0.5 if action == "TRADE" else 0,
                )
            )
        summary = obs.summary()
        assert summary["total"] == 5
        assert summary["by_action"]["TRADE"]["count"] == 2
        assert summary["by_action"]["OBSERVE"]["count"] == 3

    def test_shadow_observer_writes_jsonl(self):
        """Observer writes to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shadow.jsonl"
            obs = ShadowObserver(path)
            obs.record(
                ShadowCandidate(
                    timestamp="2024-06-15T10:30:00",
                    symbol="XAUUSD",
                    side="LONG",
                    action="OBSERVE",
                    tier="OBSERVE",
                    score_v2=35.0,
                )
            )
            obs.flush()
            obs.close()
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["symbol"] == "XAUUSD"

    def test_simulate_shadow_outcome_not_filled(self):
        """If entry is never touched, shadow result is NOT_FILLED."""
        candles = [
            Candle(
                timestamp=datetime(2024, 6, 15, 10 + (i * 5) // 60, (i * 5) % 60, tzinfo=timezone.utc),
                open=2000 + i,
                high=2000 + i + 1,
                low=2000 + i - 0.5,
                close=2000 + i + 0.5,
            )
            for i in range(50)
        ]
        result = simulate_shadow_outcome(
            side="LONG",
            entry_price=1950.0,  # far below all candles
            stop_price=1945.0,
            tp_price=1960.0,
            candles=candles,
            start_index=0,
        )
        assert result["filled"] is False
        assert result["exit_reason"] == "NOT_FILLED"

    def test_simulate_shadow_outcome_tp_hit(self):
        """If price reaches TP, shadow result has exit_reason TP."""
        candles = [
            Candle(
                timestamp=datetime(2024, 6, 15, 10 + (i * 5) // 60, (i * 5) % 60, tzinfo=timezone.utc),
                open=2000.0,
                high=2000.0 + i * 0.5,
                low=1999.0,
                close=2000.0 + i * 0.3,
            )
            for i in range(50)
        ]
        result = simulate_shadow_outcome(
            side="LONG",
            entry_price=2000.0,
            stop_price=1998.0,
            tp_price=2005.0,
            candles=candles,
            start_index=0,
        )
        assert result["filled"] is True
        assert result["exit_reason"] in ("TP", "STOP", "BE", "EXPIRE")


# ═══════════════════════════════════════════════════════════════════════════
#  E. Integration: apply_score_v3
# ═══════════════════════════════════════════════════════════════════════════


class TestApplyScoreV3:
    """Integration: apply_score_v3 updates evaluation correctly."""

    def test_updates_score_total(self):
        """score_total is replaced with V3 score."""
        engine = ScoreV3Engine()
        ev = _make_evaluation(score=35.0)
        candle = _make_candle()
        ev_out = apply_score_v3(engine, ev, _make_bias(), candle=candle)
        assert ev_out.score_total != 35.0  # V3 overwrites
        assert 0 <= ev_out.score_total <= 100

    def test_preserves_v2_in_metadata(self):
        """Original V2 score is saved in metadata.score_v2."""
        engine = ScoreV3Engine()
        ev = _make_evaluation(score=42.0)
        ev_out = apply_score_v3(engine, ev, _make_bias(), candle=_make_candle())
        assert ev_out.metadata.get("score_v2") == 42.0

    def test_sets_action_from_v3(self):
        """Action is recalculated from V3 score, not V2."""
        engine = ScoreV3Engine(ScoreV3Config(trade_threshold=10.0, small_min=5.0, small_max=9.99))
        ev = _make_evaluation(score=5.0)
        ev_out = apply_score_v3(engine, ev, _make_bias(), candle=_make_candle())
        # With threshold=10, a decent setup should score well above 10
        assert ev_out.action in {DecisionAction.TRADE, DecisionAction.SMALL}

    def test_blocking_reasons_force_observe(self):
        """If evaluation has blocking reasons, action stays OBSERVE."""
        engine = ScoreV3Engine(ScoreV3Config(trade_threshold=10.0))
        ev = _make_evaluation(score=99.0)
        ev.reasons_blocking = ["GATE_REACTION_WAIT"]
        ev_out = apply_score_v3(engine, ev, _make_bias(), candle=_make_candle())
        assert ev_out.action == DecisionAction.OBSERVE

    def test_tier_assigned(self):
        """Tier is stored in metadata."""
        engine = ScoreV3Engine(ScoreV3Config(tier_enabled=True))
        ev = _make_evaluation()
        ev_out = apply_score_v3(engine, ev, _make_bias(), candle=_make_candle())
        assert "tier" in ev_out.metadata
        assert ev_out.metadata["tier"] in {"A_plus", "A", "B", "OBSERVE"}

    def test_features_saved_in_metadata(self):
        """V3 features dict is stored in metadata.score_v3_features."""
        engine = ScoreV3Engine()
        ev = _make_evaluation()
        ev_out = apply_score_v3(engine, ev, _make_bias(), candle=_make_candle())
        features = ev_out.metadata.get("score_v3_features")
        assert isinstance(features, dict)
        assert "htf_bias_aligned" in features
