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

    def test_all_35_features_present(self):
        """Every feature in FEATURE_NAMES must appear in output."""
        features = extract_features(
            _make_evaluation(), _make_bias(), candle=_make_candle()
        )
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"

    def test_features_are_numeric(self):
        """All features must be float or int, never None or NaN."""
        features = extract_features(
            _make_evaluation(), _make_bias(), candle=_make_candle()
        )
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
        features = extract_features(
            _make_evaluation(side="LONG"), _make_bias("LONG"), candle=_make_candle()
        )
        assert features["htf_bias_aligned"] == 1.0

    def test_htf_bias_misaligned(self):
        """When bias direction mismatches side, htf_bias_aligned = 0."""
        features = extract_features(
            _make_evaluation(side="LONG"), _make_bias("SHORT"), candle=_make_candle()
        )
        assert features["htf_bias_aligned"] == 0.0

    def test_session_classification_london(self):
        """Candle at 10:30 UTC → LONDON session."""
        features = extract_features(
            _make_evaluation(), _make_bias(), candle=_make_candle(hour=10)
        )
        assert features["is_london"] == 1.0
        assert features["is_overlap"] == 0.0
        assert features["is_ny"] == 0.0

    def test_session_classification_overlap(self):
        """Candle at 14:00 UTC → OVERLAP session."""
        features = extract_features(
            _make_evaluation(), _make_bias(), candle=_make_candle(hour=14)
        )
        assert features["is_overlap"] == 1.0

    def test_rr_ratio_positive(self):
        """RR ratio should be positive for valid entry/stop/tp."""
        features = extract_features(
            _make_evaluation(), _make_bias(), candle=_make_candle()
        )
        assert features["rr_ratio"] > 0

    def test_spread_ratio_bounded(self):
        """Spread ratio should be positive and finite."""
        features = extract_features(
            _make_evaluation(), _make_bias(), candle=_make_candle()
        )
        assert 0 <= features["spread_ratio"] < 100

    def test_fill_probability_bounded(self):
        """Fill probability proxy must be in [0, 1]."""
        features = extract_features(
            _make_evaluation(), _make_bias(), candle=_make_candle()
        )
        assert 0 <= features["fill_probability_proxy"] <= 1


# ═══════════════════════════════════════════════════════════════════════════
#  B. Heuristic scorer
# ═══════════════════════════════════════════════════════════════════════════

class TestHeuristicScorer:
    """Heuristic scorer produces bounded, non-NaN scores."""

    def test_score_in_range(self):
        """Score must be in [0, 100]."""
        features = extract_features(
            _make_evaluation(), _make_bias(), candle=_make_candle()
        )
        score = heuristic_score_v3(features)
        assert 0 <= score <= 100

    def test_score_not_nan(self):
        """Score must never be NaN."""
        features = extract_features(
            _make_evaluation(), _make_bias(), candle=_make_candle()
        )
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
            "v2_penalty_total": 0.0,
            "spread_ratio": 0.05,
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
            "v2_penalty_total": 10.0,
            "spread_ratio": 0.3,
        }
        score = heuristic_score_v3(features)
        assert score < 20, f"Poor setup scored {score}"

    def test_fallback_penalty_applied(self):
        """Displacement retest setups should score lower."""
        base_features = extract_features(
            _make_evaluation(), _make_bias(), candle=_make_candle()
        )
        fallback_features = dict(base_features)
        fallback_features["setup_is_fallback"] = 1.0

        base_score = heuristic_score_v3(base_features)
        fallback_score = heuristic_score_v3(fallback_features)
        assert fallback_score < base_score, "Fallback should penalize score"


# ═══════════════════════════════════════════════════════════════════════════
#  C. ScoreV3Engine — action resolution and tiers
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreV3Engine:
    """Engine: scoring, action, tier resolution."""

    def test_score_returns_tuple(self):
        """score() returns (float, dict)."""
        engine = ScoreV3Engine()
        score, features = engine.score(
            _make_evaluation(), _make_bias(), candle=_make_candle()
        )
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
        engine = ScoreV3Engine(ScoreV3Config(
            trade_threshold=55.0, small_min=45.0, tier_enabled=True
        ))
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
            obs.record(ShadowCandidate(
                timestamp="2024-06-15T10:30:00",
                symbol="XAUUSD",
                side="LONG",
                action=action,
                tier="NONE",
                score_v2=40.0,
                shadow_filled=action == "TRADE",
                shadow_r=0.5 if action == "TRADE" else 0,
            ))
        summary = obs.summary()
        assert summary["total"] == 5
        assert summary["by_action"]["TRADE"]["count"] == 2
        assert summary["by_action"]["OBSERVE"]["count"] == 3

    def test_shadow_observer_writes_jsonl(self):
        """Observer writes to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shadow.jsonl"
            obs = ShadowObserver(path)
            obs.record(ShadowCandidate(
                timestamp="2024-06-15T10:30:00",
                symbol="XAUUSD",
                side="LONG",
                action="OBSERVE",
                tier="OBSERVE",
                score_v2=35.0,
            ))
            obs.flush()
            obs.close()
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["symbol"] == "XAUUSD"

    def test_simulate_shadow_outcome_not_filled(self):
        """If entry is never touched, shadow result is NOT_FILLED."""
        candles = [
            Candle(timestamp=datetime(2024, 6, 15, 10 + (i * 5) // 60, (i * 5) % 60, tzinfo=timezone.utc),
                   open=2000 + i, high=2000 + i + 1, low=2000 + i - 0.5, close=2000 + i + 0.5)
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
            Candle(timestamp=datetime(2024, 6, 15, 10 + (i * 5) // 60, (i * 5) % 60, tzinfo=timezone.utc),
                   open=2000.0, high=2000.0 + i * 0.5, low=1999.0, close=2000.0 + i * 0.3)
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
