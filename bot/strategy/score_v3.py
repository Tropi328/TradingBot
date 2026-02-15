"""
ScoreV3 — Enhanced signal scoring with feature extraction and ML hooks.

Architecture:
  1. FeatureExtractor:  evaluation + candles → flat feature dict
  2. ScoreV3Model:      features → (p_win, expected_R, score_v3)
  3. Calibrated tiers:  score_v3 → tier via quantile boundaries

The model operates in two modes:
  - HEURISTIC (default): improved rule-based scoring that weights features
    better than V2, with session/volatility awareness. No training needed.
  - ML (after training): LightGBM or LogisticRegression loaded from disk.

ScoreV3 replaces the V2 composite score, while keeping the same
StrategyEvaluation contract so all downstream code (gates, sizing, orders)
works unchanged.
"""
from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from bot.data.candles import Candle
from bot.strategy.contracts import (
    BiasState,
    DecisionAction,
    StrategyEvaluation,
)
from bot.strategy.shadow_observer import classify_session, compute_atr_percentile


# ═══════════════════════════════════════════════════════════════════════════
#  Feature extraction
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_NAMES: list[str] = [
    # HTF alignment (0-3)
    "htf_bias_aligned",         # 1 if h1 bias == signal side
    "htf_bos_confirmed",        # 1 if h1 BOS state is not NONE
    "htf_pd_ok",                # 1 if premium/discount filter passes
    "htf_location_score",       # 0-1, how deep in discount/premium zone

    # FVG quality (4-8)
    "fvg_present",              # 1/0
    "fvg_size_atr",             # fvg_size / atr
    "fvg_age_bars",             # bars since FVG formed
    "fvg_distance_to_price",    # |fvg_mid - current_close| / atr
    "fvg_displacement_ratio",   # displacement body / atr

    # Trigger quality (9-13)
    "trigger_confirmations",    # 0-3 count
    "mss_confirmed",            # 1/0
    "displacement_confirmed",   # 1/0
    "sweep_magnitude_atr",      # sweep_magnitude / atr
    "setup_is_fallback",        # 1 if displacement_retest

    # Volatility regime (14-18)
    "atr_m5_raw",               # raw ATR value
    "atr_percentile",           # rank in recent history (0-1)
    "spread_ratio",             # spread / atr
    "spread_score_raw",         # 0-8 from V2 spread scoring
    "vol_regime",               # 0=low, 1=normal, 2=high

    # Session / time (19-24)
    "hour_utc",                 # 0-23
    "day_of_week",              # 0=Mon, 6=Sun
    "is_london",                # binary
    "is_ny",                    # binary
    "is_overlap",               # binary
    "news_blocked",             # binary

    # Entry quality (25-30)
    "rr_ratio",                 # reward-to-risk ratio
    "risk_distance_atr",        # |entry - stop| / atr
    "entry_distance_atr",       # |entry - close| / atr
    "edge_to_cost_ratio",       # expected_gain / (spread + slippage)
    "tp_distance_atr",          # |entry - tp| / atr
    "fill_probability_proxy",   # heuristic: distance-to-entry → likely fill

    # V2 score components (31-34)
    "v2_edge_score",
    "v2_trigger_score",
    "v2_execution_score",
    "v2_penalty_total",
]


def extract_features(
    evaluation: StrategyEvaluation,
    bias: BiasState,
    *,
    candle: Candle,
    atr_m5: float | None = None,
    atr_history: list[float | None] | None = None,
    spread: float | None = None,
    assumed_spread: float = 0.2,
    entry_price: float | None = None,
    stop_price: float | None = None,
    tp_price: float | None = None,
) -> dict[str, float]:
    """Extract a flat feature dict from an evaluation + context.

    All features are numeric (float). Missing values are filled with
    sensible defaults (0.0 or 0.5 for percentiles).
    """
    meta = evaluation.metadata or {}
    snap = evaluation.snapshot or {}
    layers = evaluation.score_layers or {}
    penalties = evaluation.penalties or {}

    side = str(meta.get("side", "")).upper()
    _atr_raw = atr_m5 or meta.get("atr_m5") or snap.get("atr_m5") or 1e-9
    try:
        atr = float(_atr_raw)
    except (TypeError, ValueError):
        atr = 1e-9
    if atr <= 0:
        atr = 1e-9
    try:
        spread_val = float(spread or meta.get("spread") or snap.get("spread") or assumed_spread)
    except (TypeError, ValueError):
        spread_val = float(assumed_spread)
    close_price = float(candle.close)
    ts = candle.timestamp
    hour = ts.hour if hasattr(ts, "hour") else 12
    dow = ts.weekday() if hasattr(ts, "weekday") else 0
    session = classify_session(hour)

    # HTF alignment
    bias_aligned = 1.0 if bias.direction == side else 0.0
    bos_confirmed = 1.0 if str(meta.get("h1_bos_state", snap.get("h1_bos_state", "NONE"))).upper() != "NONE" else 0.0
    pd_fail = bool(meta.get("h1_pd_fail", False))
    pd_ok = 0.0 if pd_fail else 1.0
    h1_pd_eq = meta.get("h1_pd_eq", snap.get("h1_pd_eq"))
    h1_close = meta.get("h1_close", snap.get("h1_close"))
    location_score = 0.5
    if h1_pd_eq is not None and h1_close is not None:
        try:
            eq_f = float(h1_pd_eq)
            cl_f = float(h1_close)
            if side == "LONG":
                location_score = min(1.0, max(0.0, (eq_f - cl_f) / max(atr * 10, 1e-9) + 0.5))
            else:
                location_score = min(1.0, max(0.0, (cl_f - eq_f) / max(atr * 10, 1e-9) + 0.5))
        except (TypeError, ValueError):
            location_score = 0.5

    # FVG quality
    fvg_present = 1.0 if meta.get("fvg_detected") else 0.0
    fvg_size = float(meta.get("fvg_size", 0))
    fvg_size_atr = fvg_size / atr if atr > 0 else 0
    fvg_mid = meta.get("fvg_mid")
    fvg_dist = abs(float(fvg_mid) - close_price) / atr if fvg_mid is not None and atr > 0 else 0
    fvg_c1_idx = meta.get("fvg_c1_index", meta.get("fvg_c1_idx"))
    sweep_idx = meta.get("sweep_index", 0)
    fvg_age = 0
    if fvg_c1_idx is not None:
        try:
            fvg_age = max(0, int(len(evaluation.snapshot.get("candles_m5", [0] * 100)) if isinstance(evaluation.snapshot.get("candles_m5"), list) else 0) - int(fvg_c1_idx))
        except (TypeError, ValueError):
            fvg_age = 0
    displacement_ratio = float(meta.get("displacement_ratio", meta.get("displacement_threshold", 0)))
    if isinstance(displacement_ratio, (int, float)):
        displacement_ratio = float(displacement_ratio)
    else:
        displacement_ratio = 0.0

    # Trigger quality
    trigger_conf = int(meta.get("trigger_confirmations", 0))
    mss_ok = 1.0 if meta.get("mss_index") is not None else 0.0
    disp_ok = 1.0 if displacement_ratio > 0 else 0.0
    sweep_mag = float(meta.get("sweep_magnitude", meta.get("sweep_mag", 0)))
    sweep_atr = sweep_mag / atr if atr > 0 else 0
    is_fallback = 1.0 if str(meta.get("setup_origin", "")).upper() == "DISPLACEMENT_RETEST" else 0.0

    # Volatility regime
    atr_pct = 0.5
    if atr_history:
        atr_pct = compute_atr_percentile(atr, atr_history)
    spread_ratio = spread_val / atr if atr > 0 else 0
    vol_regime = 1.0  # normal
    if atr_pct < 0.2:
        vol_regime = 0.0  # low
    elif atr_pct > 0.8:
        vol_regime = 2.0  # high

    # Spread score (from V2 logic)
    max_spread_ratio = 0.15
    if spread_ratio <= max_spread_ratio:
        spread_score_raw = 8.0
    elif spread_ratio <= max_spread_ratio * 1.25:
        spread_score_raw = 4.0
    else:
        spread_score_raw = 0.0

    # Entry quality
    e = float(entry_price or meta.get("entry_price", meta.get("fvg_mid", close_price)))
    s = float(stop_price or meta.get("stop_price", meta.get("sweep_level", close_price - atr)))
    t = float(tp_price or meta.get("tp_price", close_price + 2 * atr))
    risk_dist = abs(e - s)
    reward_dist = abs(t - e)
    rr = reward_dist / risk_dist if risk_dist > 0 else 0
    risk_dist_atr = risk_dist / atr if atr > 0 else 0
    entry_dist = abs(e - close_price)
    entry_dist_atr = entry_dist / atr if atr > 0 else 0
    tp_dist_atr = reward_dist / atr if atr > 0 else 0
    cost = spread_val + 0.0  # no slippage in heuristic mode
    edge_to_cost = reward_dist / cost if cost > 0 else 100.0

    # Fill probability proxy: closer entries are more likely to fill
    fill_prob = max(0.0, min(1.0, 1.0 - (entry_dist_atr / 3.0)))

    # V2 components
    v2_edge = float(layers.get("edge", 0))
    v2_trigger = float(layers.get("trigger", 0))
    v2_exec = float(layers.get("execution", 0))
    v2_penalty = float(sum(penalties.values())) if penalties else 0

    return {
        "htf_bias_aligned": bias_aligned,
        "htf_bos_confirmed": bos_confirmed,
        "htf_pd_ok": pd_ok,
        "htf_location_score": round(location_score, 4),
        "fvg_present": fvg_present,
        "fvg_size_atr": round(fvg_size_atr, 4),
        "fvg_age_bars": float(fvg_age),
        "fvg_distance_to_price": round(fvg_dist, 4),
        "fvg_displacement_ratio": round(displacement_ratio, 4),
        "trigger_confirmations": float(trigger_conf),
        "mss_confirmed": mss_ok,
        "displacement_confirmed": disp_ok,
        "sweep_magnitude_atr": round(sweep_atr, 4),
        "setup_is_fallback": is_fallback,
        "atr_m5_raw": round(atr, 6),
        "atr_percentile": round(atr_pct, 4),
        "spread_ratio": round(spread_ratio, 4),
        "spread_score_raw": spread_score_raw,
        "vol_regime": vol_regime,
        "hour_utc": float(hour),
        "day_of_week": float(dow),
        "is_london": 1.0 if session == "LONDON" else 0.0,
        "is_ny": 1.0 if session == "NY" else 0.0,
        "is_overlap": 1.0 if session == "OVERLAP" else 0.0,
        "news_blocked": 1.0 if meta.get("news_blocked") or bool(snap.get("news_blocked")) else 0.0,
        "rr_ratio": round(rr, 4),
        "risk_distance_atr": round(risk_dist_atr, 4),
        "entry_distance_atr": round(entry_dist_atr, 4),
        "edge_to_cost_ratio": round(min(edge_to_cost, 100.0), 4),
        "tp_distance_atr": round(tp_dist_atr, 4),
        "fill_probability_proxy": round(fill_prob, 4),
        "v2_edge_score": v2_edge,
        "v2_trigger_score": v2_trigger,
        "v2_execution_score": v2_exec,
        "v2_penalty_total": v2_penalty,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Heuristic ScoreV3 (default, no training needed)
# ═══════════════════════════════════════════════════════════════════════════

# Weights for heuristic scoring — tuned for ICT scalp setups
_HEURISTIC_WEIGHTS: dict[str, float] = {
    # HTF (max ~25)
    "htf_bias_aligned":      10.0,
    "htf_bos_confirmed":      5.0,
    "htf_pd_ok":               4.0,
    "htf_location_score":      6.0,

    # FVG (max ~18)
    "fvg_present":             8.0,
    "fvg_size_atr":            4.0,   # capped at 2.0 → 8.0
    "fvg_displacement_ratio":  3.0,   # capped at 2.0 → 6.0

    # Trigger (max ~25)
    "trigger_confirmations":   6.0,   # per confirmation (max 3 → 18)
    "mss_confirmed":           4.0,
    "sweep_magnitude_atr":     3.0,   # capped at 1.0 → 3.0

    # Execution (max ~15)
    "spread_score_raw":        1.2,   # 0-8 → 0-9.6
    "rr_ratio":                2.5,   # capped at 2.0 → 5.0
    "fill_probability_proxy":  3.0,   # 0-1 → 0-3.0

    # Session bonus (max ~8)
    "is_london":               3.0,
    "is_overlap":              5.0,
    "is_ny":                   2.0,

    # Vol regime bonus (normal=2, high=0, low=0)
    "vol_regime_normal":       2.0,
}

# Penalties (subtracted)
_HEURISTIC_PENALTIES: dict[str, float] = {
    "setup_is_fallback":       6.0,
    "news_blocked":            8.0,
    "v2_penalty_total":        1.0,   # pass-through V2 penalties
    "spread_too_high":         5.0,   # if spread_ratio > 0.2
    "low_rr":                  4.0,   # if rr < 1.5
}


def heuristic_score_v3(features: dict[str, float]) -> float:
    """Compute a 0-100 ScoreV3 using weighted heuristic rules.

    This is the default scorer before ML training data is collected.
    It's designed to be MORE PERMISSIVE than V2 while still penalizing
    genuinely poor setups.
    """
    score = 0.0

    # HTF alignment
    score += features.get("htf_bias_aligned", 0) * _HEURISTIC_WEIGHTS["htf_bias_aligned"]
    score += features.get("htf_bos_confirmed", 0) * _HEURISTIC_WEIGHTS["htf_bos_confirmed"]
    score += features.get("htf_pd_ok", 0) * _HEURISTIC_WEIGHTS["htf_pd_ok"]
    score += features.get("htf_location_score", 0.5) * _HEURISTIC_WEIGHTS["htf_location_score"]

    # FVG quality
    score += features.get("fvg_present", 0) * _HEURISTIC_WEIGHTS["fvg_present"]
    score += min(2.0, features.get("fvg_size_atr", 0)) * _HEURISTIC_WEIGHTS["fvg_size_atr"]
    score += min(2.0, features.get("fvg_displacement_ratio", 0)) * _HEURISTIC_WEIGHTS["fvg_displacement_ratio"]

    # Trigger quality
    score += min(3.0, features.get("trigger_confirmations", 0)) * _HEURISTIC_WEIGHTS["trigger_confirmations"]
    score += features.get("mss_confirmed", 0) * _HEURISTIC_WEIGHTS["mss_confirmed"]
    score += min(1.0, features.get("sweep_magnitude_atr", 0)) * _HEURISTIC_WEIGHTS["sweep_magnitude_atr"]

    # Execution quality
    score += features.get("spread_score_raw", 0) * _HEURISTIC_WEIGHTS["spread_score_raw"]
    score += min(2.0, features.get("rr_ratio", 0)) * _HEURISTIC_WEIGHTS["rr_ratio"]
    score += features.get("fill_probability_proxy", 0) * _HEURISTIC_WEIGHTS["fill_probability_proxy"]

    # Session bonuses
    score += features.get("is_london", 0) * _HEURISTIC_WEIGHTS["is_london"]
    score += features.get("is_overlap", 0) * _HEURISTIC_WEIGHTS["is_overlap"]
    score += features.get("is_ny", 0) * _HEURISTIC_WEIGHTS["is_ny"]

    # Vol regime bonus
    vol = features.get("vol_regime", 1.0)
    if vol == 1.0:  # normal
        score += _HEURISTIC_WEIGHTS["vol_regime_normal"]

    # Penalties
    score -= features.get("setup_is_fallback", 0) * _HEURISTIC_PENALTIES["setup_is_fallback"]
    score -= features.get("news_blocked", 0) * _HEURISTIC_PENALTIES["news_blocked"]
    score -= features.get("v2_penalty_total", 0) * _HEURISTIC_PENALTIES["v2_penalty_total"]

    if features.get("spread_ratio", 0) > 0.20:
        score -= _HEURISTIC_PENALTIES["spread_too_high"]

    if features.get("rr_ratio", 2.0) < 1.5:
        score -= _HEURISTIC_PENALTIES["low_rr"]

    return max(0.0, min(100.0, round(score, 2)))


# ═══════════════════════════════════════════════════════════════════════════
#  ML-based ScoreV3 (after training)
# ═══════════════════════════════════════════════════════════════════════════

class ScoreV3Model(Protocol):
    """Protocol for a trained ScoreV3 model."""
    def predict(self, features: dict[str, float]) -> tuple[float, float]:
        """Return (p_win, expected_R)."""
        ...


@dataclass
class HeuristicScoreV3Model:
    """Default model using the heuristic scorer."""

    def predict(self, features: dict[str, float]) -> tuple[float, float]:
        score = heuristic_score_v3(features)
        # Approximate p_win and expected_R from score
        p_win = min(0.95, max(0.05, score / 100.0))
        # expected_R = p_win * avg_win_R - (1 - p_win) * avg_loss_R
        # Use conservative estimates: avg_win_R = 1.2, avg_loss_R = 0.8
        expected_r = p_win * 1.2 - (1 - p_win) * 0.8
        return round(p_win, 4), round(expected_r, 4)


@dataclass
class TrainedScoreV3Model:
    """Wrapper for a trained sklearn/lightgbm model."""

    model: Any = None
    feature_names: list[str] = field(default_factory=list)
    calibrator: Any = None

    @classmethod
    def load(cls, path: Path) -> "TrainedScoreV3Model":
        data = pickle.loads(path.read_bytes())
        return cls(
            model=data["model"],
            feature_names=data.get("feature_names", FEATURE_NAMES),
            calibrator=data.get("calibrator"),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({
            "model": self.model,
            "feature_names": self.feature_names,
            "calibrator": self.calibrator,
        }))

    def predict(self, features: dict[str, float]) -> tuple[float, float]:
        import numpy as np

        x = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        if hasattr(self.model, "predict_proba"):
            raw_prob = self.model.predict_proba(x)[0, 1]
            if self.calibrator is not None:
                raw_prob = float(self.calibrator.predict_proba(
                    np.array([[raw_prob]])
                )[0, 1])
            p_win = float(raw_prob)
        else:
            p_win = min(0.95, max(0.05, float(self.model.predict(x)[0])))

        expected_r = p_win * 1.2 - (1 - p_win) * 0.8
        return round(p_win, 4), round(expected_r, 4)


# ═══════════════════════════════════════════════════════════════════════════
#  ScoreV3 engine (replaces _compute_v2_score in the backtest engine)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScoreV3Config:
    """Configuration for ScoreV3 scoring system."""

    enabled: bool = False
    mode: str = "heuristic"      # "heuristic" or "ml"
    model_path: str = ""         # Path to trained model (for "ml" mode)

    # Heuristic thresholds (more permissive than V2)
    trade_threshold: float = 55.0    # V2 default was 62; we lower it
    small_min: float = 45.0          # V2 default was 58
    small_max: float = 54.99

    # Tier config (quantile-based)
    tier_enabled: bool = True
    tier_a_plus_pct: float = 0.10    # top 10% → A+
    tier_a_pct: float = 0.25         # next 25% → A
    tier_b_pct: float = 0.30         # next 30% → B
    # remaining → OBSERVE

    # Shadow observe
    shadow_enabled: bool = True
    shadow_output_path: str = "reports/shadow_observe.jsonl"
    shadow_simulate_outcomes: bool = True

    # Fill probability adjustment
    fill_prob_weight: float = 0.3    # weight of fill probability in final score


class ScoreV3Engine:
    """Main scoring engine — extracts features, scores, assigns tier."""

    def __init__(self, config: ScoreV3Config | None = None) -> None:
        self.config = config or ScoreV3Config()
        self._model: ScoreV3Model = self._load_model()
        self._score_history: list[float] = []
        self._quantile_boundaries: dict[str, float] | None = None

    def _load_model(self) -> ScoreV3Model:
        if self.config.mode == "ml" and self.config.model_path:
            path = Path(self.config.model_path)
            if path.exists():
                return TrainedScoreV3Model.load(path)
        return HeuristicScoreV3Model()

    def score(
        self,
        evaluation: StrategyEvaluation,
        bias: BiasState,
        *,
        candle: Candle,
        atr_m5: float | None = None,
        atr_history: list[float | None] | None = None,
        spread: float | None = None,
        assumed_spread: float = 0.2,
        entry_price: float | None = None,
        stop_price: float | None = None,
        tp_price: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Score a candidate. Returns (score_v3, features_dict).

        The score is 0-100, already adjusted for fill probability.
        """
        features = extract_features(
            evaluation, bias,
            candle=candle,
            atr_m5=atr_m5,
            atr_history=atr_history,
            spread=spread,
            assumed_spread=assumed_spread,
            entry_price=entry_price,
            stop_price=stop_price,
            tp_price=tp_price,
        )

        p_win, expected_r = self._model.predict(features)

        # Compute raw score (0-100)
        if self.config.mode == "heuristic":
            raw_score = heuristic_score_v3(features)
        else:
            # ML mode: map (p_win, expected_r) to 0-100
            raw_score = min(100.0, max(0.0, expected_r * 50.0 + 50.0))

        # Adjust by fill probability
        fill_prob = features.get("fill_probability_proxy", 0.5)
        adjusted_score = raw_score * (1.0 - self.config.fill_prob_weight + self.config.fill_prob_weight * fill_prob)
        final_score = max(0.0, min(100.0, round(adjusted_score, 2)))

        # Track for quantile computation
        self._score_history.append(final_score)

        features["score_v3"] = final_score
        features["p_win"] = p_win
        features["expected_r"] = expected_r
        features["raw_score_v3"] = raw_score

        return final_score, features

    def resolve_action(
        self,
        score: float,
        *,
        has_blocking_reasons: bool = False,
    ) -> DecisionAction:
        """Map score to TRADE / SMALL / OBSERVE action."""
        if has_blocking_reasons:
            return DecisionAction.OBSERVE
        if score >= self.config.trade_threshold:
            return DecisionAction.TRADE
        if self.config.small_min <= score <= self.config.small_max:
            return DecisionAction.SMALL
        return DecisionAction.OBSERVE

    def resolve_tier(self, score: float) -> str:
        """Resolve tier using quantile boundaries if available, else fixed."""
        if not self.config.tier_enabled:
            return "NONE"

        # Use quantile boundaries if we have enough history
        if self._quantile_boundaries and len(self._score_history) > 100:
            if score >= self._quantile_boundaries.get("a_plus", 999):
                return "A_plus"
            if score >= self._quantile_boundaries.get("a", 999):
                return "A"
            if score >= self._quantile_boundaries.get("b", 999):
                return "B"
            return "OBSERVE"

        # Fallback to fixed thresholds
        if score >= self.config.trade_threshold + 10:
            return "A_plus"
        if score >= self.config.trade_threshold:
            return "A"
        if score >= self.config.small_min:
            return "B"
        return "OBSERVE"

    def update_quantile_boundaries(self, min_samples: int = 200) -> None:
        """Recompute quantile-based tier boundaries from score history."""
        if len(self._score_history) < min_samples:
            return
        sorted_scores = sorted(self._score_history)
        n = len(sorted_scores)
        # A+ = top tier_a_plus_pct
        a_plus_idx = max(0, int(n * (1.0 - self.config.tier_a_plus_pct)))
        # A = next tier_a_pct
        a_idx = max(0, int(n * (1.0 - self.config.tier_a_plus_pct - self.config.tier_a_pct)))
        # B = next tier_b_pct
        b_idx = max(0, int(n * (1.0 - self.config.tier_a_plus_pct - self.config.tier_a_pct - self.config.tier_b_pct)))

        self._quantile_boundaries = {
            "a_plus": sorted_scores[a_plus_idx],
            "a": sorted_scores[a_idx],
            "b": sorted_scores[b_idx],
        }

    @property
    def quantile_boundaries(self) -> dict[str, float] | None:
        return self._quantile_boundaries

    @property
    def score_history_size(self) -> int:
        return len(self._score_history)


# ═══════════════════════════════════════════════════════════════════════════
#  Integration helper — applies ScoreV3 to a StrategyEvaluation
# ═══════════════════════════════════════════════════════════════════════════

def apply_score_v3(
    engine: ScoreV3Engine,
    evaluation: StrategyEvaluation,
    bias: BiasState,
    *,
    candle: Candle,
    atr_m5: float | None = None,
    atr_history: list[float | None] | None = None,
    spread: float | None = None,
    assumed_spread: float = 0.2,
    entry_price: float | None = None,
    stop_price: float | None = None,
    tp_price: float | None = None,
) -> StrategyEvaluation:
    """Apply ScoreV3 scoring to an evaluation, updating score_total and action.

    This is a drop-in replacement for _compute_v2_score + _normalize_action_for_score.
    """
    score_v3, features = engine.score(
        evaluation, bias,
        candle=candle,
        atr_m5=atr_m5,
        atr_history=atr_history,
        spread=spread,
        assumed_spread=assumed_spread,
        entry_price=entry_price,
        stop_price=stop_price,
        tp_price=tp_price,
    )

    # Preserve V2 score in metadata
    evaluation.metadata["score_v2"] = evaluation.score_total
    evaluation.metadata["score_v3"] = score_v3
    evaluation.metadata["score_v3_features"] = features

    # Update evaluation with V3 score
    evaluation.score_total = score_v3

    # Update action based on V3 score
    new_action = engine.resolve_action(
        score_v3,
        has_blocking_reasons=bool(evaluation.reasons_blocking),
    )
    evaluation.action = new_action

    # Resolve tier
    tier = engine.resolve_tier(score_v3)
    evaluation.metadata["tier"] = tier
    evaluation.metadata["tier_boundaries"] = engine.quantile_boundaries

    return evaluation
