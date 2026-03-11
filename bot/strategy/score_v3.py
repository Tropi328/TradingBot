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

import hashlib
import hmac
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

_LOG = logging.getLogger(__name__)

# Key used for HMAC integrity verification of serialised model files.
# This is NOT a cryptographic secret — it guards against accidental
# corruption / tampering on the local filesystem only.
_MODEL_HMAC_KEY = b"score_v3_model_integrity"

from bot.data.candles import Candle
from bot.strategy.contracts import (
    BiasState,
    DecisionAction,
    StrategyEvaluation,
)
from bot.strategy.shadow_observer import classify_session, compute_atr_percentile
from bot.strategy.regime import compute_trend_regime_score, compute_vol_regime_change
from bot.strategy.utils import DEFAULT_MAX_SPREAD_RATIO

# ═══════════════════════════════════════════════════════════════════════════
#  Feature extraction
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_NAMES: list[str] = [
    # HTF alignment (0-3)
    "htf_bias_aligned",  # 1 if h1 bias == signal side
    "htf_bos_confirmed",  # 1 if h1 BOS state is not NONE
    "htf_pd_ok",  # 1 if premium/discount filter passes
    "htf_location_score",  # 0-1, how deep in discount/premium zone
    # FVG quality (4-8)
    "fvg_present",  # 1/0
    "fvg_size_atr",  # fvg_size / atr
    "fvg_age_bars",  # bars since FVG formed
    "fvg_distance_to_price",  # |fvg_mid - current_close| / atr
    "fvg_displacement_ratio",  # displacement body / atr
    # Trigger quality (9-13)
    "trigger_confirmations",  # 0-3 count
    "mss_confirmed",  # 1/0
    "displacement_confirmed",  # 1/0
    "sweep_magnitude_atr",  # sweep_magnitude / atr
    "setup_is_fallback",  # 1 if displacement_retest
    # Volatility regime (14-18)
    "atr_m5_raw",  # raw ATR value
    "atr_percentile",  # rank in recent history (0-1)
    "spread_ratio",  # spread / atr
    "spread_score_raw",  # 0-8 from V2 spread scoring
    "vol_regime",  # 0=low, 1=normal, 2=high
    # Session / time (19-24)
    "hour_utc",  # 0-23
    "day_of_week",  # 0=Mon, 6=Sun
    "is_london",  # binary
    "is_ny",  # binary
    "is_overlap",  # binary
    "news_blocked",  # binary
    # Entry quality (25-30)
    "rr_ratio",  # reward-to-risk ratio
    "risk_distance_atr",  # |entry - stop| / atr
    "entry_distance_atr",  # |entry - close| / atr
    "edge_to_cost_ratio",  # expected_gain / (spread + slippage)
    "tp_distance_atr",  # |entry - tp| / atr
    "fill_probability_proxy",  # heuristic: distance-to-entry → likely fill
    # V3-native quality composites (31-37)
    "v3_edge_quality",  # 0-1: HTF alignment + location strength
    "v3_trigger_quality",  # 0-1: trigger signal strength
    "v3_execution_quality",  # 0-1: spread/cost quality
    "v3_penalty_total",  # sum of V3-computed penalties
    "daily_context_score",  # 0-1: daily gate / trend context bonus
    "vol_regime_normal",  # 1.0 if normal vol, else 0.0
    "session_quality",  # 0-1: combined session desirability
    # Orderflow features (38-43) — from evaluation.metadata["orderflow_snapshot"]
    "of_delta_ratio",  # -1..1: net buying/selling pressure
    "of_aggression",  # 0..1: trade aggression intensity
    "of_microprice_bias",  # -1..1: microprice vs mid
    "of_absorption_score",  # 0..1: absorption (wick-to-range)
    "of_chop_score",  # 0..1: market choppiness
    "of_flow_aligned",  # 1 if orderflow direction == setup side
    # Candle pattern features (44-47) — trigger candle shape analysis
    "candle_body_to_range",  # body / (high - low), 0-1
    "candle_rejection_wick_ratio",  # setup-side wick / body, 0-1 clamped
    "candle_engulfing",  # 1 if last closed candle engulfs previous
    "candle_direction_run",  # consecutive candles in setup direction, 0-1 (capped 5)
    # Sweep quality features (48-50) — sweep signal quality analysis
    "sweep_quality",  # bell-curve quality of sweep magnitude (sweet spot 0.3-1.0 ATR), 0-1
    "sweep_to_entry_atr",  # proximity of sweep level to entry (closer=higher), 0-1
    "sweep_rejection_strength",  # how far price rejected from sweep level, 0-1
    # MTF confluence features (51-54) — multi-timeframe agreement
    "mtf_ema_confluence",  # 1 if close on correct side of H1 EMA200
    "mtf_bos_agrees_side",  # 1 if H1 BOS matches side, 0.5 if NONE, 0 if opposing
    "mtf_pd_zone_depth",  # depth in favorable H1 PD zone, 0-1
    "mtf_m5_momentum",  # M5 momentum aligned with setup direction, 0-1
    # Regime detection features (55-56) — market regime awareness
    "regime_trend_score",  # trend strength: displacement from EMA / ATR, 0-1
    "regime_vol_change",  # vol expansion/compression rate, -1..1 (centred 0)
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
    recent_candles: list[Candle] | None = None,
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

    # FVG quality — require non-zero size to count as present;
    # fvg_detected=True with fvg_size=0 (missing) should not grant the presence bonus.
    fvg_size = float(meta.get("fvg_size", 0))
    fvg_present = 1.0 if meta.get("fvg_detected") and fvg_size > 0 else 0.0
    fvg_size_atr = fvg_size / atr if atr > 0 else 0
    fvg_mid = meta.get("fvg_mid")
    fvg_dist = abs(float(fvg_mid) - close_price) / atr if fvg_mid is not None and atr > 0 else 0
    fvg_c1_idx = meta.get("fvg_c1_index", meta.get("fvg_c1_idx"))
    fvg_age = 0
    if fvg_c1_idx is not None:
        try:
            fvg_age = max(
                0,
                int(
                    len(evaluation.snapshot.get("candles_m5", [0] * 100))
                    if isinstance(evaluation.snapshot.get("candles_m5"), list)
                    else 0
                )
                - int(fvg_c1_idx),
            )
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
    max_spread_ratio = DEFAULT_MAX_SPREAD_RATIO
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

    # ── V3-native quality composites ──
    # Edge quality: combines HTF alignment signals into a 0-1 composite
    _edge_signals = [bias_aligned, bos_confirmed, pd_ok, location_score]
    v3_edge_quality = sum(_edge_signals) / max(len(_edge_signals), 1)

    # Trigger quality: combines trigger confirmations, MSS, displacement, sweep
    _trigger_signals = [
        min(1.0, trigger_conf / 3.0),  # normalise 0-3 → 0-1
        mss_ok,
        disp_ok,
        min(1.0, sweep_atr),
    ]
    v3_trigger_quality = sum(_trigger_signals) / max(len(_trigger_signals), 1)

    # Execution quality: spread score + cost efficiency + fill likelihood
    _exec_signals = [
        spread_score_raw / 8.0,  # normalise 0-8 → 0-1
        min(1.0, fill_prob),
        min(1.0, rr / 3.0),  # normalise, 3.0 RR → 1.0
    ]
    v3_execution_quality = sum(_exec_signals) / max(len(_exec_signals), 1)

    # V3-native penalty total: built from raw evaluation data
    _v3_penalties: list[float] = []
    if is_fallback > 0.5:
        _v3_penalties.append(6.0)
    news_flag = 1.0 if meta.get("news_blocked") or bool(snap.get("news_blocked")) else 0.0
    if news_flag > 0.5:
        _v3_penalties.append(8.0)
    if spread_ratio > 0.20:
        _v3_penalties.append(5.0)
    if rr < 1.5:
        _v3_penalties.append(4.0)
    if bool(meta.get("near_adr_exhausted")):
        _v3_penalties.append(4.0)
    if bool(meta.get("late_retest")):
        _v3_penalties.append(3.0)
    if bool(meta.get("correlation_exposure")):
        _v3_penalties.append(3.0)
    v3_penalty_total = sum(_v3_penalties)

    # Daily context score: bonus from daily gate / trend alignment
    _daily_raw = float(meta.get("daily_context", snap.get("daily_context", 0)))
    daily_context_score = min(1.0, _daily_raw / 10.0)  # normalise 0-10 → 0-1

    # Vol regime flag
    vol_regime_normal = 1.0 if vol_regime == 1.0 else 0.0

    # Session quality: weighted composite of session flags
    is_london = 1.0 if session == "LONDON" else 0.0
    is_ny = 1.0 if session == "NY" else 0.0
    is_overlap = 1.0 if session == "OVERLAP" else 0.0
    session_quality = is_overlap * 1.0 + is_london * 0.7 + is_ny * 0.5
    session_quality = min(1.0, session_quality)

    # ── Orderflow features (from evaluation.metadata["orderflow_snapshot"]) ──
    of_snap = meta.get("orderflow_snapshot")
    if isinstance(of_snap, dict):
        _of_metrics = of_snap.get("metrics") or {}
        of_delta_ratio = float(_of_metrics.get("delta_ratio", 0.0))
        of_aggression = float(_of_metrics.get("aggression", 0.0))
        of_microprice_bias = float(_of_metrics.get("microprice_bias", 0.0))
        of_absorption_score = float(_of_metrics.get("absorption_score", 0.0))
        of_chop_score = float(_of_metrics.get("chop_score", 0.5))
        _of_direction = str(of_snap.get("direction", "NEUTRAL")).upper()
        of_flow_aligned = 1.0 if (side and _of_direction == side) else 0.0
    else:
        # No orderflow data — neutral defaults
        of_delta_ratio = 0.0
        of_aggression = 0.0
        of_microprice_bias = 0.0
        of_absorption_score = 0.0
        of_chop_score = 0.5
        of_flow_aligned = 0.0

    # ── Candle pattern features (from trigger candle + recent history) ──
    candle_range = candle.high - candle.low
    candle_body = abs(candle.close - candle.open)
    candle_body_to_range = candle_body / candle_range if candle_range > 0 else 0.5

    # Rejection wick ratio: for LONG setups the lower wick (rejection of lows)
    # is the signal; for SHORT the upper wick.  Normalised as wick/body, 0-1.
    if candle_range > 0:
        if side == "LONG":
            _rejection_wick = min(candle.open, candle.close) - candle.low
        else:
            _rejection_wick = candle.high - max(candle.open, candle.close)
        candle_rejection_wick_ratio = (
            min(1.0, _rejection_wick / candle_body)
            if candle_body > 0
            else (1.0 if _rejection_wick > candle_range * 0.3 else 0.0)
        )
    else:
        candle_rejection_wick_ratio = 0.0

    # Engulfing: last closed candle completely engulfs the previous one
    candle_engulfing = 0.0
    candle_direction_run = 0.0
    _rc = recent_candles
    if _rc and len(_rc) >= 2:
        prev = _rc[-2]
        last = _rc[-1]
        prev_body_hi = max(prev.open, prev.close)
        prev_body_lo = min(prev.open, prev.close)
        last_body_hi = max(last.open, last.close)
        last_body_lo = min(last.open, last.close)
        if (
            last_body_hi >= prev_body_hi
            and last_body_lo <= prev_body_lo
            and abs(last.close - last.open) > abs(prev.close - prev.open)
        ):
            candle_engulfing = 1.0

        # Direction run: count consecutive candles moving in the setup direction
        run = 0
        for _c in reversed(_rc):
            if side == "LONG" and _c.close > _c.open:
                run += 1
            elif side == "SHORT" and _c.close < _c.open:
                run += 1
            else:
                break
        candle_direction_run = min(1.0, run / 5.0)  # normalise 0-5 → 0-1

    # ── Sweep quality features (magnitude, proximity, rejection) ──
    # sweep_quality: bell-curve score — sweet spot 0.3-1.0 ATR
    if sweep_atr <= 0:
        sweep_quality = 0.0
    elif sweep_atr < 0.1:
        sweep_quality = sweep_atr / 0.1 * 0.3  # ramp up to 0.3 (noise zone)
    elif sweep_atr < 0.3:
        sweep_quality = 0.3 + (sweep_atr - 0.1) / 0.2 * 0.7  # ramp 0.3 → 1.0
    elif sweep_atr <= 1.0:
        sweep_quality = 1.0  # sweet spot
    elif sweep_atr <= 2.0:
        sweep_quality = 1.0 - (sweep_atr - 1.0) / 1.0 * 0.6  # decay to 0.4
    else:
        sweep_quality = max(0.0, 0.4 - (sweep_atr - 2.0) * 0.2)  # taper off

    # sweep_to_entry_atr: proximity of sweep level to entry price
    sweep_level = meta.get("sweep_level")
    if sweep_level is not None and atr > 0:
        try:
            _sweep_dist = abs(float(sweep_level) - e) / atr
            sweep_to_entry_atr = max(0.0, 1.0 - _sweep_dist)  # closer=higher
        except (TypeError, ValueError):
            sweep_to_entry_atr = 0.0
    else:
        sweep_to_entry_atr = 0.0

    # sweep_rejection_strength: how far price moved away from the sweep level
    if sweep_level is not None and atr > 0:
        try:
            _sl = float(sweep_level)
            if side == "LONG":
                _rej = close_price - _sl  # positive = price above sweep
            else:
                _rej = _sl - close_price  # positive = price below sweep
            sweep_rejection_strength = min(1.0, max(0.0, _rej / atr))
        except (TypeError, ValueError):
            sweep_rejection_strength = 0.0
    else:
        sweep_rejection_strength = 0.0

    # ── MTF confluence features (H1 structure + M5 momentum agreement) ──
    # mtf_ema_confluence: close on correct side of H1 EMA200
    _h1_ema200 = meta.get("h1_ema200") or snap.get("h1_ema200")
    if _h1_ema200 is not None:
        try:
            _ema = float(_h1_ema200)
            if side == "LONG":
                mtf_ema_confluence = 1.0 if close_price > _ema else 0.0
            else:
                mtf_ema_confluence = 1.0 if close_price < _ema else 0.0
        except (TypeError, ValueError):
            mtf_ema_confluence = 0.5
    else:
        mtf_ema_confluence = 0.5  # neutral when unavailable

    # mtf_bos_agrees_side: H1 BOS direction matches setup side
    _bos_state = str(meta.get("h1_bos_state", snap.get("h1_bos_state", "NONE"))).upper()
    if _bos_state == "NONE":
        mtf_bos_agrees_side = 0.5  # neutral
    elif (_bos_state == "BULLISH" and side == "LONG") or (_bos_state == "BEARISH" and side == "SHORT"):
        mtf_bos_agrees_side = 1.0  # agreement
    else:
        mtf_bos_agrees_side = 0.0  # opposing

    # mtf_pd_zone_depth: depth in favorable H1 PD zone
    _pd_eq = meta.get("h1_pd_eq") or snap.get("h1_pd_eq")
    _pd_low = meta.get("h1_pd_low") or snap.get("h1_pd_low")
    _pd_high = meta.get("h1_pd_high") or snap.get("h1_pd_high")
    mtf_pd_zone_depth = 0.5  # neutral default
    if _pd_eq is not None and _pd_low is not None and _pd_high is not None:
        try:
            pd_eq_f = float(_pd_eq)
            pd_lo_f = float(_pd_low)
            pd_hi_f = float(_pd_high)
            if side == "LONG" and pd_eq_f > pd_lo_f:
                # Discount zone depth: how far below EQ (deeper = better for LONG)
                mtf_pd_zone_depth = min(1.0, max(0.0, (pd_eq_f - close_price) / (pd_eq_f - pd_lo_f)))
            elif side == "SHORT" and pd_hi_f > pd_eq_f:
                # Premium zone depth: how far above EQ (deeper = better for SHORT)
                mtf_pd_zone_depth = min(1.0, max(0.0, (close_price - pd_eq_f) / (pd_hi_f - pd_eq_f)))
        except (TypeError, ValueError):
            mtf_pd_zone_depth = 0.5

    # mtf_m5_momentum: recent M5 price momentum aligned with setup direction
    mtf_m5_momentum = 0.0
    if _rc and len(_rc) >= 3 and atr > 0:
        _first_close = _rc[0].close
        _last_close = _rc[-1].close
        _raw_momentum = (_last_close - _first_close) / atr
        if side == "LONG":
            mtf_m5_momentum = min(1.0, max(0.0, _raw_momentum))
        else:
            mtf_m5_momentum = min(1.0, max(0.0, -_raw_momentum))

    # ── Regime detection features ──
    regime_trend_score = compute_trend_regime_score(_rc if _rc else [], atr)
    regime_vol_change = compute_vol_regime_change(atr_history if atr_history else [])

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
        "is_london": is_london,
        "is_ny": is_ny,
        "is_overlap": is_overlap,
        "news_blocked": news_flag,
        "rr_ratio": round(rr, 4),
        "risk_distance_atr": round(risk_dist_atr, 4),
        "entry_distance_atr": round(entry_dist_atr, 4),
        "edge_to_cost_ratio": round(min(edge_to_cost, 100.0), 4),
        "tp_distance_atr": round(tp_dist_atr, 4),
        "fill_probability_proxy": round(fill_prob, 4),
        "v3_edge_quality": round(v3_edge_quality, 4),
        "v3_trigger_quality": round(v3_trigger_quality, 4),
        "v3_execution_quality": round(v3_execution_quality, 4),
        "v3_penalty_total": round(v3_penalty_total, 2),
        "daily_context_score": round(daily_context_score, 4),
        "vol_regime_normal": vol_regime_normal,
        "session_quality": round(session_quality, 4),
        "of_delta_ratio": round(of_delta_ratio, 4),
        "of_aggression": round(of_aggression, 4),
        "of_microprice_bias": round(of_microprice_bias, 4),
        "of_absorption_score": round(of_absorption_score, 4),
        "of_chop_score": round(of_chop_score, 4),
        "of_flow_aligned": of_flow_aligned,
        # Candle pattern features (44-47)
        "candle_body_to_range": round(candle_body_to_range, 4),
        "candle_rejection_wick_ratio": round(candle_rejection_wick_ratio, 4),
        "candle_engulfing": candle_engulfing,
        "candle_direction_run": round(candle_direction_run, 4),
        # Sweep quality features (48-50)
        "sweep_quality": round(sweep_quality, 4),
        "sweep_to_entry_atr": round(sweep_to_entry_atr, 4),
        "sweep_rejection_strength": round(sweep_rejection_strength, 4),
        # MTF confluence features (51-54)
        "mtf_ema_confluence": mtf_ema_confluence,
        "mtf_bos_agrees_side": mtf_bos_agrees_side,
        "mtf_pd_zone_depth": round(mtf_pd_zone_depth, 4),
        "mtf_m5_momentum": round(mtf_m5_momentum, 4),
        # Regime detection features (55-56)
        "regime_trend_score": round(regime_trend_score, 4),
        "regime_vol_change": round(regime_vol_change, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Heuristic ScoreV3 (default, no training needed)
# ═══════════════════════════════════════════════════════════════════════════

# Weights for heuristic scoring — tuned for ICT scalp setups.
# All features are V3-native; no V2 dependency.
_HEURISTIC_WEIGHTS: dict[str, float] = {
    # HTF alignment (max ~25)
    "htf_bias_aligned": 10.0,
    "htf_bos_confirmed": 5.0,
    "htf_pd_ok": 4.0,
    "htf_location_score": 6.0,
    # FVG quality (max ~22)
    "fvg_present": 8.0,
    "fvg_size_atr": 4.0,  # capped at 2.0 → 8.0
    "fvg_displacement_ratio": 3.0,  # capped at 2.0 → 6.0
    # Trigger quality (max ~25)
    "trigger_confirmations": 6.0,  # per confirmation (max 3 → 18)
    "mss_confirmed": 4.0,
    "sweep_magnitude_atr": 3.0,  # capped at 1.0 → 3.0
    # Execution quality (max ~12.6 — fill_probability_proxy REMOVED to fix double-penalty)
    "spread_score_raw": 1.2,  # 0-8 → 0-9.6
    "rr_ratio": 2.5,  # capped at 2.0 → 5.0
    # Session bonus (max ~5, via session_quality 0-1)
    "session_quality": 5.0,
    # Vol regime bonus (max 2)
    "vol_regime_normal": 2.0,
    # V3-native composites (bonus, max ~6)
    "daily_context_score": 3.0,  # 0-1 → 0-3
    "v3_edge_quality": 3.0,  # 0-1 → 0-3 (reward well-rounded edge)
    # Orderflow (max ~8: flow_aligned 4 + absorption 2 + delta_ratio 2)
    "of_flow_aligned": 4.0,  # binary: orderflow agrees with setup side
    "of_absorption_score": 2.0,  # 0-1 → 0-2 (strong absorption = good)
    "of_delta_ratio": 2.0,  # abs value capped at 1.0 → 0-2 (strong delta)
    # Candle patterns (max ~7: body 2 + rejection 3 + engulfing 2)
    "candle_body_to_range": 2.0,  # 0-1 → 0-2 (strong body = decisive candle)
    "candle_rejection_wick_ratio": 3.0,  # 0-1 → 0-3 (rejection wick = key signal)
    "candle_engulfing": 2.0,  # binary: engulfing pattern present
    # Sweep quality (max ~8: quality 4 + proximity 2 + rejection 2)
    "sweep_quality": 4.0,  # 0-1 → 0-4 (sweet-spot magnitude)
    "sweep_to_entry_atr": 2.0,  # 0-1 → 0-2 (sweep close to entry = clean)
    "sweep_rejection_strength": 2.0,  # 0-1 → 0-2 (decisive rejection)
    # MTF confluence (max ~10: ema 3 + bos 3 + pd_depth 2 + momentum 2)
    "mtf_ema_confluence": 3.0,  # 0/0.5/1 → 0-3 (trend alignment)
    "mtf_bos_agrees_side": 3.0,  # 0/0.5/1 → 0-3 (structure agreement)
    "mtf_pd_zone_depth": 2.0,  # 0-1 → 0-2 (deeper in favorable zone)
    "mtf_m5_momentum": 2.0,  # 0-1 → 0-2 (momentum alignment)
    # Regime detection (max ~5: trend 3 + vol_expanding 2)
    "regime_trend_score": 3.0,  # 0-1 → 0-3 (stronger trend = better for ICT)
    "regime_vol_expanding": 2.0,  # positive vol_change → 0-2 (expanding = opportunity)
}

# Penalties (subtracted) — all V3-native, no V2 passthrough
_HEURISTIC_PENALTIES: dict[str, float] = {
    "setup_is_fallback": 6.0,
    "news_blocked": 8.0,
    "spread_too_high": 5.0,  # if spread_ratio > 0.2
    "low_rr": 4.0,  # if rr < 1.5
    "high_vol": 3.0,  # if vol_regime == 2.0 (high)
    "of_high_chop": 4.0,  # if of_chop_score > 0.75
    "candle_weak_body": 2.0,  # if candle_body_to_range < 0.2 (doji/indecision)
    "sweep_absent": 3.0,  # if sweep_quality == 0 (no sweep detected)
    "mtf_bos_opposing": 4.0,  # if H1 BOS opposes setup direction
    "regime_vol_compressing": 2.0,  # if vol is strongly compressing (< -0.3)
}


def heuristic_score_v3(features: dict[str, float]) -> float:
    """Compute a 0-100 ScoreV3 using weighted heuristic rules.

    This is the default scorer before ML training data is collected.
    It's designed to be MORE PERMISSIVE than V2 while still penalizing
    genuinely poor setups.  Fully standalone — no V2 dependency.
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

    # Execution quality (fill_probability_proxy removed — it's used only in
    # the multiplicative adjustment in ScoreV3Engine.score(), not here)
    score += features.get("spread_score_raw", 0) * _HEURISTIC_WEIGHTS["spread_score_raw"]
    score += min(2.0, features.get("rr_ratio", 0)) * _HEURISTIC_WEIGHTS["rr_ratio"]

    # Session bonus (single composite feature)
    score += features.get("session_quality", 0) * _HEURISTIC_WEIGHTS["session_quality"]

    # Vol regime bonus
    score += features.get("vol_regime_normal", 0) * _HEURISTIC_WEIGHTS["vol_regime_normal"]

    # V3-native composite bonuses
    score += features.get("daily_context_score", 0) * _HEURISTIC_WEIGHTS["daily_context_score"]
    score += features.get("v3_edge_quality", 0) * _HEURISTIC_WEIGHTS["v3_edge_quality"]

    # Orderflow bonuses
    score += features.get("of_flow_aligned", 0) * _HEURISTIC_WEIGHTS["of_flow_aligned"]
    score += min(1.0, features.get("of_absorption_score", 0)) * _HEURISTIC_WEIGHTS["of_absorption_score"]
    score += min(1.0, abs(features.get("of_delta_ratio", 0))) * _HEURISTIC_WEIGHTS["of_delta_ratio"]

    # Candle pattern bonuses
    score += features.get("candle_body_to_range", 0.5) * _HEURISTIC_WEIGHTS["candle_body_to_range"]
    score += features.get("candle_rejection_wick_ratio", 0) * _HEURISTIC_WEIGHTS["candle_rejection_wick_ratio"]
    score += features.get("candle_engulfing", 0) * _HEURISTIC_WEIGHTS["candle_engulfing"]

    # Sweep quality bonuses
    score += features.get("sweep_quality", 0) * _HEURISTIC_WEIGHTS["sweep_quality"]
    score += features.get("sweep_to_entry_atr", 0) * _HEURISTIC_WEIGHTS["sweep_to_entry_atr"]
    score += features.get("sweep_rejection_strength", 0) * _HEURISTIC_WEIGHTS["sweep_rejection_strength"]

    # MTF confluence bonuses
    score += features.get("mtf_ema_confluence", 0.5) * _HEURISTIC_WEIGHTS["mtf_ema_confluence"]
    score += features.get("mtf_bos_agrees_side", 0.5) * _HEURISTIC_WEIGHTS["mtf_bos_agrees_side"]
    score += features.get("mtf_pd_zone_depth", 0) * _HEURISTIC_WEIGHTS["mtf_pd_zone_depth"]
    score += features.get("mtf_m5_momentum", 0) * _HEURISTIC_WEIGHTS["mtf_m5_momentum"]

    # ── Penalties ──
    score -= features.get("setup_is_fallback", 0) * _HEURISTIC_PENALTIES["setup_is_fallback"]
    score -= features.get("news_blocked", 0) * _HEURISTIC_PENALTIES["news_blocked"]

    # Conditional penalties with SAFE defaults:
    # spread_ratio defaults to the actual computed value (already in features);
    # rr_ratio defaults to the actual computed value.
    # Only penalize if the feature was genuinely extracted as bad.
    if features.get("spread_ratio", 0.5) > 0.20:
        score -= _HEURISTIC_PENALTIES["spread_too_high"]

    if features.get("rr_ratio", 0.0) < 1.5:
        score -= _HEURISTIC_PENALTIES["low_rr"]

    # High volatility penalty (replaces V2 pass-through)
    if features.get("vol_regime", 1.0) == 2.0:
        score -= _HEURISTIC_PENALTIES["high_vol"]

    # High orderflow chop penalty (replaces V2 soft gate)
    if features.get("of_chop_score", 0.5) > 0.75:
        score -= _HEURISTIC_PENALTIES["of_high_chop"]

    # Weak candle body penalty (doji/indecision on trigger candle)
    if features.get("candle_body_to_range", 0.5) < 0.2:
        score -= _HEURISTIC_PENALTIES["candle_weak_body"]

    # No sweep detected penalty
    if features.get("sweep_quality", 0) == 0:
        score -= _HEURISTIC_PENALTIES["sweep_absent"]

    # MTF BOS opposing setup direction penalty
    if features.get("mtf_bos_agrees_side", 0.5) == 0.0:
        score -= _HEURISTIC_PENALTIES["mtf_bos_opposing"]

    # Regime detection bonuses
    score += features.get("regime_trend_score", 0) * _HEURISTIC_WEIGHTS["regime_trend_score"]
    _vol_change = features.get("regime_vol_change", 0.0)
    if _vol_change > 0:
        score += min(1.0, _vol_change) * _HEURISTIC_WEIGHTS["regime_vol_expanding"]

    # Regime volatility compressing penalty
    if _vol_change < -0.3:
        score -= _HEURISTIC_PENALTIES["regime_vol_compressing"]

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
    def load(cls, path: Path) -> TrainedScoreV3Model:
        raw = path.read_bytes()
        # Verify HMAC integrity if a .sig sidecar file exists.
        sig_path = path.with_suffix(path.suffix + ".sig")
        if sig_path.exists():
            expected_mac = sig_path.read_bytes()
            actual_mac = hmac.new(_MODEL_HMAC_KEY, raw, hashlib.sha256).digest()
            if not hmac.compare_digest(expected_mac, actual_mac):
                raise ValueError(
                    f"Model file integrity check failed for {path}. The file may have been corrupted or tampered with."
                )
        else:
            _LOG.warning(
                "No integrity signature found for model %s — loading without verification.",
                path,
            )
        data = pickle.loads(raw)
        if not isinstance(data, dict) or "model" not in data:
            raise ValueError(f"Invalid model file structure in {path}")
        return cls(
            model=data["model"],
            feature_names=data.get("feature_names", FEATURE_NAMES),
            calibrator=data.get("calibrator"),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = pickle.dumps(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "calibrator": self.calibrator,
            }
        )
        path.write_bytes(raw)
        # Write HMAC signature sidecar for integrity verification on load.
        sig_path = path.with_suffix(path.suffix + ".sig")
        sig_path.write_bytes(hmac.new(_MODEL_HMAC_KEY, raw, hashlib.sha256).digest())

    def predict(self, features: dict[str, float]) -> tuple[float, float]:
        import numpy as np

        x = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        if hasattr(self.model, "predict_proba"):
            raw_prob = self.model.predict_proba(x)[0, 1]
            if self.calibrator is not None:
                raw_prob = float(self.calibrator.predict_proba(np.array([[raw_prob]]))[0, 1])
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
    mode: str = "heuristic"  # "heuristic" or "ml"
    model_path: str = ""  # Path to trained model (for "ml" mode)

    # Heuristic thresholds (more permissive than V2)
    trade_threshold: float = 55.0  # V2 default was 62; we lower it
    small_min: float = 45.0  # V2 default was 58
    small_max: float = 54.99

    # Tier config (quantile-based)
    tier_enabled: bool = True
    tier_a_plus_pct: float = 0.10  # top 10% → A+
    tier_a_pct: float = 0.25  # next 25% → A
    tier_b_pct: float = 0.30  # next 30% → B
    # remaining → OBSERVE

    # Shadow observe
    shadow_enabled: bool = True
    shadow_output_path: str = "reports/shadow_observe.jsonl"
    shadow_simulate_outcomes: bool = True

    # Fill probability adjustment
    fill_prob_weight: float = 0.3  # weight of fill probability in final score


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
            evaluation,
            bias,
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
        evaluation,
        bias,
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
