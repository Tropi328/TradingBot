"""
Regime detection for V3 scoring.

Provides two pure-function regime metrics:
  - trend_regime_score: measures trend strength (0 = ranging, 1 = trending)
  - vol_regime_change: measures volatility direction (-1 = compressing, +1 = expanding)

These fill the gap left by the existing vol_regime (static snapshot)
and the unimplemented trend_regime metadata field.
"""

from __future__ import annotations

from bot.data.candles import Candle


def compute_trend_regime_score(
    candles: list[Candle],
    atr: float,
    ema_period: int = 20,
) -> float:
    """Compute trend strength from price displacement vs short EMA.

    Logic:
      1. Compute EMA(close, ema_period) over the candles.
      2. Take the last close minus the last EMA value.
      3. Normalise by ATR → abs(displacement) / ATR.
      4. Clamp to [0, 1].  Values > 1 ATR from EMA = strong trend.

    Returns 0.0 when data is insufficient or ATR is zero.
    """
    if not candles or atr <= 0:
        return 0.0
    closes = [c.close for c in candles]
    if len(closes) < ema_period:
        return 0.0

    # Inline EMA to avoid import-cycle risk with indicators.py.
    # Simple enough for a short list — no caching needed here.
    seed = sum(closes[:ema_period]) / ema_period
    alpha = 2.0 / (ema_period + 1)
    prev = seed
    for i in range(ema_period, len(closes)):
        prev = (closes[i] - prev) * alpha + prev

    displacement = abs(closes[-1] - prev) / atr
    return min(1.0, displacement)


def compute_vol_regime_change(
    atr_history: list[float | None],
    short_window: int = 10,
    long_window: int = 50,
) -> float:
    """Compute volatility expansion/compression rate.

    Logic:
      1. From atr_history, take the last `long_window` valid values.
      2. Compute mean of last `short_window` values (recent vol).
      3. Compute mean of all `long_window` values (baseline vol).
      4. Ratio = (short_mean / long_mean) - 1.0
         Positive = expanding, negative = compressing.
      5. Clamp to [-1, 1].

    Returns 0.0 when data is insufficient.
    """
    if not atr_history:
        return 0.0
    valid = [float(v) for v in atr_history if v is not None and float(v) > 0]
    if len(valid) < long_window:
        return 0.0

    recent = valid[-long_window:]
    long_mean = sum(recent) / len(recent)
    if long_mean <= 0:
        return 0.0

    short_vals = recent[-short_window:]
    short_mean = sum(short_vals) / len(short_vals)

    ratio = (short_mean / long_mean) - 1.0
    return max(-1.0, min(1.0, ratio))
