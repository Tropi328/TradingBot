from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from bot.data.candles import Candle
from bot.strategy.swings import SwingPoint, detect_swings


@dataclass(slots=True)
class SweepSignal:
    side: str
    swept_level: float
    reference_swing_index: int
    sweep_candle_index: int
    sweep_time: datetime
    magnitude: float
    rejected: bool


@dataclass(slots=True)
class MSSSignal:
    side: str
    broken_level: float
    source_swing_index: int
    candle_index: int
    candle_time: datetime


@dataclass(slots=True)
class FVGSignal:
    side: str
    c1_index: int
    c2_index: int
    c3_index: int
    lower: float
    upper: float
    midpoint: float
    timestamp: datetime


def detect_sweep_reject(
    candles: list[Candle],
    side: str,
    *,
    lookback_min_hours: int,
    lookback_max_hours: int,
    atr_m15: float,
    threshold_atr_multiplier: float,
    minimal_tick_buffer: float,
    fractal_left: int,
    fractal_right: int,
) -> SweepSignal | None:
    if len(candles) < 20:
        return None
    bars_per_hour = 4
    min_bars = lookback_min_hours * bars_per_hour
    max_bars = lookback_max_hours * bars_per_hour
    end_index = len(candles) - 1
    threshold = max(threshold_atr_multiplier * atr_m15, minimal_tick_buffer)

    swing_highs, swing_lows = detect_swings(
        candles, fractal_left=fractal_left, fractal_right=fractal_right
    )
    swings: list[SwingPoint] = swing_lows if side == "LONG" else swing_highs

    candidates = []
    for swing in swings:
        age = end_index - swing.index
        if min_bars <= age <= max_bars:
            candidates.append(swing)
    if not candidates:
        return None

    reference = candidates[-1]
    detected: SweepSignal | None = None
    for index in range(reference.index + 1, len(candles)):
        candle = candles[index]
        if side == "LONG":
            swept = candle.low < reference.price
            rejected = candle.close > reference.price
            magnitude = reference.price - candle.low
        else:
            swept = candle.high > reference.price
            rejected = candle.close < reference.price
            magnitude = candle.high - reference.price
        if swept and rejected and magnitude >= threshold:
            detected = SweepSignal(
                side=side,
                swept_level=reference.price,
                reference_swing_index=reference.index,
                sweep_candle_index=index,
                sweep_time=candle.timestamp,
                magnitude=magnitude,
                rejected=True,
            )
    return detected


def detect_mss(
    candles: list[Candle],
    side: str,
    *,
    since_index: int,
    fractal_left: int,
    fractal_right: int,
) -> MSSSignal | None:
    if since_index >= len(candles) - 1:
        return None
    swing_highs, swing_lows = detect_swings(
        candles, fractal_left=fractal_left, fractal_right=fractal_right
    )
    for index in range(since_index + 1, len(candles)):
        candle = candles[index]
        if side == "LONG":
            eligible = [s for s in swing_highs if since_index < s.index < index]
            if not eligible:
                continue
            ref = eligible[-1]
            if candle.close > ref.price:
                return MSSSignal(
                    side=side,
                    broken_level=ref.price,
                    source_swing_index=ref.index,
                    candle_index=index,
                    candle_time=candle.timestamp,
                )
        else:
            eligible = [s for s in swing_lows if since_index < s.index < index]
            if not eligible:
                continue
            ref = eligible[-1]
            if candle.close < ref.price:
                return MSSSignal(
                    side=side,
                    broken_level=ref.price,
                    source_swing_index=ref.index,
                    candle_index=index,
                    candle_time=candle.timestamp,
                )
    return None


def detect_latest_fvg(candles: list[Candle], side: str, *, start_index: int = 0) -> FVGSignal | None:
    if len(candles) < 3:
        return None
    signal: FVGSignal | None = None
    for i in range(max(0, start_index), len(candles) - 2):
        c1 = candles[i]
        c3 = candles[i + 2]
        if side == "LONG" and c3.low > c1.high:
            lower = c1.high
            upper = c3.low
            signal = FVGSignal(
                side=side,
                c1_index=i,
                c2_index=i + 1,
                c3_index=i + 2,
                lower=lower,
                upper=upper,
                midpoint=(lower + upper) / 2.0,
                timestamp=c3.timestamp,
            )
        elif side == "SHORT" and c3.high < c1.low:
            lower = c3.high
            upper = c1.low
            signal = FVGSignal(
                side=side,
                c1_index=i,
                c2_index=i + 1,
                c3_index=i + 2,
                lower=lower,
                upper=upper,
                midpoint=(lower + upper) / 2.0,
                timestamp=c3.timestamp,
            )
    return signal


def is_a_plus_setup(
    *,
    pd_ideal: bool,
    displacement_body: float,
    atr_m5: float,
    spread_ok: bool,
    a_plus_multiplier: float,
) -> bool:
    if atr_m5 <= 0:
        return False
    displacement_a_plus = displacement_body > (a_plus_multiplier * atr_m5)
    score = sum([pd_ideal, displacement_a_plus, spread_ok])
    return score >= 2

