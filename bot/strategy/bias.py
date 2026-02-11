from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bot.data.candles import Candle
from bot.strategy.indicators import ema, latest_value
from bot.strategy.swings import detect_swings

Bias = Literal["UP", "DOWN", "NEUTRAL"]
PDState = Literal["DISCOUNT", "PREMIUM", "EQ", "UNKNOWN"]


@dataclass(slots=True)
class BiasDecision:
    bias: Bias
    last_close: float
    ema_value: float | None
    last_swing_high: float | None
    last_swing_low: float | None


def determine_h1_bias(
    candles: list[Candle],
    ema_period: int,
    fractal_left: int,
    fractal_right: int,
) -> BiasDecision:
    if not candles:
        return BiasDecision("NEUTRAL", 0.0, None, None, None)
    closes = [c.close for c in candles]
    ema_values = ema(closes, ema_period)
    ema_now = latest_value(ema_values)
    highs, lows = detect_swings(candles, fractal_left=fractal_left, fractal_right=fractal_right)
    last_high = highs[-1].price if highs else None
    last_low = lows[-1].price if lows else None
    close_now = closes[-1]

    if ema_now is None or last_high is None or last_low is None:
        return BiasDecision("NEUTRAL", close_now, ema_now, last_high, last_low)

    if close_now > ema_now and close_now > last_high:
        return BiasDecision("UP", close_now, ema_now, last_high, last_low)
    if close_now < ema_now and close_now < last_low:
        return BiasDecision("DOWN", close_now, ema_now, last_high, last_low)
    return BiasDecision("NEUTRAL", close_now, ema_now, last_high, last_low)


def dealing_range(last_swing_high: float | None, last_swing_low: float | None) -> tuple[float | None, float | None, float | None]:
    if last_swing_high is None or last_swing_low is None:
        return None, None, None
    high = max(last_swing_high, last_swing_low)
    low = min(last_swing_high, last_swing_low)
    eq = (high + low) / 2.0
    return low, high, eq


def premium_discount_state(price: float, low: float | None, high: float | None) -> PDState:
    if low is None or high is None:
        return "UNKNOWN"
    eq = (low + high) / 2.0
    if price < eq:
        return "DISCOUNT"
    if price > eq:
        return "PREMIUM"
    return "EQ"


def pd_allows_trade(bias: Bias, pd_state: PDState) -> bool:
    if bias == "UP":
        return pd_state == "DISCOUNT"
    if bias == "DOWN":
        return pd_state == "PREMIUM"
    return False

