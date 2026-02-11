from __future__ import annotations

from bot.data.candles import Candle


def ema(values: list[float], period: int) -> list[float | None]:
    if not values:
        return []
    if period <= 0:
        raise ValueError("period must be > 0")
    output: list[float | None] = [None] * len(values)
    if len(values) < period:
        return output
    seed = sum(values[:period]) / period
    output[period - 1] = seed
    alpha = 2 / (period + 1)
    prev = seed
    for i in range(period, len(values)):
        prev = (values[i] - prev) * alpha + prev
        output[i] = prev
    return output


def atr(candles: list[Candle], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be > 0")
    if not candles:
        return []

    tr_values: list[float] = []
    prev_close = candles[0].close
    for candle in candles:
        tr = max(
            candle.high - candle.low,
            abs(candle.high - prev_close),
            abs(candle.low - prev_close),
        )
        tr_values.append(tr)
        prev_close = candle.close
    return ema(tr_values, period)


def real_body(candle: Candle) -> float:
    return abs(candle.close - candle.open)


def latest_value(values: list[float | None]) -> float | None:
    for value in reversed(values):
        if value is not None:
            return value
    return None

