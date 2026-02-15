from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

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


@dataclass(slots=True)
class _AtrCacheEntry:
    candles_obj: list[Candle]
    tr_values: list[float]
    atr_values: list[float | None]
    prev_close: float
    ema_prev: float | None


_ATR_CACHE: OrderedDict[tuple[int, int], _AtrCacheEntry] = OrderedDict()
_ATR_CACHE_MAX_ENTRIES = 32


def _atr_cache_store(key: tuple[int, int], entry: _AtrCacheEntry) -> None:
    _ATR_CACHE[key] = entry
    _ATR_CACHE.move_to_end(key)
    while len(_ATR_CACHE) > _ATR_CACHE_MAX_ENTRIES:
        _ATR_CACHE.popitem(last=False)


def atr(candles: list[Candle], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be > 0")
    if not candles:
        return []

    key = (id(candles), int(period))
    cached = _ATR_CACHE.get(key)
    if cached is not None and cached.candles_obj is candles:
        cached_len = len(cached.tr_values)
        if len(candles) == cached_len:
            _ATR_CACHE.move_to_end(key)
            return cached.atr_values
        if len(candles) > cached_len:
            alpha = 2 / (period + 1)
            prev_close = cached.prev_close
            tr_values = cached.tr_values
            atr_values = cached.atr_values
            ema_prev = cached.ema_prev
            for candle in candles[cached_len:]:
                tr = max(
                    candle.high - candle.low,
                    abs(candle.high - prev_close),
                    abs(candle.low - prev_close),
                )
                tr_values.append(tr)
                idx = len(tr_values) - 1
                if idx < (period - 1):
                    atr_values.append(None)
                elif idx == (period - 1):
                    seed = sum(tr_values[:period]) / period
                    ema_prev = seed
                    atr_values.append(seed)
                else:
                    if ema_prev is None:
                        ema_prev = sum(tr_values[:period]) / period
                    ema_prev = (tr - ema_prev) * alpha + ema_prev
                    atr_values.append(ema_prev)
                prev_close = candle.close
            cached.prev_close = prev_close
            cached.ema_prev = ema_prev
            _ATR_CACHE.move_to_end(key)
            return atr_values

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
    atr_values = ema(tr_values, period)
    ema_prev: float | None = None
    for value in reversed(atr_values):
        if value is not None:
            ema_prev = value
            break
    _atr_cache_store(
        key,
        _AtrCacheEntry(
            candles_obj=candles,
            tr_values=tr_values,
            atr_values=atr_values,
            prev_close=prev_close,
            ema_prev=ema_prev,
        ),
    )
    return atr_values


def real_body(candle: Candle) -> float:
    return abs(candle.close - candle.open)


def latest_value(values: list[float | None]) -> float | None:
    for value in reversed(values):
        if value is not None:
            return value
    return None
