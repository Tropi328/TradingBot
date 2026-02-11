from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bot.data.candles import Candle
from bot.strategy.ict import detect_latest_fvg


def _candle(ts: datetime, o: float, h: float, l: float, c: float) -> Candle:
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c)


def test_bullish_fvg_detection() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [
        _candle(start, 100, 101, 99, 100.5),
        _candle(start + timedelta(minutes=5), 101, 102, 100.2, 101.5),
        _candle(start + timedelta(minutes=10), 102, 103, 101.4, 102.7),
    ]
    signal = detect_latest_fvg(candles, side="LONG")

    assert signal is not None
    assert signal.lower == 101
    assert signal.upper == 101.4
    assert signal.midpoint == 101.2


def test_bearish_fvg_detection() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [
        _candle(start, 100, 102, 99, 101),
        _candle(start + timedelta(minutes=5), 99, 100, 97, 98),
        _candle(start + timedelta(minutes=10), 96, 98, 95, 96.5),
    ]
    signal = detect_latest_fvg(candles, side="SHORT")

    assert signal is not None
    assert signal.lower == 98
    assert signal.upper == 99
    assert signal.midpoint == 98.5

