from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bot.data.candles import Candle
from bot.strategy.ict import detect_mss


def _candle(ts: datetime, o: float, h: float, l: float, c: float) -> Candle:
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c)


def test_detect_mss_long_after_sweep() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [
        _candle(start + timedelta(minutes=5 * 0), 99, 100, 95, 98),
        _candle(start + timedelta(minutes=5 * 1), 98, 99, 90, 91),   # sweep context
        _candle(start + timedelta(minutes=5 * 2), 91, 101, 92, 100),
        _candle(start + timedelta(minutes=5 * 3), 100, 104, 97, 100),  # local swing high
        _candle(start + timedelta(minutes=5 * 4), 100, 102, 96, 101),
        _candle(start + timedelta(minutes=5 * 5), 101, 105, 98, 104.5),  # break above 104
        _candle(start + timedelta(minutes=5 * 6), 104, 106, 103, 105),
    ]
    signal = detect_mss(
        candles,
        side="LONG",
        since_index=1,
        fractal_left=1,
        fractal_right=1,
    )

    assert signal is not None
    assert signal.candle_index == 5
    assert signal.broken_level == 104


def test_detect_mss_short_after_sweep() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [
        _candle(start + timedelta(minutes=5 * 0), 101, 103, 100, 102),
        _candle(start + timedelta(minutes=5 * 1), 102, 110, 101, 109),  # sweep context
        _candle(start + timedelta(minutes=5 * 2), 109, 108, 104, 105),
        _candle(start + timedelta(minutes=5 * 3), 105, 106, 101, 103),  # local swing low
        _candle(start + timedelta(minutes=5 * 4), 103, 105, 102, 104),
        _candle(start + timedelta(minutes=5 * 5), 104, 104.5, 99, 100.5),  # break below 101
        _candle(start + timedelta(minutes=5 * 6), 100.5, 101, 98, 99),
    ]
    signal = detect_mss(
        candles,
        side="SHORT",
        since_index=1,
        fractal_left=1,
        fractal_right=1,
    )

    assert signal is not None
    assert signal.candle_index == 5
    assert signal.broken_level == 101

