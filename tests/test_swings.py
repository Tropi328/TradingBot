from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bot.data.candles import Candle
from bot.strategy.swings import detect_swings


def _candle(ts: datetime, o: float, h: float, l: float, c: float) -> Candle:
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c)


def test_detect_swings_fractal_2_2() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [
        _candle(start + timedelta(minutes=5 * i), 1.0, h, l, 1.0)
        for i, (h, l) in enumerate(
            [
                (1.0, 0.9),
                (2.0, 1.5),
                (5.0, 2.6),
                (2.2, 1.2),
                (1.4, 0.4),
                (3.0, 1.4),
                (2.0, 1.2),
            ]
        )
    ]

    highs, lows = detect_swings(candles, fractal_left=2, fractal_right=2)

    assert len(highs) == 1
    assert len(lows) == 1
    assert highs[0].index == 2
    assert highs[0].price == 5.0
    assert lows[0].index == 4
    assert lows[0].price == 0.4

