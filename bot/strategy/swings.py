from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from bot.data.candles import Candle


@dataclass(slots=True)
class SwingPoint:
    index: int
    timestamp: datetime
    price: float
    kind: str


def detect_swings(
    candles: list[Candle],
    fractal_left: int = 2,
    fractal_right: int = 2,
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    highs: list[SwingPoint] = []
    lows: list[SwingPoint] = []
    if len(candles) < (fractal_left + fractal_right + 1):
        return highs, lows

    for index in range(fractal_left, len(candles) - fractal_right):
        center = candles[index]
        left = candles[index - fractal_left : index]
        right = candles[index + 1 : index + 1 + fractal_right]

        if all(center.high > c.high for c in left + right):
            highs.append(
                SwingPoint(
                    index=index,
                    timestamp=center.timestamp,
                    price=center.high,
                    kind="HIGH",
                )
            )
        if all(center.low < c.low for c in left + right):
            lows.append(
                SwingPoint(
                    index=index,
                    timestamp=center.timestamp,
                    price=center.low,
                    kind="LOW",
                )
            )
    return highs, lows


def last_confirmed_swing_high(candles: list[Candle], fractal_left: int, fractal_right: int) -> SwingPoint | None:
    highs, _ = detect_swings(candles, fractal_left=fractal_left, fractal_right=fractal_right)
    return highs[-1] if highs else None


def last_confirmed_swing_low(candles: list[Candle], fractal_left: int, fractal_right: int) -> SwingPoint | None:
    _, lows = detect_swings(candles, fractal_left=fractal_left, fractal_right=fractal_right)
    return lows[-1] if lows else None


def index_at_or_after(candles: list[Candle], timestamp: datetime) -> int | None:
    for i, candle in enumerate(candles):
        if candle.timestamp >= timestamp:
            return i
    return None

