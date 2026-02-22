from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.config import AppConfig  # noqa: E402
from bot.data.candles import Candle  # noqa: E402


# ── Shared fixtures ──────────────────────────────────────────────


@pytest.fixture()
def app_config() -> AppConfig:
    """Return a default AppConfig – the most duplicated setup across tests."""
    return AppConfig()


@pytest.fixture()
def make_candle():
    """Factory for a single Candle with sensible XAUUSD-like defaults."""

    def _make(
        ts: datetime | None = None,
        open: float = 2000.0,
        high: float = 2002.0,
        low: float = 1998.0,
        close: float = 2000.0,
        bid: float | None = None,
        ask: float | None = None,
        volume: float = 1.0,
    ) -> Candle:
        return Candle(
            timestamp=ts or datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc),
            open=open,
            high=high,
            low=low,
            close=close,
            bid=bid,
            ask=ask,
            volume=volume,
        )

    return _make


@pytest.fixture()
def make_candle_series():
    """Generate *count* flat candles as a contiguous 5-min series with bid/ask."""

    def _make(
        count: int = 900,
        base: float = 100.0,
        spread: float = 0.2,
        start: datetime | None = None,
        step_minutes: int = 5,
    ) -> list[Candle]:
        start = start or datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        half = spread * 0.5
        candles: list[Candle] = []
        for i in range(count):
            ts = start + timedelta(minutes=step_minutes * i)
            candles.append(
                Candle(
                    timestamp=ts,
                    open=base,
                    high=base + 1.0,
                    low=base - 1.0,
                    close=base,
                    bid=base - half,
                    ask=base + half,
                    volume=1.0,
                )
            )
        return candles

    return _make

