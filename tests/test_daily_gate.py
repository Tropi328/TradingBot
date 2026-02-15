from __future__ import annotations

from datetime import datetime, timezone

from bot.data.candles import Candle
from bot.gating.daily_gate import DailyGateProvider
from bot.news.calendar_provider import Event


def _daily_close(day: int, close: float) -> Candle:
    ts = datetime(2026, 1, day, 23, 55, tzinfo=timezone.utc)
    return Candle(
        timestamp=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        bid=close - 0.05,
        ask=close + 0.05,
        volume=1.0,
    )


def test_daily_gate_no_lookahead_for_current_day() -> None:
    candles = [
        _daily_close(1, 100.0),
        _daily_close(2, 200.0),
        _daily_close(3, 200.0),
    ]
    gate = DailyGateProvider(mode="trend", thr=0.001)
    gate.refresh_from_candles(candles)

    day_2 = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)
    day_3 = datetime(2026, 1, 3, 12, 0, tzinfo=timezone.utc)
    bias_day_2 = gate.evaluate(ts=day_2, symbol="XAUUSD", spread=0.5)
    bias_day_3 = gate.evaluate(ts=day_3, symbol="XAUUSD", spread=0.5)

    assert bias_day_2.bias == "FLAT"
    assert "TREND_NEUTRAL" in bias_day_2.reasons
    assert bias_day_3.bias == "LONG"


def test_daily_gate_news_window_blocks_trading() -> None:
    candles = [
        _daily_close(1, 100.0),
        _daily_close(2, 120.0),
        _daily_close(3, 121.0),
    ]
    event_time = datetime(2026, 1, 3, 12, 0, tzinfo=timezone.utc)
    events = [
        Event(
            event_id="evt-1",
            title="US CPI",
            currency="USD",
            impact="HIGH",
            time=event_time,
            category="macro",
            source="test",
        )
    ]
    gate = DailyGateProvider(
        mode="trend_vol_news",
        thr=0.0005,
        pre_minutes=30,
        post_minutes=30,
        events=events,
    )
    gate.refresh_from_candles(candles)

    in_window = gate.evaluate(ts=datetime(2026, 1, 3, 12, 15, tzinfo=timezone.utc), symbol="XAUUSD", spread=0.5)
    assert in_window.bias == "FLAT"
    assert "NEWS_WINDOW" in in_window.reasons


def test_daily_gate_trend_bias_long_short_flat() -> None:
    gate_long = DailyGateProvider(mode="trend", thr=0.0)
    gate_long.refresh_from_candles([_daily_close(1, 100.0), _daily_close(2, 103.0), _daily_close(3, 103.0)])
    assert gate_long.evaluate(ts=datetime(2026, 1, 3, 10, 0, tzinfo=timezone.utc), symbol="XAUUSD", spread=0.5).bias == "LONG"

    gate_short = DailyGateProvider(mode="trend", thr=0.0)
    gate_short.refresh_from_candles([_daily_close(1, 100.0), _daily_close(2, 97.0), _daily_close(3, 97.0)])
    assert gate_short.evaluate(ts=datetime(2026, 1, 3, 10, 0, tzinfo=timezone.utc), symbol="XAUUSD", spread=0.5).bias == "SHORT"

    gate_flat = DailyGateProvider(mode="trend", thr=0.0)
    gate_flat.refresh_from_candles([_daily_close(1, 100.0), _daily_close(2, 100.0), _daily_close(3, 100.0)])
    assert gate_flat.evaluate(ts=datetime(2026, 1, 3, 10, 0, tzinfo=timezone.utc), symbol="XAUUSD", spread=0.5).bias == "FLAT"

