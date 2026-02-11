from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bot.data.candles import Candle
from bot.strategy.trace import (
    DecisionTrace,
    format_trace_text,
    is_new_closed_candle,
    map_reason_codes,
)


def _candle(ts: datetime, o: float, h: float, l: float, c: float) -> Candle:
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c)


def test_is_new_closed_candle_uses_previous_bar() -> None:
    start = datetime(2026, 2, 10, 10, 0, tzinfo=timezone.utc)
    candles = [
        _candle(start + timedelta(minutes=0), 1, 2, 1, 2),
        _candle(start + timedelta(minutes=5), 2, 3, 2, 3),
        _candle(start + timedelta(minutes=10), 3, 4, 3, 4),
    ]
    is_new, closed_ts = is_new_closed_candle(candles, last_processed_closed_ts=None)
    assert is_new is True
    assert closed_ts == candles[-2].timestamp

    is_new2, closed_ts2 = is_new_closed_candle(candles, last_processed_closed_ts=closed_ts)
    assert is_new2 is False
    assert closed_ts2 == closed_ts

    candles.append(_candle(start + timedelta(minutes=15), 4, 5, 4, 5))
    is_new3, closed_ts3 = is_new_closed_candle(candles, last_processed_closed_ts=closed_ts)
    assert is_new3 is True
    assert closed_ts3 == candles[-2].timestamp


def test_is_new_closed_candle_requires_two_bars() -> None:
    one = [_candle(datetime(2026, 2, 10, 10, 0, tzinfo=timezone.utc), 1, 2, 1, 2)]
    is_new, closed_ts = is_new_closed_candle(one, last_processed_closed_ts=None)
    assert is_new is False
    assert closed_ts is None


def test_reason_mapping_and_trace_text() -> None:
    mapped = map_reason_codes(["BIAS_FAIL", "BIAS_FAIL", "SWEEP_FAIL", "CUSTOM_CODE"])
    assert mapped == ["H1_BIAS_NEUTRAL", "M15_NO_SWEEP", "CUSTOM_CODE"]

    trace = DecisionTrace(asset="GOLD", created_at=datetime(2026, 2, 10, 12, 0, tzinfo=timezone.utc))
    trace.reasons = mapped
    text = format_trace_text(trace, "Europe/Warsaw")
    assert "H1:SKIP(no new close)" in text
    assert "M15:SKIP(no new close)" in text
    assert "M5:SKIP(no new close)" in text
    assert "reasons=H1_BIAS_NEUTRAL,M15_NO_SWEEP,CUSTOM_CODE" in text
