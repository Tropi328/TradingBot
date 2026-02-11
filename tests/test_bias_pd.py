from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bot.data.candles import Candle
from bot.strategy.bias import determine_h1_bias, pd_allows_trade, premium_discount_state


def _candle(ts: datetime, o: float, h: float, l: float, c: float) -> Candle:
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c)


def test_bias_up_when_above_ema_and_bos() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [
        _candle(start + timedelta(hours=0), 100, 101, 99, 100),
        _candle(start + timedelta(hours=1), 100, 103, 99.5, 102),
        _candle(start + timedelta(hours=2), 102, 104, 100, 103),
        _candle(start + timedelta(hours=3), 103, 105, 101, 104),
        _candle(start + timedelta(hours=4), 104, 103, 100.5, 102),
        _candle(start + timedelta(hours=5), 102, 106, 101, 105),
        _candle(start + timedelta(hours=6), 105, 110, 104, 109),
    ]

    decision = determine_h1_bias(
        candles,
        ema_period=3,
        fractal_left=1,
        fractal_right=1,
    )
    assert decision.bias == "UP"
    assert decision.last_swing_high is not None
    assert decision.last_close > decision.last_swing_high


def test_premium_discount_gating() -> None:
    # Dealing range low=90 high=110 -> EQ=100
    pd_state_long = premium_discount_state(price=98, low=90, high=110)
    pd_state_short = premium_discount_state(price=104, low=90, high=110)

    assert pd_state_long == "DISCOUNT"
    assert pd_state_short == "PREMIUM"
    assert pd_allows_trade("UP", pd_state_long) is True
    assert pd_allows_trade("DOWN", pd_state_short) is True
    assert pd_allows_trade("UP", pd_state_short) is False

