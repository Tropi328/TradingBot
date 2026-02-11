from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from bot.config import AppConfig
from bot.data.candles import Candle
from bot.strategy.contracts import DecisionAction, StrategyDataBundle
from bot.strategy.index_existing import IndexExistingStrategy
from bot.strategy.state_machine import StrategyEngine
from bot.strategy.trace import closed_candles


def _series(*, start: datetime, minutes: int, count: int, base: float, drift: float, amp: float) -> list[Candle]:
    candles: list[Candle] = []
    prev_close = base
    for i in range(count):
        ts = start + timedelta(minutes=minutes * i)
        close = base + drift * i + amp * math.sin(i / 6.0)
        open_price = prev_close
        high = max(open_price, close) + (amp * 0.25 + 0.1)
        low = min(open_price, close) - (amp * 0.25 + 0.1)
        candles.append(Candle(timestamp=ts, open=open_price, high=high, low=low, close=close))
        prev_close = close
    return candles


def test_index_existing_adapter_matches_legacy_engine_snapshot() -> None:
    config = AppConfig()
    symbol = "US500"
    now = datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc)
    candles_h1 = _series(start=now - timedelta(hours=400), minutes=60, count=320, base=6000, drift=0.8, amp=7.0)
    candles_m15 = _series(start=now - timedelta(hours=180), minutes=15, count=640, base=6100, drift=0.18, amp=3.2)
    candles_m5 = _series(start=now - timedelta(hours=70), minutes=5, count=840, base=6150, drift=0.08, amp=1.8)
    bundle = StrategyDataBundle(
        symbol=symbol,
        now=now,
        candles_h1=candles_h1,
        candles_m15=candles_m15,
        candles_m5=candles_m5,
        spread=0.5,
        spread_history=[0.4] * 120,
        news_blocked=False,
        entry_state="WAIT",
        h1_new_close=True,
        m15_new_close=True,
        m5_new_close=True,
        extra={"minimal_tick_buffer": 0.5},
    )

    adapter = IndexExistingStrategy(config)
    adapter.preprocess(symbol, bundle)
    adapter.compute_bias(symbol, bundle)
    candidates = adapter.detect_candidates(symbol, bundle)
    assert candidates, "Adapter should produce pipeline candidate for index strategy"
    evaluation = adapter.evaluate_candidate(symbol, candidates[0], bundle)
    generated = adapter.generate_order(symbol, evaluation, candidates[0], bundle)
    legacy_from_adapter = adapter.last_legacy_decision(symbol)
    assert legacy_from_adapter is not None

    engine = StrategyEngine(config)
    legacy_h1 = engine.evaluate_h1(closed_candles(candles_h1))
    legacy_m15 = engine.evaluate_m15(
        candles_m15=closed_candles(candles_m15),
        h1=legacy_h1,
        minimal_tick_buffer=0.5,
        now=now,
        previous=None,
    )
    legacy_direct, _ = engine.evaluate_m5(
        epic=symbol,
        candles_m5=closed_candles(candles_m5),
        current_spread=0.5,
        spread_history=[0.4] * 120,
        news_blocked=False,
        h1=legacy_h1,
        m15=legacy_m15,
        entry_state="WAIT",
    )

    assert legacy_from_adapter.reason_codes == legacy_direct.reason_codes
    assert bool(legacy_from_adapter.signal) == bool(legacy_direct.signal)
    assert evaluation.action == (
        DecisionAction.TRADE if legacy_direct.signal is not None else DecisionAction.OBSERVE
    )
    if legacy_direct.signal is not None:
        assert generated is not None
        assert generated.side == legacy_direct.signal.side
        assert generated.entry_price == legacy_direct.signal.entry_price
        assert generated.stop_price == legacy_direct.signal.stop_price
        assert generated.take_profit == legacy_direct.signal.take_profit

