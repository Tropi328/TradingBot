from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import bot.backtest.engine as engine_module
from bot.config import AppConfig
from bot.data.candles import Candle
from bot.strategy.state_machine import StrategyDecision, StrategySignal


def _base_candles(count: int = 900) -> list[Candle]:
    start = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
    candles: list[Candle] = []
    for i in range(count):
        ts = start + timedelta(minutes=5 * i)
        close = 100.0
        candles.append(
            Candle(
                timestamp=ts,
                open=100.0,
                high=100.6,
                low=99.4,
                close=close,
                bid=99.9,
                ask=100.1,
                volume=1.0,
            )
        )
    return candles


def _no_signal() -> StrategyDecision:
    return StrategyDecision(
        signal=None,
        reason_codes=[],
        bias="NEUTRAL",
        pd_state="UNKNOWN",
        sweep_ok=False,
        mss_ok=False,
        displacement_ok=False,
        fvg_ok=False,
        spread_ok=True,
        payload={},
    )


def test_fill_rules_long_entry_at_limit_price_and_tp_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    candles = _base_candles()
    entry_idx = 701
    exit_idx = 702
    # Entry touch bar.
    candles[entry_idx] = Candle(
        timestamp=candles[entry_idx].timestamp,
        open=100.0,
        high=101.0,
        low=99.2,
        close=100.1,
        bid=100.0,
        ask=100.2,
        volume=1.0,
    )
    # TP bar.
    candles[exit_idx] = Candle(
        timestamp=candles[exit_idx].timestamp,
        open=100.2,
        high=103.0,
        low=100.0,
        close=102.9,
        bid=102.8,
        ask=103.0,
        volume=1.0,
    )

    class _FakeLongEngine:
        def __init__(self, _config: AppConfig):
            self.sent = False

        def evaluate(self, **_kwargs) -> StrategyDecision:
            candles_m5 = _kwargs.get("candles_m5", [])
            if self.sent or len(candles_m5) < 701:
                return _no_signal()
            self.sent = True
            signal = StrategySignal(
                side="LONG",
                entry_price=100.0,
                stop_price=99.0,
                take_profit=102.0,
                rr=2.0,
                a_plus=False,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                reason_codes=["TEST"],
                metadata={},
            )
            return StrategyDecision(
                signal=signal,
                reason_codes=["TEST"],
                bias="UP",
                pd_state="DISCOUNT",
                sweep_ok=True,
                mss_ok=True,
                displacement_ok=True,
                fvg_ok=True,
                spread_ok=True,
                payload={},
            )

    monkeypatch.setattr(engine_module, "StrategyEngine", _FakeLongEngine)

    config = AppConfig()
    asset = config.assets[0]
    report = engine_module.run_backtest(config=config, asset=asset, candles_m5=candles, assumed_spread=0.2)
    assert report.trades == 1
    trade = report.trade_log[0]
    assert trade.side == "LONG"
    # Entry: limit price (100.0) + half spread (0.1) = 100.1
    assert trade.entry_price == pytest.approx(100.1, rel=0.0, abs=1e-9)
    # TP exit fills at position.tp (limit order), not candle.bid
    assert trade.exit_price == pytest.approx(102.0, rel=0.0, abs=1e-9)


def test_fill_rules_short_entry_at_limit_price_and_tp_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    candles = _base_candles()
    entry_idx = 701
    exit_idx = 702
    candles[entry_idx] = Candle(
        timestamp=candles[entry_idx].timestamp,
        open=100.0,
        high=100.8,
        low=99.2,
        close=99.9,
        bid=100.0,
        ask=100.2,
        volume=1.0,
    )
    candles[exit_idx] = Candle(
        timestamp=candles[exit_idx].timestamp,
        open=99.8,
        high=100.0,
        low=97.0,
        close=97.3,
        bid=97.1,
        ask=97.4,
        volume=1.0,
    )

    class _FakeShortEngine:
        def __init__(self, _config: AppConfig):
            self.sent = False

        def evaluate(self, **_kwargs) -> StrategyDecision:
            candles_m5 = _kwargs.get("candles_m5", [])
            if self.sent or len(candles_m5) < 701:
                return _no_signal()
            self.sent = True
            signal = StrategySignal(
                side="SHORT",
                entry_price=100.0,
                stop_price=101.0,
                take_profit=98.0,
                rr=2.0,
                a_plus=False,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                reason_codes=["TEST"],
                metadata={},
            )
            return StrategyDecision(
                signal=signal,
                reason_codes=["TEST"],
                bias="DOWN",
                pd_state="PREMIUM",
                sweep_ok=True,
                mss_ok=True,
                displacement_ok=True,
                fvg_ok=True,
                spread_ok=True,
                payload={},
            )

    monkeypatch.setattr(engine_module, "StrategyEngine", _FakeShortEngine)

    config = AppConfig()
    asset = config.assets[0]
    report = engine_module.run_backtest(config=config, asset=asset, candles_m5=candles, assumed_spread=0.2)
    assert report.trades == 1
    trade = report.trade_log[0]
    assert trade.side == "SHORT"
    # Entry: limit price (100.0) - half spread (0.1) = 99.9
    assert trade.entry_price == pytest.approx(99.9, rel=0.0, abs=1e-9)
    # TP exit fills at position.tp (limit order), not candle.ask
    assert trade.exit_price == pytest.approx(98.0, rel=0.0, abs=1e-9)
