from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import bot.backtest.engine as engine_module
from bot.config import AppConfig
from bot.data.candles import Candle
from bot.execution.fx import FxConverter
from bot.strategy.state_machine import StrategyDecision, StrategySignal


def _base_candles(count: int = 900) -> list[Candle]:
    start = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
    candles: list[Candle] = []
    for i in range(count):
        ts = start + timedelta(minutes=5 * i)
        candles.append(
            Candle(
                timestamp=ts,
                open=100.0,
                high=100.6,
                low=99.4,
                close=100.0,
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


def test_fx_all_in_rate_capital_example() -> None:
    converter = FxConverter(
        fee_rate=0.007,
        fee_mode="all_in_rate",
        rate_source="static",
        static_rates={"EURUSD": 1.08461},
    )
    result = converter.convert(amount=100.0, from_currency="EUR", to_currency="USD", apply_fee=True)
    assert result.spot_rate == pytest.approx(1.08461, abs=1e-9)
    assert result.all_in_rate == pytest.approx(1.08461 * (1.0 - 0.007), rel=0, abs=1e-9)
    assert result.fx_cost == pytest.approx((100.0 * 1.08461) - (100.0 * result.all_in_rate), abs=1e-9)


def test_fx_fee_only_when_conversion_occurs() -> None:
    converter = FxConverter(
        fee_rate=0.007,
        fee_mode="all_in_rate",
        rate_source="static",
        static_rates={"USDPLN": 4.0},
    )
    same = converter.convert(amount=25.0, from_currency="USD", to_currency="USD", apply_fee=True)
    assert same.fx_cost == 0.0

    converted = converter.convert(amount=25.0, from_currency="USD", to_currency="PLN", apply_fee=True)
    assert converted.fx_cost > 0.0


def test_feasibility_gate_rejects_too_small_size_for_100_pln(monkeypatch: pytest.MonkeyPatch) -> None:
    candles = _base_candles()

    class _FakeSignalEngine:
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
                stop_price=90.0,
                take_profit=120.0,
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

    monkeypatch.setattr(engine_module, "StrategyEngine", _FakeSignalEngine)

    config = AppConfig()
    config.account_currency = "PLN"
    config.fx_static_rates = {"USDPLN": 4.0}
    config.risk.equity = 100.0
    config.risk.risk_per_trade = 0.005
    asset = config.assets[0]
    asset.min_size = 1.0
    asset.size_step = 1.0
    asset.instrument_currency = "USD"

    report = engine_module.run_backtest(config=config, asset=asset, candles_m5=candles, assumed_spread=0.2)
    assert report.trades_filled == 0
    assert report.trades == 0
    assert report.rejected_by_reason.get("SIZE_TOO_SMALL", 0) > 0


def test_regression_usd_account_zero_fx_fee(monkeypatch: pytest.MonkeyPatch) -> None:
    candles = _base_candles()
    entry_idx = 701
    exit_idx = 702
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
    config.account_currency = "USD"
    config.fx_conversion_fee_rate = 0.0
    config.fx_static_rates = {}
    asset = config.assets[0]
    asset.instrument_currency = "USD"

    report = engine_module.run_backtest(config=config, asset=asset, candles_m5=candles, assumed_spread=0.2)
    assert report.trades == 1
    assert report.total_pnl == pytest.approx(130.0, abs=1e-9)
    assert report.fx_cost_sum == pytest.approx(0.0, abs=1e-12)

