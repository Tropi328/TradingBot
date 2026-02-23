from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import bot.backtest.engine as engine_module
from bot.config import AppConfig
from bot.data.candles import Candle
from bot.strategy.state_machine import StrategyDecision


def _no_signal_decision() -> StrategyDecision:
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


class _NoSignalEngine:
    def __init__(self, _config: AppConfig):
        pass

    def evaluate(self, **kwargs) -> StrategyDecision:
        return _no_signal_decision()


def _daily_sparse_candles(start_utc: datetime, days: int) -> list[Candle]:
    candles: list[Candle] = []
    for i in range(days):
        ts = start_utc + timedelta(days=i)
        mid = 100.0
        spread = 0.2
        half = spread * 0.5
        candles.append(
            Candle(
                timestamp=ts,
                open=mid,
                high=mid + 0.2,
                low=mid - 0.2,
                close=mid,
                bid=mid - half,
                ask=mid + half,
                volume=1.0,
            )
        )
    return candles


def test_backtest_capital_ramp_topups_multi_month(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "StrategyEngine", _NoSignalEngine)
    cfg = AppConfig(account_currency="PLN", capital_ramp={"enabled": True})
    cfg.risk.equity = 10_000.0
    asset = cfg.assets[0]
    asset.epic = "XAUUSD"
    candles = _daily_sparse_candles(datetime(2026, 2, 15, tzinfo=timezone.utc), 80)

    report = engine_module.run_backtest(config=cfg, asset=asset, candles_m5=candles, assumed_spread=0.2)
    assert report.capital_ramp_enabled is True
    assert report.capital_ramp_topups_count == 3
    assert report.capital_ramp_topups_total == 300.0
    assert report.capital_ramp_stopped_reason is None
    assert report.equity_end == 400.0


def test_backtest_capital_ramp_first_year_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "StrategyEngine", _NoSignalEngine)
    cfg = AppConfig(account_currency="PLN", capital_ramp={"enabled": True})
    asset = cfg.assets[0]
    asset.epic = "XAUUSD"
    candles = _daily_sparse_candles(datetime(2026, 9, 15, tzinfo=timezone.utc), 220)

    report = engine_module.run_backtest(config=cfg, asset=asset, candles_m5=candles, assumed_spread=0.2)
    assert report.capital_ramp_topups_count == 3  # Oct/Nov/Dec only
    assert report.capital_ramp_topups_total == 300.0
    assert report.capital_ramp_stopped_reason == "YEAR_END"


def test_backtest_capital_ramp_disabled_keeps_original_equity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "StrategyEngine", _NoSignalEngine)
    cfg = AppConfig(account_currency="USD", capital_ramp={"enabled": False})
    cfg.risk.equity = 777.0
    asset = cfg.assets[0]
    asset.epic = "XAUUSD"
    candles = _daily_sparse_candles(datetime(2026, 2, 15, tzinfo=timezone.utc), 140)

    report = engine_module.run_backtest(config=cfg, asset=asset, candles_m5=candles, assumed_spread=0.2)
    assert report.capital_ramp_enabled is False
    assert report.capital_ramp_topups_count == 0
    assert report.capital_ramp_topups_total == 0.0
    assert report.equity_end == pytest.approx(777.0)


def test_backtest_multi_strategy_capital_ramp_first_year_only() -> None:
    cfg = AppConfig(account_currency="PLN", capital_ramp={"enabled": True})
    cfg.risk.equity = 5000.0
    asset = cfg.assets[0]
    asset.epic = "XAUUSD"
    candles = _daily_sparse_candles(datetime(2026, 9, 15, tzinfo=timezone.utc), 420)

    report = engine_module.run_backtest_multi_strategy(
        config=cfg,
        asset=asset,
        candles_m5=candles,
        assumed_spread=0.2,
    )
    assert report.capital_ramp_enabled is True
    assert report.capital_ramp_topups_count == 3
    assert report.capital_ramp_topups_total == 300.0
    assert report.capital_ramp_stopped_reason == "YEAR_END"
