from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import bot.backtest.engine as engine_module
from bot.config import AppConfig
from bot.data.candles import Candle
from bot.execution.feasibility import RejectReason, validate_order
from bot.strategy.state_machine import StrategyDecision, StrategySignal


def _candles_with_spread(spread: float, count: int = 900) -> list[Candle]:
    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    candles: list[Candle] = []
    half = spread * 0.5
    for i in range(count):
        ts = start + timedelta(minutes=5 * i)
        mid = 100.0
        candles.append(
            Candle(
                timestamp=ts,
                open=mid,
                high=mid + 1.0,
                low=mid - 1.0,
                close=mid,
                bid=mid - half,
                ask=mid + half,
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


class _SingleSignalEngine:
    def __init__(self, _config: AppConfig):
        self.sent = False

    def evaluate(self, **kwargs) -> StrategyDecision:
        candles_m5 = kwargs.get("candles_m5", [])
        if self.sent or len(candles_m5) < 701:
            return _no_signal()
        self.sent = True
        signal = StrategySignal(
            side="LONG",
            entry_price=100.0,
            stop_price=99.0,
            take_profit=104.0,
            rr=4.0,
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


def test_spread_points_filter_rejects_wide_spread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "StrategyEngine", _SingleSignalEngine)
    candles = _candles_with_spread(spread=0.12)

    config = AppConfig()
    config.risk.equity = 100.0
    config.backtest_tuning.spread_limit_points = 6.0
    config.backtest_tuning.min_edge_to_cost_ratio = 1.0
    asset = config.assets[0]
    asset.epic = "XAUUSD"
    asset.point_size = 0.01

    report = engine_module.run_backtest(config=config, asset=asset, candles_m5=candles, assumed_spread=0.12)
    assert report.rejected_by_reason.get("SPREAD_TOO_WIDE", 0) > 0


def test_min_risk_cash_per_trade_reduces_size_too_small(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "StrategyEngine", _SingleSignalEngine)
    candles = _candles_with_spread(spread=0.02)

    config_low = AppConfig()
    config_low.risk.equity = 100.0
    config_low.risk.risk_per_trade = 0.001
    config_low.risk.min_risk_cash_per_trade = 0.0
    config_low.risk.max_risk_cash_per_trade = 2.0
    config_low.backtest_tuning.spread_limit_points = 20.0
    config_low.backtest_tuning.min_edge_to_cost_ratio = 1.0
    asset_low = config_low.assets[0]
    asset_low.min_size = 1.0
    asset_low.size_step = 1.0

    report_low = engine_module.run_backtest(config=config_low, asset=asset_low, candles_m5=candles, assumed_spread=0.02)

    config_high = AppConfig()
    config_high.risk.equity = 100.0
    config_high.risk.risk_per_trade = 0.001
    config_high.risk.min_risk_cash_per_trade = 1.2
    config_high.risk.max_risk_cash_per_trade = 2.0
    config_high.backtest_tuning.spread_limit_points = 20.0
    config_high.backtest_tuning.min_edge_to_cost_ratio = 1.0
    asset_high = config_high.assets[0]
    asset_high.min_size = 1.0
    asset_high.size_step = 1.0

    report_high = engine_module.run_backtest(config=config_high, asset=asset_high, candles_m5=candles, assumed_spread=0.02)

    assert report_high.rejected_by_reason.get("SIZE_TOO_SMALL", 0) < report_low.rejected_by_reason.get("SIZE_TOO_SMALL", 0)


def test_allow_min_size_override_if_within_risk_respects_max_risk_cash() -> None:
    rejected = validate_order(
        raw_size=0.1,
        entry_price=100.0,
        stop_price=99.5,
        take_profit=102.0,
        min_size=1.0,
        size_step=1.0,
        max_risk_cash=0.4,
        equity=100.0,
        open_positions_count=0,
        max_positions=1,
        allow_min_size_override_if_within_risk=True,
    )
    assert rejected.ok is False
    assert rejected.reason == RejectReason.SIZE_TOO_SMALL

    accepted = validate_order(
        raw_size=0.1,
        entry_price=100.0,
        stop_price=99.5,
        take_profit=102.0,
        min_size=1.0,
        size_step=1.0,
        max_risk_cash=0.6,
        equity=100.0,
        open_positions_count=0,
        max_positions=1,
        allow_min_size_override_if_within_risk=True,
    )
    assert accepted.ok is True
    assert float(accepted.details.get("rounded_size", 0.0)) == pytest.approx(1.0)
    assert bool(accepted.details.get("min_size_override_used", False)) is True
    assert float(accepted.details.get("risk_cash_rounded", 0.0)) <= 0.6


def test_margin_cap_shrinks_size_instead_of_rejecting() -> None:
    """When margin is insufficient for the requested size, the system should
    cap the position size to fit within margin instead of rejecting outright."""
    # raw_size=200 at entry=2600 → notional=520,000 → margin(5%)=26,000
    # equity/free_margin=10,000 → should cap size down to ~76 lots
    result = validate_order(
        raw_size=200.0,
        entry_price=2600.0,
        stop_price=2595.0,
        take_profit=2610.0,
        min_size=0.01,
        size_step=0.01,
        max_risk_cash=100000.0,
        equity=10000.0,
        open_positions_count=0,
        max_positions=10,
        free_margin=10000.0,
        margin_requirement_pct=5.0,
        max_leverage=20.0,
        margin_safety_factor=1.0,
    )
    assert result.ok is True, f"Expected OK, got {result.reason}: {result.details}"
    size = float(result.details["rounded_size"])
    assert size > 0
    assert size < 200.0  # must be smaller than requested
    assert result.details.get("margin_capped") is True
    # verify margin constraint: size * entry * 5% <= 10000
    assert size * 2600.0 * 0.05 <= 10000.0 + 1e-6


def test_margin_cap_rejects_if_min_size_too_large() -> None:
    """If even min_size exceeds margin, the rejection should stand."""
    result = validate_order(
        raw_size=200.0,
        entry_price=2600.0,
        stop_price=2595.0,
        take_profit=2610.0,
        min_size=100.0,  # min_size=100 → margin=13,000 > 10,000
        size_step=1.0,
        max_risk_cash=100000.0,
        equity=10000.0,
        open_positions_count=0,
        max_positions=10,
        free_margin=10000.0,
        margin_requirement_pct=5.0,
        max_leverage=20.0,
        margin_safety_factor=1.0,
    )
    assert result.ok is False
    assert result.reason == RejectReason.INSUFFICIENT_MARGIN
