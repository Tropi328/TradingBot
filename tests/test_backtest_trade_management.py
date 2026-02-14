from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bot.backtest.engine import (
    _OpenPosition,
    _apply_overnight_swap_if_due,
    _build_dynamic_assumed_spread_series,
    _calc_exit,
    _expected_rr,
    _manage_open_position,
    _next_rollover_timestamp,
    _tp2_r_for_target_total_r,
)
from bot.data.candles import Candle


def _candle(ts: datetime, *, high: float, low: float, close: float) -> Candle:
    return Candle(timestamp=ts, open=close, high=high, low=low, close=close, volume=1.0)


def test_stop_exit_respects_max_loss_cap() -> None:
    pos = _OpenPosition(
        side="LONG",
        entry=100.0,
        stop=99.0,
        tp=102.0,
        size=1.0,
        opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        initial_stop=99.0,
        initial_risk=1.0,
        max_loss_r_cap=1.0,
    )
    bar = _candle(datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc), high=100.2, low=95.0, close=96.0)
    should_close, exit_price, reason = _calc_exit(pos, bar, assumed_spread=0.2, slippage=0.5)
    assert should_close is True
    assert reason == "STOP"
    assert exit_price == pytest.approx(99.0, rel=0.0, abs=1e-9)


def test_tp1_moves_stop_to_be_and_takes_partial() -> None:
    pos = _OpenPosition(
        side="LONG",
        entry=100.0,
        stop=99.0,
        tp=102.0,
        size=1.0,
        opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        initial_stop=99.0,
        initial_risk=1.0,
        tp1_trigger_r=0.5,
        tp1_fraction=0.5,
        be_offset_r=0.0,
        be_delay_bars_after_tp1=0,
        trailing_after_tp1=False,
    )
    candles = [
        _candle(datetime(2024, 1, 1, tzinfo=timezone.utc), high=100.1, low=99.8, close=100.0),
        _candle(datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc), high=100.6, low=99.9, close=100.5),
    ]
    tp1_hit, be_moved = _manage_open_position(
        position=pos,
        candle=candles[-1],
        candles_m5=candles,
        index=1,
        slippage=0.0,
    )
    assert tp1_hit is True
    assert be_moved is True
    assert pos.tp1_taken is True
    assert pos.stop == pytest.approx(100.0, rel=0.0, abs=1e-9)
    assert pos.size == pytest.approx(0.5, rel=0.0, abs=1e-9)
    assert pos.realized_partial == pytest.approx(0.25, rel=0.0, abs=1e-9)


def test_tp1_be_delay_waits_before_be_move() -> None:
    pos = _OpenPosition(
        side="LONG",
        entry=100.0,
        stop=99.0,
        tp=102.0,
        size=1.0,
        opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        initial_stop=99.0,
        initial_risk=1.0,
        tp1_trigger_r=0.5,
        tp1_fraction=0.5,
        be_offset_r=0.0,
        be_delay_bars_after_tp1=2,
        trailing_after_tp1=False,
    )
    candles = [
        _candle(datetime(2024, 1, 1, tzinfo=timezone.utc), high=100.1, low=99.8, close=100.0),
        _candle(datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc), high=100.6, low=99.9, close=100.5),
        _candle(datetime(2024, 1, 1, 0, 10, tzinfo=timezone.utc), high=100.7, low=99.9, close=100.4),
        _candle(datetime(2024, 1, 1, 0, 15, tzinfo=timezone.utc), high=100.8, low=99.9, close=100.3),
    ]

    tp1_hit_1, be_moved_1 = _manage_open_position(
        position=pos,
        candle=candles[1],
        candles_m5=candles[:2],
        index=1,
        slippage=0.0,
    )
    assert tp1_hit_1 is True
    assert be_moved_1 is False
    assert pos.stop == pytest.approx(99.0, rel=0.0, abs=1e-9)

    tp1_hit_2, be_moved_2 = _manage_open_position(
        position=pos,
        candle=candles[2],
        candles_m5=candles[:3],
        index=2,
        slippage=0.0,
    )
    assert tp1_hit_2 is False
    assert be_moved_2 is False
    assert pos.stop == pytest.approx(99.0, rel=0.0, abs=1e-9)

    tp1_hit_3, be_moved_3 = _manage_open_position(
        position=pos,
        candle=candles[3],
        candles_m5=candles,
        index=3,
        slippage=0.0,
    )
    assert tp1_hit_3 is False
    assert be_moved_3 is True
    assert pos.stop == pytest.approx(100.0, rel=0.0, abs=1e-9)


def test_expected_rr_helper() -> None:
    long_rr = _expected_rr(side="LONG", entry=100.0, stop=99.0, target=101.5)
    short_rr = _expected_rr(side="SHORT", entry=100.0, stop=101.0, target=98.7)
    assert long_rr == pytest.approx(1.5, rel=0.0, abs=1e-9)
    assert short_rr == pytest.approx(1.3, rel=0.0, abs=1e-9)


def test_tp2_r_profile_with_partial_tp1() -> None:
    standard_price_mode = _tp2_r_for_target_total_r(
        target_total_r=2.0,
        tp1_trigger_r=1.0,
        tp1_fraction=0.5,
        mode="strict_tp_price",
    )
    aplus_price_mode = _tp2_r_for_target_total_r(
        target_total_r=3.0,
        tp1_trigger_r=1.0,
        tp1_fraction=0.5,
        mode="strict_tp_price",
    )
    assert standard_price_mode == pytest.approx(2.0, rel=0.0, abs=1e-9)
    assert aplus_price_mode == pytest.approx(3.0, rel=0.0, abs=1e-9)

    standard_total_mode = _tp2_r_for_target_total_r(
        target_total_r=2.0,
        tp1_trigger_r=1.0,
        tp1_fraction=0.5,
        mode="strict_total_rr",
    )
    aplus_total_mode = _tp2_r_for_target_total_r(
        target_total_r=3.0,
        tp1_trigger_r=1.0,
        tp1_fraction=0.5,
        mode="strict_total_rr",
    )
    assert standard_total_mode == pytest.approx(3.0, rel=0.0, abs=1e-9)
    assert aplus_total_mode == pytest.approx(5.0, rel=0.0, abs=1e-9)


def test_dynamic_assumed_spread_series_stays_within_range() -> None:
    candles = [
        _candle(datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc), high=100.1, low=99.9, close=100.0),
        _candle(datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc), high=100.3, low=99.8, close=100.1),
        _candle(datetime(2024, 1, 1, 0, 10, tzinfo=timezone.utc), high=100.5, low=99.7, close=100.2),
        _candle(datetime(2024, 1, 1, 0, 15, tzinfo=timezone.utc), high=100.8, low=99.6, close=100.3),
    ]
    spreads = _build_dynamic_assumed_spread_series(
        candles_m5=candles,
        atr_values=[0.4, 0.8, 1.2, 1.8],
        min_spread=0.5,
        max_spread=0.8,
    )
    assert len(spreads) == len(candles)
    assert min(spreads) >= 0.5
    assert max(spreads) <= 0.8
    assert spreads[0] < spreads[-1]


def test_overnight_swap_applies_at_rollover_once_per_day() -> None:
    pos = _OpenPosition(
        side="LONG",
        entry=2000.0,
        stop=1995.0,
        tp=2010.0,
        size=1.0,
        opened_at=datetime(2024, 1, 1, 22, 0, tzinfo=timezone.utc),
        initial_stop=1995.0,
        initial_risk=5.0,
    )
    pos.next_swap_ts = _next_rollover_timestamp(
        datetime(2024, 1, 1, 22, 0, tzinfo=timezone.utc),
        hour=23,
        minute=0,
    )
    # Before rollover: no fee.
    fee_before = _apply_overnight_swap_if_due(
        position=pos,
        candle_ts=datetime(2024, 1, 1, 22, 55, tzinfo=timezone.utc),
        swap_hour=23,
        swap_minute=0,
        long_swap_pct=-0.016,
        short_swap_pct=0.0076,
    )
    assert fee_before == pytest.approx(0.0, rel=0.0, abs=1e-12)
    # At rollover: one fee accrual.
    fee_rollover = _apply_overnight_swap_if_due(
        position=pos,
        candle_ts=datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc),
        swap_hour=23,
        swap_minute=0,
        long_swap_pct=-0.016,
        short_swap_pct=0.0076,
    )
    assert fee_rollover == pytest.approx(-0.32, rel=0.0, abs=1e-9)
    assert pos.swap_total == pytest.approx(-0.32, rel=0.0, abs=1e-9)
    # Same candle timestamp again should not double-charge.
    fee_same_ts = _apply_overnight_swap_if_due(
        position=pos,
        candle_ts=datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc),
        swap_hour=23,
        swap_minute=0,
        long_swap_pct=-0.016,
        short_swap_pct=0.0076,
    )
    assert fee_same_ts == pytest.approx(0.0, rel=0.0, abs=1e-12)
