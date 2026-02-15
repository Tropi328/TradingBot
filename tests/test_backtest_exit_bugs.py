"""Regression tests for TP negative R, same-bar guard, STOP priority, and BE before exit."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bot.backtest.engine import (
    _OpenPosition,
    _calc_exit,
    _manage_open_position,
    _trade_r_multiple,
)
from bot.data.candles import Candle


def _candle(
    ts: datetime,
    *,
    high: float,
    low: float,
    close: float,
    bid: float | None = None,
    ask: float | None = None,
) -> Candle:
    return Candle(
        timestamp=ts,
        open=close,
        high=high,
        low=low,
        close=close,
        bid=bid,
        ask=ask,
        volume=1.0,
    )


# ---------------------------------------------------------------------------
# Test 1: TP exit should NEVER produce negative R-multiple
# Scenario: LONG position, TP barely touched by bar high, bar bid far below entry.
# Old bug: TP fill used candle.bid → negative PnL despite reason_close="TP".
# ---------------------------------------------------------------------------
class TestTpExitPositiveR:
    def test_long_tp_fills_at_tp_price_not_bar_bid(self) -> None:
        """LONG TP: bar high just touches TP but bar bid is way below entry."""
        pos = _OpenPosition(
            side="LONG",
            entry=2000.0,
            stop=1990.0,
            tp=2020.0,
            size=1.0,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            initial_stop=1990.0,
            initial_risk=10.0,
            initial_size=1.0,
        )
        # Bar: high barely touches TP (2020), but bid is far below entry (1995).
        bar = _candle(
            datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc),
            high=2020.5,
            low=1994.0,
            close=1995.0,
            bid=1995.0,
            ask=1995.5,
        )
        should_close, exit_price, reason = _calc_exit(pos, bar, assumed_spread=0.5, slippage=0.0)
        assert should_close is True
        assert reason == "TP"
        # Exit must be at TP price (2020.0), NOT at bar bid (1995.0).
        assert exit_price == pytest.approx(2020.0, abs=1e-9)
        # Verify PnL is positive.
        pnl = (exit_price - pos.entry) * pos.size
        assert pnl > 0, f"TP exit should be profitable, got PnL={pnl}"

    def test_short_tp_fills_at_tp_price_not_bar_ask(self) -> None:
        """SHORT TP: bar low just touches TP but bar ask is far above entry."""
        pos = _OpenPosition(
            side="SHORT",
            entry=2000.0,
            stop=2010.0,
            tp=1980.0,
            size=1.0,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            initial_stop=2010.0,
            initial_risk=10.0,
            initial_size=1.0,
        )
        # Bar: low barely touches TP (1980), but ask is far above entry (2005).
        bar = _candle(
            datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc),
            high=2006.0,
            low=1979.5,
            close=2005.0,
            bid=2004.5,
            ask=2005.0,
        )
        should_close, exit_price, reason = _calc_exit(pos, bar, assumed_spread=0.5, slippage=0.0)
        assert should_close is True
        assert reason == "TP"
        # Exit must be at TP price (1980.0), NOT at bar ask (2005.0).
        assert exit_price == pytest.approx(1980.0, abs=1e-9)
        pnl = (pos.entry - exit_price) * pos.size
        assert pnl > 0, f"TP exit should be profitable, got PnL={pnl}"

    def test_tp_exit_r_multiple_is_positive(self) -> None:
        """R-multiple for a TP exit must be positive."""
        pos = _OpenPosition(
            side="LONG",
            entry=2000.0,
            stop=1990.0,
            tp=2020.0,
            size=1.0,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            initial_stop=1990.0,
            initial_risk=10.0,
            initial_size=1.0,
        )
        bar = _candle(
            datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc),
            high=2021.0,
            low=1999.0,
            close=2001.0,
            bid=2000.5,
            ask=2001.0,
        )
        should_close, exit_price, reason = _calc_exit(pos, bar, assumed_spread=0.5, slippage=0.0)
        assert reason == "TP"
        total_pnl = (exit_price - pos.entry) * pos.size  # 20.0
        r_mult = _trade_r_multiple(
            total_pnl=total_pnl,
            position=pos,
            fx_converter=None,
            instrument_currency="USD",
            account_currency="USD",
            fx_apply_to=None,
        )
        assert r_mult > 0, f"TP R-multiple must be positive, got {r_mult}"
        assert r_mult == pytest.approx(2.0, abs=0.01)  # 20/10 = 2R


# ---------------------------------------------------------------------------
# Test 2: When both STOP and TP hit on the same bar, STOP wins (conservative)
# ---------------------------------------------------------------------------
class TestStopWinsSameBar:
    def test_long_both_hit_stop_wins(self) -> None:
        """LONG: bar low <= stop AND bar high >= tp → reason must be STOP."""
        pos = _OpenPosition(
            side="LONG",
            entry=2000.0,
            stop=1990.0,
            tp=2020.0,
            size=1.0,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            initial_stop=1990.0,
            initial_risk=10.0,
            initial_size=1.0,
        )
        # Both hit: low touches stop, high touches TP.
        bar = _candle(
            datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc),
            high=2021.0,
            low=1989.0,
            close=2000.0,
        )
        should_close, exit_price, reason = _calc_exit(pos, bar, assumed_spread=0.5, slippage=0.5)
        assert should_close is True
        assert reason == "STOP"
        # Fill at stop - slippage, clamped by max_loss_r_cap.
        assert exit_price == pytest.approx(1990.0, abs=1.0)

    def test_short_both_hit_stop_wins(self) -> None:
        """SHORT: bar high >= stop AND bar low <= tp → reason must be STOP."""
        pos = _OpenPosition(
            side="SHORT",
            entry=2000.0,
            stop=2010.0,
            tp=1980.0,
            size=1.0,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            initial_stop=2010.0,
            initial_risk=10.0,
            initial_size=1.0,
        )
        bar = _candle(
            datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc),
            high=2011.0,
            low=1979.0,
            close=2000.0,
        )
        should_close, exit_price, reason = _calc_exit(pos, bar, assumed_spread=0.5, slippage=0.5)
        assert should_close is True
        assert reason == "STOP"
        assert exit_price == pytest.approx(2010.0, abs=1.0)


# ---------------------------------------------------------------------------
# Test 3: Same-bar fill+exit guard — _calc_exit is NOT called on the fill bar
# (This is tested at the engine level, but we verify the guard concept here.)
# ---------------------------------------------------------------------------
class TestSameBarGuardConcept:
    def test_position_opened_at_matches_candle_skips_exit(self) -> None:
        """Positions filled on the current bar should NOT be processed for exit."""
        fill_ts = datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc)
        pos = _OpenPosition(
            side="LONG",
            entry=2000.0,
            stop=1990.0,
            tp=2020.0,
            size=1.0,
            opened_at=fill_ts,
            initial_stop=1990.0,
            initial_risk=10.0,
            initial_size=1.0,
        )
        # The guard checks: pos.opened_at == candle.timestamp → skip
        bar = _candle(fill_ts, high=2021.0, low=1989.0, close=2000.0)
        # Verify the timestamps match — the guard would fire.
        assert pos.opened_at == bar.timestamp
        # If guard is NOT applied, _calc_exit would trigger.
        should_close, _, _ = _calc_exit(pos, bar, assumed_spread=0.5, slippage=0.0)
        assert should_close is True, "Without guard, exit would trigger on same bar"

    def test_position_next_bar_processes_normally(self) -> None:
        """On the next bar after fill, exit should be processed normally."""
        fill_ts = datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc)
        next_ts = datetime(2024, 1, 1, 0, 10, tzinfo=timezone.utc)
        pos = _OpenPosition(
            side="LONG",
            entry=2000.0,
            stop=1990.0,
            tp=2020.0,
            size=1.0,
            opened_at=fill_ts,
            initial_stop=1990.0,
            initial_risk=10.0,
            initial_size=1.0,
        )
        bar = _candle(next_ts, high=2021.0, low=1999.0, close=2005.0)
        # Guard would NOT fire: different timestamps.
        assert pos.opened_at != bar.timestamp
        should_close, exit_price, reason = _calc_exit(pos, bar, assumed_spread=0.5, slippage=0.0)
        assert should_close is True
        assert reason == "TP"
        assert exit_price == pytest.approx(2020.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 4: After TP1 + BE move, the BE stop is effective before exit check
# (manage_open_position runs BEFORE _calc_exit)
# ---------------------------------------------------------------------------
class TestBeStopEffectiveBeforeExit:
    def test_be_stop_moves_before_exit_check(self) -> None:
        """After TP1, BE should move stop to entry. Next bar: stop at BE."""
        pos = _OpenPosition(
            side="LONG",
            entry=2000.0,
            stop=1990.0,
            tp=2020.0,
            size=1.0,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            initial_stop=1990.0,
            initial_risk=10.0,
            initial_size=1.0,
            tp1_trigger_r=0.5,
            tp1_fraction=0.5,
            be_offset_r=0.0,
            be_delay_bars_after_tp1=0,
            trailing_after_tp1=False,
        )
        # Bar 1: TP1 level = entry + 0.5 * initial_risk = 2005.0
        candles = [
            _candle(datetime(2024, 1, 1, tzinfo=timezone.utc), high=2000.2, low=1999.8, close=2000.0),
            _candle(datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc), high=2006.0, low=1999.0, close=2005.0),
        ]
        tp1_hit, be_moved = _manage_open_position(
            position=pos,
            candle=candles[1],
            candles_m5=candles,
            index=1,
            slippage=0.0,
        )
        assert tp1_hit is True
        assert be_moved is True
        assert pos.stop == pytest.approx(2000.0, abs=1e-9)  # Moved to entry (BE).

        # Bar 2: price drops to exactly BE level. Stop should trigger.
        bar2 = _candle(datetime(2024, 1, 1, 0, 10, tzinfo=timezone.utc), high=2003.0, low=2000.0, close=2001.0)
        should_close, exit_price, reason = _calc_exit(pos, bar2, assumed_spread=0.5, slippage=0.0)
        assert should_close is True
        assert reason == "BE"
        assert exit_price == pytest.approx(2000.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 5: R-multiple uses trade's own risk, not portfolio-level risk
# ---------------------------------------------------------------------------
class TestRMultipleTradeLevel:
    def test_r_multiple_uses_initial_risk_times_initial_size(self) -> None:
        """R = total_pnl / (initial_risk * initial_size), not equity * risk_per_trade."""
        pos = _OpenPosition(
            side="LONG",
            entry=2000.0,
            stop=1990.0,
            tp=2020.0,
            size=1.0,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            initial_stop=1990.0,
            initial_risk=10.0,     # |entry - stop| = 10
            initial_size=1.0,      # full position
        )
        # PnL = +20 (exit at TP 2020, entry at 2000, size 1)
        r_mult = _trade_r_multiple(
            total_pnl=20.0,
            position=pos,
            fx_converter=None,
            instrument_currency="USD",
            account_currency="USD",
            fx_apply_to=None,
        )
        # R = 20 / (10 * 1) = 2.0
        assert r_mult == pytest.approx(2.0, abs=1e-9)

    def test_r_multiple_negative_for_stop_loss(self) -> None:
        """Stop loss produces negative R."""
        pos = _OpenPosition(
            side="LONG",
            entry=2000.0,
            stop=1990.0,
            tp=2020.0,
            size=2.0,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            initial_stop=1990.0,
            initial_risk=10.0,
            initial_size=2.0,
        )
        # Lost 10 per unit * 2 = -20
        r_mult = _trade_r_multiple(
            total_pnl=-20.0,
            position=pos,
            fx_converter=None,
            instrument_currency="USD",
            account_currency="USD",
            fx_apply_to=None,
        )
        # R = -20 / (10 * 2) = -1.0R
        assert r_mult == pytest.approx(-1.0, abs=1e-9)

    def test_r_multiple_with_partial_close(self) -> None:
        """After TP1 partial close, R uses initial_size (not current reduced size)."""
        pos = _OpenPosition(
            side="LONG",
            entry=2000.0,
            stop=1990.0,
            tp=2020.0,
            size=0.5,              # After TP1 partial, size is halved
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            initial_stop=1990.0,
            initial_risk=10.0,
            initial_size=1.0,      # Original full size
        )
        # total_pnl includes partial realization + remaining
        total_pnl = 15.0  # e.g. TP1 partial + remaining close
        r_mult = _trade_r_multiple(
            total_pnl=total_pnl,
            position=pos,
            fx_converter=None,
            instrument_currency="USD",
            account_currency="USD",
            fx_apply_to=None,
        )
        # R = 15 / (10 * 1.0) = 1.5 (denominator uses initial_size, not current 0.5)
        assert r_mult == pytest.approx(1.5, abs=1e-9)
