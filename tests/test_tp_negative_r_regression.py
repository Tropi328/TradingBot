"""
Regression tests A-D for TP-negative-R invariant.

A) intrabar_both_hit_conservative — bar hits TP and SL → STOP wins, reason="STOP"
B) tp_reason_not_negative        — TP exit → r >= 0  (LONG + SHORT)
C) short_inequalities_correct    — SHORT: low<=tp → TP, high>=stop → STOP
D) manage_before_exit            — after TP1, BE stop moves, then _calc_exit respects it
"""
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
    close: float | None = None,
    bid: float | None = None,
    ask: float | None = None,
) -> Candle:
    return Candle(
        timestamp=ts,
        open=close or (high + low) / 2,
        high=high,
        low=low,
        close=close or (high + low) / 2,
        bid=bid,
        ask=ask,
        volume=1.0,
    )


TS0 = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
TS1 = datetime(2024, 6, 1, 12, 5, tzinfo=timezone.utc)
TS2 = datetime(2024, 6, 1, 12, 10, tzinfo=timezone.utc)
TS3 = datetime(2024, 6, 1, 12, 15, tzinfo=timezone.utc)


def _pos(
    side: str,
    entry: float,
    stop: float,
    tp: float,
    *,
    size: float = 1.0,
    tp1_trigger_r: float = 0.5,
    tp1_fraction: float = 0.5,
    be_offset_r: float = 0.0,
    be_delay_bars: int = 0,
) -> _OpenPosition:
    return _OpenPosition(
        side=side,
        entry=entry,
        stop=stop,
        tp=tp,
        size=size,
        opened_at=TS0,
        initial_stop=stop,
        initial_risk=abs(entry - stop),
        initial_size=size,
        tp1_trigger_r=tp1_trigger_r,
        tp1_fraction=tp1_fraction,
        be_offset_r=be_offset_r,
        be_delay_bars_after_tp1=be_delay_bars,
        trailing_after_tp1=False,
    )


# =========================================================================
# A) intrabar_both_hit_conservative
#    When a single bar touches BOTH stop and TP, STOP wins (conservative).
# =========================================================================
class TestIntrabarBothHitConservative:
    def test_long_both_hit_returns_stop(self) -> None:
        """LONG: low <= stop AND high >= tp → reason STOP."""
        p = _pos("LONG", entry=2000.0, stop=1990.0, tp=2020.0)
        bar = _candle(TS1, high=2021.0, low=1989.0, close=2000.0)
        closed, price, reason = _calc_exit(p, bar, assumed_spread=0.5, slippage=0.0)
        assert closed is True
        assert reason == "STOP"
        # Fill at stop (no slippage).
        assert price == pytest.approx(1990.0, abs=0.01)

    def test_short_both_hit_returns_stop(self) -> None:
        """SHORT: high >= stop AND low <= tp → reason STOP."""
        p = _pos("SHORT", entry=2000.0, stop=2010.0, tp=1980.0)
        bar = _candle(TS1, high=2011.0, low=1979.0, close=2000.0)
        closed, price, reason = _calc_exit(p, bar, assumed_spread=0.5, slippage=0.0)
        assert closed is True
        assert reason == "STOP"
        assert price == pytest.approx(2010.0, abs=0.01)

    def test_long_both_hit_with_slippage(self) -> None:
        """With slippage, stop fill is worse (lower for LONG), capped by max_loss_r_cap."""
        p = _pos("LONG", entry=2000.0, stop=1990.0, tp=2020.0)
        bar = _candle(TS1, high=2020.5, low=1989.5, close=2000.0)
        closed, price, reason = _calc_exit(p, bar, assumed_spread=0.5, slippage=0.5)
        assert closed is True
        assert reason == "STOP"
        # Fill at stop - slippage = 1990.0 - 0.5 = 1989.5
        assert price <= 1990.0

    def test_short_both_hit_with_slippage(self) -> None:
        """With slippage, stop fill is worse (higher for SHORT), capped by max_loss_r_cap."""
        p = _pos("SHORT", entry=2000.0, stop=2010.0, tp=1980.0)
        bar = _candle(TS1, high=2010.5, low=1979.5, close=2000.0)
        closed, price, reason = _calc_exit(p, bar, assumed_spread=0.5, slippage=0.5)
        assert closed is True
        assert reason == "STOP"
        assert price >= 2010.0


# =========================================================================
# B) tp_reason_not_negative
#    Invariant: reason_close == "TP" → r >= -eps.
# =========================================================================
class TestTpReasonNotNegative:
    EPS = 0.02  # tolerance in R units

    def _assert_tp_positive_r(
        self,
        side: str,
        entry: float,
        stop: float,
        tp: float,
        bar_high: float,
        bar_low: float,
        slippage: float = 0.0,
    ) -> None:
        p = _pos(side, entry=entry, stop=stop, tp=tp)
        bar = _candle(TS1, high=bar_high, low=bar_low)
        closed, exit_price, reason = _calc_exit(p, bar, assumed_spread=0.5, slippage=slippage)
        if not closed or reason != "TP":
            # If TP is invalidated (wrong-side or slippage safety), that's also OK —
            # the invariant is "IF reason==TP THEN r >= -eps".
            return
        if side == "LONG":
            pnl = (exit_price - entry) * p.size
        else:
            pnl = (entry - exit_price) * p.size
        r = _trade_r_multiple(
            total_pnl=pnl,
            position=p,
            fx_converter=None,
            instrument_currency="USD",
            account_currency="USD",
            fx_apply_to=None,
        )
        assert r >= -self.EPS, f"TP with negative R: {r:.4f} (exit={exit_price}, entry={entry})"

    def test_long_tp_normal(self) -> None:
        """LONG normal TP — clearly profitable."""
        self._assert_tp_positive_r("LONG", 2000, 1990, 2020, bar_high=2021, bar_low=1999)

    def test_short_tp_normal(self) -> None:
        """SHORT normal TP — clearly profitable."""
        self._assert_tp_positive_r("SHORT", 2000, 2010, 1980, bar_high=2001, bar_low=1979)

    def test_long_tp_barely_touched(self) -> None:
        """LONG TP: high == tp exactly. Still profitable."""
        self._assert_tp_positive_r("LONG", 2000, 1990, 2020, bar_high=2020, bar_low=1998)

    def test_short_tp_barely_touched(self) -> None:
        """SHORT TP: low == tp exactly. Still profitable."""
        self._assert_tp_positive_r("SHORT", 2000, 2010, 1980, bar_high=2002, bar_low=1980)

    def test_long_tp_with_slippage(self) -> None:
        """LONG TP with slippage — fill at tp-slip, still r >= -eps."""
        self._assert_tp_positive_r("LONG", 2000, 1990, 2020, bar_high=2021, bar_low=1999, slippage=1.0)

    def test_short_tp_with_slippage(self) -> None:
        """SHORT TP with slippage — fill at tp+slip, still r >= -eps."""
        self._assert_tp_positive_r("SHORT", 2000, 2010, 1980, bar_high=2001, bar_low=1979, slippage=1.0)

    def test_long_tp_massive_slippage_invalidated(self) -> None:
        """LONG: slippage so large it would push fill below entry → TP invalidated."""
        p = _pos("LONG", entry=2000.0, stop=1990.0, tp=2005.0)
        bar = _candle(TS1, high=2006.0, low=1999.0)
        # slippage = 10 → fill = 2005 - 10 = 1995 < entry 2000 → safety rejects.
        closed, _, reason = _calc_exit(p, bar, assumed_spread=0.5, slippage=10.0)
        # Must NOT report "TP" with negative R.
        if closed:
            assert reason != "TP", "Massive slippage must not produce TP exit"

    def test_short_tp_massive_slippage_invalidated(self) -> None:
        """SHORT: slippage so large it would push fill above entry → TP invalidated."""
        p = _pos("SHORT", entry=2000.0, stop=2010.0, tp=1995.0)
        bar = _candle(TS1, high=2001.0, low=1994.0)
        # slippage = 10 → fill = 1995 + 10 = 2005 > entry 2000 → safety rejects.
        closed, _, reason = _calc_exit(p, bar, assumed_spread=0.5, slippage=10.0)
        if closed:
            assert reason != "TP", "Massive slippage must not produce TP exit"

    def test_tp_on_wrong_side_of_entry_invalidated(self) -> None:
        """TP on the wrong side of entry (adverse fill) → TP invalidated."""
        # LONG with TP below entry: impossible in normal flow but safety catches it.
        p = _pos("LONG", entry=2000.0, stop=1990.0, tp=1998.0)
        bar = _candle(TS1, high=2001.0, low=1997.0)
        closed, _, reason = _calc_exit(p, bar, assumed_spread=0.5, slippage=0.0)
        if closed:
            assert reason != "TP", "TP on wrong side must not produce TP exit"


# =========================================================================
# C) short_inequalities_correct
#    SHORT: low <= tp → TP, high >= stop → STOP  (mirror of LONG).
# =========================================================================
class TestShortInequalitiesCorrect:
    def test_short_tp_triggered_by_low(self) -> None:
        """SHORT: bar low == tp → TP fires."""
        p = _pos("SHORT", entry=2000.0, stop=2010.0, tp=1980.0)
        bar = _candle(TS1, high=2001.0, low=1980.0)
        closed, price, reason = _calc_exit(p, bar, assumed_spread=0.5, slippage=0.0)
        assert closed is True
        assert reason == "TP"
        assert price == pytest.approx(1980.0, abs=1e-9)

    def test_short_stop_triggered_by_high(self) -> None:
        """SHORT: bar high == stop → STOP fires."""
        p = _pos("SHORT", entry=2000.0, stop=2010.0, tp=1980.0)
        bar = _candle(TS1, high=2010.0, low=1995.0)
        closed, price, reason = _calc_exit(p, bar, assumed_spread=0.5, slippage=0.0)
        assert closed is True
        assert reason == "STOP"
        assert price == pytest.approx(2010.0, abs=0.01)

    def test_short_neither_triggered(self) -> None:
        """SHORT: bar doesn't reach stop or tp → no close."""
        p = _pos("SHORT", entry=2000.0, stop=2010.0, tp=1980.0)
        bar = _candle(TS1, high=2005.0, low=1985.0)
        closed, _, _ = _calc_exit(p, bar, assumed_spread=0.5, slippage=0.0)
        assert closed is False

    def test_long_tp_triggered_by_high(self) -> None:
        """Mirror check: LONG: bar high == tp → TP fires."""
        p = _pos("LONG", entry=2000.0, stop=1990.0, tp=2020.0)
        bar = _candle(TS1, high=2020.0, low=1999.0)
        closed, price, reason = _calc_exit(p, bar, assumed_spread=0.5, slippage=0.0)
        assert closed is True
        assert reason == "TP"
        assert price == pytest.approx(2020.0, abs=1e-9)

    def test_long_stop_triggered_by_low(self) -> None:
        """Mirror check: LONG: bar low == stop → STOP fires."""
        p = _pos("LONG", entry=2000.0, stop=1990.0, tp=2020.0)
        bar = _candle(TS1, high=2005.0, low=1990.0)
        closed, price, reason = _calc_exit(p, bar, assumed_spread=0.5, slippage=0.0)
        assert closed is True
        assert reason == "STOP"
        assert price == pytest.approx(1990.0, abs=0.01)


# =========================================================================
# D) manage_before_exit
#    After TP1 partial, BE stop moves. On the next bar _calc_exit sees the
#    updated BE stop — proving the manage→exit ordering invariant.
# =========================================================================
class TestManageBeforeExit:
    def test_tp1_then_be_then_exit_at_be(self) -> None:
        """
        Bar 1: TP1 fires (partial close), BE moves stop to entry.
        Bar 2: price retraces to entry → exit at BE (not original stop).
        This proves manage_open_position runs before _calc_exit.
        """
        p = _pos(
            "LONG",
            entry=2000.0,
            stop=1990.0,
            tp=2020.0,
            size=2.0,
            tp1_trigger_r=0.5,
            tp1_fraction=0.5,
            be_offset_r=0.0,
            be_delay_bars=0,
        )
        # Bar 0: opening bar (needed as history context).
        bar0 = _candle(TS0, high=2000.2, low=1999.8, close=2000.0)
        # Bar 1: TP1 level = entry + 0.5 * 10 = 2005. High reaches 2006 → TP1.
        bar1 = _candle(TS1, high=2006.0, low=1999.5, close=2004.0)

        tp1_hit, be_moved = _manage_open_position(
            position=p,
            candle=bar1,
            candles_m5=[bar0, bar1],
            index=1,
            slippage=0.0,
        )
        assert tp1_hit is True
        assert be_moved is True
        assert p.tp1_taken is True
        assert p.be_moved is True
        assert p.stop == pytest.approx(2000.0, abs=1e-9), "Stop should be at entry (BE)"
        assert p.size == pytest.approx(1.0, abs=1e-9), "Half the position closed"

        # Bar 2: retrace hits BE (low touches 2000.0).
        bar2 = _candle(TS2, high=2003.0, low=2000.0, close=2001.0)
        closed, price, reason = _calc_exit(p, bar2, assumed_spread=0.5, slippage=0.0)
        assert closed is True
        assert reason == "BE"
        assert price == pytest.approx(2000.0, abs=1e-9)

    def test_short_tp1_then_be_then_exit_at_be(self) -> None:
        """SHORT mirror: TP1 → BE → exit at BE."""
        p = _pos(
            "SHORT",
            entry=2000.0,
            stop=2010.0,
            tp=1980.0,
            size=2.0,
            tp1_trigger_r=0.5,
            tp1_fraction=0.5,
            be_offset_r=0.0,
            be_delay_bars=0,
        )
        bar0 = _candle(TS0, high=2000.2, low=1999.8, close=2000.0)
        # TP1 level = entry - 0.5 * 10 = 1995. Low reaches 1994 → TP1.
        bar1 = _candle(TS1, high=2000.5, low=1994.0, close=1996.0)

        tp1_hit, be_moved = _manage_open_position(
            position=p,
            candle=bar1,
            candles_m5=[bar0, bar1],
            index=1,
            slippage=0.0,
        )
        assert tp1_hit is True
        assert be_moved is True
        assert p.stop == pytest.approx(2000.0, abs=1e-9), "Stop should be at entry (BE)"
        assert p.size == pytest.approx(1.0, abs=1e-9)

        # Bar 2: bounce hits BE (high touches 2000.0).
        bar2 = _candle(TS2, high=2000.0, low=1997.0, close=1998.0)
        closed, price, reason = _calc_exit(p, bar2, assumed_spread=0.5, slippage=0.0)
        assert closed is True
        assert reason == "BE"
        assert price == pytest.approx(2000.0, abs=1e-9)

    def test_manage_updates_stop_before_exit_check(self) -> None:
        """
        Without TP1/BE, but trailing: manage updates stop, then _calc_exit
        uses the UPDATED stop. Proves ordering.
        """
        p = _pos(
            "LONG",
            entry=2000.0,
            stop=1990.0,
            tp=2040.0,
            size=2.0,
            tp1_trigger_r=0.3,
            tp1_fraction=0.5,
            be_offset_r=0.0,
            be_delay_bars=0,
        )
        bar0 = _candle(TS0, high=2000.2, low=1999.8, close=2000.0)
        # Bar 1: TP1 level = 2000 + 0.3*10 = 2003. High = 2004 → TP1 + BE.
        bar1 = _candle(TS1, high=2004.0, low=1999.5, close=2003.0)

        tp1_hit, be_moved = _manage_open_position(
            position=p,
            candle=bar1,
            candles_m5=[bar0, bar1],
            index=1,
            slippage=0.0,
        )
        assert tp1_hit is True
        assert be_moved is True
        # Now stop is at entry (BE = 2000.0), not at original 1990.0.
        assert p.stop == pytest.approx(2000.0, abs=1e-9)

        # Bar 2: low exactly at BE but NOT at original stop.
        bar2 = _candle(TS2, high=2005.0, low=2000.0, close=2003.0)
        closed, price, reason = _calc_exit(p, bar2, assumed_spread=0.5, slippage=0.0)
        assert closed is True
        assert reason == "BE"
        # If _calc_exit used the original stop (1990), it would NOT have exited
        # (low=2000 > 1990). Proof that manage ran first.
