"""Tests for session_filter module: session matching and blocking."""

from __future__ import annotations

from datetime import datetime, time, timezone

import pytest

from bot.strategy.session_filter import (
    SessionMatch,
    SessionWindow,
    build_session_windows,
    match_session,
)


# ---------------------------------------------------------------------------
# SessionWindow
# ---------------------------------------------------------------------------
class TestSessionWindow:
    def test_crosses_midnight_false(self):
        w = SessionWindow(
            name="LONDON", utc_start=time(7, 0), utc_end=time(16, 30),
            threshold_adjust=-3.0, risk_mult=1.0, weekdays=frozenset(range(5)),
        )
        assert w.crosses_midnight is False

    def test_crosses_midnight_true(self):
        w = SessionWindow(
            name="ASIA_PM", utc_start=time(22, 0), utc_end=time(2, 0),
            threshold_adjust=4.0, risk_mult=0.5, weekdays=frozenset(range(5)),
        )
        assert w.crosses_midnight is True

    def test_exact_boundary_midnight(self):
        w = SessionWindow(
            name="WRAP", utc_start=time(23, 0), utc_end=time(23, 0),
            threshold_adjust=0.0, risk_mult=1.0, weekdays=frozenset(range(5)),
        )
        # end == start → crosses_midnight True
        assert w.crosses_midnight is True


# ---------------------------------------------------------------------------
# match_session — basic matching
# ---------------------------------------------------------------------------
class TestMatchSession:
    LONDON = SessionWindow(
        name="LONDON", utc_start=time(7, 0), utc_end=time(11, 0),
        threshold_adjust=-3.0, risk_mult=1.0, weekdays=frozenset(range(5)),
    )
    NY = SessionWindow(
        name="NY", utc_start=time(13, 0), utc_end=time(20, 0),
        threshold_adjust=-2.0, risk_mult=1.0, weekdays=frozenset(range(5)),
    )
    ASIA_PM = SessionWindow(
        name="ASIA", utc_start=time(22, 0), utc_end=time(3, 0),
        threshold_adjust=4.0, risk_mult=0.5, weekdays=frozenset(range(5)),
    )

    def test_no_windows(self):
        result = match_session(datetime(2025, 1, 6, 8, 0, tzinfo=timezone.utc), [])
        assert result.matched is False
        assert result.session_name == "OFF_SESSION"
        assert result.blocked is False

    def test_match_london(self):
        now = datetime(2025, 1, 6, 9, 30, tzinfo=timezone.utc)  # Monday 09:30
        result = match_session(now, [self.LONDON, self.NY, self.ASIA_PM])
        assert result.matched is True
        assert result.session_name == "LONDON"
        assert result.threshold_adjust == -3.0
        assert result.risk_mult == 1.0

    def test_match_ny(self):
        now = datetime(2025, 1, 6, 15, 0, tzinfo=timezone.utc)  # Monday 15:00
        result = match_session(now, [self.LONDON, self.NY, self.ASIA_PM])
        assert result.matched is True
        assert result.session_name == "NY"

    def test_match_asia_crosses_midnight(self):
        now = datetime(2025, 1, 6, 23, 30, tzinfo=timezone.utc)  # Monday 23:30
        result = match_session(now, [self.LONDON, self.NY, self.ASIA_PM])
        assert result.matched is True
        assert result.session_name == "ASIA"
        assert result.risk_mult == 0.5

    def test_match_asia_after_midnight(self):
        now = datetime(2025, 1, 7, 1, 0, tzinfo=timezone.utc)  # Tuesday 01:00
        result = match_session(now, [self.LONDON, self.NY, self.ASIA_PM])
        assert result.matched is True
        assert result.session_name == "ASIA"

    def test_no_match_gap(self):
        now = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc)  # Monday 12:00 = gap
        result = match_session(now, [self.LONDON, self.NY])
        assert result.matched is False
        assert result.blocked is False

    def test_blocked_outside(self):
        now = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc)
        result = match_session(now, [self.LONDON, self.NY], block_outside=True)
        assert result.matched is False
        assert result.blocked is True

    def test_weekday_filter(self):
        # Saturday = weekday 5, not in range(5)
        now = datetime(2025, 1, 11, 9, 0, tzinfo=timezone.utc)  # Saturday 09:00
        result = match_session(now, [self.LONDON])
        assert result.matched is False

    def test_first_match_wins(self):
        overlap = SessionWindow(
            name="OVERLAP", utc_start=time(9, 0), utc_end=time(12, 0),
            threshold_adjust=-5.0, risk_mult=1.5, weekdays=frozenset(range(5)),
        )
        now = datetime(2025, 1, 6, 9, 30, tzinfo=timezone.utc)
        result = match_session(now, [self.LONDON, overlap])
        assert result.session_name == "LONDON"  # first wins

    def test_boundary_start_inclusive(self):
        now = datetime(2025, 1, 6, 7, 0, tzinfo=timezone.utc)  # exactly at start
        result = match_session(now, [self.LONDON])
        assert result.matched is True
        assert result.session_name == "LONDON"

    def test_boundary_end_exclusive(self):
        now = datetime(2025, 1, 6, 11, 0, tzinfo=timezone.utc)  # exactly at end
        result = match_session(now, [self.LONDON])
        assert result.matched is False


# ---------------------------------------------------------------------------
# build_session_windows
# ---------------------------------------------------------------------------
class TestBuildSessionWindows:
    def test_disabled(self):
        class Cfg:
            enabled = False
            sessions = []
        windows = build_session_windows(Cfg())
        assert windows == []

    def test_enabled(self):
        class SessCfg:
            name = "LONDON"
            utc_start = "07:00"
            utc_end = "11:00"
            threshold_adjust = -3.0
            risk_mult = 1.0
            weekdays = list(range(5))

        class Cfg:
            enabled = True
            sessions = [SessCfg()]

        windows = build_session_windows(Cfg())
        assert len(windows) == 1
        assert windows[0].name == "LONDON"
        assert windows[0].utc_start == time(7, 0)
        assert windows[0].utc_end == time(11, 0)
        assert windows[0].weekdays == frozenset(range(5))


# ---------------------------------------------------------------------------
# SessionMatch.no_match
# ---------------------------------------------------------------------------
class TestSessionMatchNoMatch:
    def test_default(self):
        m = SessionMatch.no_match()
        assert m.matched is False
        assert m.blocked is False
        assert m.risk_mult == 1.0
        assert m.threshold_adjust == 0.0

    def test_blocked(self):
        m = SessionMatch.no_match(blocked=True)
        assert m.blocked is True
