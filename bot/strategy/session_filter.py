"""Session-aware entry filter for PAPER/LIVE mode.

Different trading sessions (London, NY, Asia) have different
volatility, spread, and opportunity profiles. This module:

• Maps the current UTC time to a named session
• Provides per-session threshold adjustments and risk multipliers
• Optionally blocks trades outside defined session windows
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any

LOGGER = logging.getLogger("trading_bot")


@dataclass(slots=True, frozen=True)
class SessionWindow:
    """Runtime representation of a single session window."""
    name: str
    utc_start: time
    utc_end: time
    threshold_adjust: float
    risk_mult: float
    weekdays: frozenset[int]

    @property
    def crosses_midnight(self) -> bool:
        return self.utc_end <= self.utc_start


@dataclass(slots=True)
class SessionMatch:
    """Result of matching current time to a session."""
    matched: bool
    session_name: str
    threshold_adjust: float
    risk_mult: float
    blocked: bool  # True if outside all sessions AND block_outside enabled

    @staticmethod
    def no_match(*, blocked: bool = False) -> "SessionMatch":
        return SessionMatch(
            matched=False,
            session_name="OFF_SESSION",
            threshold_adjust=0.0,
            risk_mult=1.0,
            blocked=blocked,
        )


def _parse_time(s: str) -> time:
    """Parse 'HH:MM' to time object."""
    parts = s.strip().split(":")
    return time(int(parts[0]), int(parts[1]))


def build_session_windows(config: Any) -> list[SessionWindow]:
    """Convert SessionFilterConfig → list of frozen SessionWindow objects."""
    if not getattr(config, "enabled", False):
        return []
    windows: list[SessionWindow] = []
    for sess in getattr(config, "sessions", []):
        windows.append(SessionWindow(
            name=sess.name,
            utc_start=_parse_time(sess.utc_start),
            utc_end=_parse_time(sess.utc_end),
            threshold_adjust=sess.threshold_adjust,
            risk_mult=sess.risk_mult,
            weekdays=frozenset(sess.weekdays),
        ))
    return windows


def match_session(
    now_utc: datetime,
    windows: list[SessionWindow],
    *,
    block_outside: bool = False,
) -> SessionMatch:
    """Find which session window the current UTC time belongs to.

    Returns the *first* matching window (order matters for priority).
    If no window matches, returns SessionMatch.no_match().
    """
    if not windows:
        return SessionMatch.no_match()

    current_time = now_utc.time()
    weekday = now_utc.weekday()

    for w in windows:
        if weekday not in w.weekdays:
            continue
        if w.crosses_midnight:
            in_window = current_time >= w.utc_start or current_time < w.utc_end
        else:
            in_window = w.utc_start <= current_time < w.utc_end
        if in_window:
            return SessionMatch(
                matched=True,
                session_name=w.name,
                threshold_adjust=w.threshold_adjust,
                risk_mult=w.risk_mult,
                blocked=False,
            )

    return SessionMatch.no_match(blocked=block_outside)
