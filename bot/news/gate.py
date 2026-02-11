from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from bot.news.calendar_provider import Event


def blocking_events(now: datetime, events: list[Event], block_minutes: int) -> list[Event]:
    window = timedelta(minutes=block_minutes)
    return [event for event in events if abs(event.time - now) <= window]


def is_blocked(now: datetime, events: list[Event], block_minutes: int = 60) -> bool:
    return len(blocking_events(now, events, block_minutes)) > 0


def should_cancel_pending(order: dict[str, Any], now: datetime, events: list[Event], block_minutes: int = 60) -> bool:
    if order.get("status", "PENDING") != "PENDING":
        return False
    return is_blocked(now, events, block_minutes=block_minutes)

