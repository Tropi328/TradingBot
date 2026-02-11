from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(slots=True)
class TradingWindow:
    start: time
    end: time


def _parse_hhmm(raw: str) -> time:
    hour_raw, minute_raw = raw.split(":", 1)
    return time(hour=int(hour_raw), minute=int(minute_raw))


def parse_windows(raw_windows: list[str]) -> list[TradingWindow]:
    windows: list[TradingWindow] = []
    for raw in raw_windows:
        item = str(raw).strip()
        if not item or "-" not in item:
            continue
        start_raw, end_raw = item.split("-", 1)
        windows.append(TradingWindow(start=_parse_hhmm(start_raw.strip()), end=_parse_hhmm(end_raw.strip())))
    return windows


def is_schedule_open(now_utc: datetime, schedule: dict[str, Any] | None, default_timezone: str) -> bool:
    if not schedule:
        return True
    enabled = bool(schedule.get("enabled", True))
    if not enabled:
        return True
    timezone_name = str(schedule.get("timezone", default_timezone))
    try:
        local = now_utc.astimezone(ZoneInfo(timezone_name))
    except ZoneInfoNotFoundError:
        local = now_utc
    weekdays_raw = schedule.get("weekdays", [0, 1, 2, 3, 4])
    weekdays = {int(day) for day in weekdays_raw}
    if local.weekday() not in weekdays:
        return False
    windows = parse_windows(schedule.get("windows", []))
    if not windows:
        return True
    current = local.timetz().replace(tzinfo=None)
    for window in windows:
        if window.start <= current <= window.end:
            return True
    return False
