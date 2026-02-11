from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _get_zone(timezone_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(
            f"Timezone '{timezone_name}' is not available. "
            "Install tzdata in your environment: pip install tzdata"
        ) from exc


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def warsaw_now() -> datetime:
    return utc_now().astimezone(_get_zone("Europe/Warsaw"))


def to_timezone(dt: datetime, timezone_name: str) -> datetime:
    return dt.astimezone(_get_zone(timezone_name))


def trading_day(dt: datetime, timezone_name: str = "Europe/Warsaw") -> date:
    return to_timezone(dt, timezone_name).date()


def is_trading_weekday(dt: datetime, timezone_name: str = "Europe/Warsaw") -> bool:
    local_dt = to_timezone(dt, timezone_name)
    return local_dt.weekday() < 5


def next_bar_open(last_bar_time: datetime, timeframe_minutes: int) -> datetime:
    return last_bar_time + timedelta(minutes=timeframe_minutes)


def has_new_bar(previous_last: datetime | None, current_last: datetime | None) -> bool:
    if current_last is None:
        return False
    if previous_last is None:
        return True
    return current_last > previous_last


def timeframe_to_minutes(timeframe: str) -> int:
    normalized = timeframe.strip().upper()
    mapping = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "H1": 60,
        "H4": 240,
        "D1": 1440,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported timeframe {timeframe}")
    return mapping[normalized]


def expected_closed_candle_utc(
    now_utc: datetime,
    timeframe_minutes: int,
    *,
    close_grace_seconds: int,
) -> datetime:
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    anchor = now_utc.astimezone(timezone.utc) - timedelta(seconds=max(0, close_grace_seconds))
    interval_seconds = timeframe_minutes * 60
    closed_epoch = int(anchor.timestamp()) // interval_seconds * interval_seconds
    return datetime.fromtimestamp(closed_epoch, tz=timezone.utc)


def should_poll_closed_candle(
    *,
    now_utc: datetime,
    timeframe: str,
    last_processed_closed_ts: datetime | None,
    last_attempt_target_ts: datetime | None,
    last_attempt_at: datetime | None,
    close_grace_seconds: int,
    retry_seconds: int,
) -> tuple[bool, datetime]:
    target = expected_closed_candle_utc(
        now_utc,
        timeframe_to_minutes(timeframe),
        close_grace_seconds=close_grace_seconds,
    )
    if last_processed_closed_ts is not None and last_processed_closed_ts >= target:
        return False, target
    if (
        last_attempt_target_ts is not None
        and last_attempt_target_ts == target
        and last_attempt_at is not None
        and (now_utc - last_attempt_at).total_seconds() < max(1, retry_seconds)
    ):
        return False, target
    return True, target
