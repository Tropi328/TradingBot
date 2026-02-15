from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable

from bot.data.candles import Candle
from bot.news.calendar_provider import Event


@dataclass(slots=True)
class DailyGateResult:
    bias: str
    reasons: list[str] = field(default_factory=list)
    allowed_strategies: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _DailyBar:
    day: date
    open: float
    high: float
    low: float
    close: float


def _to_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _parse_hhmm(value: str | None) -> time | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    parts = raw.split(":")
    if len(parts) != 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except (TypeError, ValueError):
        return None
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return time(hour=hour, minute=minute)


class DailyGateProvider:
    def __init__(
        self,
        *,
        mode: str = "off",
        ema_fast: int = 20,
        ema_slow: int = 50,
        thr: float = 0.001,
        atr_period: int = 14,
        vol_max: float = 0.02,
        max_spread: float | None = None,
        pre_minutes: int = 30,
        post_minutes: int = 30,
        rollover_start_utc: str | None = None,
        rollover_end_utc: str | None = None,
        allowed_strategies: Iterable[str] | None = None,
        events: Iterable[Event] | None = None,
    ) -> None:
        self.mode = str(mode).strip().lower()
        self.ema_fast = max(1, int(ema_fast))
        self.ema_slow = max(self.ema_fast + 1, int(ema_slow))
        self.thr = max(0.0, float(thr))
        self.atr_period = max(1, int(atr_period))
        self.vol_max = max(0.0000001, float(vol_max))
        self.max_spread = float(max_spread) if max_spread is not None else None
        self.pre_minutes = max(0, int(pre_minutes))
        self.post_minutes = max(0, int(post_minutes))
        self.rollover_start = _parse_hhmm(rollover_start_utc)
        self.rollover_end = _parse_hhmm(rollover_end_utc)
        self.allowed_strategies = [
            str(item).strip().upper()
            for item in (allowed_strategies or [])
            if str(item).strip()
        ]
        self._events: list[Event] = sorted(list(events or []), key=lambda item: _to_utc(item.time))
        self._news_windows: list[tuple[datetime, datetime]] = []
        self._news_window_starts: list[datetime] = []
        self._rebuild_event_windows()
        self._bias_by_day: dict[date, str] = {}
        self._reasons_by_day: dict[date, list[str]] = {}
        self._last_refresh_day: date | None = None

    @property
    def enabled(self) -> bool:
        return self.mode in {"trend", "trend_vol_news"}

    def set_events(self, events: Iterable[Event]) -> None:
        self._events = sorted(list(events), key=lambda item: _to_utc(item.time))
        self._rebuild_event_windows()

    def refresh_if_needed(self, *, now: datetime, candles: list[Candle]) -> None:
        if not self.enabled:
            return
        current_day = _to_utc(now).date()
        if self._last_refresh_day == current_day:
            return
        self.refresh_from_candles(candles)
        self._last_refresh_day = current_day

    def refresh_from_candles(self, candles: list[Candle]) -> None:
        self._bias_by_day.clear()
        self._reasons_by_day.clear()
        if not candles:
            return

        daily = self._aggregate_daily(candles)
        if not daily:
            return
        days = [bar.day for bar in daily]
        for day in days:
            self._bias_by_day[day] = "FLAT"
            self._reasons_by_day[day] = ["INSUFFICIENT_DAILY_HISTORY"]

        alpha_fast = 2.0 / (self.ema_fast + 1.0)
        alpha_slow = 2.0 / (self.ema_slow + 1.0)
        ema_fast: float | None = None
        ema_slow: float | None = None
        prev_close: float | None = None
        atr_window: deque[float] = deque(maxlen=self.atr_period)
        atr_pct: float | None = None

        for idx, bar in enumerate(daily):
            close_value = float(bar.close)
            if ema_fast is None:
                ema_fast = close_value
            else:
                ema_fast = (alpha_fast * close_value) + ((1.0 - alpha_fast) * ema_fast)
            if ema_slow is None:
                ema_slow = close_value
            else:
                ema_slow = (alpha_slow * close_value) + ((1.0 - alpha_slow) * ema_slow)

            if prev_close is None:
                tr = max(0.0, float(bar.high) - float(bar.low))
            else:
                tr = max(
                    float(bar.high) - float(bar.low),
                    abs(float(bar.high) - prev_close),
                    abs(float(bar.low) - prev_close),
                )
            atr_window.append(max(0.0, tr))
            prev_close = close_value
            if close_value > 0 and atr_window:
                atr_pct = (sum(atr_window) / len(atr_window)) / close_value
            else:
                atr_pct = None

            if idx + 1 >= len(daily):
                continue
            next_day = daily[idx + 1].day
            bias, reasons = self._classify(ema_fast=ema_fast, ema_slow=ema_slow, atr_pct=atr_pct)
            self._bias_by_day[next_day] = bias
            self._reasons_by_day[next_day] = reasons

    def evaluate(
        self,
        *,
        ts: datetime,
        symbol: str,
        spread: float | None = None,
    ) -> DailyGateResult:
        if not self.enabled:
            return DailyGateResult(bias="FLAT", reasons=["DAILY_GATE_DISABLED"], allowed_strategies=[])

        ts_utc = _to_utc(ts)
        day = ts_utc.date()
        base_bias = self._bias_by_day.get(day, "FLAT")
        reasons = list(self._reasons_by_day.get(day, ["INSUFFICIENT_DAILY_HISTORY"]))
        effective_bias = base_bias

        if self.mode == "trend_vol_news":
            if self.max_spread is not None and spread is not None and spread > self.max_spread:
                effective_bias = "FLAT"
                reasons.append("SPREAD_TOO_WIDE")
            if self._is_news_window(ts_utc):
                effective_bias = "FLAT"
                reasons.append("NEWS_WINDOW")
            if self._is_rollover_window(ts_utc):
                effective_bias = "FLAT"
                reasons.append("ROLLOVER_WINDOW")

        dedup_reasons = list(dict.fromkeys(reasons))
        allowed: list[str] = []
        if effective_bias != "FLAT" and self.allowed_strategies:
            allowed = list(self.allowed_strategies)
        return DailyGateResult(
            bias=effective_bias,
            reasons=dedup_reasons,
            allowed_strategies=allowed,
        )

    def _classify(self, *, ema_fast: float | None, ema_slow: float | None, atr_pct: float | None) -> tuple[str, list[str]]:
        if ema_fast is None or ema_slow is None:
            return "FLAT", ["INSUFFICIENT_DAILY_HISTORY"]
        if ema_fast > (ema_slow * (1.0 + self.thr)):
            bias = "LONG"
            reasons: list[str] = []
        elif ema_fast < (ema_slow * (1.0 - self.thr)):
            bias = "SHORT"
            reasons = []
        else:
            bias = "FLAT"
            reasons = ["TREND_NEUTRAL"]

        if self.mode == "trend_vol_news" and atr_pct is not None and atr_pct > self.vol_max:
            return "FLAT", ["VOL_TOO_HIGH"]
        return bias, reasons

    def _aggregate_daily(self, candles: list[Candle]) -> list[_DailyBar]:
        out: list[_DailyBar] = []
        current: _DailyBar | None = None
        for candle in sorted(candles, key=lambda item: _to_utc(item.timestamp)):
            ts_utc = _to_utc(candle.timestamp)
            day = ts_utc.date()
            if current is None or current.day != day:
                if current is not None:
                    out.append(current)
                current = _DailyBar(
                    day=day,
                    open=float(candle.open),
                    high=float(candle.high),
                    low=float(candle.low),
                    close=float(candle.close),
                )
                continue
            current.high = max(current.high, float(candle.high))
            current.low = min(current.low, float(candle.low))
            current.close = float(candle.close)
        if current is not None:
            out.append(current)
        return out

    def _is_news_window(self, ts_utc: datetime) -> bool:
        if not self._news_windows:
            return False
        idx = bisect_right(self._news_window_starts, ts_utc) - 1
        if idx < 0:
            return False
        start, end = self._news_windows[idx]
        return start <= ts_utc <= end

    def _is_rollover_window(self, ts_utc: datetime) -> bool:
        if self.rollover_start is None or self.rollover_end is None:
            return False
        current = ts_utc.time()
        start = self.rollover_start
        end = self.rollover_end
        if start <= end:
            return start <= current <= end
        return current >= start or current <= end

    def _rebuild_event_windows(self) -> None:
        if not self._events:
            self._news_windows = []
            self._news_window_starts = []
            return
        windows: list[tuple[datetime, datetime]] = []
        for event in self._events:
            event_utc = _to_utc(event.time)
            start = event_utc - timedelta(minutes=self.pre_minutes)
            end = event_utc + timedelta(minutes=self.post_minutes)
            windows.append((start, end))
        windows.sort(key=lambda item: item[0])
        merged: list[tuple[datetime, datetime]] = []
        for start, end in windows:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
                continue
            prev_start, prev_end = merged[-1]
            if end > prev_end:
                merged[-1] = (prev_start, end)
        self._news_windows = merged
        self._news_window_starts = [start for start, _ in merged]
