from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import requests

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Event:
    event_id: str
    title: str
    currency: str
    impact: str
    time: datetime
    category: str = "macro"
    source: str = "unknown"


class CalendarProvider(Protocol):
    def get_high_impact_events(self, start_dt: datetime, end_dt: datetime) -> list[Event]:
        ...


def _parse_dt(value: str) -> datetime:
    normalized = value.replace(" ", "T")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_relevant_event(event: Event) -> bool:
    high_impact = event.impact.upper() in {"HIGH", "3", "HIGH_IMPACT"}
    usd_related = "USD" in event.currency.upper() or "USD" in event.title.upper()
    macro_related = event.category.lower() in {"macro", "central_bank", "rates", "inflation"}
    return high_impact and (usd_related or macro_related)


class DummyCalendarProvider:
    def __init__(self, json_path: str | Path):
        self.json_path = Path(json_path)

    def get_high_impact_events(self, start_dt: datetime, end_dt: datetime) -> list[Event]:
        if not self.json_path.exists():
            return []
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        events: list[Event] = []
        for item in payload:
            try:
                event = Event(
                    event_id=str(item.get("id", "")),
                    title=str(item.get("title", "Untitled")),
                    currency=str(item.get("currency", "USD")),
                    impact=str(item.get("impact", "HIGH")),
                    time=_parse_dt(str(item["time"])),
                    category=str(item.get("category", "macro")),
                    source="dummy_json",
                )
            except (KeyError, ValueError):
                continue
            if start_dt <= event.time <= end_dt and _is_relevant_event(event):
                events.append(event)
        return sorted(events, key=lambda x: x.time)


class HttpCalendarProvider:
    """
    Generic HTTP calendar provider.

    Expected response: list[dict] or {"events": list[dict]}.
    Fields used (fallback keys supported):
    - time: "time" | "datetime" | "date"
    - impact: "impact" | "importance"
    - currency: "currency"
    - title: "title" | "event"
    """

    def __init__(
        self,
        *,
        url: str,
        token: str | None = None,
        timeout_seconds: int = 10,
        cache_ttl_seconds: int = 300,
    ):
        self.url = url
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache_events: list[Event] = []
        self._cache_expiry: datetime = datetime.min.replace(tzinfo=timezone.utc)

    def get_high_impact_events(self, start_dt: datetime, end_dt: datetime) -> list[Event]:
        now = datetime.now(timezone.utc)
        if now >= self._cache_expiry:
            self._cache_events = self._fetch_events()
            self._cache_expiry = now + timedelta(seconds=self.cache_ttl_seconds)
        return [
            event
            for event in self._cache_events
            if start_dt <= event.time <= end_dt and _is_relevant_event(event)
        ]

    def _fetch_events(self) -> list[Event]:
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        response = requests.get(self.url, headers=headers, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        raw_events = payload.get("events", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_events, list):
            return []

        events: list[Event] = []
        for index, item in enumerate(raw_events):
            if not isinstance(item, dict):
                continue
            ts = item.get("time") or item.get("datetime") or item.get("date")
            if ts is None:
                continue
            try:
                event = Event(
                    event_id=str(item.get("id", index)),
                    title=str(item.get("title") or item.get("event") or "Untitled"),
                    currency=str(item.get("currency", "USD")),
                    impact=str(item.get("impact") or item.get("importance") or "HIGH"),
                    time=_parse_dt(str(ts)),
                    category=str(item.get("category", "macro")),
                    source="http",
                )
            except ValueError:
                continue
            if _is_relevant_event(event):
                events.append(event)
        return sorted(events, key=lambda x: x.time)


def build_calendar_provider(
    *,
    provider_name: str,
    dummy_file: str | Path,
    http_url: str | None,
    http_token: str | None,
    timeout_seconds: int,
    cache_ttl_seconds: int,
) -> CalendarProvider:
    if provider_name.lower() == "http" and http_url:
        LOGGER.info("Using HTTP calendar provider: %s", http_url)
        return HttpCalendarProvider(
            url=http_url,
            token=http_token,
            timeout_seconds=timeout_seconds,
            cache_ttl_seconds=cache_ttl_seconds,
        )
    LOGGER.info("Using dummy calendar provider: %s", dummy_file)
    return DummyCalendarProvider(dummy_file)

