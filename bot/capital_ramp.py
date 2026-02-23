from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


START_EQUITY_PLN = 100.0
MONTHLY_TOPUP_PLN = 100.0
TARGET_EQUITY_PLN = 1200.0
TOPUP_DAY = 1

STOP_REASON_TARGET = "TARGET_REACHED"
STOP_REASON_YEAR_END = "YEAR_END"
EVENT_TOPUP = "TOPUP"
EVENT_STOP_TARGET = "STOP_TARGET"
EVENT_STOP_YEAR_END = "STOP_YEAR_END"


def _zone(timezone_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(
            f"Timezone '{timezone_name}' is not available. Install tzdata: pip install tzdata"
        ) from exc


def _first_day_next_month(local_day: date) -> date:
    if local_day.month == 12:
        return date(local_day.year + 1, 1, TOPUP_DAY)
    return date(local_day.year, local_day.month + 1, TOPUP_DAY)


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


@dataclass(slots=True)
class CapitalRampState:
    scope: str
    timezone_name: str
    start_ts_utc: datetime
    start_year_local: int
    start_equity: float = START_EQUITY_PLN
    monthly_topup: float = MONTHLY_TOPUP_PLN
    target_equity: float = TARGET_EQUITY_PLN
    pnl_baseline_at_start: float = 0.0
    topups_total: float = 0.0
    topups_count: int = 0
    next_topup_date_local: date | None = None
    last_topup_date_local: date | None = None
    stopped_reason: str | None = None
    stopped_at_utc: datetime | None = None
    updated_at_utc: datetime | None = None

    @property
    def year_end_local(self) -> date:
        return date(self.start_year_local, 12, 31)


@dataclass(slots=True)
class CapitalRampEvent:
    scope: str
    event_type: str
    event_ts_utc: datetime
    local_date: date
    amount: float
    model_equity: float
    payload: dict[str, object] = field(default_factory=dict)


class CapitalRampRuntime:
    def __init__(self, state: CapitalRampState) -> None:
        self.state = state
        self._tz = _zone(state.timezone_name)

    @classmethod
    def initialize(
        cls,
        *,
        scope: str,
        now_utc: datetime,
        timezone_name: str,
        current_closed_pnl: float,
    ) -> "CapitalRampRuntime":
        now = _as_utc(now_utc)
        local_day = now.astimezone(_zone(timezone_name)).date()
        state = CapitalRampState(
            scope=scope,
            timezone_name=timezone_name,
            start_ts_utc=now,
            start_year_local=local_day.year,
            start_equity=START_EQUITY_PLN,
            monthly_topup=MONTHLY_TOPUP_PLN,
            target_equity=TARGET_EQUITY_PLN,
            pnl_baseline_at_start=float(current_closed_pnl),
            next_topup_date_local=_first_day_next_month(local_day),
            updated_at_utc=now,
        )
        return cls(state)

    def _local_date(self, now_utc: datetime) -> date:
        return _as_utc(now_utc).astimezone(self._tz).date()

    def model_equity(self, *, current_closed_pnl: float) -> float:
        delta_realized = float(current_closed_pnl) - float(self.state.pnl_baseline_at_start)
        return float(self.state.start_equity) + float(self.state.topups_total) + delta_realized

    def _stop_event_if_needed(
        self,
        *,
        now_utc: datetime,
        current_closed_pnl: float,
    ) -> CapitalRampEvent | None:
        if self.state.stopped_reason:
            return None
        now = _as_utc(now_utc)
        local_day = self._local_date(now)
        model_eq = self.model_equity(current_closed_pnl=current_closed_pnl)
        reason: str | None = None
        event_type: str | None = None
        if model_eq >= float(self.state.target_equity):
            reason = STOP_REASON_TARGET
            event_type = EVENT_STOP_TARGET
        elif local_day > self.state.year_end_local:
            reason = STOP_REASON_YEAR_END
            event_type = EVENT_STOP_YEAR_END
        if reason is None or event_type is None:
            self.state.updated_at_utc = now
            return None
        self.state.stopped_reason = reason
        self.state.stopped_at_utc = now
        self.state.updated_at_utc = now
        return CapitalRampEvent(
            scope=self.state.scope,
            event_type=event_type,
            event_ts_utc=now,
            local_date=local_day,
            amount=0.0,
            model_equity=model_eq,
            payload={"stopped_reason": reason},
        )

    def maybe_apply_topup(
        self,
        *,
        now_utc: datetime,
        current_closed_pnl: float,
    ) -> tuple[CapitalRampEvent | None, CapitalRampEvent | None]:
        stop_event = self._stop_event_if_needed(
            now_utc=now_utc,
            current_closed_pnl=current_closed_pnl,
        )
        if self.state.stopped_reason:
            return None, stop_event

        now = _as_utc(now_utc)
        local_day = self._local_date(now)
        due_day = self.state.next_topup_date_local
        if due_day is None:
            due_day = _first_day_next_month(local_day)
            self.state.next_topup_date_local = due_day
        if local_day < due_day:
            self.state.updated_at_utc = now
            return None, stop_event

        amount = float(self.state.monthly_topup)
        self.state.topups_total += amount
        self.state.topups_count += 1
        self.state.last_topup_date_local = local_day
        self.state.next_topup_date_local = _first_day_next_month(local_day)
        self.state.updated_at_utc = now

        topup_event = CapitalRampEvent(
            scope=self.state.scope,
            event_type=EVENT_TOPUP,
            event_ts_utc=now,
            local_date=local_day,
            amount=amount,
            model_equity=self.model_equity(current_closed_pnl=current_closed_pnl),
            payload={
                "next_topup_date_local": self.state.next_topup_date_local.isoformat(),
                "topups_count": self.state.topups_count,
            },
        )
        stop_after_topup = self._stop_event_if_needed(
            now_utc=now,
            current_closed_pnl=current_closed_pnl,
        )
        return topup_event, stop_after_topup

    def effective_equity(
        self,
        *,
        now_utc: datetime,
        current_closed_pnl: float,
    ) -> float:
        self._stop_event_if_needed(now_utc=now_utc, current_closed_pnl=current_closed_pnl)
        return self.model_equity(current_closed_pnl=current_closed_pnl)

    def status(
        self,
        *,
        now_utc: datetime,
        current_closed_pnl: float,
    ) -> dict[str, object]:
        self._stop_event_if_needed(now_utc=now_utc, current_closed_pnl=current_closed_pnl)
        local_day = self._local_date(now_utc)
        return {
            "scope": self.state.scope,
            "timezone": self.state.timezone_name,
            "local_date": local_day.isoformat(),
            "start_ts_utc": _as_utc(self.state.start_ts_utc).isoformat(),
            "start_year_local": int(self.state.start_year_local),
            "year_end_local": self.state.year_end_local.isoformat(),
            "start_equity": float(self.state.start_equity),
            "monthly_topup": float(self.state.monthly_topup),
            "target_equity": float(self.state.target_equity),
            "pnl_baseline_at_start": float(self.state.pnl_baseline_at_start),
            "topups_total": float(self.state.topups_total),
            "topups_count": int(self.state.topups_count),
            "next_topup_date_local": (
                self.state.next_topup_date_local.isoformat()
                if self.state.next_topup_date_local is not None
                else None
            ),
            "last_topup_date_local": (
                self.state.last_topup_date_local.isoformat()
                if self.state.last_topup_date_local is not None
                else None
            ),
            "stopped_reason": self.state.stopped_reason,
            "stopped_at_utc": (
                _as_utc(self.state.stopped_at_utc).isoformat()
                if self.state.stopped_at_utc is not None
                else None
            ),
            "model_equity": self.model_equity(current_closed_pnl=current_closed_pnl),
        }
