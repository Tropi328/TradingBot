from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class OrderRecord:
    order_id: str
    deal_reference: str | None
    request_id: str | None
    epic: str
    side: str
    size: float
    entry_price: float
    stop_price: float
    take_profit: float
    status: str
    remote_status: str | None
    filled_size: float
    expires_at: datetime
    created_at: datetime
    updated_at: datetime
    reason_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PositionRecord:
    deal_id: str
    epic: str
    side: str
    size: float
    entry_price: float
    stop_price: float
    take_profit: float
    status: str
    opened_at: datetime
    closed_at: datetime | None = None
    partial_closed_size: float = 0.0
    pnl: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DailyStats:
    trading_day: str
    epic: str = "GLOBAL"
    pnl: float = 0.0
    trades_count: int = 0
    status: str = "ON"
    updated_at: datetime | None = None


@dataclass(slots=True)
class StrategyDecisionRecord:
    created_at: datetime
    epic: str
    side: str | None
    bias: str
    pd_state: str
    sweep: bool
    mss: bool
    displacement: bool
    fvg: bool
    spread_ok: bool
    news_blocked: bool
    rr: float | None
    reason_codes: list[str]
    payload: dict[str, Any]


@dataclass(slots=True)
class RiskState:
    scope: str
    loss_streak: int = 0
    cooldown_until: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class ClosedPositionEvent:
    deal_id: str
    epic: str
    pnl: float
    closed_at: datetime
