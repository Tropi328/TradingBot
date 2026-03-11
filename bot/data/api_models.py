from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MarketSnapshot(BaseModel):
    model_config = ConfigDict(extra="allow")

    bid: float
    offer: float
    high: float | None = None
    low: float | None = None


class MarketDetailsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    snapshot: MarketSnapshot
    instrument: dict[str, Any] = Field(default_factory=dict)
