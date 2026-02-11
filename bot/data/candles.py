from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    bid: float | None = None
    ask: float | None = None
    volume: float | None = None


def resolution_to_capital(timeframe: str) -> str:
    mapping = {
        "M1": "MINUTE",
        "M5": "MINUTE_5",
        "M15": "MINUTE_15",
        "H1": "HOUR",
        "H4": "HOUR_4",
        "D1": "DAY",
    }
    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe {timeframe}")
    return mapping[timeframe]


def _mid(bid: float | None, ask: float | None, fallback: float | None = None) -> float:
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if fallback is not None:
        return float(fallback)
    if bid is not None:
        return float(bid)
    if ask is not None:
        return float(ask)
    raise ValueError("Cannot compute price midpoint")


def parse_timestamp(value: str) -> datetime:
    normalized = value.replace(" ", "T")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def candles_from_prices(prices: list[dict[str, Any]]) -> list[Candle]:
    output: list[Candle] = []
    for item in prices:
        ts_raw = item.get("snapshotTimeUTC") or item.get("snapshotTime")
        if ts_raw is None:
            continue
        open_price = item.get("openPrice", {})
        high_price = item.get("highPrice", {})
        low_price = item.get("lowPrice", {})
        close_price = item.get("closePrice", {})
        o_bid = open_price.get("bid")
        o_ask = open_price.get("ask")
        h_bid = high_price.get("bid")
        h_ask = high_price.get("ask")
        l_bid = low_price.get("bid")
        l_ask = low_price.get("ask")
        c_bid = close_price.get("bid")
        c_ask = close_price.get("ask")
        candle = Candle(
            timestamp=parse_timestamp(ts_raw),
            open=_mid(o_bid, o_ask, item.get("open")),
            high=_mid(h_bid, h_ask, item.get("high")),
            low=_mid(l_bid, l_ask, item.get("low")),
            close=_mid(c_bid, c_ask, item.get("close")),
            bid=float(c_bid) if c_bid is not None else None,
            ask=float(c_ask) if c_ask is not None else None,
            volume=float(item.get("lastTradedVolume") or item.get("volume") or 0.0),
        )
        output.append(candle)
    return sorted(output, key=lambda c: c.timestamp)

