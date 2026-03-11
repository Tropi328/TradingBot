"""Candle CSV loading and aggregation — extracted from engine.py."""

from __future__ import annotations

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path

from bot.data.candles import Candle

LOGGER = logging.getLogger(__name__)


def _parse_dt(value: str) -> datetime:
    normalized = value.replace(" ", "T")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def load_candles_csv(path: str | Path) -> list[Candle]:
    csv_path = Path(path)
    candles: list[Candle] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames:
            reader.fieldnames = [name.lstrip("\ufeff").strip() for name in reader.fieldnames]
        required = {"timestamp", "open", "high", "low", "close"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must include: timestamp,open,high,low,close")
        bad_ohlc = 0
        for line_no, row in enumerate(reader, start=2):
            o = float(row["open"])
            h = float(row["high"])
            low = float(row["low"])
            c = float(row["close"])
            if h < low or h < 0 or low < 0 or o < low or o > h or c < low or c > h:
                bad_ohlc += 1
            candles.append(
                Candle(
                    timestamp=_parse_dt(str(row["timestamp"])),
                    open=o,
                    high=h,
                    low=low,
                    close=c,
                    volume=float(row.get("volume") or 0.0),
                )
            )
    if not candles:
        raise ValueError(f"CSV file is empty: {csv_path}")
    candles.sort(key=lambda c: c.timestamp)
    # Warn about data quality issues (don't raise — allow partial bad data)
    if bad_ohlc > 0:
        LOGGER.warning(
            "CSV %s: %d/%d rows with bad OHLC (high < low or negative prices)",
            csv_path.name,
            bad_ohlc,
            len(candles),
        )
    # Detect & remove duplicate timestamps
    seen_ts: set[datetime] = set()
    deduped: list[Candle] = []
    for c in candles:
        if c.timestamp not in seen_ts:
            seen_ts.add(c.timestamp)
            deduped.append(c)
    if len(deduped) < len(candles):
        LOGGER.warning(
            "CSV %s: removed %d duplicate-timestamp rows",
            csv_path.name,
            len(candles) - len(deduped),
        )
    # Timestamp gap analysis
    if len(deduped) >= 2:
        gaps = 0
        for i in range(1, len(deduped)):
            if deduped[i].timestamp <= deduped[i - 1].timestamp:
                gaps += 1
        if gaps > 0:
            LOGGER.warning(
                "CSV %s: %d non-monotonic timestamps after sort (possible data issue)",
                csv_path.name,
                gaps,
            )
    return deduped


def _bucket_time(dt: datetime, minutes: int) -> datetime:
    unix = int(dt.timestamp())
    size = minutes * 60
    return datetime.fromtimestamp(unix - (unix % size), tz=UTC)


def aggregate_candles(candles: list[Candle], timeframe_minutes: int) -> list[Candle]:
    if not candles:
        return []
    result: list[Candle] = []
    bucket_start = _bucket_time(candles[0].timestamp, timeframe_minutes)
    open_price = candles[0].open
    high = candles[0].high
    low = candles[0].low
    close = candles[0].close
    volume = candles[0].volume or 0.0

    for candle in candles[1:]:
        current_bucket = _bucket_time(candle.timestamp, timeframe_minutes)
        if current_bucket != bucket_start:
            result.append(
                Candle(
                    timestamp=bucket_start,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                )
            )
            bucket_start = current_bucket
            open_price = candle.open
            high = candle.high
            low = candle.low
            close = candle.close
            volume = candle.volume or 0.0
            continue
        high = max(high, candle.high)
        low = min(low, candle.low)
        close = candle.close
        volume += candle.volume or 0.0

    result.append(
        Candle(
            timestamp=bucket_start,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
    )
    return result
