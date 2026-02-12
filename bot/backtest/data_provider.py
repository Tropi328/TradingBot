from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)

_SIDE_CHOICES = {"BID", "ASK", "MID"}
_PRICE_CHOICES = {"bid", "ask", "mid"}
_TF_ALIAS_TO_MINUTES = {
    "1m": 1,
    "m1": 1,
    "5m": 5,
    "m5": 5,
    "15m": 15,
    "m15": 15,
    "30m": 30,
    "m30": 30,
    "1h": 60,
    "h1": 60,
    "4h": 240,
    "h4": 240,
}
_CSV_TS_CANDIDATES = ("ts_utc", "timestamp", "datetime", "date", "time")
_CSV_OPEN_CANDIDATES = ("open", "o")
_CSV_HIGH_CANDIDATES = ("high", "h")
_CSV_LOW_CANDIDATES = ("low", "l")
_CSV_CLOSE_CANDIDATES = ("close", "c")
_CSV_VOLUME_CANDIDATES = ("volume", "vol", "tick_volume")


def normalize_timeframe(value: str) -> str:
    key = value.strip().lower()
    if key not in _TF_ALIAS_TO_MINUTES:
        raise ValueError(f"Unsupported timeframe '{value}'")
    minutes = _TF_ALIAS_TO_MINUTES[key]
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}m"


def timeframe_to_minutes(value: str) -> int:
    key = value.strip().lower()
    if key not in _TF_ALIAS_TO_MINUTES:
        raise ValueError(f"Unsupported timeframe '{value}'")
    return _TF_ALIAS_TO_MINUTES[key]


def _month_start(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, 1, tzinfo=timezone.utc)


def _next_month(dt: datetime) -> datetime:
    if dt.month == 12:
        return datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(dt.year, dt.month + 1, 1, tzinfo=timezone.utc)


def _iter_months(start_utc: datetime, end_utc: datetime) -> list[tuple[int, int]]:
    if start_utc >= end_utc:
        return []
    current = _month_start(start_utc)
    out: list[tuple[int, int]] = []
    while current < end_utc:
        out.append((current.year, current.month))
        current = _next_month(current)
    return out


def _to_utc(ts: datetime | str) -> datetime:
    if isinstance(ts, datetime):
        dt = ts
    else:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(slots=True)
class MissingDataItem:
    symbol: str
    side: str
    timeframe: str
    year: int
    month: int

    def key(self) -> tuple[str, str, str, int, int]:
        return (self.symbol, self.side, self.timeframe, self.year, self.month)

    def to_line(self) -> str:
        return (
            f"symbol={self.symbol} side={self.side} tf={self.timeframe} "
            f"year={self.year} month={self.month:02d}"
        )


class MissingDataError(RuntimeError):
    def __init__(self, missing: list[MissingDataItem]):
        unique: dict[tuple[str, str, str, int, int], MissingDataItem] = {}
        for item in missing:
            unique[item.key()] = item
        self.missing = sorted(
            unique.values(),
            key=lambda item: (item.symbol, item.side, item.timeframe, item.year, item.month),
        )
        details = "\n".join(f"- {item.to_line()}" for item in self.missing)
        super().__init__(f"Missing market data:\n{details}" if details else "Missing market data")


@dataclass(slots=True)
class DataLoadResult:
    symbol: str
    timeframe: str
    price_mode: str
    frame: pd.DataFrame
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class _SideLoadResult:
    frame: pd.DataFrame
    source_files: list[str] = field(default_factory=list)


class AutoDataLoader:
    def __init__(
        self,
        data_root: str | Path,
        *,
        source_priority: list[str] | None = None,
        file_cache_size: int = 128,
    ):
        self.data_root = Path(data_root)
        self.source_priority = [item.strip() for item in (source_priority or []) if item.strip()]
        self.file_cache_size = max(8, int(file_cache_size))
        self._file_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._csv_bootstrap_attempted_symbols: set[str] = set()

    def available_sources(self) -> list[str]:
        if not self.data_root.exists():
            return []
        detected = sorted(item.name for item in self.data_root.iterdir() if item.is_dir())
        if not self.source_priority:
            return detected
        ordered: list[str] = []
        seen: set[str] = set()
        for source in self.source_priority:
            if source in detected and source not in seen:
                ordered.append(source)
                seen.add(source)
        for source in detected:
            if source not in seen:
                ordered.append(source)
        return ordered

    def load_symbol_data(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        price_mode: str = "mid",
    ) -> DataLoadResult:
        symbol_norm = symbol.strip().upper()
        tf_norm = normalize_timeframe(timeframe)
        start_utc = _to_utc(start)
        end_utc = _to_utc(end)
        if start_utc >= end_utc:
            raise ValueError("start must be < end")

        price_mode_norm = price_mode.strip().lower()
        if price_mode_norm not in _PRICE_CHOICES:
            raise ValueError(f"Unsupported price_mode '{price_mode}'. Allowed: {sorted(_PRICE_CHOICES)}")

        missing: list[MissingDataItem] = []
        source_files: list[str] = []
        source_datasets: set[str] = set()

        bid_side: _SideLoadResult | None = None
        ask_side: _SideLoadResult | None = None
        mid_side: _SideLoadResult | None = None

        if price_mode_norm == "mid":
            # Prefer BID/ASK pair for realistic spread.
            try:
                bid_side = self._load_side(
                    symbol=symbol_norm,
                    side="BID",
                    timeframe=tf_norm,
                    start=start_utc,
                    end=end_utc,
                )
            except MissingDataError as exc:
                missing.extend(exc.missing)
                bid_side = None
            try:
                ask_side = self._load_side(
                    symbol=symbol_norm,
                    side="ASK",
                    timeframe=tf_norm,
                    start=start_utc,
                    end=end_utc,
                )
            except MissingDataError as exc:
                missing.extend(exc.missing)
                ask_side = None
            if bid_side is None and ask_side is None:
                try:
                    mid_side = self._load_side(
                        symbol=symbol_norm,
                        side="MID",
                        timeframe=tf_norm,
                        start=start_utc,
                        end=end_utc,
                    )
                except MissingDataError as exc:
                    missing.extend(exc.missing)
                    raise MissingDataError(missing) from exc
        elif price_mode_norm == "bid":
            try:
                bid_side = self._load_side(
                    symbol=symbol_norm,
                    side="BID",
                    timeframe=tf_norm,
                    start=start_utc,
                    end=end_utc,
                )
            except MissingDataError as exc:
                raise MissingDataError(exc.missing) from exc
            try:
                ask_side = self._load_side(
                    symbol=symbol_norm,
                    side="ASK",
                    timeframe=tf_norm,
                    start=start_utc,
                    end=end_utc,
                )
            except MissingDataError:
                ask_side = None
        else:
            try:
                ask_side = self._load_side(
                    symbol=symbol_norm,
                    side="ASK",
                    timeframe=tf_norm,
                    start=start_utc,
                    end=end_utc,
                )
            except MissingDataError as exc:
                raise MissingDataError(exc.missing) from exc
            try:
                bid_side = self._load_side(
                    symbol=symbol_norm,
                    side="BID",
                    timeframe=tf_norm,
                    start=start_utc,
                    end=end_utc,
                )
            except MissingDataError:
                bid_side = None

        for loaded_side in (bid_side, ask_side, mid_side):
            if loaded_side is None:
                continue
            source_files.extend(loaded_side.source_files)
            for file_path in loaded_side.source_files:
                source_name = self._source_name_from_file(file_path)
                source_datasets.add(source_name)

        combined, price_diag = self._compose_price_frame(
            symbol=symbol_norm,
            timeframe=tf_norm,
            price_mode=price_mode_norm,
            bid_df=bid_side.frame if bid_side is not None else None,
            ask_df=ask_side.frame if ask_side is not None else None,
            mid_df=mid_side.frame if mid_side is not None else None,
            start=start_utc,
            end=end_utc,
        )
        if combined.empty:
            raise MissingDataError(
                [
                    MissingDataItem(
                        symbol=symbol_norm,
                        side="MID" if price_mode_norm == "mid" else price_mode_norm.upper(),
                        timeframe=tf_norm,
                        year=start_utc.year,
                        month=start_utc.month,
                    )
                ]
            )
        return DataLoadResult(
            symbol=symbol_norm,
            timeframe=tf_norm,
            price_mode=price_mode_norm,
            frame=combined,
            diagnostics={
                "price_mode_requested": price_mode_norm,
                "source_files": sorted(set(source_files)),
                "source_datasets": sorted(source_datasets),
                "fallback_counters": price_diag,
                "data_health": self._compute_data_health(combined, tf_norm),
            },
        )

    def _compose_price_frame(
        self,
        *,
        symbol: str,
        timeframe: str,
        price_mode: str,
        bid_df: pd.DataFrame | None,
        ask_df: pd.DataFrame | None,
        mid_df: pd.DataFrame | None,
        start: datetime,
        end: datetime,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        diag = {"MID_FALLBACK_OHLC_CLOSE": 0, "MID_FALLBACK_SINGLE_SIDE": 0}
        if bid_df is not None and ask_df is not None:
            merged = bid_df.merge(
                ask_df,
                on="ts_utc",
                how="inner",
                suffixes=("_bid", "_ask"),
            )
            if merged.empty:
                raise MissingDataError(
                    [
                        MissingDataItem(symbol=symbol, side="BID", timeframe=timeframe, year=start.year, month=start.month),
                        MissingDataItem(symbol=symbol, side="ASK", timeframe=timeframe, year=start.year, month=start.month),
                    ]
                )
            frame = pd.DataFrame(
                {
                    "ts_utc": merged["ts_utc"],
                    "open_mid": (merged["open_bid"] + merged["open_ask"]) / 2.0,
                    "high_mid": (merged["high_bid"] + merged["high_ask"]) / 2.0,
                    "low_mid": (merged["low_bid"] + merged["low_ask"]) / 2.0,
                    "close_mid": (merged["close_bid"] + merged["close_ask"]) / 2.0,
                    "open_bid": merged["open_bid"],
                    "high_bid": merged["high_bid"],
                    "low_bid": merged["low_bid"],
                    "close_bid": merged["close_bid"],
                    "open_ask": merged["open_ask"],
                    "high_ask": merged["high_ask"],
                    "low_ask": merged["low_ask"],
                    "close_ask": merged["close_ask"],
                    "volume": merged["volume_bid"].fillna(0.0),
                }
            )
            frame["spread"] = frame["close_ask"] - frame["close_bid"]
        elif mid_df is not None:
            close_only = pd.to_numeric(mid_df["close"], errors="coerce")
            frame = pd.DataFrame(
                {
                    "ts_utc": mid_df["ts_utc"],
                    "open_mid": pd.to_numeric(mid_df["open"], errors="coerce").fillna(close_only),
                    "high_mid": pd.to_numeric(mid_df["high"], errors="coerce").fillna(close_only),
                    "low_mid": pd.to_numeric(mid_df["low"], errors="coerce").fillna(close_only),
                    "close_mid": close_only,
                    "volume": mid_df["volume"],
                }
            )
            frame["open_bid"] = pd.NA
            frame["high_bid"] = pd.NA
            frame["low_bid"] = pd.NA
            frame["close_bid"] = pd.NA
            frame["open_ask"] = pd.NA
            frame["high_ask"] = pd.NA
            frame["low_ask"] = pd.NA
            frame["close_ask"] = pd.NA
            frame["spread"] = pd.NA
            if price_mode == "mid":
                diag["MID_FALLBACK_OHLC_CLOSE"] += int(len(frame))
        elif price_mode == "mid" and bid_df is not None:
            frame = pd.DataFrame(
                {
                    "ts_utc": bid_df["ts_utc"],
                    "open_mid": bid_df["open"],
                    "high_mid": bid_df["high"],
                    "low_mid": bid_df["low"],
                    "close_mid": bid_df["close"],
                    "open_bid": bid_df["open"],
                    "high_bid": bid_df["high"],
                    "low_bid": bid_df["low"],
                    "close_bid": bid_df["close"],
                    "volume": bid_df["volume"],
                }
            )
            frame["open_ask"] = pd.NA
            frame["high_ask"] = pd.NA
            frame["low_ask"] = pd.NA
            frame["close_ask"] = pd.NA
            frame["spread"] = pd.NA
            diag["MID_FALLBACK_SINGLE_SIDE"] += int(len(frame))
        elif price_mode == "mid" and ask_df is not None:
            frame = pd.DataFrame(
                {
                    "ts_utc": ask_df["ts_utc"],
                    "open_mid": ask_df["open"],
                    "high_mid": ask_df["high"],
                    "low_mid": ask_df["low"],
                    "close_mid": ask_df["close"],
                    "open_ask": ask_df["open"],
                    "high_ask": ask_df["high"],
                    "low_ask": ask_df["low"],
                    "close_ask": ask_df["close"],
                    "volume": ask_df["volume"],
                }
            )
            frame["open_bid"] = pd.NA
            frame["high_bid"] = pd.NA
            frame["low_bid"] = pd.NA
            frame["close_bid"] = pd.NA
            frame["spread"] = pd.NA
            diag["MID_FALLBACK_SINGLE_SIDE"] += int(len(frame))
        elif bid_df is not None and price_mode == "bid":
            frame = pd.DataFrame(
                {
                    "ts_utc": bid_df["ts_utc"],
                    "open_mid": bid_df["open"],
                    "high_mid": bid_df["high"],
                    "low_mid": bid_df["low"],
                    "close_mid": bid_df["close"],
                    "open_bid": bid_df["open"],
                    "high_bid": bid_df["high"],
                    "low_bid": bid_df["low"],
                    "close_bid": bid_df["close"],
                    "volume": bid_df["volume"],
                }
            )
            frame["open_ask"] = pd.NA
            frame["high_ask"] = pd.NA
            frame["low_ask"] = pd.NA
            frame["close_ask"] = pd.NA
            frame["spread"] = pd.NA
        elif ask_df is not None and price_mode == "ask":
            frame = pd.DataFrame(
                {
                    "ts_utc": ask_df["ts_utc"],
                    "open_mid": ask_df["open"],
                    "high_mid": ask_df["high"],
                    "low_mid": ask_df["low"],
                    "close_mid": ask_df["close"],
                    "open_ask": ask_df["open"],
                    "high_ask": ask_df["high"],
                    "low_ask": ask_df["low"],
                    "close_ask": ask_df["close"],
                    "volume": ask_df["volume"],
                }
            )
            frame["open_bid"] = pd.NA
            frame["high_bid"] = pd.NA
            frame["low_bid"] = pd.NA
            frame["close_bid"] = pd.NA
            frame["spread"] = pd.NA
        else:
            raise MissingDataError(
                [MissingDataItem(symbol=symbol, side=price_mode.upper(), timeframe=timeframe, year=start.year, month=start.month)]
            )

        if price_mode == "mid":
            frame["open"] = frame["open_mid"]
            frame["high"] = frame["high_mid"]
            frame["low"] = frame["low_mid"]
            frame["close"] = frame["close_mid"]
        elif price_mode == "bid":
            frame["open"] = frame["open_bid"]
            frame["high"] = frame["high_bid"]
            frame["low"] = frame["low_bid"]
            frame["close"] = frame["close_bid"]
        else:
            frame["open"] = frame["open_ask"]
            frame["high"] = frame["high_ask"]
            frame["low"] = frame["low_ask"]
            frame["close"] = frame["close_ask"]

        numeric_columns = (
            "open",
            "high",
            "low",
            "close",
            "open_bid",
            "high_bid",
            "low_bid",
            "close_bid",
            "open_ask",
            "high_ask",
            "low_ask",
            "close_ask",
            "spread",
        )
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "close" in frame.columns:
            frame["open"] = frame["open"].fillna(frame["close"])
            frame["high"] = frame["high"].fillna(frame["close"])
            frame["low"] = frame["low"].fillna(frame["close"])

        frame = frame.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"]).reset_index(drop=True)
        frame = frame[(frame["ts_utc"] >= start) & (frame["ts_utc"] < end)].copy()
        frame = frame.dropna(subset=["ts_utc", "open", "high", "low", "close"]).copy()
        frame["symbol"] = symbol
        frame["timeframe"] = timeframe
        frame["price_mode"] = price_mode
        return frame, diag

    def _load_side(
        self,
        *,
        symbol: str,
        side: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        allow_csv_fallback: bool = True,
    ) -> _SideLoadResult:
        side_norm = side.strip().upper()
        if side_norm not in _SIDE_CHOICES:
            raise ValueError(f"Unsupported side '{side}'")
        tf_norm = normalize_timeframe(timeframe)
        target_minutes = timeframe_to_minutes(tf_norm)
        months = _iter_months(start, end)
        if not months:
            return _SideLoadResult(frame=pd.DataFrame(columns=["ts_utc", "open", "high", "low", "close", "volume"]))
        sources = self.available_sources()
        if not sources:
            if side_norm == "MID" and allow_csv_fallback and self._bootstrap_mid_from_csv(symbol):
                return self._load_side(
                    symbol=symbol,
                    side=side_norm,
                    timeframe=tf_norm,
                    start=start,
                    end=end,
                    allow_csv_fallback=False,
                )
            raise MissingDataError(
                [MissingDataItem(symbol=symbol, side=side_norm, timeframe=tf_norm, year=year, month=month) for year, month in months]
            )

        preferred_tf: str
        if tf_norm == "1m":
            preferred_tf = "1m"
        else:
            # Use target resolution if available for all months; otherwise fallback to 1m for all months.
            all_target = all(
                any(self._file_path(source, symbol, side_norm, tf_norm, year, month).exists() for source in sources)
                for year, month in months
            )
            if all_target:
                preferred_tf = tf_norm
            else:
                all_1m = all(
                    any(self._file_path(source, symbol, side_norm, "1m", year, month).exists() for source in sources)
                    for year, month in months
                )
                if all_1m:
                    preferred_tf = "1m"
                else:
                    missing: list[MissingDataItem] = []
                    for year, month in months:
                        has_target = any(
                            self._file_path(source, symbol, side_norm, tf_norm, year, month).exists()
                            for source in sources
                        )
                        has_1m = any(
                            self._file_path(source, symbol, side_norm, "1m", year, month).exists()
                            for source in sources
                        )
                        if not has_target and not has_1m:
                            missing.append(
                                MissingDataItem(
                                    symbol=symbol,
                                    side=side_norm,
                                    timeframe=tf_norm,
                                    year=year,
                                    month=month,
                                )
                            )
                    if side_norm == "MID" and allow_csv_fallback and self._bootstrap_mid_from_csv(symbol):
                        return self._load_side(
                            symbol=symbol,
                            side=side_norm,
                            timeframe=tf_norm,
                            start=start,
                            end=end,
                            allow_csv_fallback=False,
                        )
                    raise MissingDataError(missing)

        raw_parts: list[pd.DataFrame] = []
        source_files: list[str] = []
        for year, month in months:
            found_path: Path | None = None
            for source in sources:
                candidate = self._file_path(source, symbol, side_norm, preferred_tf, year, month)
                if candidate.exists():
                    found_path = candidate
                    break
            if found_path is None:
                if side_norm == "MID" and allow_csv_fallback and self._bootstrap_mid_from_csv(symbol):
                    return self._load_side(
                        symbol=symbol,
                        side=side_norm,
                        timeframe=tf_norm,
                        start=start,
                        end=end,
                        allow_csv_fallback=False,
                    )
                raise MissingDataError(
                    [
                        MissingDataItem(
                            symbol=symbol,
                            side=side_norm,
                            timeframe=tf_norm,
                            year=year,
                            month=month,
                        )
                    ]
                )
            source_files.append(str(found_path.resolve()))
            raw_parts.append(self._read_parquet(found_path))

        frame = pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame()
        if frame.empty:
            return _SideLoadResult(
                frame=pd.DataFrame(columns=["ts_utc", "open", "high", "low", "close", "volume"]),
                source_files=source_files,
            )
        frame = frame.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"]).reset_index(drop=True)

        if preferred_tf != tf_norm:
            frame = self._resample(frame, target_minutes)
        frame = frame[(frame["ts_utc"] >= start) & (frame["ts_utc"] < end)].copy()
        return _SideLoadResult(frame=frame, source_files=source_files)

    def _compute_data_health(self, frame: pd.DataFrame, timeframe: str) -> dict[str, object]:
        tf_minutes = timeframe_to_minutes(timeframe)
        if frame.empty:
            return {
                "bars": 0,
                "timeframe_minutes": tf_minutes,
                "min_ts_utc": None,
                "max_ts_utc": None,
                "nan_counts": {},
                "duplicate_timestamps": 0,
                "gap_count_over_1bar": 0,
                "max_gap_minutes": 0.0,
            }

        ts = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
        sorted_ts = ts.sort_values()
        diffs = sorted_ts.diff().dt.total_seconds().div(60.0)
        gap_mask = diffs > float(tf_minutes)
        max_gap = float(diffs.max()) if not diffs.empty and pd.notna(diffs.max()) else 0.0

        nan_counts: dict[str, int] = {}
        for column in ("open", "high", "low", "close", "open_bid", "close_bid", "open_ask", "close_ask", "spread"):
            if column in frame.columns:
                nan_counts[column] = int(pd.to_numeric(frame[column], errors="coerce").isna().sum())

        min_ts = sorted_ts.min()
        max_ts = sorted_ts.max()
        return {
            "bars": int(len(frame)),
            "timeframe_minutes": tf_minutes,
            "min_ts_utc": min_ts.isoformat() if pd.notna(min_ts) else None,
            "max_ts_utc": max_ts.isoformat() if pd.notna(max_ts) else None,
            "nan_counts": nan_counts,
            "duplicate_timestamps": int(ts.duplicated().sum()),
            "gap_count_over_1bar": int(gap_mask.sum()) if not diffs.empty else 0,
            "max_gap_minutes": max_gap,
        }

    def _bootstrap_mid_from_csv(self, symbol: str) -> bool:
        symbol_norm = symbol.strip().upper()
        if symbol_norm in self._csv_bootstrap_attempted_symbols:
            return False
        self._csv_bootstrap_attempted_symbols.add(symbol_norm)

        csv_path = self._find_bootstrap_csv(symbol_norm)
        if csv_path is None:
            return False

        try:
            frame = self._read_ohlcv_csv(csv_path)
            if frame.empty:
                LOGGER.warning("CSV bootstrap skipped for %s: %s is empty after parsing.", symbol_norm, csv_path)
                return False
            self._write_mid_monthly_parquet(symbol_norm, frame)
            LOGGER.info(
                "CSV bootstrap completed for %s from %s (%d rows).",
                symbol_norm,
                csv_path,
                len(frame),
            )
            return True
        except Exception:
            LOGGER.exception("CSV bootstrap failed for %s from %s.", symbol_norm, csv_path)
            return False

    def _find_bootstrap_csv(self, symbol: str) -> Path | None:
        names = [f"{symbol}_1m_data.csv"]
        if symbol.endswith("USD") and len(symbol) > 3:
            names.append(f"{symbol[:-3]}_1m_data.csv")

        roots = [
            self.data_root,
            self.data_root.parent,
            self.data_root.parent / "bot",
            self.data_root.parent / "bot" / "data",
            Path.cwd(),
            Path.cwd() / "bot",
            Path.cwd() / "bot" / "data",
        ]
        seen: set[Path] = set()
        for root in roots:
            try:
                normalized_root = root.resolve()
            except OSError:
                normalized_root = root
            if normalized_root in seen:
                continue
            seen.add(normalized_root)
            for name in names:
                candidate = root / name
                if candidate.exists() and candidate.is_file():
                    return candidate
        return None

    def _read_ohlcv_csv(self, csv_path: Path) -> pd.DataFrame:
        raw = pd.read_csv(csv_path)
        columns = {name.strip().lower(): name for name in raw.columns}

        def pick(candidates: tuple[str, ...], required: bool) -> str | None:
            for candidate in candidates:
                matched = columns.get(candidate)
                if matched is not None:
                    return matched
            if required:
                raise ValueError(f"CSV missing required column. candidates={candidates}, file={csv_path}")
            return None

        ts_col = pick(_CSV_TS_CANDIDATES, required=True)
        open_col = pick(_CSV_OPEN_CANDIDATES, required=True)
        high_col = pick(_CSV_HIGH_CANDIDATES, required=True)
        low_col = pick(_CSV_LOW_CANDIDATES, required=True)
        close_col = pick(_CSV_CLOSE_CANDIDATES, required=True)
        volume_col = pick(_CSV_VOLUME_CANDIDATES, required=False)

        out = pd.DataFrame(
            {
                "ts_utc": pd.to_datetime(raw[ts_col], utc=True, errors="coerce"),
                "open": pd.to_numeric(raw[open_col], errors="coerce"),
                "high": pd.to_numeric(raw[high_col], errors="coerce"),
                "low": pd.to_numeric(raw[low_col], errors="coerce"),
                "close": pd.to_numeric(raw[close_col], errors="coerce"),
            }
        )
        if volume_col is not None:
            out["volume"] = pd.to_numeric(raw[volume_col], errors="coerce").fillna(0.0)
        else:
            out["volume"] = 0.0

        out = out.dropna(subset=["ts_utc", "open", "high", "low", "close"])
        out = out.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="last")
        return out.reset_index(drop=True)

    def _write_mid_monthly_parquet(self, symbol: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        work = frame.copy()
        work["year"] = work["ts_utc"].dt.year
        work["month"] = work["ts_utc"].dt.month

        for (year, month), chunk in work.groupby(["year", "month"]):
            target = self._file_path("local_csv", symbol, "MID", "1m", int(year), int(month))
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = chunk[["ts_utc", "open", "high", "low", "close", "volume"]].copy()
            if target.exists():
                existing = self._read_parquet(target)
                payload = pd.concat([existing, payload], ignore_index=True)
                payload = payload.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="last")
            payload.to_parquet(target, index=False, engine="pyarrow")
            self._file_cache.pop(str(target.resolve()), None)

    def _file_path(
        self,
        source: str,
        symbol: str,
        side: str,
        timeframe: str,
        year: int,
        month: int,
    ) -> Path:
        return self.data_root / source / symbol / side / timeframe / f"{year:04d}" / f"{month:02d}.parquet"

    def _source_name_from_file(self, path: str) -> str:
        file_path = Path(path)
        try:
            rel = file_path.resolve().relative_to(self.data_root.resolve())
            return rel.parts[0] if rel.parts else "unknown"
        except Exception:  # noqa: BLE001
            pass
        parts = file_path.parts
        return parts[-6] if len(parts) >= 6 else "unknown"

    def _read_parquet(self, path: Path) -> pd.DataFrame:
        key = str(path.resolve())
        cached = self._file_cache.get(key)
        if cached is not None:
            self._file_cache.move_to_end(key)
            return cached.copy()

        frame = pd.read_parquet(path, engine="pyarrow")
        columns = {name.lower(): name for name in frame.columns}
        if "ts_utc" not in columns:
            raise ValueError(f"Parquet missing ts_utc column: {path}")
        for key in ("open", "high", "low", "close"):
            if key not in columns:
                raise ValueError(f"Parquet missing {key} column: {path}")
        out = pd.DataFrame(
            {
                "ts_utc": pd.to_datetime(frame[columns["ts_utc"]], utc=True, errors="coerce"),
                "open": pd.to_numeric(frame[columns["open"]], errors="coerce"),
                "high": pd.to_numeric(frame[columns["high"]], errors="coerce"),
                "low": pd.to_numeric(frame[columns["low"]], errors="coerce"),
                "close": pd.to_numeric(frame[columns["close"]], errors="coerce"),
            }
        )
        if "volume" in columns:
            out["volume"] = pd.to_numeric(frame[columns["volume"]], errors="coerce").fillna(0.0)
        else:
            out["volume"] = 0.0

        out = out.dropna(subset=["ts_utc", "open", "high", "low", "close"]).sort_values("ts_utc")
        out = out.reset_index(drop=True)

        self._file_cache[key] = out
        self._file_cache.move_to_end(key)
        while len(self._file_cache) > self.file_cache_size:
            self._file_cache.popitem(last=False)
        return out.copy()

    @staticmethod
    def _resample(frame: pd.DataFrame, target_minutes: int) -> pd.DataFrame:
        if frame.empty:
            return frame
        rule = f"{target_minutes}min"
        resampled = (
            frame.set_index("ts_utc")
            .sort_index()
            .resample(rule, label="left", closed="left")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
        return resampled

    @staticmethod
    def missing_lines(items: Iterable[MissingDataItem]) -> list[str]:
        return [item.to_line() for item in items]
