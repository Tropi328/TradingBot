from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from bot.backtest.data_provider import AutoDataLoader


def _write_month(
    root: Path,
    *,
    source: str,
    symbol: str,
    side: str,
    timeframe: str,
    year: int,
    month: int,
    frame: pd.DataFrame,
) -> None:
    target_dir = root / source / symbol / side / timeframe / f"{year:04d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target_dir / f"{month:02d}.parquet", index=False, engine="pyarrow")


def test_resample_1m_to_5m_ohlc(tmp_path: Path) -> None:
    start = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(10):
        ts = start + timedelta(minutes=i)
        op = 100.0 + i
        rows.append(
            {
                "ts_utc": ts,
                "open": op,
                "high": op + 0.5,
                "low": op - 0.5,
                "close": op + 0.2,
                "volume": 1.0,
            }
        )
    frame = pd.DataFrame(rows)
    _write_month(
        tmp_path / "data",
        source="coinbase",
        symbol="BTCUSD",
        side="MID",
        timeframe="1m",
        year=2023,
        month=1,
        frame=frame,
    )

    loader = AutoDataLoader(tmp_path / "data")
    loaded = loader.load_symbol_data(
        symbol="BTCUSD",
        timeframe="5m",
        start=start,
        end=start + timedelta(minutes=10),
        price_mode="mid",
    )
    got = loaded.frame
    assert len(got) == 2

    first = got.iloc[0]
    second = got.iloc[1]
    assert first["open"] == 100.0
    assert first["high"] == 104.5
    assert first["low"] == 99.5
    assert first["close"] == 104.2
    assert first["volume"] == 5.0

    assert second["open"] == 105.0
    assert second["high"] == 109.5
    assert second["low"] == 104.5
    assert second["close"] == 109.2
    assert second["volume"] == 5.0


def test_mid_and_spread_from_bid_ask(tmp_path: Path) -> None:
    start = datetime(2023, 2, 1, 0, 0, tzinfo=timezone.utc)
    bid = pd.DataFrame(
        [
            {"ts_utc": start, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0},
            {"ts_utc": start + timedelta(minutes=5), "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 11.0},
        ]
    )
    ask = pd.DataFrame(
        [
            {"ts_utc": start, "open": 100.4, "high": 101.4, "low": 99.4, "close": 100.9, "volume": 10.0},
            {"ts_utc": start + timedelta(minutes=5), "open": 101.4, "high": 102.4, "low": 100.4, "close": 101.9, "volume": 11.0},
        ]
    )
    data_root = tmp_path / "data"
    _write_month(
        data_root,
        source="dukascopy",
        symbol="XAUUSD",
        side="BID",
        timeframe="5m",
        year=2023,
        month=2,
        frame=bid,
    )
    _write_month(
        data_root,
        source="dukascopy",
        symbol="XAUUSD",
        side="ASK",
        timeframe="5m",
        year=2023,
        month=2,
        frame=ask,
    )

    loader = AutoDataLoader(data_root)
    loaded = loader.load_symbol_data(
        symbol="XAUUSD",
        timeframe="5m",
        start=start,
        end=start + timedelta(minutes=10),
        price_mode="mid",
    )
    got = loaded.frame.reset_index(drop=True)

    assert got.loc[0, "close"] == 100.7
    assert got.loc[1, "close"] == 101.7
    assert got.loc[0, "spread"] == pytest.approx(0.4, rel=0.0, abs=1e-9)
    assert got.loc[1, "spread"] == pytest.approx(0.4, rel=0.0, abs=1e-9)


def test_bootstrap_csv_to_mid_parquet_when_missing(tmp_path: Path) -> None:
    start = datetime(2025, 8, 31, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(10):
        ts = start + timedelta(minutes=i)
        rows.append(
            {
                "Date": ts.isoformat().replace("+00:00", "Z"),
                "Open": 3300.0 + i,
                "High": 3300.2 + i,
                "Low": 3299.8 + i,
                "Close": 3300.1 + i,
                "Volume": 10 + i,
            }
        )

    csv_dir = tmp_path / "bot" / "data"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "XAU_1m_data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    data_root = tmp_path / "data"
    loader = AutoDataLoader(data_root)
    loaded = loader.load_symbol_data(
        symbol="XAUUSD",
        timeframe="5m",
        start=start,
        end=start + timedelta(minutes=10),
        price_mode="mid",
    )

    got = loaded.frame.reset_index(drop=True)
    assert len(got) == 2
    assert got.loc[0, "open"] == pytest.approx(3300.0, rel=0.0, abs=1e-9)
    assert got.loc[1, "close"] == pytest.approx(3309.1, rel=0.0, abs=1e-9)

    target = data_root / "local_csv" / "XAUUSD" / "MID" / "1m" / "2025" / "08.parquet"
    assert target.exists()


def test_mid_mode_falls_back_to_single_side_with_diagnostics(tmp_path: Path) -> None:
    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    bid = pd.DataFrame(
        [
            {"ts_utc": start, "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.2, "volume": 1.0},
            {"ts_utc": start + timedelta(minutes=5), "open": 10.2, "high": 10.7, "low": 9.9, "close": 10.4, "volume": 1.0},
        ]
    )
    data_root = tmp_path / "data"
    _write_month(
        data_root,
        source="dukascopy",
        symbol="XAUUSD",
        side="BID",
        timeframe="5m",
        year=2024,
        month=1,
        frame=bid,
    )

    loader = AutoDataLoader(data_root)
    loaded = loader.load_symbol_data(
        symbol="XAUUSD",
        timeframe="5m",
        start=start,
        end=start + timedelta(minutes=10),
        price_mode="mid",
    )
    got = loaded.frame.reset_index(drop=True)
    assert len(got) == 2
    assert got.loc[0, "close"] == pytest.approx(10.2, rel=0.0, abs=1e-9)
    assert loaded.diagnostics["fallback_counters"]["MID_FALLBACK_SINGLE_SIDE"] == 2
    assert loaded.diagnostics["data_health"]["bars"] == 2


def test_split_frame_by_gaps_creates_segments(tmp_path: Path) -> None:
    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    frame = pd.DataFrame(
        [
            {"ts_utc": start, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0},
            {"ts_utc": start + timedelta(minutes=5), "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0},
            {"ts_utc": start + timedelta(minutes=30), "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0},
            {"ts_utc": start + timedelta(minutes=35), "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0},
        ]
    )
    loader = AutoDataLoader(tmp_path / "data")
    segments, stats = loader.split_frame_by_gaps(frame, "5m", gap_bars=3)

    assert len(segments) == 2
    assert [len(seg) for seg in segments] == [2, 2]
    assert int(stats["gap_count_over_threshold"]) == 1


def test_split_frame_by_gaps_soft_hard_thresholds(tmp_path: Path) -> None:
    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    frame = pd.DataFrame(
        [
            {"ts_utc": start, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0},
            {"ts_utc": start + timedelta(minutes=5), "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0},
            {"ts_utc": start + timedelta(minutes=190), "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0},
            {"ts_utc": start + timedelta(minutes=195), "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0},
            {"ts_utc": start + timedelta(minutes=840), "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0},
            {"ts_utc": start + timedelta(minutes=845), "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0},
        ]
    )
    loader = AutoDataLoader(tmp_path / "data")
    segments, stats = loader.split_frame_by_gaps(
        frame,
        "5m",
        gap_bars=3,
        soft_gap_minutes=120,
        hard_gap_minutes=600,
    )

    assert len(segments) == 2
    assert [len(seg) for seg in segments] == [4, 2]
    assert int(stats["gap_count_soft_only"]) == 1
    assert int(stats["gap_count_over_threshold"]) == 1
