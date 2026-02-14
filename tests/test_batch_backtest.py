from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bot.batch_backtest import (
    ChunkSpec,
    iter_months,
    merge_results,
    partition_resume_chunks,
    resolve_parquet_files,
)


def _write_month_parquet(
    data_root: Path,
    *,
    symbol: str,
    price_mode: str,
    timeframe: str,
    year: int,
    month: int,
) -> Path:
    target = (
        data_root
        / "local_csv"
        / symbol
        / price_mode
        / timeframe
        / f"{year:04d}"
        / f"{month:02d}.parquet"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "ts_utc": datetime(year, month, 1, tzinfo=timezone.utc),
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
            }
        ]
    )
    frame.to_parquet(target, index=False, engine="pyarrow")
    return target


def _write_success_chunk(out_dir: Path, trades: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(out_dir / "trades.parquet", index=False, engine="pyarrow")
    (out_dir / "metrics.json").write_text(json.dumps({"trades": int(len(trades))}), encoding="utf-8")
    (out_dir / "SUCCESS.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")


def test_iter_months_handles_range_across_month_boundaries() -> None:
    start = datetime(2004, 6, 1, tzinfo=timezone.utc)
    end = datetime(2004, 9, 1, tzinfo=timezone.utc)
    assert iter_months(start, end) == [(2004, 6), (2004, 7), (2004, 8)]


def test_resolve_parquet_files_includes_warmup_spill_month(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_month_parquet(
        data_root,
        symbol="XAUUSD",
        price_mode="MID",
        timeframe="1m",
        year=2023,
        month=12,
    )
    _write_month_parquet(
        data_root,
        symbol="XAUUSD",
        price_mode="MID",
        timeframe="1m",
        year=2024,
        month=1,
    )

    files, missing, warmup_start = resolve_parquet_files(
        data_root=data_root,
        symbol="XAUUSD",
        price_mode="MID",
        timeframe="1m",
        start=datetime(2024, 1, 15, tzinfo=timezone.utc),
        end=datetime(2024, 2, 1, tzinfo=timezone.utc),
        warmup_days=20,
    )

    assert warmup_start == datetime(2023, 12, 26, tzinfo=timezone.utc)
    assert missing == []
    assert len(files) == 2
    assert files[0].name == "12.parquet"
    assert files[1].name == "01.parquet"


def test_merge_results_sorts_and_deduplicates_trade_id(tmp_path: Path) -> None:
    out_root = tmp_path / "runs"
    cols = [
        "symbol",
        "timeframe",
        "price_mode",
        "open_time_utc",
        "close_time_utc",
        "side",
        "entry_price",
        "exit_price",
        "size",
        "pnl",
        "fees",
        "r_multiple",
        "score",
        "forced_exit",
        "reason_open",
        "reason_close",
        "trade_id",
    ]

    jan = pd.DataFrame(
        [
            {
                "symbol": "XAUUSD",
                "timeframe": "1m",
                "price_mode": "MID",
                "open_time_utc": "2024-01-01T00:00:00+00:00",
                "close_time_utc": "2024-01-01T01:00:00+00:00",
                "side": "long",
                "entry_price": 2000.0,
                "exit_price": 2001.0,
                "size": 1.0,
                "pnl": 10.0,
                "fees": 1.0,
                "r_multiple": 1.0,
                "score": 70.0,
                "forced_exit": False,
                "reason_open": "SIGNAL",
                "reason_close": "TP",
                "trade_id": "same-trade",
            }
        ],
        columns=cols,
    )
    feb = pd.DataFrame(
        [
            {
                "symbol": "XAUUSD",
                "timeframe": "1m",
                "price_mode": "MID",
                "open_time_utc": "2024-02-01T00:00:00+00:00",
                "close_time_utc": "2024-02-01T01:00:00+00:00",
                "side": "short",
                "entry_price": 2100.0,
                "exit_price": 2098.0,
                "size": 1.0,
                "pnl": 20.0,
                "fees": 1.0,
                "r_multiple": 2.0,
                "score": 75.0,
                "forced_exit": False,
                "reason_open": "SIGNAL",
                "reason_close": "TP",
                "trade_id": "trade-2",
            },
            {
                "symbol": "XAUUSD",
                "timeframe": "1m",
                "price_mode": "MID",
                "open_time_utc": "2024-01-01T00:00:00+00:00",
                "close_time_utc": "2024-01-01T01:00:00+00:00",
                "side": "long",
                "entry_price": 2000.0,
                "exit_price": 2001.0,
                "size": 1.0,
                "pnl": 10.0,
                "fees": 1.0,
                "r_multiple": 1.0,
                "score": 70.0,
                "forced_exit": False,
                "reason_open": "SIGNAL",
                "reason_close": "TP",
                "trade_id": "same-trade",
            },
        ],
        columns=cols,
    )

    _write_success_chunk(out_root / "2024-01", jan)
    _write_success_chunk(out_root / "2024-02", feb)

    metrics = merge_results(out_root, initial_equity=10000.0)

    combined = pd.read_parquet(out_root / "ALL" / "combined_trades.parquet", engine="pyarrow")
    assert list(combined["trade_id"]) == ["same-trade", "trade-2"]
    assert int(metrics["trades"]) == 2
    assert float(metrics["net_pnl"]) == 28.0


def test_resume_skips_successful_chunks(tmp_path: Path) -> None:
    c1 = ChunkSpec(
        chunk_id="2024-01",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 2, 1, tzinfo=timezone.utc),
        out_dir=tmp_path / "2024-01",
    )
    c2 = ChunkSpec(
        chunk_id="2024-02",
        start=datetime(2024, 2, 1, tzinfo=timezone.utc),
        end=datetime(2024, 3, 1, tzinfo=timezone.utc),
        out_dir=tmp_path / "2024-02",
    )

    _write_success_chunk(c1.out_dir, pd.DataFrame(columns=["trade_id"]))

    pending, skipped = partition_resume_chunks([c1, c2])

    assert skipped == ["2024-01"]
    assert [item.chunk_id for item in pending] == ["2024-02"]
