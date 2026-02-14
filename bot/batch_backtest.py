from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd

from bot.backtest.data_provider import normalize_timeframe


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iter_months(dt_start: datetime, dt_end: datetime) -> list[tuple[int, int]]:
    start = _to_utc(dt_start)
    end = _to_utc(dt_end)
    if start >= end:
        return []
    current = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    out: list[tuple[int, int]] = []
    while current < end:
        out.append((current.year, current.month))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            current = datetime(current.year, current.month + 1, 1, tzinfo=timezone.utc)
    return out


def parquet_path(
    data_root: str | Path,
    symbol: str,
    price_mode: str,
    timeframe: str,
    year: int,
    month: int,
) -> Path:
    tf_norm = normalize_timeframe(timeframe)
    return (
        Path(data_root)
        / "local_csv"
        / symbol.strip().upper()
        / price_mode.strip().upper()
        / tf_norm
        / f"{year:04d}"
        / f"{month:02d}.parquet"
    )


def resolve_parquet_files(
    data_root: str | Path,
    symbol: str,
    price_mode: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    warmup_days: int,
) -> tuple[list[Path], list[str], datetime]:
    start_utc = _to_utc(start)
    end_utc = _to_utc(end)
    warmup_start = start_utc - timedelta(days=max(0, int(warmup_days)))
    files: list[Path] = []
    missing: list[str] = []
    for year, month in iter_months(warmup_start, end_utc):
        path = parquet_path(data_root, symbol, price_mode, timeframe, year, month)
        if path.exists():
            files.append(path)
        else:
            missing.append(f"symbol={symbol.upper()} side={price_mode.upper()} tf={normalize_timeframe(timeframe)} year={year} month={month:02d}")
    return files, missing, warmup_start


def _month_start(dt: datetime) -> datetime:
    dt = _to_utc(dt)
    return datetime(dt.year, dt.month, 1, tzinfo=timezone.utc)


def _next_month(dt: datetime) -> datetime:
    if dt.month == 12:
        return datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(dt.year, dt.month + 1, 1, tzinfo=timezone.utc)


@dataclass(slots=True)
class ChunkSpec:
    chunk_id: str
    start: datetime
    end: datetime
    out_dir: Path


@dataclass(slots=True)
class WorkerResult:
    chunk_id: str
    out_dir: Path
    success: bool
    returncode: int
    stdout: str
    stderr: str
    command: list[str]


def build_chunks(start: datetime, end: datetime, *, chunk: str, out_root: Path) -> list[ChunkSpec]:
    start_utc = _to_utc(start)
    end_utc = _to_utc(end)
    mode = chunk.strip().lower()
    if mode not in {"monthly", "weekly", "daily"}:
        raise ValueError("chunk must be one of: monthly, weekly, daily")
    chunks: list[ChunkSpec] = []
    if mode == "monthly":
        current = _month_start(start_utc)
        while current < end_utc:
            nxt = _next_month(current)
            chunk_start = current if current >= start_utc else start_utc
            chunk_end = nxt if nxt <= end_utc else end_utc
            chunk_id = f"{current.year:04d}-{current.month:02d}"
            chunks.append(ChunkSpec(chunk_id=chunk_id, start=chunk_start, end=chunk_end, out_dir=out_root / chunk_id))
            current = nxt
        return chunks

    step = timedelta(days=7 if mode == "weekly" else 1)
    current = start_utc
    idx = 0
    while current < end_utc:
        nxt = current + step
        chunk_end = nxt if nxt <= end_utc else end_utc
        chunk_id = (
            f"{current.year:04d}-W{current.isocalendar().week:02d}"
            if mode == "weekly"
            else f"{current.year:04d}-{current.month:02d}-{current.day:02d}"
        )
        if chunk_id in {c.chunk_id for c in chunks}:
            chunk_id = f"{chunk_id}-{idx:02d}"
        chunks.append(ChunkSpec(chunk_id=chunk_id, start=current, end=chunk_end, out_dir=out_root / chunk_id))
        current = chunk_end
        idx += 1
    return chunks


def chunk_is_successful(out_dir: Path) -> bool:
    return (
        (out_dir / "SUCCESS.json").exists()
        and (out_dir / "trades.parquet").exists()
        and (out_dir / "metrics.json").exists()
    )


def partition_resume_chunks(chunks: list[ChunkSpec]) -> tuple[list[ChunkSpec], list[str]]:
    pending: list[ChunkSpec] = []
    skipped: list[str] = []
    for spec in chunks:
        if chunk_is_successful(spec.out_dir):
            skipped.append(spec.chunk_id)
        else:
            pending.append(spec)
    return pending, skipped


def run_worker_subprocess(
    *,
    main_script: Path,
    config_path: Path,
    data_root: Path,
    chunk: ChunkSpec,
    symbol: str,
    price_mode: str,
    timeframe: str,
    warmup_days: int,
    initial_equity: float,
) -> WorkerResult:
    chunk.out_dir.mkdir(parents=True, exist_ok=True)
    state_path = chunk.out_dir / "state.db"
    command = [
        sys.executable,
        str(main_script),
        "--batch-worker",
        "--symbol",
        symbol.upper(),
        "--price-mode",
        price_mode.upper(),
        "--timeframe",
        normalize_timeframe(timeframe),
        "--start",
        chunk.start.isoformat(),
        "--end",
        chunk.end.isoformat(),
        "--warmup-days",
        str(int(warmup_days)),
        "--out-dir",
        str(chunk.out_dir),
        "--state-path",
        str(state_path),
        "--initial-equity",
        str(float(initial_equity)),
        "--backtest-data-root",
        str(data_root),
        "--config",
        str(config_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    return WorkerResult(
        chunk_id=chunk.chunk_id,
        out_dir=chunk.out_dir,
        success=completed.returncode == 0 and chunk_is_successful(chunk.out_dir),
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        command=command,
    )


def merge_results(out_root: str | Path, initial_equity: float) -> dict[str, Any]:
    root = Path(out_root)
    all_dir = root / "ALL"
    all_dir.mkdir(parents=True, exist_ok=True)

    trade_frames: list[pd.DataFrame] = []
    chunk_ids: list[str] = []
    for chunk_dir in sorted(item for item in root.iterdir() if item.is_dir() and item.name != "ALL"):
        if not chunk_is_successful(chunk_dir):
            continue
        trades_path = chunk_dir / "trades.parquet"
        if not trades_path.exists():
            continue
        frame = pd.read_parquet(trades_path, engine="pyarrow")
        frame["chunk_id"] = chunk_dir.name
        trade_frames.append(frame)
        chunk_ids.append(chunk_dir.name)

    required_cols = [
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

    if not trade_frames:
        empty = pd.DataFrame(columns=required_cols + ["chunk_id"])
        empty.to_parquet(all_dir / "combined_trades.parquet", index=False, engine="pyarrow")
        pd.DataFrame(columns=["idx", "trade_id", "close_time_utc", "equity"]).to_csv(
            all_dir / "combined_equity.csv",
            index=False,
        )
        metrics = {
            "chunks_merged": 0,
            "chunk_ids": [],
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_fees": 0.0,
            "net_pnl": 0.0,
            "expectancy": 0.0,
            "avg_r": 0.0,
            "max_drawdown": 0.0,
            "initial_equity": float(initial_equity),
            "equity_end": float(initial_equity),
        }
        (all_dir / "combined_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")
        return metrics

    combined = pd.concat(trade_frames, ignore_index=True)
    for col in required_cols:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined["open_time_utc"] = pd.to_datetime(combined["open_time_utc"], utc=True, errors="coerce")
    combined["close_time_utc"] = pd.to_datetime(combined["close_time_utc"], utc=True, errors="coerce")
    combined["sort_time"] = combined["close_time_utc"].fillna(combined["open_time_utc"])
    combined = combined.sort_values(["sort_time", "open_time_utc"], kind="mergesort").reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["trade_id"], keep="first").reset_index(drop=True)
    combined.drop(columns=["sort_time"], inplace=True)

    combined.to_parquet(all_dir / "combined_trades.parquet", index=False, engine="pyarrow")

    pnl = pd.to_numeric(combined["pnl"], errors="coerce").fillna(0.0)
    fees = pd.to_numeric(combined["fees"], errors="coerce").fillna(0.0)
    net = pnl - fees
    equity_values = [float(initial_equity)]
    for value in net:
        equity_values.append(equity_values[-1] + float(value))
    replay = pd.DataFrame(
        {
            "idx": range(len(combined)),
            "trade_id": combined["trade_id"].astype(str),
            "close_time_utc": combined["close_time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "equity": equity_values[1:],
        }
    )
    replay.to_csv(all_dir / "combined_equity.csv", index=False)

    curve = pd.Series(equity_values[1:] if len(equity_values) > 1 else [float(initial_equity)], dtype=float)
    rolling_peak = curve.cummax()
    drawdown = (rolling_peak - curve).max()
    wins = int((pnl > 0).sum())
    losses = int((pnl <= 0).sum())
    trades = int(len(combined))
    total_pnl = float(pnl.sum())
    total_fees = float(fees.sum())
    net_pnl = float(net.sum())
    avg_r = float(pd.to_numeric(combined["r_multiple"], errors="coerce").dropna().mean()) if trades else 0.0
    metrics = {
        "chunks_merged": len(chunk_ids),
        "chunk_ids": chunk_ids,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / trades) if trades else 0.0,
        "total_pnl": total_pnl,
        "total_fees": total_fees,
        "net_pnl": net_pnl,
        "expectancy": (net_pnl / trades) if trades else 0.0,
        "avg_r": avg_r if avg_r == avg_r else 0.0,
        "max_drawdown": float(drawdown) if drawdown == drawdown else 0.0,
        "initial_equity": float(initial_equity),
        "equity_end": float(initial_equity + net_pnl),
    }
    (all_dir / "combined_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")
    return metrics


def make_trade_id(*, open_time_utc: str, side: str, entry_price: float, chunk_id: str) -> str:
    raw = f"{open_time_utc}|{side}|{entry_price:.8f}|{chunk_id}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def orchestrate_batch(
    *,
    main_script: Path,
    config_path: Path,
    data_root: Path,
    symbol: str,
    price_mode: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    chunk: str,
    workers: int,
    warmup_days: int,
    out_root: Path,
    initial_equity: float,
    continue_on_error: bool,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    specs = build_chunks(start, end, chunk=chunk, out_root=out_root)
    pending, skipped = partition_resume_chunks(specs)

    results: list[WorkerResult] = []
    failed: list[WorkerResult] = []
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            future_map: dict[Future[WorkerResult], ChunkSpec] = {
                pool.submit(
                    run_worker_subprocess,
                    main_script=main_script,
                    config_path=config_path,
                    data_root=data_root,
                    chunk=spec,
                    symbol=symbol,
                    price_mode=price_mode,
                    timeframe=timeframe,
                    warmup_days=warmup_days,
                    initial_equity=initial_equity,
                ): spec
                for spec in pending
            }
            for future in as_completed(future_map):
                result = future.result()
                results.append(result)
                if not result.success:
                    failed.append(result)
                    if not continue_on_error:
                        for other in future_map:
                            if other is future:
                                continue
                            other.cancel()
                        break

    for result in results:
        marker = result.out_dir / ("SUCCESS.json" if result.success else "ERROR.json")
        payload = {
            "chunk_id": result.chunk_id,
            "returncode": result.returncode,
            "success": result.success,
            "command": result.command,
        }
        if not result.success:
            payload["stderr"] = result.stderr[-4000:]
            payload["stdout"] = result.stdout[-2000:]
        marker.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    if failed and not continue_on_error:
        first = failed[0]
        raise RuntimeError(
            f"Batch worker failed for chunk {first.chunk_id}: rc={first.returncode}\n"
            f"stderr={first.stderr[-1200:]}"
        )

    combined = merge_results(out_root, initial_equity)
    summary = {
        "symbol": symbol.upper(),
        "price_mode": price_mode.upper(),
        "timeframe": normalize_timeframe(timeframe),
        "start": _to_utc(start).isoformat(),
        "end": _to_utc(end).isoformat(),
        "chunk": chunk,
        "workers": max(1, int(workers)),
        "warmup_days": max(0, int(warmup_days)),
        "out_root": str(out_root),
        "chunks_total": len(specs),
        "chunks_skipped_resume": len(skipped),
        "chunks_ran": len(pending),
        "chunks_failed": len(failed),
        "combined_metrics": combined,
    }
    (out_root / "batch_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary
