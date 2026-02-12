from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bot.backtest.data_provider import AutoDataLoader, MissingDataError, normalize_timeframe
from bot.backtest.runner import BacktestRunner
from bot.config import load_config


def _parse_symbols(value: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in value.split(","):
        symbol = item.strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _is_date_only(value: str) -> bool:
    raw = value.strip()
    return len(raw) == 10 and raw[4] == "-" and raw[7] == "-"


def _parse_datetime(value: str) -> datetime:
    raw = value.strip()
    normalized = raw.replace(" ", "T")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_start(value: str) -> datetime:
    return _parse_datetime(value)


def _parse_end(value: str) -> datetime:
    dt = _parse_datetime(value)
    if _is_date_only(value):
        return dt + timedelta(days=1)
    return dt


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return base / path


def _autofetch(
    *,
    script_path: Path,
    symbols: list[str],
    timeframe: str,
    start_raw: str,
    end_raw: str,
) -> None:
    if not script_path.exists():
        raise RuntimeError(f"Autofetch requested, but script not found: {script_path}")
    cmd = [
        sys.executable,
        str(script_path),
        "--symbols",
        ",".join(symbols),
        "--tf",
        timeframe,
        "--start",
        start_raw,
        "--end",
        end_raw,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        details = stderr or stdout or f"exit={completed.returncode}"
        raise RuntimeError(f"Autofetch failed: {details}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-asset parquet backtest runner")
    parser.add_argument("--symbols", default="XAUUSD,EURUSD,US100,US500,BTCUSD")
    parser.add_argument("--tf", default="5m")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--price", choices=["mid", "bid", "ask"], default="mid")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--source-priority", default="")
    parser.add_argument("--slippage-points", type=float, default=0.0)
    parser.add_argument("--slippage-atr-multiplier", type=float, default=0.0)
    parser.add_argument("--autofetch", action="store_true")
    parser.add_argument("--fetch-script", default="fetch_market_data.py")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config_path = _resolve_path(project_root, args.config)
    data_root = _resolve_path(project_root, args.data_root)
    fetch_script = _resolve_path(project_root, args.fetch_script)
    symbols = _parse_symbols(args.symbols)
    if not symbols:
        parser.error("--symbols resolved to empty set")
    timeframe = normalize_timeframe(args.tf)
    start = _parse_start(args.start)
    end = _parse_end(args.end)
    if start >= end:
        parser.error("--start must be before --end")

    source_priority = [item.strip() for item in args.source_priority.split(",") if item.strip()]
    config = load_config(config_path)
    loader = AutoDataLoader(data_root=data_root, source_priority=source_priority)
    runner = BacktestRunner(config=config, data_loader=loader)

    attempted_fetch = False
    while True:
        try:
            batch = runner.run(
                symbols=symbols,
                timeframe=timeframe,
                start=start,
                end=end,
                price_mode=args.price,
                slippage_points=args.slippage_points,
                slippage_atr_multiplier=args.slippage_atr_multiplier,
            )
            print(json.dumps(batch.to_dict(), indent=2, ensure_ascii=True))
            return 0
        except MissingDataError as exc:
            for item in exc.missing:
                print(f"MISSING: {item.to_line()}", file=sys.stderr)
            if args.autofetch and not attempted_fetch:
                attempted_fetch = True
                try:
                    _autofetch(
                        script_path=fetch_script,
                        symbols=symbols,
                        timeframe=timeframe,
                        start_raw=args.start,
                        end_raw=args.end,
                    )
                    continue
                except Exception as fetch_exc:  # noqa: BLE001
                    print(f"AUTOFETCH_ERROR: {fetch_exc}", file=sys.stderr)
                    return 2
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
