from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .html import build_html_report
from .metrics import compute_drawdown_series, compute_metrics
from .plots import generate_plots

LOGGER = logging.getLogger(__name__)


_TRADE_FIELD_ORDER = [
    "entry_ts",
    "exit_ts",
    "side",
    "entry_price",
    "exit_price",
    "pnl",
    "r",
    "fees",
    "spread_cost",
    "slippage_cost",
    "commission_cost",
    "swap_cost",
    "fx_cost",
    "symbol",
    "timeframe",
    "reason_open",
    "reason_close",
    "forced_exit",
]
_EQUITY_FIELD_ORDER = ["idx", "ts", "equity", "drawdown", "drawdown_pct"]
_HEADLINE_KEYS = [
    "trades_count",
    "wins",
    "losses",
    "win_rate_pct",
    "total_pnl",
    "profit_factor",
    "payoff_ratio",
    "equity_start",
    "equity_end",
    "max_drawdown",
    "max_drawdown_pct",
]


@dataclass(slots=True)
class BacktestMeta:
    symbol: str
    timeframe: str
    start: str
    end: str
    variant: str
    mode: str = "backtest"
    price: str = ""
    initial_equity: float = 0.0
    config: str = ""
    data_root: str = ""
    generated_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["symbol"] = str(self.symbol).upper()
        payload["timeframe"] = str(self.timeframe)
        payload["variant"] = str(self.variant)
        payload["mode"] = str(self.mode)
        return payload


@dataclass(slots=True)
class BacktestRun:
    meta: BacktestMeta
    trades: list[Any] = field(default_factory=list)
    equity: list[Any] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class BacktestReporter:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.last_output_dir: Path | None = None

    def generate(
        self,
        run: BacktestRun,
        formats: tuple[str, ...] | list[str] | set[str] | str = ("json", "csv", "png", "html"),
    ) -> dict[str, Any]:
        format_set = _normalize_formats(formats)
        outdir = _next_unique_dir(self.base_dir / _run_folder_name(run.meta))
        outdir.mkdir(parents=True, exist_ok=True)
        charts_dir = outdir / "charts"

        trades = [_normalize_trade(item, run.meta) for item in run.trades]
        equity = _normalize_equity_points(run.equity)
        if not equity:
            equity = _equity_from_trades(
                trades=trades,
                initial_equity=float(run.meta.initial_equity),
                start_ts=run.meta.start,
            )
        equity_series = compute_drawdown_series(equity)
        metrics = compute_metrics(trades, equity_series)

        artifacts: dict[str, Any] = {"charts": {}}
        if "csv" in format_set:
            _write_csv(outdir / "trades.csv", trades, _TRADE_FIELD_ORDER)
            _write_csv(outdir / "equity.csv", equity_series, _EQUITY_FIELD_ORDER)

        if "png" in format_set:
            try:
                chart_abs_paths = generate_plots(trades=trades, equity=equity_series, outdir=charts_dir)
            except RuntimeError as exc:
                LOGGER.warning("PNG report generation skipped: %s", exc)
                chart_abs_paths = {}
            for name, abs_path in chart_abs_paths.items():
                try:
                    relative = Path(abs_path).relative_to(outdir).as_posix()
                except ValueError:
                    relative = Path(abs_path).name
                artifacts["charts"][name] = relative

        meta_payload = run.meta.to_dict()
        report_payload = {
            "meta": meta_payload,
            "metrics": metrics,
            "artifacts": artifacts,
            "extra": run.extra,
        }
        summary_payload = {"meta": meta_payload, "metrics": {key: metrics.get(key) for key in _HEADLINE_KEYS}}

        if "json" in format_set:
            _write_json(outdir / "report.json", report_payload)
            _write_json(outdir / "summary.json", summary_payload)

        if "html" in format_set:
            html_text = build_html_report(
                meta=meta_payload,
                metrics=metrics,
                chart_paths=artifacts.get("charts", {}),
            )
            (outdir / "report.html").write_text(html_text, encoding="utf-8")

        self.last_output_dir = outdir
        return metrics


def _normalize_formats(formats: tuple[str, ...] | list[str] | set[str] | str) -> set[str]:
    allowed = {"json", "csv", "png", "html"}
    if isinstance(formats, str):
        parts = [item.strip().lower() for item in formats.split(",") if item.strip()]
    else:
        parts = [str(item).strip().lower() for item in formats if str(item).strip()]
    selected = {item for item in parts if item in allowed}
    if not selected:
        return {"json", "csv", "png", "html"}
    return selected


def _sanitize_path_part(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "na"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw)
    cleaned = cleaned.strip("-._")
    return cleaned or "na"


def _run_folder_name(meta: BacktestMeta) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return "_".join(
        [
            _sanitize_path_part(meta.symbol).upper(),
            _sanitize_path_part(meta.timeframe),
            _sanitize_path_part(meta.start),
            _sanitize_path_part(meta.end),
            _sanitize_path_part(meta.variant),
            timestamp,
        ]
    )


def _next_unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 1
    while True:
        candidate = Path(f"{path}_{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _to_iso_timestamp(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        dt = value.astimezone(timezone.utc) if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    raw = str(value).strip()
    if not raw:
        return ""
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
        dt = dt.astimezone(timezone.utc) if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return str(value)


def _extract_value(item: Any, keys: tuple[str, ...], default: Any = None) -> Any:
    if isinstance(item, dict):
        for key in keys:
            if key in item:
                return item.get(key)
        return default
    for key in keys:
        if hasattr(item, key):
            return getattr(item, key)
    return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    if converted != converted:  # NaN
        return default
    return converted


def _to_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if converted != converted:  # NaN
        return None
    return converted


def _normalize_trade(item: Any, meta: BacktestMeta) -> dict[str, Any]:
    side_raw = _extract_value(item, ("side",), "")
    symbol_raw = _extract_value(item, ("symbol", "epic"), meta.symbol)
    timeframe_raw = _extract_value(item, ("timeframe", "tf"), meta.timeframe)
    trade = {
        "entry_ts": _to_iso_timestamp(_extract_value(item, ("entry_ts", "entry_time", "open_time_utc", "open_time"))),
        "exit_ts": _to_iso_timestamp(_extract_value(item, ("exit_ts", "exit_time", "close_time_utc", "close_time"))),
        "side": str(side_raw or "").upper(),
        "entry_price": _to_optional_float(_extract_value(item, ("entry_price",))),
        "exit_price": _to_optional_float(_extract_value(item, ("exit_price",))),
        "pnl": _to_float(_extract_value(item, ("pnl",), 0.0)),
        "r": _to_optional_float(_extract_value(item, ("r", "r_multiple"))),
        "fees": _to_float(_extract_value(item, ("fees",), 0.0)),
        "spread_cost": _to_float(_extract_value(item, ("spread_cost",), 0.0)),
        "slippage_cost": _to_float(_extract_value(item, ("slippage_cost",), 0.0)),
        "commission_cost": _to_float(_extract_value(item, ("commission_cost",), 0.0)),
        "swap_cost": _to_float(_extract_value(item, ("swap_cost",), 0.0)),
        "fx_cost": _to_float(_extract_value(item, ("fx_cost",), 0.0)),
        "symbol": str(symbol_raw or meta.symbol).upper(),
        "timeframe": str(timeframe_raw or meta.timeframe),
        "reason_open": str(_extract_value(item, ("reason_open",), "") or ""),
        "reason_close": str(_extract_value(item, ("reason_close", "reason"), "") or ""),
        "forced_exit": bool(_extract_value(item, ("forced_exit",), False)),
    }
    return trade


def _normalize_equity_points(points: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, point in enumerate(points):
        ts = ""
        equity_value: float | None = None
        if isinstance(point, dict):
            ts = _to_iso_timestamp(
                point.get("ts")
                or point.get("timestamp")
                or point.get("time")
                or point.get("exit_ts")
                or point.get("close_time_utc")
            )
            equity_value = _to_optional_float(point.get("equity") or point.get("value") or point.get("balance"))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            ts = _to_iso_timestamp(point[0])
            equity_value = _to_optional_float(point[1])
        else:
            equity_value = _to_optional_float(point)

        if equity_value is None:
            continue
        normalized.append({"idx": idx, "ts": ts, "equity": equity_value})
    return normalized


def _equity_from_trades(trades: list[dict[str, Any]], initial_equity: float, start_ts: str) -> list[dict[str, Any]]:
    equity = float(initial_equity)
    out: list[dict[str, Any]] = [{"idx": 0, "ts": _to_iso_timestamp(start_ts), "equity": equity}]
    for trade in trades:
        equity += float(trade.get("pnl", 0.0) or 0.0)
        out.append(
            {
                "idx": len(out),
                "ts": str(trade.get("exit_ts", "") or ""),
                "equity": equity,
            }
        )
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _ordered_fields(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    seen = set(preferred)
    out = list(preferred)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                out.append(key)
                seen.add(key)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str]) -> None:
    fieldnames = _ordered_fields(rows, preferred_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
