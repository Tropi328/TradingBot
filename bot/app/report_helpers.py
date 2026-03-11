"""Backtest report formatting, variant parsing, and CSV export — extracted from app_main.py."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import webbrowser
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bot.app.config_helpers import parse_epics_csv
from bot.app.factories import _asset_from_template
from bot.backtest.engine import BacktestVariant
from bot.config import AppConfig, AssetConfig
from bot.research.objective import OBJECTIVE_FAIL_VALUE, augment_report

LOGGER = logging.getLogger("trading_bot")


def _is_date_only(value: str) -> bool:
    raw = value.strip()
    return len(raw) == 10 and raw[4] == "-" and raw[7] == "-"


def _parse_backtest_datetime(value: str, *, end_value: bool = False) -> datetime:
    raw = value.strip()
    normalized = raw.replace(" ", "T")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC)
    if end_value and _is_date_only(raw):
        dt += timedelta(days=1)
    return dt


def _validate_batch_data_root(data_root: Path) -> None:
    if not data_root.exists():
        raise RuntimeError(
            f"Batch data root does not exist: {data_root}. "
            "Expected e.g. --data-root data with shards under data/local_csv/<SYMBOL>/<PRICE_MODE>/<TF>/YYYY/MM.parquet."
        )
    local_csv = data_root / "local_csv"
    if not local_csv.exists():
        LOGGER.warning(
            "Batch data root has no local_csv directory: %s (expected local_csv/<SYMBOL>/<PRICE_MODE>/<TF>/YYYY/MM.parquet)",
            data_root,
        )


def _autofetch_backtest_data(
    *,
    fetch_script: Path,
    symbols: list[str],
    timeframe: str,
    start_raw: str,
    end_raw: str,
) -> None:
    if not fetch_script.exists():
        raise RuntimeError(f"--backtest-autofetch requested, but script not found: {fetch_script}")
    command = [
        sys.executable,
        str(fetch_script),
        "--symbols",
        ",".join(symbols),
        "--tf",
        timeframe,
        "--start",
        start_raw,
        "--end",
        end_raw,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip() or f"exit={result.returncode}"
        raise RuntimeError(f"Autofetch failed: {details}")


def _backtest_symbols(args: argparse.Namespace, assets: list[AssetConfig]) -> list[str]:
    symbols = parse_epics_csv(args.backtest_symbols)
    if symbols:
        return symbols
    if args.backtest_epic:
        return [str(args.backtest_epic).strip().upper()]
    env_epic = (os.getenv("CAPITAL_EPIC") or "").strip().upper()
    if env_epic:
        return [env_epic]
    trade_enabled = [asset.epic for asset in assets if asset.trade_enabled]
    if trade_enabled:
        return trade_enabled
    return [assets[0].epic]


def _asset_map_for_symbols(symbols: list[str], assets: list[AssetConfig], config: AppConfig) -> dict[str, AssetConfig]:
    by_epic = {asset.epic.upper(): asset for asset in assets}
    template = assets[0] if assets else AssetConfig(**config.instrument.model_dump(), trade_enabled=True)
    out: dict[str, AssetConfig] = {}
    for symbol in symbols:
        if symbol in by_epic:
            out[symbol] = by_epic[symbol]
        else:
            out[symbol] = _asset_from_template(symbol, template, True)
    return out


def _parse_backtest_variants(raw: str) -> list[BacktestVariant]:
    mapping: dict[str, BacktestVariant] = {
        "W0": BacktestVariant(
            code="W0",
            reaction_timeout_reset=False,
            soft_reason_penalties=False,
            thresholds_v2=False,
            dynamic_threshold_bump=False,
        ),
        "W1": BacktestVariant(
            code="W1",
            reaction_timeout_reset=True,
            soft_reason_penalties=False,
            thresholds_v2=False,
            dynamic_threshold_bump=False,
        ),
        "W2": BacktestVariant(
            code="W2",
            reaction_timeout_reset=True,
            soft_reason_penalties=True,
            thresholds_v2=False,
            dynamic_threshold_bump=False,
        ),
        "W3": BacktestVariant(
            code="W3",
            reaction_timeout_reset=True,
            soft_reason_penalties=True,
            thresholds_v2=True,
            dynamic_threshold_bump=True,
        ),
    }
    variants: list[BacktestVariant] = []
    seen: set[str] = set()
    for item in str(raw or "W0").split(","):
        code = item.strip().upper()
        if not code or code in seen:
            continue
        if code not in mapping:
            raise RuntimeError(f"Unknown backtest variant '{code}'. Allowed: W0,W1,W2,W3")
        variants.append(mapping[code])
        seen.add(code)
    if not variants:
        variants.append(mapping["W0"])
    return variants


def _first_variant_code(raw: str) -> str:
    try:
        variants = _parse_backtest_variants(raw)
    except (ValueError, KeyError, IndexError):
        return "W0"
    return variants[0].code if variants else "W0"


def _parse_report_formats(raw: str) -> tuple[str, ...]:
    allowed = {"json", "csv", "png", "html"}
    parts = [item.strip().lower() for item in str(raw or "").split(",") if item.strip()]
    selected: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part not in allowed:
            LOGGER.warning("Unknown report format '%s' ignored (allowed: json,csv,png,html)", part)
            continue
        if part in seen:
            continue
        selected.append(part)
        seen.add(part)
    if not selected:
        return ("json", "csv", "png", "html")
    return tuple(selected)


def _open_report_html(path: Path) -> None:
    try:
        webbrowser.open(path.resolve().as_uri())
    except Exception as exc:
        LOGGER.warning("Failed to open report HTML '%s': %s", path, exc)


def _coerce_iso_utc(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        dt = value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return dt.isoformat()
    raw = str(value).strip()
    if not raw:
        return ""
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return str(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.isoformat()


def _trade_value(item: object, keys: tuple[str, ...]) -> object:
    if isinstance(item, dict):
        for key in keys:
            if key in item:
                return item.get(key)
        return None
    for key in keys:
        if hasattr(item, key):
            return getattr(item, key)
    return None


def _trade_time_bounds(trades: list[object], *, default_start: str, default_end: str) -> tuple[str, str]:
    entry_ts: list[str] = []
    exit_ts: list[str] = []
    for trade in trades:
        entry_raw = _trade_value(trade, ("entry_ts", "entry_time", "open_time_utc", "open_time"))
        exit_raw = _trade_value(trade, ("exit_ts", "exit_time", "close_time_utc", "close_time"))
        entry_iso = _coerce_iso_utc(entry_raw)
        exit_iso = _coerce_iso_utc(exit_raw)
        if entry_iso:
            entry_ts.append(entry_iso)
        if exit_iso:
            exit_ts.append(exit_iso)
    start = min(entry_ts) if entry_ts else default_start
    end = max(exit_ts) if exit_ts else default_end
    return start, end


def _month_label(value: str, fallback: datetime) -> str:
    raw = value.strip()
    if len(raw) >= 7 and raw[4] == "-":
        return raw[:7]
    return fallback.strftime("%Y-%m")


def _variant_report_filename(
    *, variant: BacktestVariant, start_raw: str, end_raw: str, start_dt: datetime, end_dt: datetime, symbol: str
) -> str:
    start_label = _month_label(start_raw, start_dt)
    end_label = _month_label(end_raw, end_dt)
    return f"{variant.code}_{start_label}_{end_label}_{symbol}.json"


def _top3_blockers(report_dict: dict[str, object]) -> str:
    blockers_raw = report_dict.get("top_blockers")
    if not isinstance(blockers_raw, dict) or not blockers_raw:
        return "-"
    items = list(blockers_raw.items())[:3]
    return ",".join(f"{k}:{v}" for k, v in items)


def _log_variant_comparison(*, variant_payloads: dict[str, dict[str, object]], symbols: list[str]) -> None:
    LOGGER.info("Backtest variant comparison:")
    LOGGER.info(
        "variant | symbol | trades | win_rate | pnl | expectancy | avg_r | payoff | pf | max_dd | candidates | top3_blockers"
    )
    for code, payload in variant_payloads.items():
        reports_raw = payload.get("reports")
        if not isinstance(reports_raw, dict):
            continue
        for symbol in symbols:
            report = reports_raw.get(symbol)
            if not isinstance(report, dict):
                continue
            LOGGER.info(
                "%s | %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %s | %s",
                code,
                symbol,
                report.get("trades", 0),
                float(report.get("win_rate", 0.0)),
                float(report.get("total_pnl", 0.0)),
                float(report.get("expectancy", 0.0)),
                float(report.get("avg_r", 0.0)),
                float(report.get("payoff_ratio", 0.0)),
                float(report.get("profit_factor", 0.0)),
                float(report.get("max_drawdown", 0.0)),
                report.get("signal_candidates", 0),
                _top3_blockers(report),
            )


def _log_daily_gate_comparison(*, gate_payloads: dict[str, dict[str, object]], symbols: list[str]) -> None:
    LOGGER.info("Daily gate comparison:")
    LOGGER.info(
        "gate_mode | symbol | trades | win_rate | pnl | max_dd | expectancy | avg_r | flat_days | long_days | short_days | blocked_by_gate"
    )
    for gate_mode, payload in gate_payloads.items():
        reports_raw = payload.get("reports")
        if not isinstance(reports_raw, dict):
            continue
        for symbol in symbols:
            report = reports_raw.get(symbol)
            if not isinstance(report, dict):
                continue
            days = report.get("daily_gate_bias_days", {})
            if not isinstance(days, dict):
                days = {}
            LOGGER.info(
                "%s | %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %s | %s | %s | %s",
                gate_mode,
                symbol,
                report.get("trades", 0),
                float(report.get("win_rate", 0.0)),
                float(report.get("total_pnl", 0.0)),
                float(report.get("max_drawdown", 0.0)),
                float(report.get("expectancy", 0.0)),
                float(report.get("avg_r", 0.0)),
                int(days.get("FLAT", 0)),
                int(days.get("LONG", 0)),
                int(days.get("SHORT", 0)),
                int(report.get("blocked_by_gate", 0)),
            )


def _augment_report_with_research_fields(
    report_dict: dict[str, object],
    *,
    config: AppConfig,
    oos_pass: bool | None = None,
) -> dict[str, object]:
    return augment_report(
        report_dict,
        initial_equity=float(config.risk.equity),
        dd_cap_pct=float(config.research.dd_cap_pct),
        dd_cap_basis=str(config.research.dd_cap_basis),
        min_trades_oos=int(config.research.min_trades_oos),
        objective_mode=str(config.research.objective_mode),
        oos_pass=oos_pass,
    )


def _extract_source_reports_from_report_dir(report_dir: Path) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    if not report_dir.exists():
        return reports
    for report_path in sorted(report_dir.rglob("report.json")):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            LOGGER.warning("Research: failed to parse %s", report_path)
            continue
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        source_report: dict[str, object] | None = None
        extra = payload.get("extra")
        if isinstance(extra, dict):
            src = extra.get("source_report")
            if isinstance(src, dict):
                if isinstance(src.get("aggregate"), dict):
                    source_report = dict(src.get("aggregate") or {})
                else:
                    source_report = dict(src)
        if source_report is None:
            source_report = {
                "trades": metrics.get("trades_count", 0),
                "wins": metrics.get("wins", 0),
                "losses": metrics.get("losses", 0),
                "total_pnl": metrics.get("total_pnl", 0.0),
                "total_pnl_net": metrics.get("total_pnl_net", metrics.get("total_pnl", 0.0)),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "max_drawdown_pct_peak": metrics.get("max_drawdown_pct_peak", metrics.get("max_drawdown_pct", 0.0)),
                "max_drawdown_pct_initial": metrics.get("max_drawdown_pct_initial", 0.0),
                "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
                "expectancy": metrics.get("avg_pnl", 0.0),
                "avg_r": 0.0,
                "spread_cost_sum": metrics.get("spread_cost_sum", 0.0),
                "slippage_cost_sum": metrics.get("slippage_cost_sum", 0.0),
                "commission_cost_sum": metrics.get("commission_cost_sum", 0.0),
                "swap_cost_sum": metrics.get("swap_cost_sum", 0.0),
                "fx_cost_sum": metrics.get("fx_cost_sum", 0.0),
                "constraint_dd_cap_pass_peak": metrics.get("constraint_dd_cap_pass_peak"),
                "constraint_dd_cap_pass_initial": metrics.get("constraint_dd_cap_pass_initial"),
                "constraint_dd_cap_pass": metrics.get("constraint_dd_cap_pass"),
                "objective_value": metrics.get("objective_value"),
                "blocked_by_reason": {},
                "anomaly_flags": metrics.get("anomaly_flags", []),
                "orders_submitted": metrics.get("orders_submitted", 0),
                "trades_filled": metrics.get("trades_filled", 0),
                "profit_factor_net": metrics.get("profit_factor_net", metrics.get("profit_factor", 0.0)),
                "payoff_ratio": metrics.get("payoff_ratio", 0.0),
            }
        source_report.setdefault("anomaly_flags", metrics.get("anomaly_flags", []))
        source_report.setdefault("orders_submitted", metrics.get("orders_submitted", 0))
        source_report.setdefault("trades_filled", metrics.get("trades_filled", 0))
        source_report.setdefault(
            "profit_factor_net", metrics.get("profit_factor_net", metrics.get("profit_factor", 0.0))
        )
        source_report.setdefault("payoff_ratio", metrics.get("payoff_ratio", 0.0))
        meta = payload.get("meta")
        if isinstance(meta, dict):
            symbol = str(meta.get("symbol", "")).strip().upper()
            if symbol and "symbol" not in source_report:
                source_report["symbol"] = symbol
        reports.append(source_report)
    return reports


def _empty_aggregate_summary() -> dict[str, object]:
    return {
        "reports_count": 0,
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "total_pnl_net": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_pct_peak": 0.0,
        "max_drawdown_pct_initial": 0.0,
        "dd_ref_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "expectancy": 0.0,
        "expectancy_net": 0.0,
        "avg_r": 0.0,
        "profit_factor_net": 0.0,
        "payoff_ratio": 0.0,
        "cost_breakdown_net": {},
        "blocked_by_reason": {},
        "anomaly_flags": [],
        "orders_submitted": 0,
        "trades_filled": 0,
        "oos_pass": False,
        "constraint_dd_cap_pass_peak": False,
        "constraint_dd_cap_pass_initial": False,
        "constraint_dd_cap_pass": False,
        "quality_pass": False,
        "quality_reasons": [],
        "anomaly_flags_is": [],
        "anomaly_flags_oos": [],
        "objective_value": OBJECTIVE_FAIL_VALUE,
    }


def _safe_json_primitive(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_json_primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_primitive(v) for v in value]
    return str(value)


def _write_csv(path: Path, *, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _safe_json_primitive(row.get(key)) for key in fieldnames})
