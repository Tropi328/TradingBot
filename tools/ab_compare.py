#!/usr/bin/env python
"""
ab_compare.py – A/B backtest comparison between baseline and safety-preset configs.

Usage
-----
    python tools/ab_compare.py                              # defaults
    python tools/ab_compare.py --symbol XAUUSD --start 2024-01-01 --end 2025-02-01
    python tools/ab_compare.py --config-a config.yaml --config-b configs/variants/preset_pnl_safe.yaml

Output
------
- Console table with side-by-side metrics
- reports/ab_compare_<timestamp>.json with full details
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure the bot package is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from bot.backtest.data_provider import AutoDataLoader
from bot.backtest.engine import BacktestVariant, run_backtest_multi_strategy
from bot.backtest.runner import BacktestRunner
from bot.config import AppConfig, AssetConfig, load_config

LOGGER = logging.getLogger("ab_compare")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B backtest comparison")
    p.add_argument("--config-a", default="config.yaml", help="Baseline config (A)")
    p.add_argument("--config-b", default="configs/variants/preset_pnl_safe.yaml", help="New/safe config (B)")
    p.add_argument("--label-a", default="BASELINE", help="Label for config A")
    p.add_argument("--label-b", default="PNL_SAFE", help="Label for config B")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2025-02-01")
    p.add_argument("--price-mode", default="mid", choices=["mid", "bid", "ask"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="reports")
    p.add_argument("--slippage-points", type=float, default=0.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading (shared across both runs)
# ---------------------------------------------------------------------------
def _load_candles(
    *,
    data_root: str,
    symbol: str,
    start: datetime,
    end: datetime,
    price_mode: str,
) -> tuple[list[Any], float, str, dict[str, Any]]:
    """Load candles once and return (candles, assumed_spread, spread_mode, diagnostics)."""
    loader = AutoDataLoader(data_root=data_root)
    loaded = loader.load_symbol_data(
        symbol=symbol,
        timeframe="5m",
        start=start,
        end=end,
        price_mode=price_mode,
    )
    candles = BacktestRunner._frame_to_candles(loaded.frame)
    data_health = loaded.diagnostics.get("data_health", {})
    LOGGER.info(
        "Data loaded: symbol=%s bars=%s min_ts=%s max_ts=%s",
        symbol,
        data_health.get("bars"),
        data_health.get("min_ts_utc"),
        data_health.get("max_ts_utc"),
    )

    # Determine assumed spread from data or config
    spread_series = loaded.frame.get("spread")
    spread_from_data = (
        float(spread_series.dropna().median())
        if spread_series is not None and hasattr(spread_series, "dropna") and not spread_series.dropna().empty
        else None
    )
    assumed_spread = spread_from_data if spread_from_data is not None else 0.65

    nan_counts = data_health.get("nan_counts", {}) if isinstance(data_health, dict) else {}
    bars_count = int(data_health.get("bars", 0)) if isinstance(data_health, dict) else 0
    close_bid_nan = int(nan_counts.get("close_bid", bars_count)) if isinstance(nan_counts, dict) else bars_count
    close_ask_nan = int(nan_counts.get("close_ask", bars_count)) if isinstance(nan_counts, dict) else bars_count
    spread_mode = "ASSUMED_OHLC" if bars_count > 0 and close_bid_nan >= bars_count and close_ask_nan >= bars_count else "REAL_BIDASK"

    return candles, assumed_spread, spread_mode, loaded.diagnostics


# ---------------------------------------------------------------------------
# Run a single backtest
# ---------------------------------------------------------------------------
def _run_single(
    *,
    label: str,
    config: AppConfig,
    symbol: str,
    candles: list[Any],
    assumed_spread: float,
    spread_mode: str,
    slippage_points: float,
) -> dict[str, Any]:
    """Run a backtest and return the report dict augmented with timing info."""
    asset = _find_asset(config, symbol)
    data_context = {
        "symbol": symbol,
        "spread_mode": spread_mode,
        "assumed_spread_used": assumed_spread,
    }
    LOGGER.info("Running backtest [%s] …", label)
    t0 = time.perf_counter()
    report = run_backtest_multi_strategy(
        config=config,
        asset=asset,
        candles_m5=candles,
        assumed_spread=assumed_spread,
        slippage_points=slippage_points,
        variant=BacktestVariant(),
        data_context=data_context,
    )
    elapsed = time.perf_counter() - t0
    LOGGER.info("  [%s] done in %.1fs – trades=%d PnL=%.2f DD=%.2f", label, elapsed, report.trades, report.total_pnl, report.max_drawdown)
    rd = report.to_dict()
    rd["_label"] = label
    rd["_elapsed_seconds"] = round(elapsed, 2)
    return rd


def _find_asset(config: AppConfig, symbol: str) -> AssetConfig:
    """Resolve the AssetConfig for the requested symbol (handle XAUUSD/GOLD alias)."""
    sym_up = symbol.strip().upper()
    for asset in config.assets:
        if asset.epic.upper() == sym_up:
            return asset
    # Fallback to the single-instrument field
    if config.instrument.epic.upper() == sym_up:
        return config.instrument
    raise ValueError(f"Symbol {sym_up} not found in config assets or instrument")


# ---------------------------------------------------------------------------
# Metric extraction & comparison
# ---------------------------------------------------------------------------
_METRIC_KEYS = [
    ("trades", "Trades", "{:.0f}", False),
    ("wins", "Wins", "{:.0f}", False),
    ("losses", "Losses", "{:.0f}", False),
    ("win_rate", "Win Rate %", "{:.1f}", True),
    ("total_pnl", "Total PnL", "{:.2f}", True),
    ("total_pnl_net", "PnL (net)", "{:.2f}", True),
    ("max_drawdown", "Max Drawdown", "{:.2f}", False),
    ("max_drawdown_net", "Max DD (net)", "{:.2f}", False),
    ("expectancy", "Expectancy", "{:.2f}", True),
    ("profit_factor", "Profit Factor", "{:.2f}", True),
    ("avg_r", "Avg R", "{:.3f}", True),
    ("avg_win_R", "Avg Win R", "{:.3f}", True),
    ("avg_loss_R", "Avg Loss R", "{:.3f}", False),
    ("payoff_R", "Payoff R", "{:.2f}", True),
    ("equity_end", "Equity End", "{:.2f}", True),
    ("signal_candidates", "Signal Candidates", "{:.0f}", False),
    ("orders_submitted", "Orders Submitted", "{:.0f}", False),
    ("trades_filled", "Trades Filled", "{:.0f}", False),
    ("count_TP1_hits", "TP1 Hits", "{:.0f}", False),
    ("count_BE_moves", "BE Moves", "{:.0f}", False),
    ("time_in_market_bars", "Time in Market (bars)", "{:.0f}", False),
    ("spread_cost_sum", "Spread Cost", "{:.2f}", False),
    ("swap_cost_sum", "Swap Cost", "{:.2f}", False),
    ("forced_closes_count", "Forced Closes", "{:.0f}", False),
]

_FUNNEL_KEYS = [
    ("fill_rate_pct", "Fill Rate %", "{:.1f}", True),
    ("avg_concurrent_positions", "Avg Concurrent", "{:.2f}", False),
    ("signal_candidates", "Funnel: Signals", "{:.0f}", False),
    ("orders_placed", "Funnel: Orders", "{:.0f}", False),
    ("filled_orders", "Funnel: Fills", "{:.0f}", False),
    ("trades_opened", "Funnel: Opens", "{:.0f}", False),
    ("trades_closed", "Funnel: Closes", "{:.0f}", False),
    ("blocked_by_risk_budget", "Funnel: Budget Block", "{:.0f}", False),
    ("blocked_by_supervisor", "Funnel: Supervisor Block", "{:.0f}", False),
    ("orders_expired_ttl", "Funnel: TTL Expired", "{:.0f}", False),
]


def _val(report: dict[str, Any], key: str, sub: str | None = None) -> float:
    """Safely extract a numeric metric."""
    src = report if sub is None else report.get(sub, {})
    v = src.get(key)
    if v is None:
        return 0.0
    return float(v)


def _delta_str(a: float, b: float, higher_is_better: bool) -> str:
    d = b - a
    if abs(d) < 1e-9:
        return "  =="
    sign = "+" if d > 0 else ""
    arrow = "▲" if (d > 0 and higher_is_better) or (d < 0 and not higher_is_better) else "▼"
    return f"{sign}{d:.2f} {arrow}"


def _print_table(a: dict[str, Any], b: dict[str, Any], label_a: str, label_b: str) -> list[dict[str, Any]]:
    """Print side-by-side comparison table and return structured rows."""
    rows: list[dict[str, Any]] = []
    header = f"{'Metric':<25s} │ {label_a:>14s} │ {label_b:>14s} │ {'Delta':>14s}"
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for key, name, fmt, hib in _METRIC_KEYS:
        va = _val(a, key)
        vb = _val(b, key)
        fa = fmt.format(va)
        fb = fmt.format(vb)
        delta = _delta_str(va, vb, hib)
        print(f"{name:<25s} │ {fa:>14s} │ {fb:>14s} │ {delta:>14s}")
        rows.append({"metric": name, label_a: va, label_b: vb, "delta": vb - va})

    # Decision funnel section
    funnel_a = a.get("decision_funnel", {})
    funnel_b = b.get("decision_funnel", {})
    if funnel_a or funnel_b:
        print(sep)
        print(f"{'DECISION FUNNEL':<25s} │ {label_a:>14s} │ {label_b:>14s} │ {'Delta':>14s}")
        print(sep)
        for key, name, fmt, hib in _FUNNEL_KEYS:
            va = _val(a, key, sub="decision_funnel")
            vb = _val(b, key, sub="decision_funnel")
            fa = fmt.format(va)
            fb = fmt.format(vb)
            delta = _delta_str(va, vb, hib)
            print(f"{name:<25s} │ {fa:>14s} │ {fb:>14s} │ {delta:>14s}")
            rows.append({"metric": f"funnel_{name}", label_a: va, label_b: vb, "delta": vb - va})

    print(sep)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    root = ROOT

    config_path_a = root / args.config_a
    config_path_b = root / args.config_b
    if not config_path_a.exists():
        LOGGER.error("Config A not found: %s", config_path_a)
        sys.exit(1)
    if not config_path_b.exists():
        LOGGER.error("Config B not found: %s", config_path_b)
        sys.exit(1)

    config_a = load_config(config_path_a)
    config_b = load_config(config_path_b)

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    symbol = args.symbol.strip().upper()

    data_root = str(root / args.data_root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data once ──────────────────────────────────────────────
    candles, assumed_spread, spread_mode, diagnostics = _load_candles(
        data_root=data_root,
        symbol=symbol,
        start=start,
        end=end,
        price_mode=args.price_mode,
    )

    # Check config-specific spread overrides
    for label, cfg in [("A", config_a), ("B", config_b)]:
        sym_spread = cfg.backtest_tuning.assumed_spread_by_symbol.get(symbol)
        if sym_spread is not None:
            LOGGER.info("[%s] Config override assumed_spread=%s for %s", label, sym_spread, symbol)

    LOGGER.info(
        "Shared data: %d candles, assumed_spread=%.3f, spread_mode=%s",
        len(candles), assumed_spread, spread_mode,
    )

    # ── 2. Run both backtests ──────────────────────────────────────────
    report_a = _run_single(
        label=args.label_a,
        config=config_a,
        symbol=symbol,
        candles=candles,
        assumed_spread=assumed_spread,
        spread_mode=spread_mode,
        slippage_points=args.slippage_points,
    )
    report_b = _run_single(
        label=args.label_b,
        config=config_b,
        symbol=symbol,
        candles=candles,
        assumed_spread=assumed_spread,
        spread_mode=spread_mode,
        slippage_points=args.slippage_points,
    )

    # ── 3. Compare & print table ───────────────────────────────────────
    rows = _print_table(report_a, report_b, args.label_a, args.label_b)

    # ── 4. Verdict ─────────────────────────────────────────────────────
    pnl_a = _val(report_a, "total_pnl")
    pnl_b = _val(report_b, "total_pnl")
    dd_a = _val(report_a, "max_drawdown")
    dd_b = _val(report_b, "max_drawdown")
    pf_a = _val(report_a, "profit_factor")
    pf_b = _val(report_b, "profit_factor")

    verdicts: list[str] = []
    if pnl_b > pnl_a:
        verdicts.append(f"PnL improved +{pnl_b - pnl_a:.2f}")
    elif pnl_b < pnl_a:
        verdicts.append(f"PnL decreased {pnl_b - pnl_a:.2f}")
    else:
        verdicts.append("PnL unchanged")

    if dd_b < dd_a:
        verdicts.append(f"DD reduced -{dd_a - dd_b:.2f}")
    elif dd_b > dd_a:
        verdicts.append(f"DD increased +{dd_b - dd_a:.2f}")
    else:
        verdicts.append("DD unchanged")

    print(f"\n{'='*60}")
    print(f"VERDICT: {' | '.join(verdicts)}")
    if pnl_b >= pnl_a and dd_b <= dd_a:
        print("✅ Config B dominates: higher PnL with equal or lower drawdown")
    elif pnl_b >= pnl_a and dd_b > dd_a:
        print("⚠️  Config B trades more PnL for more risk")
    elif pnl_b < pnl_a and dd_b < dd_a:
        print("⚠️  Config B is more conservative: less PnL but less risk")
    else:
        print("❌ Config B is worse on both PnL and drawdown")
    print(f"{'='*60}\n")

    # ── 5. Save JSON ───────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ab_compare_{ts}.json"
    payload = {
        "timestamp": ts,
        "symbol": symbol,
        "start": args.start,
        "end": args.end,
        "config_a": str(args.config_a),
        "config_b": str(args.config_b),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "report_a": report_a,
        "report_b": report_b,
        "comparison_rows": rows,
        "verdict": verdicts,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=True), encoding="utf-8")
    LOGGER.info("Full comparison saved: %s", out_path)


if __name__ == "__main__":
    main()
