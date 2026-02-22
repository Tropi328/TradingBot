"""Diagnostic script: run a short backtest and print pipeline rejection summary."""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bot.config import load_config, AssetConfig
from bot.backtest.engine import (
    BacktestVariant,
    load_candles_csv,
    run_backtest_multi_strategy,
)

def main() -> None:
    config_path = PROJECT_ROOT / "config.yaml"
    config = load_config(config_path)

    # Try to find XAUUSD data
    data_dir = PROJECT_ROOT / "data"
    csv_path = data_dir / "local_csv"
    
    # List available CSVs
    print("=== Available CSV files ===")
    for p in sorted(data_dir.rglob("*.csv")):
        print(f"  {p.relative_to(PROJECT_ROOT)}")
    
    # Try loading from local_csv or data root
    candles = None
    tried = []
    for candidate in [
        csv_path / "XAUUSD_5m.csv",
        csv_path / "GOLD_5m.csv",
        csv_path / "XAUUSD_1m.csv",
        csv_path / "GOLD_1m.csv",
        data_dir / "XAUUSD_5m.csv",
        data_dir / "XAU_1m_data.csv",
    ]:
        tried.append(str(candidate))
        if candidate.exists():
            print(f"\nLoading candles from: {candidate}")
            candles = load_candles_csv(str(candidate))
            print(f"  Loaded {len(candles)} candles, range: {candles[0].timestamp} → {candles[-1].timestamp}")
            break
    
    if candles is None:
        # Try the bot/data directory
        bot_data = PROJECT_ROOT / "bot" / "data" / "XAU_1m_data.csv"
        tried.append(str(bot_data))
        if bot_data.exists():
            print(f"\nLoading candles from: {bot_data}")
            candles = load_candles_csv(str(bot_data))
            print(f"  Loaded {len(candles)} candles, range: {candles[0].timestamp} → {candles[-1].timestamp}")
    
    if candles is None:
        print(f"\nNo XAUUSD data found. Tried: {tried}")
        print("Looking for any CSV...")
        for p in sorted(data_dir.rglob("*.csv"))[:1]:
            print(f"  Trying: {p}")
            candles = load_candles_csv(str(p))
            print(f"  Loaded {len(candles)} candles")
            break
    
    if candles is None:
        print("FATAL: No data found at all")
        return

    # Resolve asset
    asset = None
    for a in config.assets:
        if a.epic.upper() in ("XAUUSD", "GOLD"):
            asset = a
            break
    if asset is None:
        asset = config.assets[0] if config.assets else AssetConfig(
            epic="XAUUSD", currency="USD", instrument_currency="USD",
            point_size=0.01, min_size=0.01, size_step=0.01,
        )

    # Use a smaller slice for faster diag (last ~5000 bars = ~2.5 weeks of 5m)
    if len(candles) > 5000:
        candles = candles[-5000:]
        print(f"\nUsing last 5000 candles: {candles[0].timestamp} → {candles[-1].timestamp}")

    print(f"\n=== Running backtest ===")
    print(f"  Asset: {asset.epic}")
    print(f"  Equity: {config.risk.equity}")
    print(f"  Risk/trade: {config.risk.risk_per_trade}")
    print(f"  Min size: {asset.min_size}, Step: {asset.size_step}")
    print(f"  Point size: {asset.point_size}")
    print(f"  Trade threshold: {config.decision_policy.trade_score_threshold}")
    print(f"  Small range: {config.decision_policy.small_score_min} - {config.decision_policy.small_score_max}")
    print(f"  V2 thresholds: trade={config.backtest_tuning.thresholds_v2_trade}, "
          f"small_min={config.backtest_tuning.thresholds_v2_small_min}, "
          f"small_max={config.backtest_tuning.thresholds_v2_small_max}")

    report = run_backtest_multi_strategy(
        config=config,
        asset=asset,
        candles_m5=candles,
        assumed_spread=0.65,
        variant=BacktestVariant(
            code="DIAG",
            reaction_timeout_reset=True,
            soft_reason_penalties=True,
            thresholds_v2=True,
            dynamic_threshold_bump=True,
        ),
        data_context={
            "spread_mode": "ASSUMED_OHLC",
            "assumed_spread_used": 0.65,
        },
    )

    print(f"\n=== BACKTEST RESULTS ===")
    print(f"  Trades: {report.trades}")
    print(f"  Signal candidates: {report.signal_candidates}")
    print(f"  Orders submitted: {report.orders_submitted}")
    print(f"  Trades filled: {report.trades_filled}")
    print(f"  Win rate: {report.win_rate:.2%}")
    print(f"  Total PnL: {report.total_pnl:.2f}")
    print(f"  Equity end: {report.equity_end:.2f}")

    print(f"\n=== DECISION COUNTS ===")
    for key, count in sorted(report.decision_counts.items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}")

    print(f"\n=== TOP BLOCKERS (sorted by count) ===")
    for key, count in sorted(report.top_blockers.items(), key=lambda x: -x[1])[:30]:
        print(f"  {key}: {count}")

    print(f"\n=== EXECUTION FAIL BREAKDOWN ===")
    for key, count in sorted(report.execution_fail_breakdown.items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}")

    print(f"\n=== MISSING FEATURE COUNTS ===")
    for key, count in sorted(report.missing_feature_counts.items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}")

    print(f"\n=== REJECTED BY REASON ===")
    for key, count in sorted(report.rejected_by_reason.items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}")

    print(f"\n=== SCORE BINS ===")
    for key, count in sorted(report.score_bins.items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}")

    print(f"\n=== GATE BLOCK COUNTS ===")
    for key, count in sorted(report.gate_block_counts.items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}")

    print(f"\n=== WAIT METRICS ===")
    for key, val in sorted(report.wait_metrics.items()):
        print(f"  {key}: {val}")

    print(f"\n=== SPREAD GATE ADJUSTMENTS ===")
    for key, count in sorted(report.spread_gate_adjustments.items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}")

    print(f"\n=== DAILY GATE ===")
    print(f"  Mode: {report.daily_gate_mode}")
    print(f"  Bias bars: {report.daily_gate_bias_bars}")
    print(f"  Blocked by gate: {report.blocked_by_gate}")
    print(f"  Gate reasons: {report.blocked_by_gate_reasons}")

    # Quick sizing check
    print(f"\n=== QUICK SIZING CHECK ===")
    equity = config.risk.equity
    risk_pct = config.risk.risk_per_trade
    risk_cash = equity * risk_pct
    # Typical XAUUSD SL distance ~3-5 USD
    for sl_dist in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
        raw_size = risk_cash / sl_dist
        import math
        floored = math.floor(raw_size / asset.size_step) * asset.size_step
        print(f"  SL={sl_dist:.1f}: risk_cash={risk_cash:.2f}, raw_size={raw_size:.4f}, "
              f"floored={floored:.4f}, min_size={asset.min_size}, "
              f"OK={floored >= asset.min_size}")

    print(f"\n=== AVG SCORE ===")
    print(f"  Avg score: {report.avg_score}")

    # Forced closes
    print(f"\n=== FORCED CLOSES ===")
    print(f"  Forced closes: {report.forced_closes_count}")
    print(f"  Min size overrides: {report.min_size_overrides_count}")
    print(f"  Margin capped: {report.margin_capped_count}")


if __name__ == "__main__":
    main()
