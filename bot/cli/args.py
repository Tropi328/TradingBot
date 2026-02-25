from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capital.com DEMO multi-asset trading bot")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dry-run", action="store_true", help="No order placement, only logging/journaling")
    mode_group.add_argument("--paper", action="store_true", help="Place orders on Capital.com DEMO API")

    parser.add_argument("--test-order", action="store_true", help="Place one synthetic test LIMIT order immediately and exit.")
    parser.add_argument("--test-side", choices=["LONG", "SHORT"], default="LONG")
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--test-epic", default=None)

    parser.add_argument("--backtest", action="store_true", help="Run offline backtest from CSV and exit.")
    parser.add_argument("--backtest-data", default=None)
    parser.add_argument("--backtest-epic", default=None)
    parser.add_argument("--backtest-spread", type=float, default=0.2)
    parser.add_argument("--backtest-symbols", default=None, help="Comma-separated symbols for auto data mode.")
    parser.add_argument("--backtest-start", default=None, help="UTC start (e.g. 2023-01-01)")
    parser.add_argument("--backtest-end", default=None, help="UTC end (exclusive if datetime, inclusive date if YYYY-MM-DD)")
    parser.add_argument("--backtest-tf", default="5m", help="Target timeframe, e.g. 5m")
    parser.add_argument("--backtest-price", choices=["mid", "bid", "ask"], default="mid")
    parser.add_argument("--backtest-data-root", default="data")
    parser.add_argument("--backtest-source-priority", default="", help="Comma-separated source priority.")
    parser.add_argument("--backtest-slippage-points", type=float, default=0.0)
    parser.add_argument("--backtest-slippage-atr-multiplier", type=float, default=0.0)
    parser.add_argument("--daily-gate", choices=["off", "trend", "trend_vol_news"], default=None)
    parser.add_argument("--daily-gate-thr", type=float, default=None)
    parser.add_argument("--daily-gate-pre-minutes", type=int, default=None)
    parser.add_argument("--daily-gate-post-minutes", type=int, default=None)
    parser.add_argument("--daily-gate-vol-max", type=float, default=None)
    parser.add_argument("--daily-gate-max-spread", type=float, default=None)
    parser.add_argument("--daily-gate-ab", action="store_true", help="Run backtest comparison for off/trend/trend_vol_news.")
    parser.add_argument("--daily-gate-grid-search", action="store_true", help="Run gate parameter grid search (backtest mode).")
    parser.add_argument("--daily-gate-grid-limit", type=int, default=0, help="Optional limit of tested parameter sets per gate mode.")
    parser.add_argument("--backtest-autofetch", action="store_true")
    parser.add_argument("--backtest-fetch-script", default="fetch_market_data.py")
    parser.add_argument("--backtest-variants", default="W0", help="Comma-separated variants: W0,W1,W2[,W3]")
    parser.add_argument("--backtest-reports-dir", default="reports")
    parser.add_argument("--report", dest="report", action="store_true", default=True, help="Generate detailed backtest artifacts.")
    parser.add_argument("--no-report", dest="report", action="store_false", help="Disable detailed backtest artifacts.")
    parser.add_argument("--report-dir", default="reports/backtest", help="Base directory for detailed backtest reports.")
    parser.add_argument(
        "--report-formats",
        default="json,csv,png,html",
        help="Comma-separated formats for detailed reports: json,csv,png,html",
    )
    parser.add_argument("--report-open", action="store_true", help="Open generated report.html after backtest completion.")
    parser.add_argument("--research-run", action="store_true", help="Run research A/B experiments with DD-cap objective.")
    parser.add_argument("--research-optimize", action="store_true", help="Run deep IS/OOS optimizer pipeline.")
    parser.add_argument("--ops-healthcheck", action="store_true", help="Run one-shot ops healthcheck and exit with 0/1.")
    parser.add_argument("--ops-backup-now", action="store_true", help="Run one-shot state backup with manifest and validation.")
    parser.add_argument(
        "--ops-restore-verify",
        default=None,
        metavar="BACKUP_DIR",
        help="Verify backup integrity/manifest without restoring runtime DB files.",
    )
    parser.add_argument(
        "--research-runtime-budget",
        choices=["quick", "medium", "deep"],
        default=None,
        help="Optimizer runtime budget for search-space size.",
    )
    parser.add_argument(
        "--research-benchmark-symbols",
        default=None,
        help="Comma-separated symbols for optimizer benchmark (default from config.research.symbols).",
    )
    parser.add_argument(
        "--research-symbols",
        default=None,
        help="Comma-separated symbols for research run (default from config.research.symbols).",
    )
    parser.add_argument(
        "--research-objective",
        default=None,
        help="Research objective mode (e.g. pnl_dd_cap).",
    )
    parser.add_argument(
        "--research-dd-cap",
        type=float,
        default=None,
        help="Hard max drawdown cap in percent for research ranking.",
    )
    parser.add_argument(
        "--research-dd-cap-basis",
        choices=["initial", "peak", "both"],
        default=None,
        help="DD-cap basis used in research objective: initial, peak or both.",
    )
    parser.add_argument(
        "--research-workers",
        type=int,
        default=None,
        help="Max parallel research workers (capped to 3).",
    )
    parser.add_argument(
        "--research-capitals",
        default=None,
        help="Comma-separated optimizer capitals in amount:CCY format, e.g. 10000:USD,100:PLN.",
    )
    parser.add_argument(
        "--research-capital-run-mode",
        choices=["sequential", "parallel"],
        default=None,
        help="How to run optimizer across multiple capitals.",
    )
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--wf-splits", type=int, default=4)

    parser.add_argument("--batch-backtest", action="store_true", help="Run parquet-sharded batch backtest orchestrator.")
    parser.add_argument("--batch-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--price-mode", default="MID")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--chunk", default="monthly")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--warmup-days", type=int, default=60)
    parser.add_argument("--out-root", default="runs/batch")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--initial-equity", type=float, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--data-root", default=None, help="Batch mode alias for --backtest-data-root")

    parser.add_argument("--state-log", choices=["text", "json"], default="text")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--diagnostics-export",
        default=None,
        metavar="PATH",
        help="Export signal_candidates diagnostics JSON/CSV on shutdown (PAPER/DRY mode).",
    )
    parser.add_argument(
        "--diagnostics-format",
        choices=["json", "csv"],
        default="json",
        help="Format for --diagnostics-export output.",
    )
    parser.add_argument(
        "--decision-trace",
        default=None,
        metavar="PATH",
        help="Write decision-trace JSONL to PATH (e.g. logs/decision_trace.jsonl). "
        "Overrides diagnostics.decision_trace_path in config.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        default=False,
        help="Open web dashboard after backtest (default: off).",
    )
    parser.add_argument(
        "--no-dashboard",
        dest="dashboard",
        action="store_false",
        help="Disable auto-opening the web dashboard after backtest.",
    )
    parser.add_argument(
        "--mc-viewer",
        action="store_true",
        default=None,
        help="Force-enable Monte Carlo live viewer window (overrides config).",
    )
    parser.add_argument(
        "--no-mc-viewer",
        dest="mc_viewer",
        action="store_false",
        help="Disable Monte Carlo live viewer window.",
    )
    parser.add_argument(
        "--hold-viewers",
        action="store_true",
        default=False,
        help="Keep process alive after backtest when dashboard/MC viewer is running.",
    )
    return parser.parse_args()
