from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from bot.batch_backtest import make_trade_id, orchestrate_batch
from bot.backtest.data_provider import AutoDataLoader, MissingDataError, normalize_timeframe
from bot.backtest.engine import (
    BacktestVariant,
    aggregate_backtest_reports,
    run_backtest,
    run_backtest_from_csv,
    run_backtest_multi_strategy,
    run_walk_forward,
    run_walk_forward_from_csv,
    run_walk_forward_multi_strategy,
)
from bot.backtest.monte_carlo import MCAdaptiveModel, run_monte_carlo_simulation
from bot.backtest.runner import BacktestRunner
from bot.clock import (
    is_symbol_market_open,
    should_poll_closed_candle,
    trading_day,
    utc_now,
)
from bot.config import AppConfig, AssetConfig, load_config
from bot.data.capital_client import CapitalAPIError, CapitalClient
from bot.data.market_data import MarketDataService
from bot.execution.orders import OrderExecutor
from bot.execution.position_manager import MultiTPProfile, PositionManager, build_multi_tp_profile
from bot.execution.feasibility import estimate_required_margin, validate_order
from bot.execution.order_validation import compute_risk_cash_plan, price_to_points
from bot.execution.sizing import (
    compute_compound_equity,
    resolve_score_tier,
    tier_risk_multiplier,
)
from bot.execution.paper_costs import (
    PaperCostConfig,
    SlippageModelConfig,
    compute_be_offset,
    compute_fill_prices,
    estimate_roundtrip_cost,
    estimate_roundtrip_cost_points,
)
from bot.execution.micro_loss_defense import (
    MicroLossCheckResult,
    MicroLossDefenseConfig,
    MicroLossMetrics,
    run_micro_loss_checks,
)
from bot.monitoring.alerts import AlertConfig, AlertDispatcher
from bot.monitoring.dashboard import DashboardWriter
from bot.monitoring.signal_candidates import (
    SignalCandidate,
    SignalCandidateAggregator,
    SignalCandidateLogger,
    export_diagnostics,
    init_signal_candidates_table,
)
from bot.ops_runtime import run_backup_now, run_ops_healthcheck, run_restore_verify
from bot.news.calendar_provider import CalendarProvider, Event, build_calendar_provider
from bot.news.gate import is_blocked, should_cancel_pending
from bot.gating.daily_gate import DailyGateProvider
from bot.gating.adaptive import (
    AdaptiveThresholdConfig,
    ReentryState,
    SoftGateResult,
    apply_soft_gates,
    build_adaptive_config,
    compute_adaptive_threshold,
    normalize_action_adaptive,
)
from bot.strategy.session_filter import (
    SessionMatch,
    SessionWindow,
    build_session_windows,
    match_session,
)
from bot.reporting.backtest_reporter import BacktestMeta, BacktestReporter, BacktestRun
from bot.research.objective import OBJECTIVE_FAIL_VALUE, aggregate_reports, augment_report, objective_rank_key
from bot.research.optimizer import (
    build_search_space_payload,
    build_stage_a_gate_candidates,
    build_stage_b_candidates,
    build_stage_b_summary,
    build_time_split,
    failed_stage_summary,
    get_checkpoint_record,
    load_checkpoint,
    normalize_runtime_budget,
    optimizer_rank_key,
    save_checkpoint,
    upsert_checkpoint_record,
)
from bot.storage.db import get_connection, init_db
from bot.storage.journal import Journal
from bot.storage.models import ClosedPositionEvent, DailyStats, StrategyDecisionRecord
from bot.strategy.contracts import (
    BiasState,
    DecisionAction,
    SetupCandidate,
    StrategyDataBundle,
    StrategyEvaluation,
    StrategyOutcome,
    StrategyPlugin,
)
from bot.strategy.candidate_queue import CandidateQueue
from bot.strategy.decision_core import (
    MAIN_HARD_GATE_POLICY,
    MAIN_SCORE_POLICY,
    apply_orderflow_small_soft_gate as _apply_orderflow_small_soft_gate_core,
    clamp_value as _clamp_core,
    compute_v2_score_core,
    default_observe_evaluation as _default_observe_evaluation_core,
    evaluate_hard_gates_core,
    normalize_action_fixed_threshold,
    orderflow_param as _orderflow_param_core,
    pick_best_candidate as _pick_best_candidate_core,
    quality_gate_reasons_core,
    resolve_orderflow_mode as _resolve_orderflow_mode_core,
    risk_multiplier_for as _risk_multiplier_for_core,
)
from bot.strategy.index_existing import IndexExistingStrategy
from bot.strategy.orb_h4_retest import OrbH4RetestStrategy
from bot.strategy.portfolio_supervisor import EntryProposal, PortfolioSupervisor
from bot.strategy.orderflow import (
    CompositeOrderflowProvider,
    OrderflowProvider,
    OrderflowSnapshot,
)
from bot.strategy.ranker import rank_score
from bot.strategy.risk import RiskEngine
from bot.strategy.router import StrategyRouter
from bot.strategy.scalp_ict_pa import ScalpIctPriceActionStrategy
from bot.strategy.schedule import is_schedule_open
from bot.strategy.state_machine import (
    H1Snapshot,
    M15Snapshot,
    M5Snapshot,
    StrategyDecision,
    StrategySignal,
)
from bot.strategy.trend_pullback_m15 import TrendPullbackM15Strategy
from bot.strategy.trace import (
    DecisionTrace,
    closed_candles,
    format_trace_text,
    is_new_closed_candle,
    map_reason_codes,
    trace_to_json,
)

LOGGER = logging.getLogger("trading_bot")


@dataclass(slots=True)
class AssetRuntimeState:
    asset: AssetConfig
    strategy_name: str = "UNKNOWN"
    cache: dict[str, list] = field(default_factory=dict)
    last_processed_closed_ts: dict[str, datetime | None] = field(default_factory=dict)
    last_poll_target_ts: dict[str, datetime | None] = field(default_factory=dict)
    last_poll_attempt_at: dict[str, datetime | None] = field(default_factory=dict)
    quote: tuple[float, float, float] | None = None
    quote_last_fetch_at: datetime | None = None
    last_reason_codes: list[str] = field(default_factory=list)
    stale_data: bool = False
    h1_snapshot: H1Snapshot | None = None
    m15_snapshot: M15Snapshot | None = None
    m5_snapshot: M5Snapshot | None = None
    bias_state: BiasState | None = None
    last_evaluation: StrategyEvaluation | None = None
    last_candidate: SetupCandidate | None = None
    pending_outcome: StrategyOutcome | None = None
    entry_state: str = "WAIT"
    last_trace_signature: str = ""
    reentry: ReentryState = field(default_factory=ReentryState)


@dataclass(slots=True)
class DailyRuntimeSummary:
    trading_day: str
    cycles: int = 0
    signal_candidates: int = 0
    blockers: Counter[str] = field(default_factory=Counter)
    api_requests_start: int = 0
    api_retries_start: int = 0
    api_429_start: int = 0

    def top_blockers(self, limit: int = 5) -> str:
        if not self.blockers:
            return "-"
        return ",".join(f"{key}:{value}" for key, value in self.blockers.most_common(limit))


@dataclass(slots=True)
class PendingOrderIntent:
    symbol: str
    state: AssetRuntimeState
    route_priority: int
    cooldown_seconds: int
    route_risk: dict[str, object]
    outcome: StrategyOutcome
    signal: StrategySignal
    risk_multiplier: float
    rank_score: float
    asset_stats_snapshot: DailyStats


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


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_epics_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        epic = part.strip().upper()
        if not epic or epic in seen:
            continue
        seen.add(epic)
        out.append(epic)
    return out


def _apply_cli_overrides(args: argparse.Namespace, config: AppConfig) -> None:
    if args.initial_equity is not None:
        config.risk.equity = float(args.initial_equity)
    if args.research_benchmark_symbols is not None:
        config.research.symbols = parse_epics_csv(args.research_benchmark_symbols)
    if args.research_symbols is not None:
        config.research.symbols = parse_epics_csv(args.research_symbols)
    if args.research_objective is not None:
        config.research.objective_mode = str(args.research_objective).strip().lower()
    if args.research_dd_cap is not None:
        config.research.dd_cap_pct = float(args.research_dd_cap)
    if args.research_dd_cap_basis is not None:
        config.research.dd_cap_basis = str(args.research_dd_cap_basis).strip().lower()
        config.research.optimize.dd_cap_basis = str(args.research_dd_cap_basis).strip().lower()
    if args.research_runtime_budget is not None:
        config.research.optimize.runtime_budget = str(args.research_runtime_budget).strip().lower()
    if args.research_workers is not None:
        workers = max(1, min(3, int(args.research_workers)))
        config.research.max_workers = workers
        config.research.optimize.max_workers = workers
        config.backtest_runtime.parallel_workers = workers
    if args.daily_gate is not None:
        config.daily_gate.mode = str(args.daily_gate).strip().lower()
    if args.daily_gate_thr is not None:
        config.daily_gate.thr = float(args.daily_gate_thr)
    if args.daily_gate_pre_minutes is not None:
        config.daily_gate.pre_minutes = int(args.daily_gate_pre_minutes)
    if args.daily_gate_post_minutes is not None:
        config.daily_gate.post_minutes = int(args.daily_gate_post_minutes)
    if args.daily_gate_vol_max is not None:
        config.daily_gate.vol_max = float(args.daily_gate_vol_max)
    if args.daily_gate_max_spread is not None:
        config.daily_gate.max_spread = float(args.daily_gate_max_spread)
    # Decision-trace CLI override
    if getattr(args, "decision_trace", None) is not None:
        config.diagnostics.decision_trace_enabled = True
        config.diagnostics.decision_trace_path = str(args.decision_trace)

    # Dashboard implies live decision trace stream.
    if getattr(args, "dashboard", False):
        config.diagnostics.decision_trace_enabled = True

    # Optional config-based auto-enable for backtest mode.
    if (
        getattr(args, "backtest", False)
        and bool(config.diagnostics.decision_trace_auto_enable_backtest)
        and not config.diagnostics.decision_trace_enabled
    ):
        config.diagnostics.decision_trace_enabled = True


def _daily_gate_mode(config: AppConfig) -> str:
    return str(config.daily_gate.mode).strip().lower()


def _build_daily_gate_provider(
    *,
    config: AppConfig,
    mode: str,
    candles: list | None = None,
    events: list[Event] | None = None,
    overrides: dict[str, float | int | None] | None = None,
) -> DailyGateProvider | None:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "off":
        return None
    params = {
        "thr": float(config.daily_gate.thr),
        "pre_minutes": int(config.daily_gate.pre_minutes),
        "post_minutes": int(config.daily_gate.post_minutes),
        "vol_max": float(config.daily_gate.vol_max),
        "max_spread": (float(config.daily_gate.max_spread) if config.daily_gate.max_spread is not None else None),
    }
    if overrides:
        params.update(overrides)
    provider = DailyGateProvider(
        mode=normalized_mode,
        ema_fast=int(config.daily_gate.ema_fast),
        ema_slow=int(config.daily_gate.ema_slow),
        thr=float(params["thr"]),
        atr_period=int(config.daily_gate.atr_period),
        vol_max=float(params["vol_max"]),
        max_spread=(float(params["max_spread"]) if params["max_spread"] is not None else None),
        pre_minutes=int(params["pre_minutes"]),
        post_minutes=int(params["post_minutes"]),
        rollover_start_utc=config.daily_gate.rollover_start_utc,
        rollover_end_utc=config.daily_gate.rollover_end_utc,
        allowed_strategies=config.daily_gate.allowed_strategies,
        events=events or [],
    )
    if candles:
        provider.refresh_from_candles(candles)
    return provider


def _gate_param_space_for_grid(config: AppConfig) -> list[dict[str, float | int | None]]:
    base_spread = float(config.daily_gate.max_spread) if config.daily_gate.max_spread is not None else None
    spread_candidates: list[float | None]
    if base_spread is None:
        spread_candidates = [None]
    else:
        spread_candidates = [base_spread, base_spread * 1.5]
    grid: list[dict[str, float | int | None]] = []
    for thr in [0.0005, 0.0010, 0.0020]:
        for pre_minutes in [15, 30, 45]:
            for post_minutes in [15, 30, 45]:
                for vol_max in [0.015, 0.020, 0.025]:
                    for max_spread in spread_candidates:
                        grid.append(
                            {
                                "thr": thr,
                                "pre_minutes": pre_minutes,
                                "post_minutes": post_minutes,
                                "vol_max": vol_max,
                                "max_spread": max_spread,
                            }
                        )
    return grid


def _asset_from_template(epic: str, template: AssetConfig, trade_enabled: bool) -> AssetConfig:
    return AssetConfig(
        epic=epic,
        currency=template.currency,
        instrument_currency=template.instrument_currency,
        point_size=template.point_size,
        minimal_tick_buffer=template.minimal_tick_buffer,
        min_size=template.min_size,
        size_step=template.size_step,
        trade_enabled=trade_enabled,
    )


def _estimated_open_margin(*, positions: list, config: AppConfig) -> float:
    total = 0.0
    margin_pct = float(config.backtest_tuning.broker_margin_requirement_pct)
    leverage = float(config.backtest_tuning.broker_leverage)
    for position in positions:
        try:
            entry = float(getattr(position, "entry_price", 0.0))
            size = float(getattr(position, "size", 0.0))
        except (TypeError, ValueError):
            continue
        total += estimate_required_margin(
            entry_price=entry,
            size=size,
            margin_requirement_pct=margin_pct,
            max_leverage=leverage,
        )
    return total


def build_asset_universe(config: AppConfig) -> list[AssetConfig]:
    assets = [asset.model_copy(deep=True) for asset in config.assets]
    if not assets:
        assets = [AssetConfig(**config.instrument.model_dump(), trade_enabled=True)]

    template = assets[0]
    by_epic = {a.epic.upper(): a for a in assets}

    primary = (os.getenv("CAPITAL_EPIC") or "").strip().upper()
    trade_epics = parse_epics_csv(os.getenv("CAPITAL_TRADE_EPICS"))
    watch_epics = parse_epics_csv(os.getenv("CAPITAL_WATCH_EPICS"))

    if primary:
        if primary not in by_epic:
            by_epic[primary] = _asset_from_template(primary, template, True)
        by_epic[primary].trade_enabled = True

    if trade_epics:
        for item in by_epic.values():
            item.trade_enabled = item.epic in trade_epics
        for epic in trade_epics:
            if epic not in by_epic:
                by_epic[epic] = _asset_from_template(epic, template, True)

    for epic in watch_epics:
        if epic not in by_epic:
            by_epic[epic] = _asset_from_template(epic, template, False)

    trading = sorted((a for a in by_epic.values() if a.trade_enabled), key=lambda a: a.epic)
    observing = sorted((a for a in by_epic.values() if not a.trade_enabled), key=lambda a: a.epic)
    return trading + observing


def build_client(config: AppConfig, paper_mode: bool) -> CapitalClient | None:
    base_url = os.getenv("CAPITAL_BASE_URL", config.capital.demo_base_url)
    api_key = os.getenv("CAPITAL_API_KEY")
    identifier = os.getenv("CAPITAL_IDENTIFIER")
    password = os.getenv("CAPITAL_API_PASSWORD") or os.getenv("CAPITAL_PASSWORD")
    account_id = os.getenv("CAPITAL_ACCOUNT_ID")

    if paper_mode and not (api_key and identifier and password):
        raise RuntimeError("Paper mode requires API credentials in .env")
    if not (api_key and identifier and password):
        LOGGER.warning("Credentials missing. Running without live market data.")
        return None
    return CapitalClient(
        base_url=base_url,
        api_key=api_key,
        identifier=identifier,
        password=password,
        account_id=account_id,
        rate_limit_rps=float(os.getenv("CAPITAL_RATE_LIMIT_RPS", str(config.capital.rate_limit_rps))),
        rate_limit_burst=int(os.getenv("CAPITAL_RATE_LIMIT_BURST", str(config.capital.rate_limit_burst))),
        request_max_attempts=int(os.getenv("CAPITAL_REQUEST_MAX_ATTEMPTS", str(config.capital.request_max_attempts))),
        backoff_base_seconds=float(os.getenv("CAPITAL_BACKOFF_BASE_SECONDS", str(config.capital.backoff_base_seconds))),
        backoff_max_seconds=float(os.getenv("CAPITAL_BACKOFF_MAX_SECONDS", str(config.capital.backoff_max_seconds))),
        reconnect_short_retries=int(os.getenv("CAPITAL_RECONNECT_SHORT_RETRIES", str(config.capital.reconnect_short_retries))),
        session_refresh_min_interval_seconds=int(
            os.getenv(
                "CAPITAL_SESSION_REFRESH_MIN_INTERVAL_SECONDS",
                str(config.capital.session_refresh_min_interval_seconds),
            )
        ),
    )


def build_news_provider(config: AppConfig, root: Path) -> CalendarProvider:
    provider_name = os.getenv("NEWS_PROVIDER", config.calendar.provider)
    dummy_file = Path(config.calendar.dummy_file)
    if not dummy_file.is_absolute():
        dummy_file = root / dummy_file
    return build_calendar_provider(
        provider_name=provider_name,
        dummy_file=dummy_file,
        http_url=os.getenv("NEWS_HTTP_URL"),
        http_token=os.getenv("NEWS_HTTP_TOKEN"),
        timeout_seconds=config.calendar.http_timeout_seconds,
        cache_ttl_seconds=config.calendar.http_cache_ttl_seconds,
    )


def build_alert_dispatcher(config: AppConfig) -> AlertDispatcher:
    return AlertDispatcher(
        AlertConfig(
            enabled=config.monitoring.alerts_enabled,
            discord_webhook=os.getenv("ALERT_DISCORD_WEBHOOK"),
            telegram_bot_token=os.getenv("ALERT_TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("ALERT_TELEGRAM_CHAT_ID"),
            cooldown_seconds=int(os.getenv("ALERT_COOLDOWN_SECONDS", "30")),
        )
    )


def create_decision_record(decision: StrategyDecision, epic: str, side: str | None, news_blocked: bool) -> StrategyDecisionRecord:
    return StrategyDecisionRecord(
        created_at=datetime.now(timezone.utc),
        epic=epic,
        side=side,
        bias=decision.bias,
        pd_state=decision.pd_state,
        sweep=decision.sweep_ok,
        mss=decision.mss_ok,
        displacement=decision.displacement_ok,
        fvg=decision.fvg_ok,
        spread_ok=decision.spread_ok,
        news_blocked=news_blocked,
        rr=decision.signal.rr if decision.signal else None,
        reason_codes=decision.reason_codes,
        payload=decision.payload,
    )


def _bias_to_legacy_label(direction: str) -> str:
    if direction == "LONG":
        return "UP"
    if direction == "SHORT":
        return "DOWN"
    return "NEUTRAL"


from bot.strategy.tp_profile import tp2_r_for_target_total_r as _tp2_r_for_target_total_r  # noqa: E402


def _apply_rr_profile_to_signal(
    signal: StrategySignal,
    *,
    tp1_trigger_r: float,
    tp1_fraction: float,
    tp_profile_mode: str,
) -> bool:
    risk_distance = abs(float(signal.entry_price) - float(signal.stop_price))
    if risk_distance <= 0:
        return False

    target_total_r = 3.0 if bool(signal.a_plus) or float(signal.rr) >= 3.0 else 2.0
    target_tp2_r = _tp2_r_for_target_total_r(
        target_total_r=target_total_r,
        tp1_trigger_r=tp1_trigger_r,
        tp1_fraction=tp1_fraction,
        mode=tp_profile_mode,
    )

    if signal.side == "LONG":
        signal.take_profit = float(signal.entry_price) + (target_tp2_r * risk_distance)
    else:
        signal.take_profit = float(signal.entry_price) - (target_tp2_r * risk_distance)

    signal.rr = target_total_r
    meta = dict(signal.metadata or {})
    meta["tp_target_profile"] = "A_PLUS_3R" if target_total_r >= 3.0 else "STANDARD_2R"
    meta["target_r_profile_total"] = round(target_total_r, 4)
    meta["target_r_tp2"] = round(target_tp2_r, 4)
    meta["tp1_trigger_r"] = float(tp1_trigger_r)
    meta["tp1_fraction"] = float(tp1_fraction)
    meta["tp_profile_mode"] = str(tp_profile_mode).strip().lower()
    signal.metadata = meta
    return True


def create_decision_record_from_outcome(
    *,
    outcome: StrategyOutcome,
    news_blocked: bool,
) -> StrategyDecisionRecord:
    side = outcome.order_request.side if outcome.order_request is not None else outcome.candidate.side if outcome.candidate is not None else None
    reason_codes = list(outcome.reason_codes)
    payload = dict(outcome.payload)
    payload["strategy_name"] = outcome.strategy_name
    payload["score_total"] = outcome.evaluation.score_total
    payload["score_layers"] = outcome.evaluation.score_layers
    payload["score_breakdown"] = outcome.evaluation.score_breakdown
    payload["penalties"] = outcome.evaluation.penalties
    payload["gates"] = outcome.evaluation.gates
    payload["gate_blocked"] = outcome.evaluation.gate_blocked
    payload["reasons_blocking"] = outcome.evaluation.reasons_blocking
    payload["would_enter_if"] = outcome.evaluation.would_enter_if
    payload["snapshot"] = outcome.evaluation.snapshot

    has_sweep = bool(
        payload.get("sweep")
        or payload.get("sweep_level")
        or payload.get("m15_setup_state") == "ARMED"
    )
    has_mss = bool(payload.get("mss") or payload.get("mss_index"))
    has_disp = bool(payload.get("displacement") or payload.get("displacement_ratio"))
    has_fvg = bool(payload.get("fvg") or payload.get("fvg_mid"))
    spread_ok = "SCALP_SPREAD_ELEVATED" not in reason_codes and "M5_SPREAD_FAIL" not in reason_codes
    pd_state = str(payload.get("pd_state") or payload.get("h1_pd_state") or "UNKNOWN")

    return StrategyDecisionRecord(
        created_at=datetime.now(timezone.utc),
        epic=outcome.symbol,
        side=side,
        bias=_bias_to_legacy_label(outcome.bias.direction),
        pd_state=pd_state,
        sweep=has_sweep,
        mss=has_mss,
        displacement=has_disp,
        fvg=has_fvg,
        spread_ok=spread_ok,
        news_blocked=news_blocked,
        rr=outcome.order_request.rr if outcome.order_request else None,
        reason_codes=reason_codes,
        payload=payload,
    )


def apply_closed_events(events: list[ClosedPositionEvent], trading_day_str: str, journal: Journal, risk_engine: RiskEngine, now: datetime, alerts: AlertDispatcher) -> None:
    for event in events:
        journal.add_daily_pnl(trading_day_str, event.pnl, epic=event.epic)
        journal.add_daily_pnl(trading_day_str, event.pnl, epic="GLOBAL")
        for scope in (f"ASSET:{event.epic}", "GLOBAL"):
            state = journal.get_risk_state(scope)
            if event.pnl < 0:
                state.loss_streak += 1
                if state.loss_streak >= risk_engine.risk.cooldown_loss_streak:
                    state.cooldown_until = now + timedelta(minutes=risk_engine.risk.cooldown_minutes)
            elif event.pnl > 0:
                state.loss_streak = 0
                state.cooldown_until = None
            state.updated_at = now
            journal.upsert_risk_state(state)
        alerts.send(event="POSITION_CLOSED", message=f"{event.epic} deal={event.deal_id} pnl={event.pnl:.2f}", dedupe_key=f"close-{event.deal_id}")


def place_single_test_order(order_executor: OrderExecutor, market_data: MarketDataService, assets: list[AssetConfig], config: AppConfig, dry_run: bool, side: str, test_size: float | None, test_epic: str | None) -> None:
    epic = (test_epic or next((a.epic for a in assets if a.trade_enabled), assets[0].epic)).strip().upper()
    asset = next((a for a in assets if a.epic == epic), None)
    if asset is None:
        raise RuntimeError(f"Unknown test epic: {epic}")

    bid, ask, _ = market_data.fetch_quote_and_spread(epic)
    if bid is None or ask is None:
        raise RuntimeError("Cannot place test order: missing current bid/ask quote")

    point = max(asset.point_size, 0.01)
    now = utc_now()
    risk_distance = 200 * point
    if side == "LONG":
        entry = ask + (10 * point) if dry_run else ask - (20 * point)
        stop = entry - risk_distance
        take_profit = entry + (2 * risk_distance)
    else:
        entry = bid - (10 * point) if dry_run else bid + (20 * point)
        stop = entry + risk_distance
        take_profit = entry - (2 * risk_distance)

    size = test_size if test_size is not None else asset.min_size
    signal = StrategySignal(
        side=side,
        entry_price=entry,
        stop_price=stop,
        take_profit=take_profit,
        rr=2.0,
        a_plus=False,
        expires_at=now + timedelta(minutes=config.execution.limit_ttl_bars * 5),
        reason_codes=["TEST_ORDER"],
        metadata={"test_order": True, "dry_run": dry_run, "source_bid": bid, "source_ask": ask},
    )
    order = order_executor.place_limit_order(signal, size=size, epic=asset.epic, currency=asset.currency, idempotency_key=f"TEST-{asset.epic}-{int(now.timestamp())}")
    LOGGER.info("Test LIMIT order placed: id=%s epic=%s side=%s size=%.4f", order.order_id, order.epic, order.side, order.size)
    if dry_run:
        filled = order_executor.process_pending_fills(quotes_by_epic={asset.epic: (bid, ask, ask - bid)}, now=now)
        LOGGER.info("Dry-run test fill=%s", bool(filled))


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
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    if end_value and _is_date_only(raw):
        dt += timedelta(days=1)
    return dt


def _resolve_runtime_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return root / path


def _resolve_config_path(root: Path, value: str) -> Path:
    direct = _resolve_runtime_path(root, value)
    if direct.exists():
        return direct
    raw = Path(value)
    candidates = [
        root / "configs" / raw,
        root / "configs" / "variants" / raw,
        root / "configs" / raw.name,
        root / "configs" / "variants" / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return direct


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
        "W0": BacktestVariant(code="W0", reaction_timeout_reset=False, soft_reason_penalties=False, thresholds_v2=False, dynamic_threshold_bump=False),
        "W1": BacktestVariant(code="W1", reaction_timeout_reset=True, soft_reason_penalties=False, thresholds_v2=False, dynamic_threshold_bump=False),
        "W2": BacktestVariant(code="W2", reaction_timeout_reset=True, soft_reason_penalties=True, thresholds_v2=False, dynamic_threshold_bump=False),
        "W3": BacktestVariant(code="W3", reaction_timeout_reset=True, soft_reason_penalties=True, thresholds_v2=True, dynamic_threshold_bump=True),
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
        dt = value.astimezone(timezone.utc) if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
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
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
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


def _variant_report_filename(*, variant: BacktestVariant, start_raw: str, end_raw: str, start_dt: datetime, end_dt: datetime, symbol: str) -> str:
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
    LOGGER.info("variant | symbol | trades | win_rate | pnl | expectancy | avg_r | payoff | pf | max_dd | candidates | top3_blockers")
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
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                continue
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
            }
        meta = payload.get("meta")
        if isinstance(meta, dict):
            symbol = str(meta.get("symbol", "")).strip().upper()
            if symbol and "symbol" not in source_report:
                source_report["symbol"] = symbol
        reports.append(source_report)
    return reports


def _build_research_subprocess_command(
    *,
    root: Path,
    args: argparse.Namespace,
    config_path: Path,
    symbols: list[str],
    gate_mode: str,
    run_report_dir: Path,
    run_auto_reports_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(root / "main.py"),
        "--backtest",
        "--backtest-symbols",
        ",".join(symbols),
        "--backtest-start",
        str(args.backtest_start),
        "--backtest-end",
        str(args.backtest_end),
        "--backtest-tf",
        str(args.backtest_tf),
        "--backtest-price",
        str(args.backtest_price),
        "--backtest-data-root",
        str(args.backtest_data_root),
        "--backtest-spread",
        str(args.backtest_spread),
        "--backtest-slippage-points",
        str(args.backtest_slippage_points),
        "--backtest-slippage-atr-multiplier",
        str(args.backtest_slippage_atr_multiplier),
        "--backtest-variants",
        str(args.backtest_variants),
        "--daily-gate",
        str(gate_mode),
        "--report",
        "--report-formats",
        "json",
        "--report-dir",
        str(run_report_dir),
        "--backtest-reports-dir",
        str(run_auto_reports_dir),
        "--config",
        str(config_path),
        "--no-dashboard",
        "--no-mc-viewer",
    ]
    if str(args.backtest_source_priority or "").strip():
        command.extend(["--backtest-source-priority", str(args.backtest_source_priority)])
    if bool(args.backtest_autofetch):
        command.append("--backtest-autofetch")
    if bool(args.walk_forward):
        command.extend(["--walk-forward", "--wf-splits", str(args.wf_splits)])
    if args.initial_equity is not None:
        command.extend(["--initial-equity", str(args.initial_equity)])
    return command


def _run_single_research_candidate(
    *,
    root: Path,
    args: argparse.Namespace,
    config: AppConfig,
    config_path: Path,
    symbols: list[str],
    gate_mode: str,
    run_dir: Path,
) -> dict[str, object]:
    report_dir = run_dir / "detailed_reports"
    auto_reports_dir = run_dir / "auto_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    auto_reports_dir.mkdir(parents=True, exist_ok=True)

    command = _build_research_subprocess_command(
        root=root,
        args=args,
        config_path=config_path,
        symbols=symbols,
        gate_mode=gate_mode,
        run_report_dir=report_dir,
        run_auto_reports_dir=auto_reports_dir,
    )
    env = os.environ.copy()
    if config.backtest_runtime.deterministic:
        env["PYTHONHASHSEED"] = str(config.research.seed)

    started_at = time.time()
    result = subprocess.run(command, cwd=str(root), capture_output=True, text=True, env=env)
    elapsed_seconds = round(time.time() - started_at, 3)

    source_reports = _extract_source_reports_from_report_dir(report_dir)
    if source_reports:
        summary = aggregate_reports(
            source_reports,
            initial_equity=float(config.risk.equity),
            dd_cap_pct=float(config.research.dd_cap_pct),
            dd_cap_basis=str(config.research.dd_cap_basis),
            min_trades_oos=int(config.research.min_trades_oos),
            objective_mode=str(config.research.objective_mode),
        )
    else:
        summary = {
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
            "max_drawdown_pct": 0.0,
            "expectancy": 0.0,
            "expectancy_net": 0.0,
            "avg_r": 0.0,
            "cost_breakdown_net": {},
            "blocked_by_reason": {},
            "oos_pass": False,
            "constraint_dd_cap_pass_peak": False,
            "constraint_dd_cap_pass_initial": False,
            "constraint_dd_cap_pass": False,
            "objective_value": OBJECTIVE_FAIL_VALUE,
        }

    available_symbols = sorted(
        {
            str(item.get("symbol", "")).strip().upper()
            for item in source_reports
            if isinstance(item, dict) and str(item.get("symbol", "")).strip()
        }
    )
    missing_symbols = sorted(set(symbols) - set(available_symbols)) if available_symbols else list(symbols)
    partial_result = bool(result.returncode != 0 and bool(source_reports))
    if partial_result:
        summary = dict(summary)
        summary["partial_result"] = True
        summary["available_symbols"] = available_symbols
        summary["missing_symbols"] = missing_symbols

    return {
        "gate_mode": gate_mode,
        "command": command,
        "returncode": int(result.returncode),
        "partial_result": partial_result,
        "available_symbols": available_symbols,
        "missing_symbols": missing_symbols,
        "elapsed_seconds": elapsed_seconds,
        "stdout_tail": "\n".join((result.stdout or "").splitlines()[-20:]),
        "stderr_tail": "\n".join((result.stderr or "").splitlines()[-20:]),
        "run_dir": str(run_dir),
        "report_dir": str(report_dir),
        "auto_reports_dir": str(auto_reports_dir),
        "summary": summary,
    }


def _write_research_summary_files(research_dir: Path, payload: dict[str, object]) -> None:
    research_dir.mkdir(parents=True, exist_ok=True)
    summary_path = research_dir / "research_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return
    csv_path = research_dir / "research_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "gate_mode",
                "returncode",
                "partial_result",
                "available_symbols",
                "missing_symbols",
                "trades",
                "total_pnl_net",
                "max_drawdown_pct_peak",
                "max_drawdown_pct_initial",
                "max_drawdown_pct",
                "expectancy_net",
                "objective_value",
                "constraint_dd_cap_pass_peak",
                "constraint_dd_cap_pass_initial",
                "constraint_dd_cap_pass",
                "oos_pass",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        for idx, item in enumerate(candidates, start=1):
            if not isinstance(item, dict):
                continue
            summary = item.get("summary", {})
            if not isinstance(summary, dict):
                summary = {}
            writer.writerow(
                {
                    "rank": idx,
                    "gate_mode": item.get("gate_mode"),
                    "returncode": item.get("returncode"),
                    "partial_result": item.get("partial_result", False),
                    "available_symbols": ",".join(str(v) for v in item.get("available_symbols", [])),
                    "missing_symbols": ",".join(str(v) for v in item.get("missing_symbols", [])),
                    "trades": summary.get("trades", 0),
                    "total_pnl_net": summary.get("total_pnl_net", 0.0),
                    "max_drawdown_pct_peak": summary.get("max_drawdown_pct_peak", summary.get("max_drawdown_pct", 0.0)),
                    "max_drawdown_pct_initial": summary.get("max_drawdown_pct_initial", 0.0),
                    "max_drawdown_pct": summary.get("max_drawdown_pct", 0.0),
                    "expectancy_net": summary.get("expectancy_net", summary.get("expectancy", 0.0)),
                    "objective_value": summary.get("objective_value", OBJECTIVE_FAIL_VALUE),
                    "constraint_dd_cap_pass_peak": summary.get("constraint_dd_cap_pass_peak", False),
                    "constraint_dd_cap_pass_initial": summary.get("constraint_dd_cap_pass_initial", False),
                    "constraint_dd_cap_pass": summary.get("constraint_dd_cap_pass", False),
                    "oos_pass": summary.get("oos_pass", False),
                    "elapsed_seconds": item.get("elapsed_seconds", 0.0),
                }
            )


def _write_research_winner_config(
    *,
    root: Path,
    base_config: AppConfig,
    source_config_path: Path,
    research_dir: Path,
    best_candidate: dict[str, object] | None,
) -> Path | None:
    if not isinstance(best_candidate, dict):
        return None
    winner_mode = str(best_candidate.get("gate_mode", "")).strip().lower()
    if winner_mode not in {"off", "trend", "trend_vol_news"}:
        return None

    winner_config = base_config.model_copy(deep=True)
    winner_config.daily_gate.mode = winner_mode

    target_path = root / "configs" / "variants" / "config.variant_RESEARCH_WINNER.yaml"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = [
        "# Auto-generated by --research-run",
        f"# generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"# source_config: {source_config_path}",
        f"# research_report_dir: {research_dir}",
        f"# winner_gate_mode: {winner_mode}",
        "",
    ]
    payload = winner_config.model_dump()
    body = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    target_path.write_text("\n".join(header_lines) + body, encoding="utf-8")
    return target_path


def run_research_mode(args: argparse.Namespace, config: AppConfig, assets: list[AssetConfig], root: Path) -> None:
    if not args.backtest_start or not args.backtest_end:
        raise RuntimeError("--research-run requires --backtest-start and --backtest-end")

    symbols = parse_epics_csv(args.research_symbols) if args.research_symbols else list(config.research.symbols)
    if not symbols:
        symbols = _backtest_symbols(args, assets)
    workers = max(1, min(3, int(config.research.max_workers)))
    objective_mode = str(config.research.objective_mode).strip().lower()
    dd_cap_pct = float(config.research.dd_cap_pct)
    dd_cap_basis = str(config.research.dd_cap_basis).strip().lower()
    min_trades_oos = int(config.research.min_trades_oos)

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    research_dir = _resolve_runtime_path(root, f"reports/research/{run_stamp}")
    config_path = _resolve_config_path(root, str(args.config))
    candidate_modes = ["off", "trend", "trend_vol_news"]

    LOGGER.info(
        "Research run started | objective=%s dd_cap_pct=%.2f dd_cap_basis=%s min_trades_oos=%d symbols=%s workers=%d",
        objective_mode,
        dd_cap_pct,
        dd_cap_basis,
        min_trades_oos,
        ",".join(symbols),
        workers,
    )

    futures = []
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for mode in candidate_modes:
            candidate_dir = research_dir / mode
            futures.append(
                executor.submit(
                    _run_single_research_candidate,
                    root=root,
                    args=args,
                    config=config,
                    config_path=config_path,
                    symbols=symbols,
                    gate_mode=mode,
                    run_dir=candidate_dir,
                )
            )
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            summary = result.get("summary", {})
            if not isinstance(summary, dict):
                summary = {}
            LOGGER.info(
                "Research candidate done | mode=%s rc=%s partial=%s reports=%s pnl_net=%.4f dd_peak=%.4f dd_initial=%.4f objective=%.4f",
                result.get("gate_mode"),
                result.get("returncode"),
                bool(result.get("partial_result", False)),
                int(summary.get("reports_count", 0)),
                float(summary.get("total_pnl_net", 0.0)),
                float(summary.get("max_drawdown_pct_peak", summary.get("max_drawdown_pct", 0.0))),
                float(summary.get("max_drawdown_pct_initial", 0.0)),
                float(summary.get("objective_value", OBJECTIVE_FAIL_VALUE)),
            )
            if bool(result.get("partial_result", False)):
                LOGGER.warning(
                    "Research partial result | mode=%s available_symbols=%s missing_symbols=%s",
                    result.get("gate_mode"),
                    ",".join(str(item) for item in result.get("available_symbols", [])),
                    ",".join(str(item) for item in result.get("missing_symbols", [])),
                )

    results.sort(key=lambda item: objective_rank_key(item.get("summary", {})))
    best = results[0] if results else None
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective_mode": objective_mode,
        "dd_cap_pct": dd_cap_pct,
        "dd_cap_basis": dd_cap_basis,
        "min_trades_oos": min_trades_oos,
        "symbols": symbols,
        "max_workers": workers,
        "seed": int(config.research.seed),
        "config": str(config_path),
        "backtest": {
            "start": str(args.backtest_start),
            "end": str(args.backtest_end),
            "timeframe": str(args.backtest_tf),
            "price": str(args.backtest_price),
            "variants": str(args.backtest_variants),
            "initial_equity": float(args.initial_equity if args.initial_equity is not None else config.risk.equity),
        },
        "candidates": results,
        "best": best,
    }
    _write_research_summary_files(research_dir, payload)
    winner_config_path = _write_research_winner_config(
        root=root,
        base_config=config,
        source_config_path=config_path,
        research_dir=research_dir,
        best_candidate=best,
    )

    LOGGER.info("Research summary saved: %s", research_dir)
    if winner_config_path is not None:
        LOGGER.info("Research winner config saved: %s", winner_config_path)
    LOGGER.info("Research ranking (best->worst):")
    LOGGER.info(
        "rank | mode | pnl_net | max_dd_peak | max_dd_initial | objective | dd_cap_pass | oos_pass | trades"
    )
    for idx, item in enumerate(results, start=1):
        summary = item.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        LOGGER.info(
            "%d | %s | %.4f | %.4f | %.4f | %.4f | %s | %s | %s",
            idx,
            item.get("gate_mode"),
            float(summary.get("total_pnl_net", 0.0)),
            float(summary.get("max_drawdown_pct_peak", summary.get("max_drawdown_pct", 0.0))),
            float(summary.get("max_drawdown_pct_initial", 0.0)),
            float(summary.get("objective_value", OBJECTIVE_FAIL_VALUE)),
            bool(summary.get("constraint_dd_cap_pass", False)),
            bool(summary.get("oos_pass", False)),
            int(summary.get("trades", 0)),
        )


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
        "cost_breakdown_net": {},
        "blocked_by_reason": {},
        "oos_pass": False,
        "constraint_dd_cap_pass_peak": False,
        "constraint_dd_cap_pass_initial": False,
        "constraint_dd_cap_pass": False,
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


def _build_optimizer_candidate_config(
    *,
    base_config: AppConfig,
    gate_mode: str,
    gate_params: dict[str, object],
    risk_profile: dict[str, object] | None,
) -> dict[str, object]:
    payload = base_config.model_dump()

    daily_gate = payload.setdefault("daily_gate", {})
    if not isinstance(daily_gate, dict):
        daily_gate = {}
        payload["daily_gate"] = daily_gate
    daily_gate["mode"] = str(gate_mode).strip().lower()
    if "thr" in gate_params:
        daily_gate["thr"] = float(gate_params["thr"])
    if "pre_minutes" in gate_params:
        daily_gate["pre_minutes"] = int(gate_params["pre_minutes"])
    if "post_minutes" in gate_params:
        daily_gate["post_minutes"] = int(gate_params["post_minutes"])
    if "vol_max" in gate_params:
        daily_gate["vol_max"] = float(gate_params["vol_max"])
    if "max_spread" in gate_params:
        daily_gate["max_spread"] = float(gate_params["max_spread"])
    if "max_spread_mult" in gate_params:
        base_spread = daily_gate.get("max_spread", base_config.daily_gate.max_spread)
        if base_spread is not None:
            daily_gate["max_spread"] = float(base_spread) * float(gate_params["max_spread_mult"])

    if risk_profile:
        risk = payload.setdefault("risk", {})
        if not isinstance(risk, dict):
            risk = {}
            payload["risk"] = risk
        if "risk_per_trade" in risk_profile:
            risk["risk_per_trade"] = float(risk_profile["risk_per_trade"])
        if "max_trades_per_day" in risk_profile:
            risk["max_trades_per_day"] = int(risk_profile["max_trades_per_day"])
        if "max_total_risk_pct" in risk_profile:
            risk["max_total_risk_pct"] = float(risk_profile["max_total_risk_pct"])
        if "daily_stop_pct" in risk_profile:
            risk["daily_stop_pct"] = float(risk_profile["daily_stop_pct"])

    return payload


def _build_backtest_subprocess_command_for_optimizer(
    *,
    root: Path,
    args: argparse.Namespace,
    config_path: Path,
    symbols: list[str],
    start: str,
    end: str,
    run_report_dir: Path,
    run_auto_reports_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(root / "main.py"),
        "--backtest",
        "--backtest-symbols",
        ",".join(symbols),
        "--backtest-start",
        str(start),
        "--backtest-end",
        str(end),
        "--backtest-tf",
        str(args.backtest_tf),
        "--backtest-price",
        str(args.backtest_price),
        "--backtest-data-root",
        str(args.backtest_data_root),
        "--backtest-spread",
        str(args.backtest_spread),
        "--backtest-slippage-points",
        str(args.backtest_slippage_points),
        "--backtest-slippage-atr-multiplier",
        str(args.backtest_slippage_atr_multiplier),
        "--backtest-variants",
        str(args.backtest_variants),
        "--report",
        "--report-formats",
        "json",
        "--report-dir",
        str(run_report_dir),
        "--backtest-reports-dir",
        str(run_auto_reports_dir),
        "--config",
        str(config_path),
        "--no-dashboard",
        "--no-mc-viewer",
    ]
    if str(args.backtest_source_priority or "").strip():
        command.extend(["--backtest-source-priority", str(args.backtest_source_priority)])
    if bool(args.backtest_autofetch):
        command.append("--backtest-autofetch")
    if bool(args.walk_forward):
        command.extend(["--walk-forward", "--wf-splits", str(args.wf_splits)])
    if args.initial_equity is not None:
        command.extend(["--initial-equity", str(args.initial_equity)])
    return command


def _run_optimizer_window_candidate(
    *,
    root: Path,
    args: argparse.Namespace,
    config: AppConfig,
    config_path: Path,
    symbols: list[str],
    start: str,
    end: str,
    run_dir: Path,
    dd_cap_pct: float,
    dd_cap_basis: str,
    min_trades_oos: int,
    objective_mode: str,
    seed: int,
) -> dict[str, object]:
    report_dir = run_dir / "detailed_reports"
    auto_reports_dir = run_dir / "auto_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    auto_reports_dir.mkdir(parents=True, exist_ok=True)

    command = _build_backtest_subprocess_command_for_optimizer(
        root=root,
        args=args,
        config_path=config_path,
        symbols=symbols,
        start=start,
        end=end,
        run_report_dir=report_dir,
        run_auto_reports_dir=auto_reports_dir,
    )

    env = os.environ.copy()
    if config.backtest_runtime.deterministic:
        env["PYTHONHASHSEED"] = str(seed)

    started_at = time.time()
    result = subprocess.run(command, cwd=str(root), capture_output=True, text=True, env=env)
    elapsed_seconds = round(time.time() - started_at, 3)

    source_reports = _extract_source_reports_from_report_dir(report_dir)
    if result.returncode != 0 and not source_reports:
        # One retry for transient failures.
        retry = subprocess.run(command, cwd=str(root), capture_output=True, text=True, env=env)
        elapsed_seconds = round(time.time() - started_at, 3)
        result = retry
        source_reports = _extract_source_reports_from_report_dir(report_dir)

    if source_reports:
        summary = aggregate_reports(
            source_reports,
            initial_equity=float(config.risk.equity),
            dd_cap_pct=dd_cap_pct,
            dd_cap_basis=dd_cap_basis,
            min_trades_oos=min_trades_oos,
            objective_mode=objective_mode,
        )
    else:
        summary = _empty_aggregate_summary()

    return {
        "command": command,
        "returncode": int(result.returncode),
        "elapsed_seconds": elapsed_seconds,
        "stdout_tail": "\n".join((result.stdout or "").splitlines()[-20:]),
        "stderr_tail": "\n".join((result.stderr or "").splitlines()[-20:]),
        "report_dir": str(report_dir),
        "auto_reports_dir": str(auto_reports_dir),
        "summary": summary,
    }


def _find_resumable_research_opt_dir(base_dir: Path, *, resume_key: dict[str, object]) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = sorted((path for path in base_dir.iterdir() if path.is_dir()), reverse=True)
    for path in candidates:
        checkpoint_path = path / "checkpoint.json"
        if not checkpoint_path.exists():
            continue
        checkpoint = load_checkpoint(checkpoint_path)
        metadata = checkpoint.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        if bool(metadata.get("completed", False)):
            continue
        key = metadata.get("resume_key")
        if isinstance(key, dict) and key == resume_key:
            return path
    return None


def _summarize_stage_progress(
    *,
    stage: str,
    completed: int,
    total: int,
    started_at: float,
) -> None:
    elapsed = max(1e-9, time.time() - started_at)
    throughput_per_min = (completed / elapsed) * 60.0
    remaining = max(0, total - completed)
    eta_minutes = (remaining / throughput_per_min) if throughput_per_min > 1e-9 else 0.0
    LOGGER.info(
        "Research optimize progress | stage=%s done=%d/%d throughput=%.2f cand/min eta=%.2f min",
        stage,
        completed,
        total,
        throughput_per_min,
        eta_minutes,
    )


def _write_optimizer_best_config(
    *,
    root: Path,
    base_config: AppConfig,
    source_config_path: Path,
    report_dir: Path,
    best_record: dict[str, object] | None,
    split_payload: dict[str, object],
    objective_mode: str,
    dd_cap_pct: float,
    dd_cap_basis: str,
) -> Path | None:
    if not isinstance(best_record, dict):
        return None
    gate_mode = str(best_record.get("gate_mode", "")).strip().lower()
    if gate_mode not in {"off", "trend", "trend_vol_news"}:
        return None
    gate_params = best_record.get("gate_params", {})
    if not isinstance(gate_params, dict):
        gate_params = {}
    risk_profile = best_record.get("risk_profile", {})
    if not isinstance(risk_profile, dict):
        risk_profile = {}

    winner_payload = _build_optimizer_candidate_config(
        base_config=base_config,
        gate_mode=gate_mode,
        gate_params=gate_params,
        risk_profile=risk_profile,
    )

    target_path = root / "configs" / "variants" / "config.variant_RESEARCH_OPT_BEST.yaml"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Auto-generated by --research-optimize",
        f"# generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"# source_config: {source_config_path}",
        f"# research_opt_report_dir: {report_dir}",
        f"# objective_mode: {objective_mode}",
        f"# dd_cap_pct: {dd_cap_pct}",
        f"# dd_cap_basis: {dd_cap_basis}",
        f"# split: {json.dumps(split_payload, ensure_ascii=True)}",
        "",
    ]
    body = yaml.safe_dump(winner_payload, sort_keys=False, allow_unicode=False)
    target_path.write_text("\n".join(header) + body, encoding="utf-8")
    return target_path


def run_research_optimize_mode(args: argparse.Namespace, config: AppConfig, assets: list[AssetConfig], root: Path) -> None:
    if not args.backtest_start or not args.backtest_end:
        raise RuntimeError("--research-optimize requires --backtest-start and --backtest-end")

    optimize_cfg = config.research.optimize
    runtime_budget = normalize_runtime_budget(
        args.research_runtime_budget if args.research_runtime_budget is not None else optimize_cfg.runtime_budget
    )
    symbols = parse_epics_csv(args.research_benchmark_symbols) if args.research_benchmark_symbols else list(config.research.symbols)
    if not symbols:
        symbols = _backtest_symbols(args, assets)
    if not symbols:
        raise RuntimeError("No symbols available for --research-optimize")

    workers = max(1, min(3, int(optimize_cfg.max_workers)))
    dd_cap_pct = float(config.research.dd_cap_pct)
    dd_cap_basis = str(optimize_cfg.dd_cap_basis or config.research.dd_cap_basis).strip().lower()
    min_trades_oos = int(config.research.min_trades_oos)
    objective_mode = str(optimize_cfg.objective_mode).strip().lower()
    seed = int(optimize_cfg.seed)
    config_path = _resolve_config_path(root, str(args.config))

    split = build_time_split(
        backtest_start=str(args.backtest_start),
        backtest_end=str(args.backtest_end),
        split_ratio_is=float(optimize_cfg.split_ratio_is),
        min_days_is=int(optimize_cfg.min_days_is),
        min_days_oos=int(optimize_cfg.min_days_oos),
    )
    split_payload = split.to_dict()

    search_space_payload = config.research.search_space.model_dump()
    gate_space = dict(search_space_payload.get("gate", {}))
    risk_profiles = list(search_space_payload.get("risk_profiles", []))
    stage_a_candidates = build_stage_a_gate_candidates(
        search_space_gate=gate_space,
        runtime_budget=runtime_budget,
    )
    top_gate_keep = min(int(optimize_cfg.top_gate_keep), len(stage_a_candidates))
    top_final_keep = max(1, int(optimize_cfg.top_final_keep))

    resume_key = {
        "config": str(config_path),
        "symbols": symbols,
        "backtest_start": str(args.backtest_start),
        "backtest_end": str(args.backtest_end),
        "timeframe": str(args.backtest_tf),
        "price": str(args.backtest_price),
        "runtime_budget": runtime_budget,
        "dd_cap_pct": dd_cap_pct,
        "dd_cap_basis": dd_cap_basis,
        "objective_mode": objective_mode,
    }
    base_dir = _resolve_runtime_path(root, "reports/research_opt")
    resumable = _find_resumable_research_opt_dir(base_dir, resume_key=resume_key)
    if resumable is not None:
        run_dir = resumable
        resumed = True
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_dir = base_dir / stamp
        resumed = False
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_meta = checkpoint.setdefault("metadata", {})
    if not isinstance(checkpoint_meta, dict):
        checkpoint_meta = {}
        checkpoint["metadata"] = checkpoint_meta
    checkpoint_meta["resume_key"] = resume_key
    checkpoint_meta["runtime_budget"] = runtime_budget
    checkpoint_meta["objective_mode"] = objective_mode
    checkpoint_meta["dd_cap_pct"] = dd_cap_pct
    checkpoint_meta["dd_cap_basis"] = dd_cap_basis
    checkpoint_meta["completed"] = False
    save_checkpoint(checkpoint_path, checkpoint)

    search_space_out = build_search_space_payload(
        runtime_budget=runtime_budget,
        gate_space=gate_space,
        risk_profiles=risk_profiles,
        stage_a_candidates=stage_a_candidates,
    )
    (run_dir / "search_space.json").write_text(json.dumps(search_space_out, indent=2, ensure_ascii=True), encoding="utf-8")
    (run_dir / "split_info.json").write_text(json.dumps(split_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    LOGGER.info(
        "Research optimize started | resumed=%s budget=%s symbols=%s workers=%d stageA=%d",
        resumed,
        runtime_budget,
        ",".join(symbols),
        workers,
        len(stage_a_candidates),
    )

    stage_a_results: list[dict[str, object]] = []
    stage_a_started_at = time.time()
    stage_a_total = len(stage_a_candidates)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[object, tuple[dict[str, object], Path]] = {}
        for candidate in stage_a_candidates:
            candidate_id = str(candidate.get("candidate_id", ""))
            cached = get_checkpoint_record(checkpoint, stage="A", candidate_id=candidate_id)
            if cached is not None and cached.get("status") == "done":
                stage_a_results.append(cached)
                continue
            candidate_dir = run_dir / "stage_a" / candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate_config_payload = _build_optimizer_candidate_config(
                base_config=config,
                gate_mode=str(candidate.get("gate_mode", "off")),
                gate_params=dict(candidate.get("gate_params", {})),
                risk_profile=None,
            )
            candidate_config_path = candidate_dir / "candidate_config.yaml"
            candidate_config_path.write_text(
                yaml.safe_dump(candidate_config_payload, sort_keys=False, allow_unicode=False),
                encoding="utf-8",
            )
            future = executor.submit(
                _run_optimizer_window_candidate,
                root=root,
                args=args,
                config=config,
                config_path=candidate_config_path,
                symbols=symbols,
                start=split.is_start,
                end=split.is_end,
                run_dir=candidate_dir / "is",
                dd_cap_pct=dd_cap_pct,
                dd_cap_basis=dd_cap_basis,
                min_trades_oos=min_trades_oos,
                objective_mode=objective_mode,
                seed=seed,
            )
            futures[future] = (candidate, candidate_config_path)

        completed = len(stage_a_results)
        if completed:
            _summarize_stage_progress(stage="A", completed=completed, total=stage_a_total, started_at=stage_a_started_at)
        for future in as_completed(futures):
            candidate, candidate_config_path = futures[future]
            result = future.result()
            record = {
                "status": "done",
                "stage": "A",
                "candidate_id": candidate.get("candidate_id"),
                "gate_mode": candidate.get("gate_mode"),
                "gate_params": candidate.get("gate_params", {}),
                "risk_profile": None,
                "config_path": str(candidate_config_path),
                "returncode": result.get("returncode"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "summary": result.get("summary", _empty_aggregate_summary()),
                "stdout_tail": result.get("stdout_tail", ""),
                "stderr_tail": result.get("stderr_tail", ""),
                "report_dir": result.get("report_dir"),
            }
            stage_a_results.append(record)
            upsert_checkpoint_record(checkpoint, stage="A", candidate_id=str(record.get("candidate_id", "")), record=record)
            save_checkpoint(checkpoint_path, checkpoint)
            completed += 1
            if completed % 5 == 0 or completed == stage_a_total:
                _summarize_stage_progress(stage="A", completed=completed, total=stage_a_total, started_at=stage_a_started_at)

    stage_a_results.sort(key=lambda item: optimizer_rank_key(item.get("summary", {})))
    top_gate_candidates = [
        {
            "candidate_id": item.get("candidate_id"),
            "gate_mode": item.get("gate_mode"),
            "gate_params": item.get("gate_params", {}),
        }
        for item in stage_a_results[:top_gate_keep]
    ]

    stage_a_csv_rows: list[dict[str, object]] = []
    for rank, item in enumerate(stage_a_results, start=1):
        summary = item.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        stage_a_csv_rows.append(
            {
                "rank": rank,
                "candidate_id": item.get("candidate_id"),
                "gate_mode": item.get("gate_mode"),
                "gate_params": json.dumps(item.get("gate_params", {}), ensure_ascii=True, sort_keys=True),
                "returncode": item.get("returncode"),
                "elapsed_seconds": item.get("elapsed_seconds", 0.0),
                "trades": summary.get("trades", 0),
                "total_pnl_net": summary.get("total_pnl_net", 0.0),
                "max_drawdown_pct_peak": summary.get("max_drawdown_pct_peak", summary.get("max_drawdown_pct", 0.0)),
                "max_drawdown_pct_initial": summary.get("max_drawdown_pct_initial", 0.0),
                "dd_ref_pct": summary.get("dd_ref_pct", 0.0),
                "constraint_dd_cap_pass": summary.get("constraint_dd_cap_pass", False),
                "oos_pass": summary.get("oos_pass", False),
                "objective_value": summary.get("objective_value", OBJECTIVE_FAIL_VALUE),
            }
        )
    _write_csv(
        run_dir / "stage_a_gate_is.csv",
        fieldnames=[
            "rank",
            "candidate_id",
            "gate_mode",
            "gate_params",
            "returncode",
            "elapsed_seconds",
            "trades",
            "total_pnl_net",
            "max_drawdown_pct_peak",
            "max_drawdown_pct_initial",
            "dd_ref_pct",
            "constraint_dd_cap_pass",
            "oos_pass",
            "objective_value",
        ],
        rows=stage_a_csv_rows,
    )

    stage_b_candidates = build_stage_b_candidates(
        top_gate_candidates=top_gate_candidates,
        risk_profiles=risk_profiles,
        runtime_budget=runtime_budget,
    )
    stage_b_total = len(stage_b_candidates)
    used_risk_profiles = len(
        {str(item.get("risk_profile", {}).get("name", "")) for item in stage_b_candidates if isinstance(item, dict)}
    )
    LOGGER.info(
        "Research optimize stage B | top_gate_keep=%d risk_profiles=%d candidates=%d",
        len(top_gate_candidates),
        used_risk_profiles,
        stage_b_total,
    )

    stage_b_results: list[dict[str, object]] = []
    stage_b_started_at = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[object, tuple[str, dict[str, object], Path]] = {}
        for candidate in stage_b_candidates:
            candidate_id = str(candidate.get("candidate_id", ""))
            cached = get_checkpoint_record(checkpoint, stage="B", candidate_id=candidate_id)
            if cached is not None and cached.get("status") == "done":
                stage_b_results.append(cached)
                continue

            candidate_dir = run_dir / "stage_b" / candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate_config_payload = _build_optimizer_candidate_config(
                base_config=config,
                gate_mode=str(candidate.get("gate_mode", "off")),
                gate_params=dict(candidate.get("gate_params", {})),
                risk_profile=dict(candidate.get("risk_profile", {})),
            )
            candidate_config_path = candidate_dir / "candidate_config.yaml"
            candidate_config_path.write_text(
                yaml.safe_dump(candidate_config_payload, sort_keys=False, allow_unicode=False),
                encoding="utf-8",
            )

            future_is = executor.submit(
                _run_optimizer_window_candidate,
                root=root,
                args=args,
                config=config,
                config_path=candidate_config_path,
                symbols=symbols,
                start=split.is_start,
                end=split.is_end,
                run_dir=candidate_dir / "is",
                dd_cap_pct=dd_cap_pct,
                dd_cap_basis=dd_cap_basis,
                min_trades_oos=min_trades_oos,
                objective_mode=objective_mode,
                seed=seed,
            )
            futures[future_is] = ("is", candidate, candidate_config_path)

            future_oos = executor.submit(
                _run_optimizer_window_candidate,
                root=root,
                args=args,
                config=config,
                config_path=candidate_config_path,
                symbols=symbols,
                start=split.oos_start,
                end=split.oos_end,
                run_dir=candidate_dir / "oos",
                dd_cap_pct=dd_cap_pct,
                dd_cap_basis=dd_cap_basis,
                min_trades_oos=min_trades_oos,
                objective_mode=objective_mode,
                seed=seed,
            )
            futures[future_oos] = ("oos", candidate, candidate_config_path)

        interim: dict[str, dict[str, object]] = {}
        completed = len(stage_b_results)
        if completed:
            _summarize_stage_progress(stage="B", completed=completed, total=stage_b_total, started_at=stage_b_started_at)

        for future in as_completed(futures):
            window_name, candidate, candidate_config_path = futures[future]
            result = future.result()
            candidate_id = str(candidate.get("candidate_id", ""))
            bucket = interim.setdefault(candidate_id, {"candidate": candidate, "config_path": candidate_config_path})
            bucket[window_name] = result
            if "is" in bucket and "oos" in bucket:
                is_summary = bucket["is"].get("summary", _empty_aggregate_summary())
                oos_summary = bucket["oos"].get("summary", _empty_aggregate_summary())
                if not isinstance(is_summary, dict):
                    is_summary = _empty_aggregate_summary()
                if not isinstance(oos_summary, dict):
                    oos_summary = _empty_aggregate_summary()
                summary = build_stage_b_summary(
                    is_summary=is_summary,
                    oos_summary=oos_summary,
                    dd_cap_pct=dd_cap_pct,
                    dd_cap_basis=dd_cap_basis,
                    min_trades_oos=min_trades_oos,
                    objective_mode=objective_mode,
                )
                combined = {
                    "status": "done",
                    "stage": "B",
                    "candidate_id": candidate.get("candidate_id"),
                    "gate_candidate_id": candidate.get("gate_candidate_id"),
                    "gate_mode": candidate.get("gate_mode"),
                    "gate_params": candidate.get("gate_params", {}),
                    "risk_profile": candidate.get("risk_profile", {}),
                    "risk_profile_name": dict(candidate.get("risk_profile", {})).get("name", ""),
                    "config_path": str(candidate_config_path),
                    "is_returncode": bucket["is"].get("returncode"),
                    "oos_returncode": bucket["oos"].get("returncode"),
                    "elapsed_seconds": float(bucket["is"].get("elapsed_seconds", 0.0))
                    + float(bucket["oos"].get("elapsed_seconds", 0.0)),
                    "is_summary": is_summary,
                    "oos_summary": oos_summary,
                    "summary": summary,
                    "stderr_tail_is": bucket["is"].get("stderr_tail", ""),
                    "stderr_tail_oos": bucket["oos"].get("stderr_tail", ""),
                }
                stage_b_results.append(combined)
                upsert_checkpoint_record(checkpoint, stage="B", candidate_id=candidate_id, record=combined)
                save_checkpoint(checkpoint_path, checkpoint)
                interim.pop(candidate_id, None)
                completed += 1
                if completed % 5 == 0 or completed == stage_b_total:
                    _summarize_stage_progress(stage="B", completed=completed, total=stage_b_total, started_at=stage_b_started_at)

    stage_b_results.sort(key=lambda item: optimizer_rank_key(item.get("summary", failed_stage_summary())))
    top_records = stage_b_results[: min(top_final_keep, len(stage_b_results))]
    best_record = top_records[0] if top_records else None

    stage_b_csv_rows: list[dict[str, object]] = []
    for rank, item in enumerate(stage_b_results, start=1):
        summary = item.get("summary", {})
        if not isinstance(summary, dict):
            summary = failed_stage_summary()
        stage_b_csv_rows.append(
            {
                "rank": rank,
                "candidate_id": item.get("candidate_id"),
                "gate_candidate_id": item.get("gate_candidate_id"),
                "gate_mode": item.get("gate_mode"),
                "gate_params": json.dumps(item.get("gate_params", {}), ensure_ascii=True, sort_keys=True),
                "risk_profile_name": item.get("risk_profile_name"),
                "risk_profile": json.dumps(item.get("risk_profile", {}), ensure_ascii=True, sort_keys=True),
                "is_returncode": item.get("is_returncode"),
                "oos_returncode": item.get("oos_returncode"),
                "elapsed_seconds": item.get("elapsed_seconds", 0.0),
                "is_total_pnl_net": summary.get("is_total_pnl_net", 0.0),
                "is_dd_ref_pct": summary.get("is_dd_ref_pct", 0.0),
                "oos_total_pnl_net": summary.get("oos_total_pnl_net", 0.0),
                "oos_dd_ref_pct": summary.get("oos_dd_ref_pct", 0.0),
                "oos_expectancy_net": summary.get("oos_expectancy_net", 0.0),
                "oos_trades": summary.get("oos_trades", 0),
                "constraint_dd_cap_pass_peak": summary.get("constraint_dd_cap_pass_peak", False),
                "constraint_dd_cap_pass_initial": summary.get("constraint_dd_cap_pass_initial", False),
                "constraint_dd_cap_pass": summary.get("constraint_dd_cap_pass", False),
                "oos_pass": summary.get("oos_pass", False),
                "objective_value": summary.get("objective_value", OBJECTIVE_FAIL_VALUE),
            }
        )
    _write_csv(
        run_dir / "stage_b_gate_risk_is_oos.csv",
        fieldnames=[
            "rank",
            "candidate_id",
            "gate_candidate_id",
            "gate_mode",
            "gate_params",
            "risk_profile_name",
            "risk_profile",
            "is_returncode",
            "oos_returncode",
            "elapsed_seconds",
            "is_total_pnl_net",
            "is_dd_ref_pct",
            "oos_total_pnl_net",
            "oos_dd_ref_pct",
            "oos_expectancy_net",
            "oos_trades",
            "constraint_dd_cap_pass_peak",
            "constraint_dd_cap_pass_initial",
            "constraint_dd_cap_pass",
            "oos_pass",
            "objective_value",
        ],
        rows=stage_b_csv_rows,
    )

    top20_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_budget": runtime_budget,
        "objective_mode": objective_mode,
        "dd_cap_pct": dd_cap_pct,
        "dd_cap_basis": dd_cap_basis,
        "split": split_payload,
        "top": top_records,
    }
    (run_dir / "top20.json").write_text(json.dumps(top20_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    (run_dir / "best.json").write_text(
        json.dumps({"best": best_record, "split": split_payload, "objective_mode": objective_mode}, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    winner_path = _write_optimizer_best_config(
        root=root,
        base_config=config,
        source_config_path=config_path,
        report_dir=run_dir,
        best_record=best_record,
        split_payload=split_payload,
        objective_mode=objective_mode,
        dd_cap_pct=dd_cap_pct,
        dd_cap_basis=dd_cap_basis,
    )

    checkpoint["metadata"]["completed"] = True
    checkpoint["metadata"]["stage_a_candidates"] = len(stage_a_candidates)
    checkpoint["metadata"]["stage_b_candidates"] = len(stage_b_candidates)
    save_checkpoint(checkpoint_path, checkpoint)

    LOGGER.info("Research optimize summary saved: %s", run_dir)
    if winner_path is not None:
        LOGGER.info("Research optimize winner config saved: %s", winner_path)
    LOGGER.info("Research optimize ranking (best->worst):")
    LOGGER.info("rank | gate | risk | oos_pnl_net | oos_dd_ref | objective | dd_pass | oos_pass | oos_trades")
    for idx, record in enumerate(top_records[:20], start=1):
        summary = record.get("summary", {})
        if not isinstance(summary, dict):
            summary = failed_stage_summary()
        LOGGER.info(
            "%d | %s | %s | %.4f | %.4f | %.4f | %s | %s | %d",
            idx,
            record.get("gate_mode"),
            record.get("risk_profile_name"),
            float(summary.get("oos_total_pnl_net", 0.0)),
            float(summary.get("oos_dd_ref_pct", 0.0)),
            float(summary.get("objective_value", OBJECTIVE_FAIL_VALUE)),
            bool(summary.get("constraint_dd_cap_pass", False)),
            bool(summary.get("oos_pass", False)),
            int(summary.get("oos_trades", 0)),
        )


def _run_multi_strategy_segmented(
    *,
    config: AppConfig,
    asset: AssetConfig,
    frame,
    loader: AutoDataLoader,
    timeframe: str,
    assumed_spread: float,
    slippage_points: float,
    slippage_atr_multiplier: float,
    variant: BacktestVariant,
    execution_debug_path: Path | None,
    no_price_debug_path: Path | None,
    reaction_timeout_debug_path: Path | None,
    data_context: dict[str, object],
    trade_start_utc: datetime | None = None,
    flatten_at_chunk_end: bool = False,
    daily_gate: DailyGateProvider | None = None,
    daily_gate_prepared: bool = False,
    mc_model: MCAdaptiveModel | None = None,
):
    segments, segment_info = loader.split_frame_by_gaps(
        frame,
        timeframe,
        gap_bars=3,
        soft_gap_minutes=int(config.backtest_tuning.segment_soft_gap_minutes),
        hard_gap_minutes=int(config.backtest_tuning.segment_hard_gap_minutes),
    )
    if not segments:
        segments = [frame]
        segment_info = {
            "segment_count": 1,
            "segment_sizes": [int(len(frame))],
            "gap_threshold_bars": 3,
            "gap_threshold_minutes": 0.0,
            "soft_gap_minutes": float(config.backtest_tuning.segment_soft_gap_minutes),
            "hard_gap_minutes": float(config.backtest_tuning.segment_hard_gap_minutes),
            "gap_count_over_threshold": 0,
            "gap_count_soft_only": 0,
            "gaps_over_threshold": [],
            "gaps_soft_only": [],
        }

    reports = []
    skipped_small = 0
    rolling_equity = float(config.risk.equity)
    warmup_bars_per_segment = max(260, int(config.execution.history_bars.m5))
    for idx, segment in enumerate(segments):
        segment_input = segment
        if idx > 0 and warmup_bars_per_segment > 0:
            prior = pd.concat(segments[:idx], ignore_index=True).tail(warmup_bars_per_segment)
            if not prior.empty:
                segment_input = (
                    pd.concat([prior, segment], ignore_index=True)
                    .sort_values("ts_utc")
                    .drop_duplicates(subset=["ts_utc"], keep="last")
                    .reset_index(drop=True)
                )
        candles = BacktestRunner._frame_to_candles(segment_input)
        if len(candles) < 260:
            skipped_small += 1
            continue
        segment_start_ts = pd.to_datetime(segment["ts_utc"].iloc[0], utc=True, errors="coerce")
        segment_trade_start = trade_start_utc
        if pd.notna(segment_start_ts):
            segment_start = segment_start_ts.to_pydatetime()
            if segment_trade_start is None or segment_trade_start < segment_start:
                segment_trade_start = segment_start
        segment_context = dict(data_context)
        segment_context["segment_index"] = idx + 1
        segment_context["segment_count"] = len(segments)
        segment_context["segment_start_utc"] = segment["ts_utc"].iloc[0].isoformat()
        segment_context["segment_end_utc"] = segment["ts_utc"].iloc[-1].isoformat()
        segment_context["segment_input_bars"] = int(len(segment_input))
        segment_context["segment_trade_bars"] = int(len(segment))
        segment_context["segment_start_equity"] = float(rolling_equity)
        segment_config = config.model_copy(deep=True)
        segment_config.risk.equity = float(rolling_equity)
        report = run_backtest_multi_strategy(
            config=segment_config,
            asset=asset,
            candles_m5=candles,
            assumed_spread=assumed_spread,
            slippage_points=slippage_points,
            slippage_atr_multiplier=slippage_atr_multiplier,
            variant=variant,
            execution_debug_path=execution_debug_path,
            no_price_debug_path=no_price_debug_path,
            reaction_timeout_debug_path=reaction_timeout_debug_path,
            data_context=segment_context,
            trade_start_utc=segment_trade_start,
            flatten_at_chunk_end=flatten_at_chunk_end,
            daily_gate=daily_gate,
            daily_gate_prepared=daily_gate_prepared,
            mc_model=mc_model,
        )
        reports.append(report)
        rolling_equity = float(getattr(report, "equity_end", rolling_equity))

    if not reports:
        candles_all = BacktestRunner._frame_to_candles(frame)
        report = run_backtest_multi_strategy(
            config=config,
            asset=asset,
            candles_m5=candles_all,
            assumed_spread=assumed_spread,
            slippage_points=slippage_points,
            slippage_atr_multiplier=slippage_atr_multiplier,
            variant=variant,
            execution_debug_path=execution_debug_path,
            no_price_debug_path=no_price_debug_path,
            reaction_timeout_debug_path=reaction_timeout_debug_path,
            data_context=data_context,
            trade_start_utc=trade_start_utc,
            flatten_at_chunk_end=flatten_at_chunk_end,
            daily_gate=daily_gate,
            daily_gate_prepared=daily_gate_prepared,
            mc_model=mc_model,
        )
        reports = [report]

    if len(reports) == 1:
        merged = reports[0]
    else:
        merged = aggregate_backtest_reports(
            config=config,
            asset=asset,
            reports=reports,
        )

    segment_meta = dict(segment_info)
    segment_meta["segment_run_count"] = len(reports)
    segment_meta["segment_skipped_small"] = skipped_small
    segment_meta["equity_start"] = float(config.risk.equity)
    segment_meta["equity_end"] = float(rolling_equity)
    return merged, segment_meta


_dashboard_server = None  # module-level so both helpers share state


def _maybe_start_dashboard(args: argparse.Namespace, config: AppConfig) -> None:
    """Start the web dashboard in the background (non-blocking).

    Called *before* the backtest runs so the user can watch data live.
    The JSONL file doesn't need to exist yet – the reader thread will
    wait for it to appear.
    """
    global _dashboard_server
    if not getattr(args, "dashboard", False):
        return
    if not config.diagnostics.decision_trace_enabled:
        return
    trace_path = Path(config.diagnostics.decision_trace_path)
    try:
        from tools.termviz_web import start_dashboard
        LOGGER.info("Starting decision-trace dashboard on http://localhost:8777 ...")
        _dashboard_server = start_dashboard(
            path=trace_path,
            port=8777,
            open_browser=True,
            blocking=False,   # returns immediately
        )
    except (ImportError, OSError):
        LOGGER.exception("Could not start dashboard")


def _maybe_block_dashboard(*, hold_open: bool = False) -> None:
    """Optionally keep process alive so dashboard/MC viewer stay open.

    Called *after* backtest. When hold_open is False, viewers are closed and
    function returns immediately. When hold_open is True, process blocks until
    dashboard/viewer are closed or interrupted.
    """
    global _dashboard_server
    global _mc_viewer_proc

    has_dashboard = _dashboard_server is not None
    has_mc = _mc_viewer_proc is not None and _mc_viewer_proc.poll() is None

    if not has_dashboard and not has_mc:
        return

    if not hold_open:
        if _dashboard_server is not None:
            _dashboard_server.shutdown()
            _dashboard_server = None
        _maybe_stop_mc_viewer()
        return

    parts: list[str] = []
    if has_dashboard:
        parts.append("dashboard")
    if has_mc:
        parts.append("MC viewer")
    running_label = " + ".join(parts)

    print(f"\n  Backtest complete – {running_label} still running.")
    print("  Press Ctrl+C to stop.\n")
    try:
        while True:
            # If MC viewer was closed by the user (window closed), note it
            if _mc_viewer_proc is not None and _mc_viewer_proc.poll() is not None:
                _mc_viewer_proc = None
            # If dashboard is gone and MC viewer is gone, stop blocking
            if _dashboard_server is None and (_mc_viewer_proc is None or _mc_viewer_proc.poll() is not None):
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if _dashboard_server is not None:
            _dashboard_server.shutdown()
            _dashboard_server = None
        _maybe_stop_mc_viewer()
        print("Stopped.")


# ---------------------------------------------------------------------------
# Monte Carlo live viewer auto-launch
# ---------------------------------------------------------------------------
_mc_viewer_proc: subprocess.Popen | None = None
_mc_viewer_stderr_fh = None  # file handle for MC viewer stderr log


def _maybe_start_mc_viewer(config: AppConfig, root: Path, cli_override: bool | None = None) -> None:
    """Spawn the Monte Carlo live viewer as a child process (non-blocking).

    Starts when ``monte_carlo.live_window.enabled`` is True and
    ``viewer_mode`` is ``"process"`` or ``"terminal"`` — or when forced
    via *cli_override*.  Failures are logged and swallowed.
    """
    global _mc_viewer_proc, _mc_viewer_stderr_fh
    lw = config.monte_carlo.live_window

    # CLI --mc-viewer / --no-mc-viewer override config
    if cli_override is not None:
        if not cli_override:
            return
        # cli_override True → force-start regardless of config flags
    else:
        if not lw.enabled or lw.viewer_mode not in ("process", "terminal") or not lw.open_on_start:
            return

    png_path = root / lw.png_path
    json_path = root / lw.json_path
    # Ensure the parent directory exists so the bot can later write files
    png_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Terminal mode: launch the plotext-based terminal viewer ----------
    if lw.viewer_mode == "terminal":
        viewer_script = root / "tools" / "termviz_mc.py"
        if not viewer_script.exists():
            LOGGER.warning("MC terminal viewer script not found: %s — skipping", viewer_script)
            return
        cmd = [
            sys.executable,
            str(viewer_script),
            "--json", str(json_path),
            "--refresh", str(lw.refresh_seconds),
            "--title", lw.window_title,
        ]
        mc_log = root / "logs" / "mc_viewer.log"
        mc_log.parent.mkdir(parents=True, exist_ok=True)
        stderr_fh = None
        try:
            stderr_fh = open(mc_log, "w", encoding="utf-8")  # noqa: SIM115
            # CREATE_NEW_CONSOLE on Windows gives the viewer its own terminal window
            creation_flags = 0
            if sys.platform == "win32":
                creation_flags = subprocess.CREATE_NEW_CONSOLE
            _mc_viewer_proc = subprocess.Popen(
                cmd,
                stderr=stderr_fh,
                creationflags=creation_flags,
            )
            _mc_viewer_stderr_fh = stderr_fh
            LOGGER.info("MC terminal viewer started (PID %s)", _mc_viewer_proc.pid)
            time.sleep(0.5)
            if _mc_viewer_proc.poll() is not None:
                rc = _mc_viewer_proc.returncode
                _mc_viewer_proc = None
                LOGGER.warning(
                    "MC terminal viewer exited immediately (code %s) — check %s", rc, mc_log,
                )
        except Exception:
            LOGGER.exception("Could not start MC terminal viewer")
            if stderr_fh is not None and _mc_viewer_proc is None:
                try:
                    stderr_fh.close()
                except OSError:
                    pass
                _mc_viewer_stderr_fh = None
        return

    # -- Process mode: launch the matplotlib PNG viewer -------------------
    viewer_script = root / "tools" / "monte_carlo_live_viewer.py"
    if not viewer_script.exists():
        LOGGER.warning("MC viewer script not found: %s — skipping auto-launch", viewer_script)
        return

    cmd = [
        sys.executable,
        str(viewer_script),
        "--png", str(png_path),
        "--json", str(json_path),
        "--refresh", str(lw.refresh_seconds),
        "--title", lw.window_title,
        "--max-fps", str(lw.max_fps),
    ]
    if not lw.show_stats_overlay:
        cmd.append("--no-overlay")

    # Log stderr to file so viewer errors are visible for debugging
    mc_log = root / "logs" / "mc_viewer.log"
    mc_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_fh = None
    try:
        stderr_fh = open(mc_log, "w", encoding="utf-8")  # noqa: SIM115
        _mc_viewer_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_fh,
        )
        _mc_viewer_stderr_fh = stderr_fh  # store for cleanup
        LOGGER.info("Monte Carlo live viewer started (PID %s)", _mc_viewer_proc.pid)
        # Brief health-check: give the process a moment to die on import errors
        time.sleep(0.5)
        if _mc_viewer_proc.poll() is not None:
            rc = _mc_viewer_proc.returncode
            _mc_viewer_proc = None
            LOGGER.warning(
                "MC viewer exited immediately (code %s) — check %s for details", rc, mc_log,
            )
    except Exception:
        LOGGER.exception("Could not start Monte Carlo live viewer")
        # Close stderr_fh if Popen failed so we don't leak the handle
        if stderr_fh is not None and _mc_viewer_proc is None:
            try:
                stderr_fh.close()
            except OSError:
                pass
            _mc_viewer_stderr_fh = None


def _maybe_stop_mc_viewer() -> None:
    """Terminate the MC viewer child process if it is still running."""
    global _mc_viewer_proc, _mc_viewer_stderr_fh
    if _mc_viewer_proc is None:
        return
    try:
        _mc_viewer_proc.terminate()
        _mc_viewer_proc.wait(timeout=3)
    except Exception:
        pass
    _mc_viewer_proc = None
    # Close the stderr log file handle
    if _mc_viewer_stderr_fh is not None:
        try:
            _mc_viewer_stderr_fh.close()
        except OSError:
            pass
        _mc_viewer_stderr_fh = None


def _resolve_monte_carlo_starting_equity(
    *,
    mode: str,
    configured_initial_equity: float,
    initial_equity: float | None,
    current_equity: float | None,
    fallback_current_equity: float | None,
) -> float:
    mode_norm = str(mode or "initial").strip().lower()
    if mode_norm not in {"initial", "current"}:
        mode_norm = "initial"

    configured = float(configured_initial_equity) if configured_initial_equity > 0 else 1.0
    initial = configured
    if initial_equity is not None:
        try:
            initial_candidate = float(initial_equity)
        except (TypeError, ValueError):
            initial_candidate = None
        if initial_candidate is not None and initial_candidate > 0:
            initial = initial_candidate

    if mode_norm == "current":
        for candidate in (current_equity, fallback_current_equity):
            if candidate is None:
                continue
            try:
                current_val = float(candidate)
            except (TypeError, ValueError):
                continue
            if current_val > 0:
                return current_val

    return initial


def _run_monte_carlo_for_payloads(
    config: AppConfig,
    root: Path,
    gate_payloads: dict[str, dict[str, object]],
) -> None:
    """Extract trade PnLs from gate_payloads and run Monte Carlo simulation.

    When multiple gate modes are present (e.g. off / trend / trend_vol_news),
    uses only the **primary** configured mode — mixing trades from different
    strategy variants would produce meaningless MC results.
    Falls back to the first available mode if the configured mode is absent.
    """
    mc = config.monte_carlo
    if not mc.enabled:
        return

    # Pick the single gate mode to simulate: configured mode first,
    # else the first mode with actual reports.
    primary_mode = _daily_gate_mode(config)
    if primary_mode not in gate_payloads:
        primary_mode = next(iter(gate_payloads), None)
    if primary_mode is None:
        LOGGER.info("Monte Carlo: no gate payloads — skipping simulation.")
        return

    payload = gate_payloads[primary_mode]
    reports_map = payload.get("reports")
    if not isinstance(reports_map, dict):
        LOGGER.info("Monte Carlo: no reports in gate mode %s — skipping.", primary_mode)
        return

    # Collect PnLs from selected gate mode only
    pnls: list[float] = []
    for _sym, report in reports_map.items():
        if not isinstance(report, dict):
            continue
        # Walk-forward reports nest under 'aggregate'
        src = report.get("aggregate", report)
        # Prefer pre-computed _trade_pnls list (raw trade PnLs)
        trade_pnls = src.get("_trade_pnls") or report.get("_trade_pnls")
        if trade_pnls:
            pnls.extend(float(v) for v in trade_pnls)
            continue
        # Fallback: try to extract from serialised trade_log dicts
        trade_log = src.get("trade_log", [])
        for trade in trade_log:
            if isinstance(trade, dict):
                pnl_val = trade.get("pnl")
                if pnl_val is not None:
                    pnls.append(float(pnl_val))

    if not pnls:
        LOGGER.info("Monte Carlo: no trades in mode %s — skipping simulation.", primary_mode)
        return

    # Extract both initial and current equity from reports, then choose
    # starting_equity via a single mode resolver.
    initial_equity_report: float | None = None
    current_equity_report: float | None = None
    for _sym, report in reports_map.items():
        if not isinstance(report, dict):
            continue
        src = report.get("aggregate", report)
        if initial_equity_report is None:
            _init = src.get("initial_equity")
            if _init is None:
                _init = report.get("initial_equity")
            try:
                _init_f = float(_init)
            except (TypeError, ValueError):
                _init_f = None
            if _init_f is not None and _init_f > 0:
                initial_equity_report = _init_f
        if current_equity_report is None:
            _eq = src.get("equity_end")
            if _eq is None:
                _eq = report.get("equity_end")
            try:
                _eq_f = float(_eq)
            except (TypeError, ValueError):
                _eq_f = None
            if _eq_f is not None and _eq_f > 0:
                current_equity_report = _eq_f
    configured_initial = float(config.risk.equity)
    fallback_current = configured_initial + float(sum(pnls))
    starting_equity = _resolve_monte_carlo_starting_equity(
        mode=mc.equity_mode_backtest,
        configured_initial_equity=configured_initial,
        initial_equity=initial_equity_report,
        current_equity=current_equity_report,
        fallback_current_equity=fallback_current,
    )

    png_path = _resolve_runtime_path(root, mc.live_window.png_path)
    json_path = _resolve_runtime_path(root, mc.live_window.json_path)

    try:
        run_monte_carlo_simulation(
            trade_pnls=pnls,
            starting_equity=starting_equity,
            png_path=png_path,
            json_path=json_path,
            num_simulations=mc.num_simulations,
            ruin_dd_threshold=mc.ruin_dd_threshold,
            seed=mc.seed,
            max_paths_plotted=mc.max_paths_plotted,
            sampling_mode=mc.sampling_mode,
            block_size=mc.block_size,
            equity_mode=mc.equity_mode_backtest,
            ruin_equity_floor_pct=mc.ruin_equity_floor_pct,
            ruin_equity_floor_abs=mc.ruin_equity_floor_abs,
            count_breakeven_as_loss=mc.count_breakeven_as_loss,
        )
    except Exception:
        LOGGER.exception("Monte Carlo simulation failed")


def _run_monte_carlo_for_trades(
    config: AppConfig,
    root: Path,
    trade_log: list[object],
) -> None:
    """Run Monte Carlo directly from a list of BacktestTrade objects."""
    mc = config.monte_carlo
    if not mc.enabled:
        return

    pnls: list[float] = []
    for trade in trade_log:
        pnl_val = getattr(trade, "pnl", None)
        if pnl_val is not None:
            pnls.append(float(pnl_val))

    if not pnls:
        LOGGER.info("Monte Carlo: no trades found — skipping simulation.")
        return

    configured_initial = float(config.risk.equity)
    current_equity = configured_initial + float(sum(pnls))
    starting_equity = _resolve_monte_carlo_starting_equity(
        mode=mc.equity_mode_backtest,
        configured_initial_equity=configured_initial,
        initial_equity=configured_initial,
        current_equity=current_equity,
        fallback_current_equity=current_equity,
    )

    png_path = _resolve_runtime_path(root, mc.live_window.png_path)
    json_path = _resolve_runtime_path(root, mc.live_window.json_path)

    try:
        run_monte_carlo_simulation(
            trade_pnls=pnls,
            starting_equity=starting_equity,
            png_path=png_path,
            json_path=json_path,
            num_simulations=mc.num_simulations,
            ruin_dd_threshold=mc.ruin_dd_threshold,
            seed=mc.seed,
            max_paths_plotted=mc.max_paths_plotted,
            sampling_mode=mc.sampling_mode,
            block_size=mc.block_size,
            equity_mode=mc.equity_mode_backtest,
            ruin_equity_floor_pct=mc.ruin_equity_floor_pct,
            ruin_equity_floor_abs=mc.ruin_equity_floor_abs,
            count_breakeven_as_loss=mc.count_breakeven_as_loss,
        )
    except Exception:
        LOGGER.exception("Monte Carlo simulation failed")


def _run_monte_carlo_for_batch(
    config: AppConfig,
    root: Path,
    out_root: Path,
    summary: dict[str, object],
) -> None:
    """Run Monte Carlo on the combined batch-backtest trades (from parquet)."""
    mc = config.monte_carlo
    if not mc.enabled:
        return

    parquet_path = out_root / "all" / "combined_trades.parquet"
    if not parquet_path.exists():
        LOGGER.info("Monte Carlo: no combined_trades.parquet — skipping.")
        return

    try:
        import pyarrow.parquet as pq
        table = pq.read_table(str(parquet_path), columns=["pnl"])
        pnls = [float(v) for v in table.column("pnl").to_pylist() if v is not None]
    except Exception:
        LOGGER.exception("Monte Carlo: failed to read batch trade PnLs from parquet")
        return

    if not pnls:
        LOGGER.info("Monte Carlo: no trades in batch results — skipping.")
        return

    combined_metrics = summary.get("combined_metrics", {})
    if not isinstance(combined_metrics, dict):
        combined_metrics = {}
    try:
        initial_equity = float(combined_metrics.get("initial_equity", config.risk.equity))
    except (TypeError, ValueError):
        initial_equity = float(config.risk.equity)
    try:
        current_equity = float(combined_metrics.get("equity_end"))
    except (TypeError, ValueError):
        current_equity = None
    fallback_current = initial_equity + float(sum(pnls))
    starting_equity = _resolve_monte_carlo_starting_equity(
        mode=mc.equity_mode_backtest,
        configured_initial_equity=float(config.risk.equity),
        initial_equity=initial_equity,
        current_equity=current_equity,
        fallback_current_equity=fallback_current,
    )

    png_path = _resolve_runtime_path(root, mc.live_window.png_path)
    json_path = _resolve_runtime_path(root, mc.live_window.json_path)

    try:
        run_monte_carlo_simulation(
            trade_pnls=pnls,
            starting_equity=starting_equity,
            png_path=png_path,
            json_path=json_path,
            num_simulations=mc.num_simulations,
            ruin_dd_threshold=mc.ruin_dd_threshold,
            seed=mc.seed,
            max_paths_plotted=mc.max_paths_plotted,
            sampling_mode=mc.sampling_mode,
            block_size=mc.block_size,
            equity_mode=mc.equity_mode_backtest,
            ruin_equity_floor_pct=mc.ruin_equity_floor_pct,
            ruin_equity_floor_abs=mc.ruin_equity_floor_abs,
            count_breakeven_as_loss=mc.count_breakeven_as_loss,
        )
    except Exception:
        LOGGER.exception("Monte Carlo simulation failed")


def run_backtest_mode(args: argparse.Namespace, config: AppConfig, assets: list[AssetConfig], root: Path) -> None:
    # Start the live dashboard BEFORE the backtest so we can watch in real-time
    _maybe_start_dashboard(args, config)

    # Start Monte Carlo live viewer (non-blocking subprocess)
    _maybe_start_mc_viewer(config, root, cli_override=getattr(args, "mc_viewer", None))

    report_formats = _parse_report_formats(args.report_formats)
    reporter = BacktestReporter(_resolve_runtime_path(root, str(args.report_dir))) if args.report else None
    generated_report_dirs: list[Path] = []
    selected_gate_modes = (
        ["off", "trend", "trend_vol_news"]
        if args.daily_gate_ab
        else [_daily_gate_mode(config)]
    )

    def _emit_detailed_report(
        *,
        symbol: str,
        timeframe: str,
        start_raw: str,
        end_raw: str,
        variant_code: str,
        mode: str,
        trades: list[object],
        payload: dict[str, object],
        data_root_value: str,
    ) -> None:
        if reporter is None:
            return
        meta = BacktestMeta(
            symbol=symbol,
            timeframe=timeframe,
            start=start_raw,
            end=end_raw,
            variant=variant_code,
            mode=mode,
            price=str(args.backtest_price),
            initial_equity=float(config.risk.equity),
            config=str(args.config),
            data_root=data_root_value,
        )
        run = BacktestRun(
            meta=meta,
            trades=trades,
            equity=[],
            extra={
                "source_report": payload.get("aggregate", payload) if isinstance(payload, dict) else payload,
                "source_report_full": payload,
            },
        )
        reporter.generate(run=run, formats=report_formats)
        if reporter.last_output_dir is not None:
            generated_report_dirs.append(reporter.last_output_dir)
            LOGGER.info("Backtest detailed report saved: %s", reporter.last_output_dir)

    def _maybe_open_reports() -> None:
        if not args.report_open or "html" not in report_formats:
            return
        opened = 0
        seen: set[str] = set()
        for report_dir in generated_report_dirs:
            html_path = report_dir / "report.html"
            key = str(html_path.resolve())
            if key in seen or not html_path.exists():
                continue
            seen.add(key)
            _open_report_html(html_path)
            opened += 1
        if opened == 0:
            LOGGER.warning("No report.html generated to open.")

    # Legacy CSV path mode (kept for backward compatibility).
    if args.backtest_data:
        epic = (args.backtest_epic or os.getenv("CAPITAL_EPIC") or assets[0].epic).strip().upper()
        selected = next((a for a in assets if a.epic == epic), assets[0])
        csv_path = _resolve_runtime_path(root, str(args.backtest_data))
        gate_payloads: dict[str, dict[str, object]] = {}
        for gate_mode in selected_gate_modes:
            daily_gate = _build_daily_gate_provider(config=config, mode=gate_mode, events=[])
            if args.walk_forward:
                report = run_walk_forward_from_csv(
                    config=config,
                    asset=selected,
                    csv_path=csv_path,
                    wf_splits=args.wf_splits,
                    assumed_spread=args.backtest_spread,
                    slippage_points=args.backtest_slippage_points,
                    slippage_atr_multiplier=args.backtest_slippage_atr_multiplier,
                    daily_gate=daily_gate,
                )
                report_dict = report.to_dict()
                aggregate_payload = report_dict.get("aggregate")
                if isinstance(aggregate_payload, dict):
                    report_dict["aggregate"] = _augment_report_with_research_fields(
                        aggregate_payload,
                        config=config,
                        oos_pass=None,
                    )
                LOGGER.info(
                    "Walk-forward report (daily_gate=%s): %s",
                    gate_mode,
                    json.dumps(report_dict, indent=2, ensure_ascii=True) if args.report else "summary-only (--no-report)",
                )
                aggregate_report = report.aggregate
                default_start = str(args.backtest_start or "csv-start")
                default_end = str(args.backtest_end or "csv-end")
                start_label, end_label = _trade_time_bounds(
                    list(aggregate_report.trade_log),
                    default_start=default_start,
                    default_end=default_end,
                )
                _emit_detailed_report(
                    symbol=selected.epic,
                    timeframe=normalize_timeframe(args.backtest_tf),
                    start_raw=start_label,
                    end_raw=end_label,
                    variant_code=f"{_first_variant_code(args.backtest_variants)}_{gate_mode}",
                    mode="walk-forward",
                    trades=list(aggregate_report.trade_log),
                    payload=report_dict,
                    data_root_value=str(csv_path),
                )
                _wf_agg = report_dict.get("aggregate", {})
                _wf_agg["_trade_pnls"] = [float(t.pnl) for t in aggregate_report.trade_log]
                gate_payloads[gate_mode] = {
                    "variant": _first_variant_code(args.backtest_variants),
                    "mode": "walk-forward",
                    "symbols": [selected.epic],
                    "reports": {selected.epic: _wf_agg},
                }
            else:
                report = run_backtest_from_csv(
                    config=config,
                    asset=selected,
                    csv_path=csv_path,
                    assumed_spread=args.backtest_spread,
                    slippage_points=args.backtest_slippage_points,
                    slippage_atr_multiplier=args.backtest_slippage_atr_multiplier,
                    daily_gate=daily_gate,
                )
                report_dict = _augment_report_with_research_fields(
                    report.to_dict(),
                    config=config,
                    oos_pass=None,
                )
                LOGGER.info(
                    "Backtest report (daily_gate=%s): %s",
                    gate_mode,
                    json.dumps(report_dict, indent=2, ensure_ascii=True) if args.report else "summary-only (--no-report)",
                )
                default_start = str(args.backtest_start or "csv-start")
                default_end = str(args.backtest_end or "csv-end")
                start_label, end_label = _trade_time_bounds(
                    list(report.trade_log),
                    default_start=default_start,
                    default_end=default_end,
                )
                _emit_detailed_report(
                    symbol=selected.epic,
                    timeframe=normalize_timeframe(args.backtest_tf),
                    start_raw=start_label,
                    end_raw=end_label,
                    variant_code=f"{_first_variant_code(args.backtest_variants)}_{gate_mode}",
                    mode="backtest",
                    trades=list(report.trade_log),
                    payload=report_dict,
                    data_root_value=str(csv_path),
                )
                report_dict["_trade_pnls"] = [float(t.pnl) for t in report.trade_log]
                gate_payloads[gate_mode] = {
                    "variant": _first_variant_code(args.backtest_variants),
                    "mode": "backtest",
                    "symbols": [selected.epic],
                    "reports": {selected.epic: report_dict},
                }
        if len(gate_payloads) > 1:
            _log_daily_gate_comparison(gate_payloads=gate_payloads, symbols=[selected.epic])
        _run_monte_carlo_for_payloads(config, root, gate_payloads)
        _maybe_open_reports()
        _maybe_block_dashboard(hold_open=bool(getattr(args, "hold_viewers", False)))
        return

    # New automatic parquet data mode (without --backtest-data).
    if not args.backtest_start or not args.backtest_end:
        raise RuntimeError("--backtest-start and --backtest-end are required when --backtest-data is not provided")

    timeframe = normalize_timeframe(args.backtest_tf)
    start = _parse_backtest_datetime(args.backtest_start, end_value=False)
    end = _parse_backtest_datetime(args.backtest_end, end_value=True)
    if start >= end:
        raise RuntimeError("--backtest-start must be before --backtest-end")

    symbols = _backtest_symbols(args, assets)
    source_priority = [item.strip() for item in str(args.backtest_source_priority).split(",") if item.strip()]
    data_root = _resolve_runtime_path(root, str(args.backtest_data_root))
    fetch_script = _resolve_runtime_path(root, str(args.backtest_fetch_script))
    reports_dir = _resolve_runtime_path(root, str(args.backtest_reports_dir))
    reports_dir.mkdir(parents=True, exist_ok=True)
    loader = AutoDataLoader(data_root=data_root, source_priority=source_priority)
    asset_map = _asset_map_for_symbols(symbols, assets, config)
    variants = _parse_backtest_variants(args.backtest_variants)
    if args.walk_forward and len(variants) > 1:
        raise RuntimeError("Walk-forward supports one variant at a time. Use --backtest-variants W0 or W3.")
    all_modes = list(dict.fromkeys(selected_gate_modes))
    if args.daily_gate_grid_search:
        all_modes = ["off", "trend", "trend_vol_news"]

    symbol_run_cache: dict[str, dict[str, object]] = {}

    def _prepare_symbol_run_data(symbol: str) -> dict[str, object]:
        cached = symbol_run_cache.get(symbol)
        if cached is not None:
            return cached
        loaded = loader.load_symbol_data(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            price_mode=args.backtest_price,
        )
        data_health = loaded.diagnostics.get("data_health", {})
        LOGGER.info(
            "Backtest data health | symbol=%s tf=%s bars=%s min_ts=%s max_ts=%s dups=%s gaps=%s",
            symbol,
            timeframe,
            data_health.get("bars"),
            data_health.get("min_ts_utc"),
            data_health.get("max_ts_utc"),
            data_health.get("duplicate_timestamps"),
            data_health.get("gap_count_over_1bar"),
        )
        frame = loaded.frame
        candles = BacktestRunner._frame_to_candles(frame)
        spread_series = frame.get("spread")
        spread_from_data = (
            float(spread_series.dropna().median())
            if spread_series is not None and hasattr(spread_series, "dropna") and not spread_series.dropna().empty
            else None
        )
        symbol_spread_map = config.backtest_tuning.assumed_spread_by_symbol
        symbol_spread_default = symbol_spread_map.get(symbol.upper())
        if symbol_spread_default is None:
            symbol_spread_default = symbol_spread_map.get(asset_map[symbol].epic.upper())
        if spread_from_data is not None:
            assumed_spread = spread_from_data
        elif symbol_spread_default is not None:
            assumed_spread = float(symbol_spread_default)
        else:
            assumed_spread = float(args.backtest_spread)

        nan_counts = data_health.get("nan_counts", {}) if isinstance(data_health, dict) else {}
        bars_count = int(data_health.get("bars", 0)) if isinstance(data_health, dict) and data_health.get("bars") is not None else 0
        close_bid_nan = int(nan_counts.get("close_bid", bars_count)) if isinstance(nan_counts, dict) else bars_count
        close_ask_nan = int(nan_counts.get("close_ask", bars_count)) if isinstance(nan_counts, dict) else bars_count
        spread_mode = "ASSUMED_OHLC" if bars_count > 0 and close_bid_nan >= bars_count and close_ask_nan >= bars_count else "REAL_BIDASK"

        prepared = {
            "loaded": loaded,
            "frame": frame,
            "candles": candles,
            "assumed_spread": float(assumed_spread),
            "spread_mode": spread_mode,
        }
        symbol_run_cache[symbol] = prepared
        return prepared

    needs_news = any(mode == "trend_vol_news" for mode in all_modes)
    gate_events: list[Event] = []
    if needs_news:
        news_provider = build_news_provider(config, root)
        gate_events = news_provider.get_high_impact_events(
            start - timedelta(minutes=max(config.daily_gate.pre_minutes, 1)),
            end + timedelta(minutes=max(config.daily_gate.post_minutes, 1)),
        )

    def _summarize_payloads(payloads: dict[str, dict[str, object]]) -> dict[str, float]:
        total_pnl = 0.0
        max_drawdown = 0.0
        expectancy_values: list[float] = []
        for payload in payloads.values():
            reports_raw = payload.get("reports")
            if not isinstance(reports_raw, dict):
                continue
            for report in reports_raw.values():
                if not isinstance(report, dict):
                    continue
                source = report
                aggregate = report.get("aggregate")
                if isinstance(aggregate, dict):
                    source = aggregate
                total_pnl += float(source.get("total_pnl", 0.0))
                max_drawdown = max(max_drawdown, float(source.get("max_drawdown", 0.0)))
                expectancy_values.append(float(source.get("expectancy", 0.0)))
        expectancy = (sum(expectancy_values) / len(expectancy_values)) if expectancy_values else 0.0
        return {
            "total_pnl": total_pnl,
            "max_drawdown": max_drawdown,
            "expectancy": expectancy,
        }

    def _run_auto_for_gate(
        *,
        gate_mode: str,
        gate_overrides: dict[str, float | int | None] | None = None,
        emit_reports: bool = True,
    ) -> dict[str, dict[str, object]]:
        payloads: dict[str, dict[str, object]] = {}
        for variant in variants:
            reports: dict[str, dict[str, object]] = {}
            for symbol in symbols:
                prepared = _prepare_symbol_run_data(symbol)
                loaded = prepared["loaded"]
                frame = prepared["frame"]
                candles_for_gate = prepared["candles"]
                daily_gate = _build_daily_gate_provider(
                    config=config,
                    mode=gate_mode,
                    candles=candles_for_gate,
                    events=gate_events,
                    overrides=gate_overrides,
                )
                assumed_spread = float(prepared["assumed_spread"])
                spread_mode = str(prepared["spread_mode"])

                debug_file = (reports_dir / f"{variant.code}_{gate_mode}_debug_exec_{symbol}.jsonl") if emit_reports else None
                no_price_debug_file = (reports_dir / f"{variant.code}_{gate_mode}_debug_no_price_{symbol}.jsonl") if emit_reports else None
                reaction_timeout_debug_file = (
                    reports_dir / f"{variant.code}_{gate_mode}_debug_reaction_timeout_{symbol}.jsonl"
                ) if emit_reports else None
                data_context = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "price_mode_requested": args.backtest_price,
                    "spread_mode": spread_mode,
                    "assumed_spread_used": float(assumed_spread),
                    "daily_gate_mode": gate_mode,
                    **(loaded.diagnostics if isinstance(getattr(loaded, "diagnostics", None), dict) else {}),
                }
                if args.walk_forward:
                    report = run_walk_forward_multi_strategy(
                        config=config,
                        asset=asset_map[symbol],
                        candles_m5=candles_for_gate,
                        wf_splits=args.wf_splits,
                        assumed_spread=assumed_spread,
                        slippage_points=args.backtest_slippage_points,
                        slippage_atr_multiplier=args.backtest_slippage_atr_multiplier,
                        variant=variant,
                        execution_debug_path=debug_file,
                        no_price_debug_path=no_price_debug_file,
                        reaction_timeout_debug_path=reaction_timeout_debug_file,
                        data_context=data_context,
                        daily_gate=daily_gate,
                    )
                    report_dict = report.to_dict()
                    aggregate_payload = report_dict.get("aggregate")
                    if isinstance(aggregate_payload, dict):
                        report_dict["aggregate"] = _augment_report_with_research_fields(
                            aggregate_payload,
                            config=config,
                            oos_pass=None,
                        )
                    report_trades = list(report.aggregate.trade_log)
                else:
                    # Create MC adaptive model if enabled — shares state
                    # across segments so risk scaling accumulates correctly.
                    _mc_model: MCAdaptiveModel | None = None
                    if config.monte_carlo.enabled and config.monte_carlo.adaptive.enabled:
                        _mc_png = _resolve_runtime_path(root, config.monte_carlo.live_window.png_path)
                        _mc_json = _resolve_runtime_path(root, config.monte_carlo.live_window.json_path)
                        _mc_model = MCAdaptiveModel.from_config(
                            config.monte_carlo, png_path=_mc_png, json_path=_mc_json,
                        )
                    report, segment_meta = _run_multi_strategy_segmented(
                        config=config,
                        asset=asset_map[symbol],
                        frame=frame,
                        loader=loader,
                        timeframe=timeframe,
                        assumed_spread=assumed_spread,
                        slippage_points=args.backtest_slippage_points,
                        slippage_atr_multiplier=args.backtest_slippage_atr_multiplier,
                        variant=variant,
                        execution_debug_path=debug_file,
                        no_price_debug_path=no_price_debug_file,
                        reaction_timeout_debug_path=reaction_timeout_debug_file,
                        data_context=data_context,
                        daily_gate=daily_gate,
                        daily_gate_prepared=True,
                        mc_model=_mc_model,
                    )
                    report_dict = _augment_report_with_research_fields(
                        report.to_dict(),
                        config=config,
                        oos_pass=None,
                    )
                    report_dict["segment_health"] = segment_meta
                    report_trades = list(report.trade_log)
                report_payload: dict[str, object] = report_dict
                aggregate_payload = report_dict.get("aggregate")
                if args.walk_forward and isinstance(aggregate_payload, dict):
                    report_payload = aggregate_payload

                report_payload["data_health"] = loaded.diagnostics.get("data_health", {})
                report_payload["price_diagnostics"] = {
                    "price_mode_requested": loaded.diagnostics.get("price_mode_requested", args.backtest_price),
                    "source_datasets": loaded.diagnostics.get("source_datasets", []),
                    "source_files_count": len(loaded.diagnostics.get("source_files", [])),
                    "fallback_counters": loaded.diagnostics.get("fallback_counters", {}),
                    "gap_segments": loaded.diagnostics.get("gap_segments", {}),
                    "spread_mode": spread_mode,
                    "assumed_spread_used": float(report_payload.get("assumed_spread_used", assumed_spread)),
                }
                report_payload["daily_gate_mode"] = gate_mode
                if gate_overrides:
                    report_payload["daily_gate_params"] = gate_overrides
                reports[symbol] = report_payload
                if emit_reports:
                    _emit_detailed_report(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_raw=str(args.backtest_start),
                        end_raw=str(args.backtest_end),
                        variant_code=f"{variant.code}_{gate_mode}",
                        mode="walk-forward" if args.walk_forward else "backtest",
                        trades=report_trades,
                        payload=report_dict,
                        data_root_value=str(data_root),
                    )

                    filename = _variant_report_filename(
                        variant=variant,
                        start_raw=str(args.backtest_start),
                        end_raw=str(args.backtest_end),
                        start_dt=start,
                        end_dt=end,
                        symbol=symbol,
                    )
                    suffix = f"_{gate_mode}"
                    target = reports_dir / filename.replace(".json", f"{suffix}.json")
                    # Strip internal _trade_pnls before writing to disk
                    _disk_dict = {k: v for k, v in report_dict.items() if not k.startswith("_")}
                    target.write_text(json.dumps(_disk_dict, indent=2, ensure_ascii=True), encoding="utf-8")

                # Stash raw trade PnLs AFTER file-write so they don't leak
                # to JSON report files.  MC helper reads them from the dict.
                trade_pnls = [float(t.pnl) for t in report_trades]
                report_payload["_trade_pnls"] = trade_pnls
                if report_payload is not report_dict:
                    report_dict["_trade_pnls"] = trade_pnls

            payloads[variant.code] = {
                "variant": variant.code,
                "mode": "walk-forward" if args.walk_forward else "backtest",
                "daily_gate_mode": gate_mode,
                "daily_gate_params": gate_overrides or {},
                "symbols": symbols,
                "timeframe": timeframe,
                "price": args.backtest_price,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "data_root": str(data_root),
                "reports": reports,
            }
        return payloads

    def _run_grid_for_mode(mode: str) -> tuple[dict[str, float | int | None], list[dict[str, object]]]:
        ranked: list[dict[str, object]] = []
        grid_space = _gate_param_space_for_grid(config)
        if int(args.daily_gate_grid_limit) > 0:
            grid_space = grid_space[: int(args.daily_gate_grid_limit)]
        for params in grid_space:
            payloads = _run_auto_for_gate(gate_mode=mode, gate_overrides=params, emit_reports=False)
            summary = _summarize_payloads(payloads)
            ranked.append(
                {
                    "mode": mode,
                    "params": dict(params),
                    "total_pnl": float(summary["total_pnl"]),
                    "max_drawdown": float(summary["max_drawdown"]),
                    "expectancy": float(summary["expectancy"]),
                }
            )
        ranked.sort(key=lambda item: (-float(item["total_pnl"]), float(item["max_drawdown"]), -float(item["expectancy"])))
        best = ranked[0] if ranked else {"params": {}}
        best_params = dict(best.get("params", {}))
        LOGGER.info("Daily gate grid top5 (%s): %s", mode, json.dumps(ranked[:5], ensure_ascii=True))
        return best_params, ranked[:5]

    attempt = 0
    while True:
        try:
            mode_overrides: dict[str, dict[str, float | int | None]] = {}
            grid_report: dict[str, object] = {}
            run_modes = list(all_modes)
            if args.daily_gate_grid_search:
                run_modes = ["off", "trend", "trend_vol_news"]
                for mode in ["trend", "trend_vol_news"]:
                    best_params, top5 = _run_grid_for_mode(mode)
                    mode_overrides[mode] = best_params
                    grid_report[mode] = {"best_params": best_params, "top5": top5}
                grid_path = reports_dir / "daily_gate_grid_search.json"
                grid_path.write_text(json.dumps(grid_report, indent=2, ensure_ascii=True), encoding="utf-8")
                LOGGER.info("Daily gate grid-search report saved: %s", grid_path)

            gate_payloads: dict[str, dict[str, object]] = {}
            for gate_mode in run_modes:
                variant_payloads = _run_auto_for_gate(
                    gate_mode=gate_mode,
                    gate_overrides=mode_overrides.get(gate_mode),
                    emit_reports=True,
                )
                merged_reports: dict[str, dict[str, object]] = {}
                for payload in variant_payloads.values():
                    reports_raw = payload.get("reports")
                    if isinstance(reports_raw, dict):
                        for symbol, report in reports_raw.items():
                            if isinstance(report, dict):
                                merged_reports[str(symbol)] = report
                gate_payloads[gate_mode] = {
                    "variant": ",".join(variant_payloads.keys()),
                    "mode": "walk-forward" if args.walk_forward else "backtest",
                    "daily_gate_mode": gate_mode,
                    "symbols": symbols,
                    "reports": merged_reports,
                }
                if len(variant_payloads) == 1:
                    payload = next(iter(variant_payloads.values()))
                    if args.report:
                        LOGGER.info(
                            "Backtest auto-data report (daily_gate=%s): %s",
                            gate_mode,
                            json.dumps(payload, indent=2, ensure_ascii=True),
                        )
                    else:
                        LOGGER.info(
                            "Backtest auto-data summary (daily_gate=%s): variant=%s symbols=%s",
                            gate_mode,
                            payload.get("variant"),
                            ",".join(str(item) for item in payload.get("symbols", [])),
                        )
                        _log_variant_comparison(variant_payloads=variant_payloads, symbols=symbols)
                else:
                    LOGGER.info(
                        "Backtest auto-data variants (daily_gate=%s): %s",
                        gate_mode,
                        ",".join(variant_payloads.keys()),
                    )
                    _log_variant_comparison(variant_payloads=variant_payloads, symbols=symbols)

            if len(gate_payloads) > 1:
                _log_daily_gate_comparison(gate_payloads=gate_payloads, symbols=symbols)
                comparison_path = reports_dir / "daily_gate_comparison.json"
                comparison_path.write_text(json.dumps(gate_payloads, indent=2, ensure_ascii=True), encoding="utf-8")
                LOGGER.info("Daily gate comparison report saved: %s", comparison_path)
            _run_monte_carlo_for_payloads(config, root, gate_payloads)
            _maybe_open_reports()
            _maybe_block_dashboard(hold_open=bool(getattr(args, "hold_viewers", False)))
            return
        except MissingDataError as exc:
            for item in exc.missing:
                LOGGER.error("Missing data: %s", item.to_line())
            if args.backtest_autofetch and attempt == 0:
                attempt += 1
                _autofetch_backtest_data(
                    fetch_script=fetch_script,
                    symbols=symbols,
                    timeframe=timeframe,
                    start_raw=args.backtest_start,
                    end_raw=args.backtest_end,
                )
                continue
            raise RuntimeError("Backtest data missing. Use --backtest-autofetch or provide --backtest-data CSV.") from exc


def _batch_trade_rows(
    *,
    report,
    symbol: str,
    timeframe: str,
    price_mode: str,
    chunk_id: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trade in report.trade_log:
        open_time = trade.entry_time.astimezone(timezone.utc).isoformat()
        close_time = trade.exit_time.astimezone(timezone.utc).isoformat()
        side = str(trade.side).lower()
        entry_price = float(trade.entry_price)
        trade_id = make_trade_id(
            open_time_utc=open_time,
            side=side,
            entry_price=entry_price,
            chunk_id=chunk_id,
        )
        rows.append(
            {
                "symbol": symbol.upper(),
                "timeframe": normalize_timeframe(timeframe),
                "price_mode": price_mode.upper(),
                "open_time_utc": open_time,
                "close_time_utc": close_time,
                "side": side,
                "entry_price": entry_price,
                "exit_price": float(trade.exit_price),
                "qty": float(trade.size),
                "size": float(trade.size),
                "pnl": float(trade.pnl),
                "fees": float(getattr(trade, "fees", 0.0) or 0.0),
                "spread_cost": float(getattr(trade, "spread_cost", 0.0) or 0.0),
                "slippage_cost": float(getattr(trade, "slippage_cost", 0.0) or 0.0),
                "commission_cost": float(getattr(trade, "commission_cost", 0.0) or 0.0),
                "swap_cost": float(getattr(trade, "swap_cost", 0.0) or 0.0),
                "fx_cost": float(getattr(trade, "fx_cost", 0.0) or 0.0),
                "r_multiple": float(trade.r_multiple) if trade.r_multiple is not None else None,
                "score": float(trade.score) if getattr(trade, "score", None) is not None else None,
                "forced_exit": bool(getattr(trade, "forced_exit", False)),
                "reason_open": str(getattr(trade, "reason_open", "SIGNAL") or "SIGNAL"),
                "reason_close": str(getattr(trade, "reason_close", "") or getattr(trade, "reason", "")),
                "gate_bias": str(getattr(trade, "gate_bias", "") or ""),
                "trade_id": trade_id,
            }
        )
    return rows


def run_batch_worker_mode(args: argparse.Namespace, config: AppConfig, assets: list[AssetConfig], root: Path) -> None:
    if not args.symbol:
        raise RuntimeError("--batch-worker requires --symbol")
    if not args.start or not args.end:
        raise RuntimeError("--batch-worker requires --start and --end")
    if not args.out_dir:
        raise RuntimeError("--batch-worker requires --out-dir")

    symbol = str(args.symbol).strip().upper()
    timeframe = normalize_timeframe(str(args.timeframe))
    price_mode = str(args.price_mode).strip().upper()
    if price_mode not in {"MID", "BID", "ASK"}:
        raise RuntimeError("--price-mode must be MID, BID, or ASK")
    start = _parse_backtest_datetime(str(args.start), end_value=False)
    end = _parse_backtest_datetime(str(args.end), end_value=False)
    if start >= end:
        raise RuntimeError("--end must be greater than --start")

    out_dir = _resolve_runtime_path(root, str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_id = out_dir.name
    marker_success = out_dir / "SUCCESS.json"
    marker_error = out_dir / "ERROR.json"
    if marker_success.exists():
        marker_success.unlink()
    if marker_error.exists():
        marker_error.unlink()

    try:
        data_root_arg = str(args.data_root) if args.data_root else str(args.backtest_data_root)
        data_root = _resolve_runtime_path(root, data_root_arg)
        _validate_batch_data_root(data_root)
        loader = AutoDataLoader(data_root=data_root, source_priority=["local_csv"])
        loaded = loader.load_symbol_data_range(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            price_mode=price_mode.lower(),
            warmup_days=int(args.warmup_days),
            source="local_csv",
        )
        missing_shards = loaded.diagnostics.get("missing_shards", [])
        if isinstance(missing_shards, list):
            for item in missing_shards:
                LOGGER.warning("Batch shard gap: %s", item)
        frame = loaded.frame
        if frame.empty:
            raise RuntimeError(f"No candles loaded for chunk={chunk_id}")

        by_epic = {asset.epic.upper(): asset for asset in assets}
        template = assets[0] if assets else AssetConfig(**config.instrument.model_dump(), trade_enabled=True)
        asset = by_epic.get(symbol, _asset_from_template(symbol, template, True))

        spread_series = frame.get("spread")
        spread_from_data = (
            float(spread_series.dropna().median())
            if spread_series is not None and hasattr(spread_series, "dropna") and not spread_series.dropna().empty
            else None
        )
        symbol_spread_map = config.backtest_tuning.assumed_spread_by_symbol
        configured_spread = symbol_spread_map.get(symbol.upper()) or symbol_spread_map.get(asset.epic.upper())
        if spread_from_data is not None:
            assumed_spread = spread_from_data
            spread_mode = "REAL_BIDASK"
        else:
            assumed_spread = float(configured_spread if configured_spread is not None else args.backtest_spread)
            spread_mode = "ASSUMED_OHLC"

        # Create MC adaptive model if enabled
        _mc_model: MCAdaptiveModel | None = None
        if config.monte_carlo.enabled and config.monte_carlo.adaptive.enabled:
            _mc_png = _resolve_runtime_path(root, config.monte_carlo.live_window.png_path)
            _mc_json = _resolve_runtime_path(root, config.monte_carlo.live_window.json_path)
            _mc_model = MCAdaptiveModel.from_config(
                config.monte_carlo, png_path=_mc_png, json_path=_mc_json,
            )

        report, segment_meta = _run_multi_strategy_segmented(
            config=config,
            asset=asset,
            frame=frame,
            loader=loader,
            timeframe=timeframe,
            assumed_spread=assumed_spread,
            slippage_points=args.backtest_slippage_points,
            slippage_atr_multiplier=args.backtest_slippage_atr_multiplier,
            variant=BacktestVariant(code="BATCH-W0"),
            execution_debug_path=out_dir / "debug_exec.jsonl",
            no_price_debug_path=out_dir / "debug_no_price.jsonl",
            reaction_timeout_debug_path=out_dir / "debug_reaction_timeout.jsonl",
            trade_start_utc=start,
            flatten_at_chunk_end=True,
            mc_model=_mc_model,
            data_context={
                "symbol": symbol,
                "timeframe": timeframe,
                "price_mode_requested": price_mode.lower(),
                "spread_mode": spread_mode,
                "assumed_spread_used": assumed_spread,
                **(loaded.diagnostics if isinstance(loaded.diagnostics, dict) else {}),
            },
        )

        rows = _batch_trade_rows(
            report=report,
            symbol=symbol,
            timeframe=timeframe,
            price_mode=price_mode,
            chunk_id=chunk_id,
        )
        columns = [
            "symbol",
            "timeframe",
            "price_mode",
            "open_time_utc",
            "close_time_utc",
            "side",
            "entry_price",
            "exit_price",
            "qty",
            "size",
            "pnl",
            "fees",
            "spread_cost",
            "slippage_cost",
            "commission_cost",
            "swap_cost",
            "fx_cost",
            "r_multiple",
            "score",
            "forced_exit",
            "reason_open",
            "reason_close",
            "trade_id",
        ]
        import pandas as pd

        trades_df = pd.DataFrame(rows, columns=columns)
        trades_df.to_parquet(out_dir / "trades.parquet", index=False, engine="pyarrow")

        report_dict = report.to_dict()
        report_dict = _augment_report_with_research_fields(
            report_dict,
            config=config,
            oos_pass=None,
        )
        report_dict["data_health"] = loaded.diagnostics.get("data_health", {})
        report_dict["segment_health"] = segment_meta
        report_dict["price_diagnostics"] = {
            "price_mode_requested": loaded.diagnostics.get("price_mode_requested", price_mode.lower()),
            "source_datasets": loaded.diagnostics.get("source_datasets", ["local_csv"]),
            "source_files_count": len(loaded.diagnostics.get("source_files", [])),
            "fallback_counters": loaded.diagnostics.get("fallback_counters", {}),
            "missing_shards": loaded.diagnostics.get("missing_shards", []),
            "gap_segments": loaded.diagnostics.get("gap_segments", {}),
            "spread_mode": spread_mode,
            "assumed_spread_used": float(report_dict.get("assumed_spread_used", assumed_spread)),
        }
        (out_dir / "report.json").write_text(json.dumps(report_dict, indent=2, ensure_ascii=True), encoding="utf-8")

        metrics = {
            "chunk_id": chunk_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "price_mode": price_mode,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "warmup_days": int(args.warmup_days),
            "trades": report_dict.get("trades", 0),
            "wins": report_dict.get("wins", 0),
            "losses": report_dict.get("losses", 0),
            "win_rate": report_dict.get("win_rate", 0.0),
            "total_pnl": report_dict.get("total_pnl", 0.0),
            "expectancy": report_dict.get("expectancy", 0.0),
            "avg_r": report_dict.get("avg_r", 0.0),
            "max_drawdown": report_dict.get("max_drawdown", 0.0),
            "signal_candidates": report_dict.get("signal_candidates", 0),
            "avg_win": report_dict.get("avg_win", 0.0),
            "avg_loss": report_dict.get("avg_loss", 0.0),
            "payoff_ratio": report_dict.get("payoff_ratio", 0.0),
            "profit_factor": report_dict.get("profit_factor", 0.0),
            "spread_mode": spread_mode,
            "assumed_spread_used": float(report_dict.get("assumed_spread_used", assumed_spread)),
            "data_health": report_dict.get("data_health", {}),
            "gate_block_counts": report_dict.get("gate_block_counts", {}),
            "segment_health": segment_meta,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")
        marker_success.write_text(
            json.dumps(
                {
                    "chunk_id": chunk_id,
                    "symbol": symbol,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "status": "ok",
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        marker_error.write_text(
            json.dumps(
                {
                    "symbol": str(args.symbol),
                    "start": str(args.start),
                    "end": str(args.end),
                    "error": str(exc),
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        raise


def run_batch_backtest_mode(args: argparse.Namespace, config: AppConfig, root: Path) -> None:
    # Start Monte Carlo live viewer for batch backtest mode
    _maybe_start_mc_viewer(config, root, cli_override=getattr(args, "mc_viewer", None))

    if not args.symbol:
        raise RuntimeError("--batch-backtest requires --symbol")
    if not args.start or not args.end:
        raise RuntimeError("--batch-backtest requires --start and --end")
    symbol = str(args.symbol).strip().upper()
    timeframe = normalize_timeframe(str(args.timeframe))
    price_mode = str(args.price_mode).strip().upper()
    if price_mode not in {"MID", "BID", "ASK"}:
        raise RuntimeError("--price-mode must be MID, BID, or ASK")
    start = _parse_backtest_datetime(str(args.start), end_value=False)
    end = _parse_backtest_datetime(str(args.end), end_value=True)
    if start >= end:
        raise RuntimeError("--end must be after --start")

    out_root = _resolve_runtime_path(root, str(args.out_root))
    data_root_arg = str(args.data_root) if args.data_root else str(args.backtest_data_root)
    data_root = _resolve_runtime_path(root, data_root_arg)
    _validate_batch_data_root(data_root)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    summary = orchestrate_batch(
        main_script=root / "main.py",
        config_path=config_path,
        data_root=data_root,
        symbol=symbol,
        price_mode=price_mode,
        timeframe=timeframe,
        start=start,
        end=end,
        chunk=str(args.chunk),
        workers=max(1, min(3, int(args.workers))),
        warmup_days=max(0, int(args.warmup_days)),
        out_root=out_root,
        initial_equity=float(args.initial_equity if args.initial_equity is not None else config.risk.equity),
        continue_on_error=bool(args.continue_on_error),
    )
    LOGGER.info("Batch backtest summary: %s", json.dumps(summary, indent=2, ensure_ascii=True))

    # Run Monte Carlo on the combined batch trades
    _run_monte_carlo_for_batch(config, root, out_root, summary)

    _maybe_stop_mc_viewer()


def _timeframe_history(config: AppConfig) -> dict[str, int]:
    return {
        config.timeframes.h1: config.execution.history_bars.h1,
        config.timeframes.m15: config.execution.history_bars.m15,
        config.timeframes.m5: config.execution.history_bars.m5,
    }


def refresh_timeframe_cache(
    *,
    market_data: MarketDataService,
    state: AssetRuntimeState,
    now: datetime,
    timeframe: str,
    history_count: int,
    close_grace_seconds: int,
    retry_seconds: int,
) -> tuple[bool, datetime | None]:
    should_poll, target_closed_ts = should_poll_closed_candle(
        now_utc=now,
        timeframe=timeframe,
        last_processed_closed_ts=state.last_processed_closed_ts.get(timeframe),
        last_attempt_target_ts=state.last_poll_target_ts.get(timeframe),
        last_attempt_at=state.last_poll_attempt_at.get(timeframe),
        close_grace_seconds=close_grace_seconds,
        retry_seconds=retry_seconds,
    )
    state.last_poll_target_ts[timeframe] = target_closed_ts
    if not should_poll:
        return False, target_closed_ts

    state.last_poll_attempt_at[timeframe] = now
    full = market_data.fetch_candles(state.asset.epic, timeframe, max_points=history_count)
    if not full:
        return False, target_closed_ts
    is_new, closed_ts = is_new_closed_candle(
        full,
        state.last_processed_closed_ts.get(timeframe),
    )
    if not is_new:
        return False, closed_ts or target_closed_ts
    state.cache[timeframe] = full
    state.last_processed_closed_ts[timeframe] = closed_ts
    return True, closed_ts


def derive_entry_state(previous: str, *, has_open: bool, has_pending: bool) -> str:
    if has_open:
        return "FILLED"
    if has_pending:
        return "ORDER_PLACED"
    if previous == "ORDER_PLACED":
        return "EXPIRED"
    if previous == "EXPIRED":
        return "WAIT"
    return "WAIT"


def _bias_for_trace(h1: H1Snapshot | None) -> str:
    if h1 is None:
        return "NEUTRAL"
    if h1.side == "LONG":
        return "LONG"
    if h1.side == "SHORT":
        return "SHORT"
    return "NEUTRAL"


def _build_trace(
    *,
    state: AssetRuntimeState,
    now: datetime,
    h1_last_closed: datetime | None,
    h1_new_close: bool,
    m15_last_closed: datetime | None,
    m15_new_close: bool,
    m5_last_closed: datetime | None,
    m5_new_close: bool,
    strategy_name: str,
    evaluation: StrategyEvaluation | None,
    final_decision: str,
    reasons: list[str],
) -> DecisionTrace:
    trace = DecisionTrace(
        asset=state.asset.epic,
        created_at=now,
        strategy_name=strategy_name,
        score_total=evaluation.score_total if evaluation is not None else None,
        score_layers=dict(evaluation.score_layers) if evaluation is not None else {},
        score_breakdown=dict(evaluation.score_breakdown) if evaluation is not None else {},
        penalties=dict(evaluation.penalties) if evaluation is not None else {},
        gates=dict(evaluation.gates) if evaluation is not None else {},
        gate_blocked=evaluation.gate_blocked if evaluation is not None else None,
        reasons_blocking=list(evaluation.reasons_blocking) if evaluation is not None else [],
        would_enter_if=list(evaluation.would_enter_if) if evaluation is not None else [],
        snapshot=dict(evaluation.snapshot) if evaluation is not None else {},
        h1_last_closed_ts=h1_last_closed,
        h1_new_close=h1_new_close,
        m15_last_closed_ts=m15_last_closed,
        m15_new_close=m15_new_close,
        m5_last_closed_ts=m5_last_closed,
        m5_new_close=m5_new_close,
        final_decision=final_decision,
        reasons=map_reason_codes(reasons),
    )
    if state.h1_snapshot is not None:
        trace.h1.updated = h1_new_close
        trace.h1.bias_state = _bias_for_trace(state.h1_snapshot)
        trace.h1.safe_mode = state.h1_snapshot.safe_mode
        trace.h1.ema200_ready = state.h1_snapshot.ema200_ready
        trace.h1.ema200_value = state.h1_snapshot.ema200_value
        trace.h1.bos_state = state.h1_snapshot.bos_state
        trace.h1.bos_age = state.h1_snapshot.bos_age
        trace.h1.bars = state.h1_snapshot.bars
        trace.h1.required_bars = state.h1_snapshot.required_bars
        trace.h1.pd_state = state.h1_snapshot.pd_state
        trace.h1.close = state.h1_snapshot.last_close
        trace.h1.eq = state.h1_snapshot.eq
        trace.h1.dealing_low = state.h1_snapshot.dealing_low
        trace.h1.dealing_high = state.h1_snapshot.dealing_high
    if state.m15_snapshot is not None:
        trace.m15.updated = m15_new_close
        trace.m15.setup_state = state.m15_snapshot.setup_state
        trace.m15.sweep_dir = state.m15_snapshot.sweep_dir
        trace.m15.reject_ok = state.m15_snapshot.reject_ok
        trace.m15.sweep_level = state.m15_snapshot.sweep_level
        trace.m15.invalidation_level = state.m15_snapshot.invalidation_level
        trace.m15.setup_age_minutes = state.m15_snapshot.setup_age_minutes
    if state.m5_snapshot is not None:
        trace.m5.updated = m5_new_close
        trace.m5.mss_ok = state.m5_snapshot.mss_ok
        trace.m5.displacement_ok = state.m5_snapshot.displacement_ok
        trace.m5.fvg_ok = state.m5_snapshot.fvg_ok
        trace.m5.fvg_range = state.m5_snapshot.fvg_range
        trace.m5.fvg_mid = state.m5_snapshot.fvg_mid
        trace.m5.limit_price = state.m5_snapshot.limit_price
    trace.m5.entry_state = state.entry_state
    return trace


def _trace_signature(trace: DecisionTrace) -> str:
    payload = {
        "asset": trace.asset,
        "strategy": trace.strategy_name,
        "score_total": trace.score_total,
        "score_layers": trace.score_layers,
        "score_breakdown": trace.score_breakdown,
        "penalties": trace.penalties,
        "gates": trace.gates,
        "gate_blocked": trace.gate_blocked,
        "reasons_blocking": trace.reasons_blocking,
        "h1_new": trace.h1_new_close,
        "m15_new": trace.m15_new_close,
        "m5_new": trace.m5_new_close,
        "h1": {
            "bias": trace.h1.bias_state,
            "safe_mode": trace.h1.safe_mode,
            "ema_ready": trace.h1.ema200_ready,
            "bos": trace.h1.bos_state,
            "bos_age": trace.h1.bos_age,
            "bars": trace.h1.bars,
            "required": trace.h1.required_bars,
            "pd_state": trace.h1.pd_state,
        },
        "m15": {
            "setup": trace.m15.setup_state,
            "sweep": trace.m15.sweep_dir,
            "reject": trace.m15.reject_ok,
            "age": trace.m15.setup_age_minutes,
        },
        "m5": {
            "entry": trace.m5.entry_state,
            "mss": trace.m5.mss_ok,
            "disp": trace.m5.displacement_ok,
            "fvg": trace.m5.fvg_ok,
            "fvg_range": trace.m5.fvg_range,
            "fvg_mid": trace.m5.fvg_mid,
        },
        "final": trace.final_decision,
        "reasons": trace.reasons,
        "snapshot": trace.snapshot,
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def resolve_db_path(root: Path, *, paper_mode: bool) -> str:
    mode = "paper" if paper_mode else "dry"
    template = os.getenv("SQLITE_PATH_TEMPLATE")
    if template:
        db_path = template.replace("{mode}", mode)
    else:
        raw_path = os.getenv("SQLITE_PATH")
        if raw_path:
            raw_path = raw_path.strip()
            if "{mode}" in raw_path:
                db_path = raw_path.replace("{mode}", mode)
            else:
                base = Path(raw_path)
                suffix = base.suffix or ".db"
                db_path = str(base.with_name(f"{base.stem}_{mode}{suffix}"))
        else:
            db_path = "bot_state_paper.db" if paper_mode else "bot_state_dry.db"
    path = Path(db_path)
    if not path.is_absolute():
        path = root / path
    return str(path)


def should_refresh_quote(
    *,
    now: datetime,
    last_fetch_at: datetime | None,
    interval_seconds: int,
) -> bool:
    if last_fetch_at is None:
        return True
    return (now - last_fetch_at).total_seconds() >= max(1, interval_seconds)


def _quote_refresh_interval_seconds(
    *,
    config: AppConfig,
    trade_enabled: bool,
) -> int:
    default_value = (
        config.execution.quote_refresh_seconds_trade
        if trade_enabled
        else config.execution.quote_refresh_seconds_observe
    )
    env_name = "QUOTE_REFRESH_TRADE_SECONDS" if trade_enabled else "QUOTE_REFRESH_OBSERVE_SECONDS"
    return int(os.getenv(env_name, str(default_value)))


def _log_daily_summary(
    *,
    summary: DailyRuntimeSummary,
    client: CapitalClient | None,
) -> None:
    api_requests = 0
    api_retries = 0
    api_429 = 0
    if client is not None:
        metrics = client.metrics_snapshot()
        api_requests = metrics.get("total_requests", 0) - summary.api_requests_start
        api_retries = metrics.get("total_retries", 0) - summary.api_retries_start
        api_429 = metrics.get("http_429_count", 0) - summary.api_429_start
    LOGGER.info(
        "Daily summary day=%s cycles=%d signal_candidates=%d top_blockers=%s api_requests=%d retries=%d http429=%d",
        summary.trading_day,
        summary.cycles,
        summary.signal_candidates,
        summary.top_blockers(),
        api_requests,
        api_retries,
        api_429,
    )


def _default_observe_evaluation(*, symbol: str, reason: str) -> StrategyEvaluation:
    return _default_observe_evaluation_core(symbol=symbol, reason=reason)


def _pick_best_candidate(
    *,
    strategy: StrategyPlugin,
    symbol: str,
    candidates: list[SetupCandidate],
    data: StrategyDataBundle,
) -> tuple[SetupCandidate | None, StrategyEvaluation]:
    return _pick_best_candidate_core(
        strategy=strategy,
        symbol=symbol,
        candidates=candidates,
        data=data,
    )


def _normalize_action_for_score(
    *,
    evaluation: StrategyEvaluation,
    config: AppConfig,
    adaptive_cfg: AdaptiveThresholdConfig | None = None,
    soft_gate_result: SoftGateResult | None = None,
    session_threshold_adjust: float = 0.0,
) -> StrategyEvaluation:
    if evaluation.score_total is None:
        return evaluation
    if evaluation.reasons_blocking:
        evaluation.action = DecisionAction.OBSERVE
        return evaluation
    score = float(evaluation.score_total)

    # ── Adaptive threshold path ──────────────────────────
    if adaptive_cfg is not None and adaptive_cfg.enabled:
        # Apply soft-gate penalty to score (reversible metadata, not permanent)
        penalty = soft_gate_result.total_penalty if soft_gate_result else 0.0
        adjusted_score = score - penalty
        evaluation.metadata["adaptive_score_before_penalty"] = score
        evaluation.metadata["adaptive_soft_penalty"] = penalty
        evaluation.metadata["adaptive_score_adjusted"] = adjusted_score

        trend_regime = str(evaluation.metadata.get("trend_regime", "UNKNOWN"))
        vol_regime = str(evaluation.metadata.get("volatility_regime", "NORMAL"))
        threshold = compute_adaptive_threshold(
            config=adaptive_cfg,
            trend_regime=trend_regime,
            vol_regime=vol_regime,
        )
        # Apply session-based threshold adjustment
        threshold += session_threshold_adjust
        evaluation.metadata["adaptive_threshold"] = threshold
        evaluation.metadata["session_threshold_adjust"] = session_threshold_adjust

        action_str = normalize_action_adaptive(
            score=adjusted_score,
            threshold=threshold,
            small_band=5.0,
        )
        if action_str == "TRADE":
            evaluation.action = DecisionAction.TRADE
        elif action_str == "SMALL":
            evaluation.action = DecisionAction.SMALL
        else:
            evaluation.action = DecisionAction.OBSERVE
            if "SCORE_BELOW_MIN" not in evaluation.reasons_blocking:
                evaluation.reasons_blocking.append("SCORE_BELOW_MIN")
        return evaluation

    # ── Default fixed-threshold path ─────────────────────
    return normalize_action_fixed_threshold(
        evaluation=evaluation,
        config=config,
    )


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return _clamp_core(value, min_value, max_value)


def _resolve_orderflow_mode(*, symbol: str, route_params: dict[str, object], default_mode: str, full_symbols: set[str]) -> str:
    return _resolve_orderflow_mode_core(
        symbol=symbol,
        route_params=route_params,
        default_mode=default_mode,
        full_symbols=full_symbols,
    )


def _orderflow_param(
    *,
    route_params: dict[str, object],
    settings: dict[str, float] | None,
    key: str,
    default: float,
) -> float:
    return _orderflow_param_core(
        route_params=route_params,
        settings=settings,
        key=key,
        default=default,
    )


def _compute_v2_score(
    *,
    symbol: str = "",
    strategy_name: str,
    bias: BiasState,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    news_blocked: bool,
    schedule_open: bool,
    orderflow_snapshot: OrderflowSnapshot | None = None,
    setup_side: str | None = None,
    orderflow_settings: dict[str, float] | None = None,
) -> StrategyEvaluation:
    del symbol
    return compute_v2_score_core(
        strategy_name=strategy_name,
        bias=bias,
        route_params=route_params,
        evaluation=evaluation,
        news_blocked=news_blocked,
        schedule_open=schedule_open,
        policy=MAIN_SCORE_POLICY,
        orderflow_snapshot=orderflow_snapshot,
        setup_side=setup_side,
        orderflow_settings=orderflow_settings,
    )


def _evaluate_hard_gates(
    *,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    now: datetime,
    timezone_name: str,
) -> tuple[dict[str, bool], list[str]]:
    result = evaluate_hard_gates_core(
        route_params=route_params,
        evaluation=evaluation,
        now=now,
        timezone_name=timezone_name,
        policy=MAIN_HARD_GATE_POLICY,
    )
    return result.gates, result.reasons


def _quality_gate_reasons(
    *,
    symbol: str,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    now: datetime,
    timezone_name: str,
) -> list[str]:
    del symbol
    return quality_gate_reasons_core(
        route_params=route_params,
        evaluation=evaluation,
        now=now,
        timezone_name=timezone_name,
        policy=MAIN_HARD_GATE_POLICY,
    )


def _apply_orderflow_small_soft_gate(
    *,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    orderflow_settings: dict[str, float] | None,
) -> StrategyEvaluation:
    return _apply_orderflow_small_soft_gate_core(
        route_params=route_params,
        evaluation=evaluation,
        orderflow_settings=orderflow_settings,
    )


def _risk_multiplier_for(
    *,
    evaluation: StrategyEvaluation,
    route_risk: dict[str, object],
    config: AppConfig,
) -> float:
    return _risk_multiplier_for_core(
        evaluation=evaluation,
        route_risk=route_risk,
        config=config,
    )


def run_multi_strategy_loop(
    *,
    args: argparse.Namespace,
    config: AppConfig,
    journal: Journal,
    states: dict[str, AssetRuntimeState],
    client: CapitalClient | None,
    market_data: MarketDataService,
    news_provider: CalendarProvider,
    risk_engine: RiskEngine,
    strategy_router: StrategyRouter,
    strategy_plugins: dict[str, StrategyPlugin],
    orderflow_provider: OrderflowProvider,
    portfolio_supervisor: PortfolioSupervisor,
    order_executor: OrderExecutor,
    position_manager: PositionManager,
    dashboard_writer: DashboardWriter,
    alerts: AlertDispatcher,
    close_grace_seconds: int,
    candle_retry_seconds: int,
    sync_pending_seconds: int,
    sync_positions_seconds: int,
    tf_history: dict[str, int],
    sc_logger: SignalCandidateLogger | None = None,
    sc_aggregator: SignalCandidateAggregator | None = None,
    micro_loss_metrics: MicroLossMetrics | None = None,
) -> None:
    metrics_start = client.metrics_snapshot() if client is not None else {}
    daily_summary = DailyRuntimeSummary(
        trading_day=trading_day(utc_now(), config.timezone).isoformat(),
        api_requests_start=metrics_start.get("total_requests", 0),
        api_retries_start=metrics_start.get("total_retries", 0),
        api_429_start=metrics_start.get("http_429_count", 0),
    )
    stop_event = threading.Event()
    candidate_queue = CandidateQueue()
    last_heartbeat = time.monotonic()
    last_dashboard = time.monotonic()
    last_pending_sync_at: datetime | None = None
    last_positions_sync_at: datetime | None = None
    cycle = 0
    orderflow_settings = {
        "trigger_bonus_max": float(config.orderflow.trigger_bonus_max),
        "execution_bonus_max": float(config.orderflow.execution_bonus_max),
        "divergence_penalty_min": float(config.orderflow.divergence_penalty_min),
        "divergence_penalty_max": float(config.orderflow.divergence_penalty_max),
        "small_soft_gate_confidence": float(config.orderflow.small_soft_gate_confidence),
        "small_soft_gate_chop": float(config.orderflow.small_soft_gate_chop),
    }
    orderflow_full_symbols = set(config.orderflow.full_symbols)
    orderflow_default_mode = config.orderflow.default_mode
    orderflow_default_window = int(config.orderflow.default_window)
    # Paper cost model
    paper_cost_cfg = PaperCostConfig(
        enabled=config.paper_costs.enabled,
        slippage_market=SlippageModelConfig(
            base_ticks=config.paper_costs.slippage_market.base_ticks,
            beta_spread=config.paper_costs.slippage_market.beta_spread,
            beta_atr=config.paper_costs.slippage_market.beta_atr,
        ),
        slippage_stop=SlippageModelConfig(
            base_ticks=config.paper_costs.slippage_stop.base_ticks,
            beta_spread=config.paper_costs.slippage_stop.beta_spread,
            beta_atr=config.paper_costs.slippage_stop.beta_atr,
        ),
        slippage_limit=SlippageModelConfig(
            base_ticks=config.paper_costs.slippage_limit.base_ticks,
            beta_spread=config.paper_costs.slippage_limit.beta_spread,
            beta_atr=config.paper_costs.slippage_limit.beta_atr,
        ),
        commission_per_side=config.paper_costs.commission_per_side,
        swap_per_day=config.paper_costs.swap_per_day,
        use_bid_ask_fills=config.paper_costs.use_bid_ask_fills,
    )
    # Micro-loss defense config
    ml_defense_cfg = MicroLossDefenseConfig(
        enabled=config.micro_loss_defense.enabled,
        micro_loss_k=config.micro_loss_defense.micro_loss_k,
        min_stop_spread_mult=config.micro_loss_defense.min_stop_spread_mult,
        min_stop_atr_mult=config.micro_loss_defense.min_stop_atr_mult,
        edge_mult=config.micro_loss_defense.edge_mult,
        be_buffer_atr_frac=config.micro_loss_defense.be_buffer_atr_frac,
        be_buffer_ticks=config.micro_loss_defense.be_buffer_ticks,
    )
    # Adaptive threshold config
    adaptive_cfg: AdaptiveThresholdConfig | None = None
    if config.adaptive_threshold.enabled:
        adaptive_cfg = build_adaptive_config(config.adaptive_threshold)
        LOGGER.info(
            "Adaptive threshold ENABLED  base=%.1f  range_adj=%.1f  trend_adj=%.1f  "
            "soft_gates=%s  penalty=%.1f",
            adaptive_cfg.base_threshold, adaptive_cfg.range_adjust,
            adaptive_cfg.trend_adjust, adaptive_cfg.soft_gates_enabled,
            adaptive_cfg.soft_gate_penalty,
        )
    elif config.adaptive_threshold.soft_gates_enabled:
        # Soft gates can be on independently of adaptive threshold
        adaptive_cfg = build_adaptive_config(config.adaptive_threshold)
        LOGGER.info("Soft gates ENABLED (adaptive threshold OFF)")
    # ── Session-aware windows ────────────────────────────
    session_windows: list[SessionWindow] = build_session_windows(config.session_filter)
    if session_windows:
        LOGGER.info(
            "Session filter ENABLED  windows=%d  block_outside=%s",
            len(session_windows),
            config.session_filter.block_outside_sessions,
        )
    session_start_utc = utc_now()
    daily_gate_mode = _daily_gate_mode(config)
    daily_gate_services: dict[str, DailyGateProvider] = {}
    if daily_gate_mode != "off":
        for epic in states.keys():
            provider = _build_daily_gate_provider(config=config, mode=daily_gate_mode, events=[])
            if provider is not None:
                daily_gate_services[epic] = provider

    def _stop(signum: int, _frame: object) -> None:
        LOGGER.info("Received signal %s, shutting down.", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)

    while not stop_event.is_set():
        cycle += 1
        now = utc_now()
        day = trading_day(now, config.timezone).isoformat()
        if day != daily_summary.trading_day:
            _log_daily_summary(summary=daily_summary, client=client)
            metrics = client.metrics_snapshot() if client is not None else {}
            daily_summary = DailyRuntimeSummary(
                trading_day=day,
                api_requests_start=metrics.get("total_requests", 0),
                api_retries_start=metrics.get("total_retries", 0),
                api_429_start=metrics.get("http_429_count", 0),
            )
        daily_summary.cycles += 1
        # ── Session matching (multi-strategy loop) ────
        session_match: SessionMatch = match_session(
            now, session_windows,
            block_outside=bool(config.session_filter.block_outside_sessions),
        ) if session_windows else SessionMatch.no_match()
        try:
            symbol_market_open = {
                epic: is_symbol_market_open(
                    now,
                    symbol=epic,
                    timezone_name=config.timezone,
                    default_profile=config.market_hours.default_profile,
                    symbol_profiles=config.market_hours.symbol_profiles,
                )
                for epic in states.keys()
            }
            if not any(symbol_market_open.values()):
                stop_event.wait(config.execution.loop_seconds)
                continue

            global_stats = journal.get_daily_stats(day, epic="GLOBAL")
            if risk_engine.should_turn_off_for_day(global_stats.pnl):
                if global_stats.status != "OFF":
                    journal.set_daily_status(day, "OFF", epic="GLOBAL")
                    alerts.send(
                        event="DAILY_STOP_GLOBAL",
                        level="warning",
                        message=f"Global daily stop triggered pnl={global_stats.pnl:.2f}",
                        dedupe_key=f"daily-stop-global-{day}",
                    )

            if should_refresh_quote(now=now, last_fetch_at=last_pending_sync_at, interval_seconds=sync_pending_seconds):
                order_executor.sync_remote_pending_orders()
                last_pending_sync_at = now

            closed_sync: list[ClosedPositionEvent] = []
            if should_refresh_quote(now=now, last_fetch_at=last_positions_sync_at, interval_seconds=sync_positions_seconds):
                closed_sync = position_manager.sync_positions_from_api()
                last_positions_sync_at = now

            quotes: dict[str, tuple[float, float, float]] = {}
            for epic, state in states.items():
                if not symbol_market_open.get(epic, True):
                    continue
                quote_interval = _quote_refresh_interval_seconds(
                    config=config,
                    trade_enabled=state.asset.trade_enabled,
                )
                if should_refresh_quote(now=now, last_fetch_at=state.quote_last_fetch_at, interval_seconds=quote_interval):
                    bid, ask, spread = market_data.fetch_quote_and_spread(epic)
                    if bid is not None and ask is not None and spread is not None:
                        state.quote = (bid, ask, spread)
                        state.quote_last_fetch_at = now
                if state.quote is not None:
                    quotes[epic] = state.quote

            if config.watchlist.log_quotes:
                rows = []
                for epic, state in states.items():
                    if state.asset.trade_enabled:
                        continue
                    q = quotes.get(epic)
                    if q is None:
                        continue
                    b, a, s = q
                    rows.append(f"{epic} bid={b:.2f} ask={a:.2f} spr={s:.2f}")
                if rows:
                    LOGGER.info("Watchlist quotes | %s", " | ".join(rows))

            expired = order_executor.cancel_expired_orders(now)
            for order_id in expired:
                alerts.send(
                    event="ORDER_CANCELLED",
                    level="warning",
                    message=f"Pending order cancelled by TTL: {order_id}",
                    dedupe_key=f"cancel-ttl-{order_id}",
                )

            events = news_provider.get_high_impact_events(
                now - timedelta(minutes=config.news_gate.block_minutes + 1),
                now + timedelta(minutes=config.news_gate.block_minutes + 1),
            )
            if daily_gate_services:
                for provider in daily_gate_services.values():
                    provider.set_events(events)
            news_blocked = is_blocked(now, events, block_minutes=config.news_gate.block_minutes)
            if news_blocked:
                for order in order_executor.get_pending_orders():
                    if should_cancel_pending({"status": order.status}, now, events, block_minutes=config.news_gate.block_minutes):
                        order_executor.cancel_order(order.order_id)
                        alerts.send(
                            event="ORDER_CANCELLED",
                            level="warning",
                            message=f"Pending order cancelled by news gate: {order.order_id}",
                            dedupe_key=f"cancel-news-{order.order_id}",
                        )

            filled = order_executor.process_pending_fills(quotes_by_epic=quotes, now=now)
            for pos in filled:
                journal.increment_daily_trades(day, epic=pos.epic)
                journal.increment_daily_trades(day, epic="GLOBAL")
                alerts.send(
                    event="ORDER_FILLED",
                    level="info",
                    message=f"{pos.epic} deal={pos.deal_id} side={pos.side}",
                    dedupe_key=f"filled-{pos.deal_id}",
                )

            closed_manage = position_manager.manage_open_positions(now=now, quotes_by_epic=quotes)
            closed = closed_sync + closed_manage
            if closed:
                apply_closed_events(closed, day, journal, risk_engine, now, alerts)
                # ── Update re-entry state per asset ──────────
                for _cev in closed:
                    _rs = states.get(_cev.epic)
                    if _rs is not None:
                        _rs.reentry.record_close(
                            side=_cev.side or "UNKNOWN",
                            exit_type=_cev.exit_type or "UNKNOWN",
                            pnl=_cev.pnl,
                            closed_at=_cev.closed_at,
                        )

            close_meta: dict[str, tuple[bool, datetime | None, bool, datetime | None, bool, datetime | None]] = {}
            pending_intents: list[PendingOrderIntent] = []
            final_decision: dict[str, str] = {}
            for epic, state in states.items():
                if not symbol_market_open.get(epic, True):
                    final_decision[epic] = "MARKET_CLOSED"
                    state.last_reason_codes = ["MARKET_CLOSED"]
                    continue
                asset_stats = journal.get_daily_stats(day, epic=epic)
                if risk_engine.should_turn_off_for_day(asset_stats.pnl) and asset_stats.status != "OFF":
                    journal.set_daily_status(day, "OFF", epic=epic)
                h1_new, h1_closed = refresh_timeframe_cache(
                    market_data=market_data,
                    state=state,
                    now=now,
                    timeframe=config.timeframes.h1,
                    history_count=tf_history[config.timeframes.h1],
                    close_grace_seconds=close_grace_seconds,
                    retry_seconds=candle_retry_seconds,
                )
                m15_new, m15_closed = refresh_timeframe_cache(
                    market_data=market_data,
                    state=state,
                    now=now,
                    timeframe=config.timeframes.m15,
                    history_count=tf_history[config.timeframes.m15],
                    close_grace_seconds=close_grace_seconds,
                    retry_seconds=candle_retry_seconds,
                )
                m5_new, m5_closed = refresh_timeframe_cache(
                    market_data=market_data,
                    state=state,
                    now=now,
                    timeframe=config.timeframes.m5,
                    history_count=tf_history[config.timeframes.m5],
                    close_grace_seconds=close_grace_seconds,
                    retry_seconds=candle_retry_seconds,
                )
                close_meta[epic] = (h1_new, h1_closed, m15_new, m15_closed, m5_new, m5_closed)

                open_asset = position_manager.get_open_positions(epic=epic)
                pending_asset = order_executor.get_pending_orders(epic=epic)
                state.entry_state = derive_entry_state(
                    state.entry_state,
                    has_open=bool(open_asset),
                    has_pending=bool(pending_asset),
                )
                q = quotes.get(epic)
                spread = q[2] if q is not None else None
                gate_provider = daily_gate_services.get(epic)
                gate_result = None
                if gate_provider is not None:
                    m5_gate_candles = closed_candles(state.cache.get(config.timeframes.m5, []))
                    gate_provider.refresh_if_needed(now=now, candles=m5_gate_candles)
                    gate_result = gate_provider.evaluate(ts=now, symbol=epic, spread=spread)
                routes = strategy_router.routes_for(epic)
                best_outcome: StrategyOutcome | None = None
                best_route = routes[0]
                best_rank = float("-inf")
                route_summaries: list[dict[str, object]] = []

                for route in routes:
                    strategy = strategy_plugins.get(route.strategy)
                    if strategy is None:
                        evaluation = _default_observe_evaluation(
                            symbol=epic,
                            reason=f"UNKNOWN_STRATEGY_{route.strategy}",
                        )
                        outcome = StrategyOutcome(
                            symbol=epic,
                            strategy_name=route.strategy,
                            bias=BiasState(epic, route.strategy, "NEUTRAL", config.timeframes.m5, now, {}),
                            candidate=None,
                            evaluation=evaluation,
                            order_request=None,
                            reason_codes=[f"UNKNOWN_STRATEGY_{route.strategy}"],
                            payload={"strategy_name": route.strategy},
                        )
                        rank_value = rank_score(evaluation)
                    else:
                        bundle = StrategyDataBundle(
                            symbol=epic,
                            now=now,
                            candles_h1=state.cache.get(config.timeframes.h1, []),
                            candles_m15=state.cache.get(config.timeframes.m15, []),
                            candles_m5=state.cache.get(config.timeframes.m5, []),
                            spread=spread,
                            spread_history=market_data.spread_history(epic),
                            news_blocked=news_blocked,
                            entry_state=state.entry_state,
                            h1_new_close=h1_new,
                            m15_new_close=m15_new,
                            m5_new_close=m5_new,
                            quote=q,
                            extra={
                                "minimal_tick_buffer": state.asset.minimal_tick_buffer,
                                "strategy_params": route.params,
                                "strategy_risk": route.risk,
                                "origin_strategy": route.strategy,
                            },
                        )
                        strategy.preprocess(epic, bundle)
                        bias = strategy.compute_bias(epic, bundle)
                        raw_candidates = strategy.detect_candidates(epic, bundle)
                        candidates = candidate_queue.put_many(
                            symbol=epic,
                            strategy=route.strategy,
                            candidates=raw_candidates,
                            now=now,
                        )
                        candidate, evaluation = _pick_best_candidate(
                            strategy=strategy,
                            symbol=epic,
                            candidates=candidates,
                            data=bundle,
                        )
                        schedule_cfg = route.params.get("schedule")
                        schedule_open = True
                        if isinstance(schedule_cfg, dict):
                            schedule_open = is_schedule_open(now, schedule_cfg, config.timezone)
                        mode_override = _resolve_orderflow_mode(
                            symbol=epic,
                            route_params=route.params,
                            default_mode=orderflow_default_mode,
                            full_symbols=orderflow_full_symbols,
                        )
                        route_of = route.params.get("orderflow")
                        window = orderflow_default_window
                        if isinstance(route_of, dict):
                            try:
                                window = max(8, int(route_of.get("window", orderflow_default_window)))
                            except (TypeError, ValueError):
                                window = orderflow_default_window
                        atr_for_of = evaluation.metadata.get("atr_m5", evaluation.snapshot.get("atr_m5"))
                        atr_for_of_float: float | None
                        try:
                            atr_for_of_float = float(atr_for_of) if atr_for_of is not None else None
                        except (TypeError, ValueError):
                            atr_for_of_float = None
                        orderflow_snapshot = orderflow_provider.get_snapshot(
                            symbol=epic,
                            tf=config.timeframes.m5,
                            window=window,
                            candles=state.cache.get(config.timeframes.m5, []),
                            spread=spread,
                            quote=q,
                            atr_value=atr_for_of_float,
                            extra=bundle.extra,
                            mode_override=mode_override,
                        )
                        evaluation = _compute_v2_score(
                            symbol=epic,
                            strategy_name=route.strategy,
                            bias=bias,
                            route_params=route.params,
                            evaluation=evaluation,
                            news_blocked=news_blocked,
                            schedule_open=schedule_open,
                            orderflow_snapshot=orderflow_snapshot,
                            setup_side=candidate.side if candidate is not None else None,
                            orderflow_settings=orderflow_settings,
                        )
                        _ = _normalize_action_for_score(
                            evaluation=evaluation, config=config,
                            session_threshold_adjust=session_match.threshold_adjust,
                        )
                        gate_reasons = _quality_gate_reasons(
                            symbol=epic,
                            route_params=route.params,
                            evaluation=evaluation,
                            now=now,
                            timezone_name=config.timezone,
                        )
                        # ── Soft-gate conversion ─────────────────
                        _sg_result: SoftGateResult | None = None
                        if adaptive_cfg is not None and adaptive_cfg.soft_gates_enabled and gate_reasons:
                            _sg_result = apply_soft_gates(
                                gate_reasons,
                                soft_gates_enabled=True,
                                soft_gate_penalty=adaptive_cfg.soft_gate_penalty,
                            )
                            evaluation.metadata["soft_gate_converted"] = _sg_result.converted_gates
                            evaluation.metadata["soft_gate_penalty"] = _sg_result.total_penalty
                            # Only hard reasons block; soft ones become penalties
                            if _sg_result.hard_reasons:
                                evaluation.action = DecisionAction.OBSERVE
                                for code in _sg_result.hard_reasons:
                                    if code not in evaluation.reasons_blocking:
                                        evaluation.reasons_blocking.append(code)
                            # Re-normalize with adaptive threshold + soft penalty
                            if not _sg_result.hard_reasons:
                                evaluation.reasons_blocking.clear()
                                _ = _normalize_action_for_score(
                                    evaluation=evaluation,
                                    config=config,
                                    adaptive_cfg=adaptive_cfg,
                                    soft_gate_result=_sg_result,
                                    session_threshold_adjust=session_match.threshold_adjust,
                                )
                        elif gate_reasons:
                            evaluation.action = DecisionAction.OBSERVE
                            for code in gate_reasons:
                                if code not in evaluation.reasons_blocking:
                                    evaluation.reasons_blocking.append(code)
                        elif adaptive_cfg is not None and adaptive_cfg.enabled:
                            # No gates hit → still re-normalize with adaptive threshold
                            evaluation.reasons_blocking.clear()
                            _ = _normalize_action_for_score(
                                evaluation=evaluation,
                                config=config,
                                adaptive_cfg=adaptive_cfg,
                                session_threshold_adjust=session_match.threshold_adjust,
                            )
                        evaluation = _apply_orderflow_small_soft_gate(
                            route_params=route.params,
                            evaluation=evaluation,
                            orderflow_settings=orderflow_settings,
                        )

                        signal_request = (
                            strategy.generate_order(epic, evaluation, candidate, bundle)
                            if candidate is not None and evaluation.action in {DecisionAction.TRADE, DecisionAction.SMALL}
                            else None
                        )
                        outcome = StrategyOutcome(
                            symbol=epic,
                            strategy_name=route.strategy,
                            bias=bias,
                            candidate=candidate,
                            evaluation=evaluation,
                            order_request=signal_request,
                            reason_codes=list(evaluation.reasons_blocking),
                            payload={
                                "strategy_name": route.strategy,
                                "score_total": evaluation.score_total,
                                "score_breakdown": evaluation.score_breakdown,
                                "snapshot": evaluation.snapshot,
                                "candidate_id": candidate.candidate_id if candidate is not None else None,
                                **evaluation.metadata,
                            },
                        )
                        soft_reasons = evaluation.metadata.get("soft_reasons")
                        if isinstance(soft_reasons, list):
                            for soft_reason in soft_reasons:
                                code = f"SOFT_REASON_{str(soft_reason).upper()}"
                                if code not in outcome.reason_codes:
                                    outcome.reason_codes.append(code)
                        if isinstance(strategy, IndexExistingStrategy):
                            state.h1_snapshot, state.m15_snapshot, state.m5_snapshot = strategy.last_snapshots(epic)
                            legacy = strategy.last_legacy_decision(epic)
                            if legacy is not None:
                                for code in legacy.reason_codes:
                                    if code not in outcome.reason_codes:
                                        outcome.reason_codes.append(code)
                        rank_value = rank_score(evaluation) + (route.priority * 0.01)

                    # --- SignalCandidate logging (telemetry) ---
                    if sc_logger is not None:
                        _atr_val = outcome.evaluation.metadata.get(
                            "atr_m5", outcome.evaluation.snapshot.get("atr_m5")
                        )
                        _atr_float: float | None = None
                        try:
                            _atr_float = float(_atr_val) if _atr_val is not None else None
                        except (TypeError, ValueError):
                            pass
                        _sl_dist: float | None = None
                        _tp_dist: float | None = None
                        _expected_rr: float | None = None
                        _estimated_rtc: float | None = None
                        if outcome.order_request is not None:
                            sig = outcome.order_request
                            _sl_dist = abs(sig.entry_price - sig.stop_price) if sig.entry_price and sig.stop_price else None
                            _tp_dist = abs(sig.take_profit - sig.entry_price) if sig.take_profit and sig.entry_price else None
                            if _sl_dist and _sl_dist > 0 and _tp_dist is not None:
                                _expected_rr = _tp_dist / _sl_dist
                        if spread is not None and paper_cost_cfg.enabled:
                            _estimated_rtc = estimate_roundtrip_cost_points(
                                spread=spread, atr=_atr_float, config=paper_cost_cfg
                            )
                        _setup_name = "NONE"
                        if outcome.candidate is not None:
                            _setup_name = outcome.candidate.setup_type or outcome.candidate.strategy_name
                        _trend_regime = str(outcome.evaluation.metadata.get("trend_regime", "UNKNOWN"))
                        _vol_regime = str(outcome.evaluation.metadata.get("volatility_regime", "UNKNOWN"))
                        _session_name = str(outcome.evaluation.metadata.get("session_name", "UNKNOWN"))
                        sc_logger.log(SignalCandidate(
                            timestamp=now,
                            symbol=epic,
                            timeframe=config.timeframes.m5,
                            strategy_name=outcome.strategy_name,
                            setup_name=_setup_name,
                            side=outcome.order_request.side if outcome.order_request else (outcome.candidate.side if outcome.candidate else None),
                            score=outcome.evaluation.score_total,
                            spread=spread,
                            atr=_atr_float,
                            trend_regime=_trend_regime,
                            volatility_regime=_vol_regime,
                            session_name=_session_name,
                            bias_direction=outcome.bias.direction if outcome.bias else None,
                            sl_distance=_sl_dist,
                            tp_distance=_tp_dist,
                            expected_rr=_expected_rr,
                            expected_move=_tp_dist,
                            estimated_roundtrip_cost=_estimated_rtc,
                            action=outcome.evaluation.action.value,
                            accepted=outcome.order_request is not None,
                            rejection_reasons=list(outcome.evaluation.reasons_blocking),
                            score_breakdown=dict(outcome.evaluation.score_breakdown),
                            features=dict(outcome.evaluation.snapshot) if outcome.evaluation.snapshot else {},
                            gates=dict(outcome.evaluation.gates),
                            metadata={"route_strategy": route.strategy, "route_priority": route.priority},
                        ))

                    route_summaries.append(
                        {
                            "strategy": route.strategy,
                            "score": outcome.evaluation.score_total,
                            "action": outcome.evaluation.action.value,
                            "has_order": outcome.order_request is not None,
                            "rank": round(rank_value, 4),
                        }
                    )
                    if best_outcome is None:
                        best_outcome = outcome
                        best_route = route
                        best_rank = rank_value
                        continue
                    best_has_order = best_outcome.order_request is not None
                    current_has_order = outcome.order_request is not None
                    if current_has_order and not best_has_order:
                        best_outcome = outcome
                        best_route = route
                        best_rank = rank_value
                        continue
                    if current_has_order == best_has_order and rank_value > best_rank:
                        best_outcome = outcome
                        best_route = route
                        best_rank = rank_value

                if best_outcome is None:
                    final_decision[epic] = "NO_SIGNAL"
                    continue
                state.strategy_name = best_outcome.strategy_name
                state.bias_state = best_outcome.bias
                state.last_candidate = best_outcome.candidate
                state.last_evaluation = best_outcome.evaluation
                best_outcome.payload["route_rankings"] = route_summaries
                state.pending_outcome = best_outcome
                final_decision[epic] = "MANAGE" if state.entry_state == "FILLED" else "NO_SIGNAL"
                if gate_result is not None:
                    best_outcome.payload["daily_gate"] = {
                        "mode": daily_gate_mode,
                        "bias": gate_result.bias,
                        "reasons": list(gate_result.reasons),
                        "allowed_strategies": list(gate_result.allowed_strategies),
                    }
                    best_outcome.evaluation.gates.setdefault("DailyGate", True)

                if best_outcome.order_request is None:
                    continue
                daily_summary.signal_candidates += 1
                if gate_result is not None:
                    gate_reasons: list[str] = list(gate_result.reasons)
                    gate_bias = str(gate_result.bias).upper()
                    side = str(best_outcome.order_request.side).upper()
                    if gate_bias == "FLAT":
                        gate_reasons.append("DAILY_GATE_FLAT")
                    elif gate_bias == "LONG" and side != "LONG":
                        gate_reasons.append("DAILY_GATE_LONG_ONLY")
                    elif gate_bias == "SHORT" and side != "SHORT":
                        gate_reasons.append("DAILY_GATE_SHORT_ONLY")
                    if gate_result.allowed_strategies:
                        allowed = {str(item).upper() for item in gate_result.allowed_strategies}
                        if str(best_outcome.strategy_name).upper() not in allowed:
                            gate_reasons.append("DAILY_GATE_STRATEGY_BLOCKED")
                    if gate_reasons:
                        best_outcome.evaluation.gates["DailyGate"] = False
                        if best_outcome.evaluation.gate_blocked is None:
                            best_outcome.evaluation.gate_blocked = "DailyGate"
                        for code in list(dict.fromkeys(gate_reasons)):
                            if code not in best_outcome.reason_codes:
                                best_outcome.reason_codes.append(code)
                        final_decision[epic] = "NO_SIGNAL"
                        continue
                if not state.asset.trade_enabled:
                    best_outcome.reason_codes.append("OBSERVE_ONLY_ASSET")
                    continue
                if pending_asset:
                    best_outcome.reason_codes.append("PENDING_EXISTS")
                    final_decision[epic] = "WAIT_LIMIT_FILL"
                    continue
                max_trades_symbol = int(best_route.risk.get("max_trades_per_day", config.risk.max_trades_per_day))
                if asset_stats.trades_count >= max_trades_symbol:
                    best_outcome.reason_codes.append("RISK_GATE_MAX_TRADES_DAY")
                    best_outcome.evaluation.gates.setdefault("RiskGate", True)
                    best_outcome.evaluation.gates["RiskGate"] = False
                    if best_outcome.evaluation.gate_blocked is None:
                        best_outcome.evaluation.gate_blocked = "RiskGate"
                    final_decision[epic] = "NO_SIGNAL"
                    continue
                risk_multiplier = _risk_multiplier_for(
                    evaluation=best_outcome.evaluation,
                    route_risk=best_route.risk,
                    config=config,
                )
                pending_intents.append(
                    PendingOrderIntent(
                        symbol=epic,
                        state=state,
                        route_priority=best_route.priority,
                        cooldown_seconds=best_route.cooldown_seconds,
                        route_risk=best_route.risk,
                        outcome=best_outcome,
                        signal=best_outcome.order_request,
                        risk_multiplier=risk_multiplier,
                        rank_score=best_rank,
                        asset_stats_snapshot=asset_stats,
                    )
                )
                final_decision[epic] = "PLACE_LIMIT_PENDING_SUPERVISOR"

            supervisor_input = [
                EntryProposal(
                    symbol=intent.symbol,
                    strategy_name=intent.outcome.strategy_name,
                    priority=intent.route_priority,
                    score_total=intent.outcome.evaluation.score_total,
                    rank_score=intent.rank_score,
                    risk_r=intent.risk_multiplier,
                    cooldown_seconds=intent.cooldown_seconds,
                    payload=intent.outcome.payload,
                )
                for intent in pending_intents
            ]
            supervisor_result = portfolio_supervisor.evaluate_entries(
                now=now,
                trading_day=day,
                proposals=supervisor_input,
                open_positions=position_manager.get_open_positions(),
            )
            selected_symbols = {item.symbol for item in supervisor_result.selected}

            for intent in pending_intents:
                if intent.symbol not in selected_symbols:
                    blocked = supervisor_result.blocked.get(intent.symbol, ["SUPERVISOR_REJECTED"])
                    for code in blocked:
                        if code not in intent.outcome.reason_codes:
                            intent.outcome.reason_codes.append(code)
                    intent.outcome.reason_codes.append("RISK_GATE_SUPERVISOR")
                    intent.outcome.evaluation.gates.setdefault("RiskGate", True)
                    intent.outcome.evaluation.gates["RiskGate"] = False
                    if intent.outcome.evaluation.gate_blocked is None:
                        intent.outcome.evaluation.gate_blocked = "RiskGate"
                    final_decision[intent.symbol] = "NO_SIGNAL"
                    continue

                global_stats = journal.get_daily_stats(day, epic="GLOBAL")
                cooldown = journal.get_risk_state(f"ASSET:{intent.symbol}").cooldown_until
                open_asset_now = position_manager.get_open_positions(epic=intent.symbol)
                open_all_now = position_manager.get_open_positions()

                # ── Session block check ──────────────────
                if session_match.blocked:
                    intent.outcome.reason_codes.append("SESSION_BLOCKED")
                    intent.outcome.payload["session"] = {"name": session_match.session_name, "blocked": True}
                    final_decision[intent.symbol] = "NO_SIGNAL"
                    continue

                # ── Tier-based sizing ────────────────────
                _score_for_tier = float(intent.outcome.evaluation.score_total or 0.0)
                _tier_label = resolve_score_tier(
                    _score_for_tier,
                    tier_cfg=config.tier_sizing if config.tier_sizing.enabled else None,
                )
                _tier_mult = tier_risk_multiplier(
                    _tier_label,
                    tier_cfg=config.tier_sizing if config.tier_sizing.enabled else None,
                )
                if config.tier_sizing.enabled and _tier_mult <= 0:
                    intent.outcome.reason_codes.append("TIER_OBSERVE_SKIP")
                    intent.outcome.payload["tier"] = {"label": _tier_label, "mult": 0.0}
                    final_decision[intent.symbol] = "NO_SIGNAL"
                    continue

                # ── Compound equity ──────────────────────
                _sizing_equity = float(config.risk.equity)
                if config.compound_equity.enabled:
                    _sizing_equity = compute_compound_equity(
                        _sizing_equity,
                        floor_equity=float(config.compound_equity.floor_equity),
                        cap_equity=float(config.compound_equity.cap_equity),
                    )

                # ── Session + tier adjusted risk ─────────
                _session_risk_mult = session_match.risk_mult
                effective_risk_per_trade = risk_engine.effective_risk_per_trade(
                    risk_multiplier=intent.risk_multiplier,
                    equity=_sizing_equity,
                )
                effective_risk_per_trade *= _tier_mult * _session_risk_mult
                intent.outcome.payload["sizing_meta"] = {
                    "tier": _tier_label, "tier_mult": _tier_mult,
                    "session": session_match.session_name,
                    "session_risk_mult": _session_risk_mult,
                    "sizing_equity": _sizing_equity,
                }

                risk_check = risk_engine.can_open_new_trade_multi(
                    now=now,
                    asset_epic=intent.symbol,
                    asset_stats=intent.asset_stats_snapshot,
                    global_stats=global_stats,
                    asset_open_positions=open_asset_now,
                    all_open_positions=open_all_now,
                    new_trade_risk_amount=_sizing_equity * effective_risk_per_trade,
                    cooldown_until=cooldown,
                    equity=_sizing_equity,
                )
                if not risk_check.allowed:
                    for code in risk_check.reason_codes:
                        if code not in intent.outcome.reason_codes:
                            intent.outcome.reason_codes.append(code)
                    intent.outcome.reason_codes.append("RISK_GATE_LIMITS")
                    intent.outcome.evaluation.gates.setdefault("RiskGate", True)
                    intent.outcome.evaluation.gates["RiskGate"] = False
                    if intent.outcome.evaluation.gate_blocked is None:
                        intent.outcome.evaluation.gate_blocked = "RiskGate"
                    intent.outcome.payload["risk"] = risk_check.metadata
                    final_decision[intent.symbol] = "NO_SIGNAL"
                    continue

                risk_distance = abs(float(intent.signal.entry_price) - float(intent.signal.stop_price))
                risk_cash_plan = compute_risk_cash_plan(
                    risk=config.risk,
                    equity=_sizing_equity,
                    effective_risk_per_trade=effective_risk_per_trade,
                )
                raw_size = (float(risk_cash_plan.target_risk_cash) / risk_distance) if risk_distance > 0 else 0.0
                spread_now = intent.state.quote[2] if intent.state.quote is not None else None
                spread_points_now = (
                    price_to_points(float(spread_now), point_size=float(intent.state.asset.point_size))
                    if spread_now is not None
                    else None
                )
                open_margin = _estimated_open_margin(positions=open_all_now, config=config)
                free_margin = max(0.0, _sizing_equity - float(open_margin))
                spread_limit_points = config.backtest_tuning.spread_limit_points
                feasibility = validate_order(
                    raw_size=raw_size,
                    entry_price=float(intent.signal.entry_price),
                    stop_price=float(intent.signal.stop_price),
                    take_profit=float(intent.signal.take_profit),
                    min_size=float(intent.state.asset.min_size),
                    size_step=float(intent.state.asset.size_step),
                    max_risk_cash=float(risk_cash_plan.max_risk_cash),
                    equity=float(_sizing_equity),
                    open_positions_count=len(open_asset_now),
                    max_positions=int(config.risk.max_positions),
                    spread=(float(spread_points_now) if spread_points_now is not None else None),
                    spread_limit=(float(spread_limit_points) if spread_limit_points is not None else None),
                    min_stop_distance=float(intent.state.asset.minimal_tick_buffer),
                    free_margin=free_margin,
                    margin_requirement_pct=float(config.backtest_tuning.broker_margin_requirement_pct),
                    max_leverage=float(config.backtest_tuning.broker_leverage),
                    margin_safety_factor=1.0,
                    allow_min_size_override_if_within_risk=bool(config.risk.allow_min_size_override_if_within_risk),
                    cooldown_blocked=bool(cooldown is not None and now < cooldown),
                    news_blocked=bool(news_blocked),
                )
                if not feasibility.ok:
                    reject_code = feasibility.reason.value if feasibility.reason is not None else "ORDER_FEASIBILITY_REJECT"
                    intent.outcome.reason_codes.append(reject_code)
                    intent.outcome.reason_codes.append("RISK_GATE_SIZE_INVALID")
                    intent.outcome.evaluation.gates.setdefault("RiskGate", True)
                    intent.outcome.evaluation.gates["RiskGate"] = False
                    if intent.outcome.evaluation.gate_blocked is None:
                        intent.outcome.evaluation.gate_blocked = "RiskGate"
                    intent.outcome.payload["feasibility"] = feasibility.details
                    final_decision[intent.symbol] = "NO_SIGNAL"
                    continue
                size = float(feasibility.details.get("rounded_size", 0.0))
                if size <= 0:
                    intent.outcome.reason_codes.append("SIZE_TOO_SMALL")
                    intent.outcome.reason_codes.append("RISK_GATE_SIZE_INVALID")
                    intent.outcome.evaluation.gates.setdefault("RiskGate", True)
                    intent.outcome.evaluation.gates["RiskGate"] = False
                    if intent.outcome.evaluation.gate_blocked is None:
                        intent.outcome.evaluation.gate_blocked = "RiskGate"
                    intent.outcome.payload["feasibility"] = feasibility.details
                    final_decision[intent.symbol] = "NO_SIGNAL"
                    continue

                if not _apply_rr_profile_to_signal(
                    intent.signal,
                    tp1_trigger_r=float(config.backtest_tuning.tp1_trigger_r),
                    tp1_fraction=float(config.backtest_tuning.tp1_fraction),
                    tp_profile_mode=str(config.backtest_tuning.tp_profile_mode),
                ):
                    intent.outcome.reason_codes.append("ORDER_INVALID_RISK")
                    intent.outcome.reason_codes.append("RISK_GATE_SIZE_INVALID")
                    intent.outcome.evaluation.gates.setdefault("RiskGate", True)
                    intent.outcome.evaluation.gates["RiskGate"] = False
                    if intent.outcome.evaluation.gate_blocked is None:
                        intent.outcome.evaluation.gate_blocked = "RiskGate"
                    final_decision[intent.symbol] = "NO_SIGNAL"
                    continue

                # --- Re-entry check ---
                if adaptive_cfg is not None:
                    _re_state = states.get(intent.symbol)
                    if _re_state is not None:
                        _re_side = intent.signal.side if intent.signal else "UNKNOWN"
                        _re_ok, _re_reason = _re_state.reentry.can_reenter(_re_side, now)
                        if not _re_ok:
                            if _re_reason not in intent.outcome.reason_codes:
                                intent.outcome.reason_codes.append(_re_reason)
                            intent.outcome.payload["reentry_blocked"] = _re_reason
                            final_decision[intent.symbol] = "NO_SIGNAL"
                            LOGGER.info(
                                "Re-entry blocked %s side=%s reason=%s",
                                intent.symbol, _re_side, _re_reason,
                            )
                            continue

                # --- Micro-loss defense: min SL + edge filter ---
                if ml_defense_cfg.enabled:
                    _ml_spread = float(spread_now) if spread_now is not None else 0.0
                    _ml_atr_raw = intent.outcome.evaluation.metadata.get(
                        "atr_m5", intent.outcome.evaluation.snapshot.get("atr_m5")
                    )
                    _ml_atr: float | None = None
                    try:
                        _ml_atr = float(_ml_atr_raw) if _ml_atr_raw is not None else None
                    except (TypeError, ValueError):
                        pass
                    _ml_sl_dist = abs(float(intent.signal.entry_price) - float(intent.signal.stop_price))
                    _ml_tp_dist = abs(float(intent.signal.take_profit) - float(intent.signal.entry_price))
                    _ml_rtc = estimate_roundtrip_cost_points(
                        spread=_ml_spread, atr=_ml_atr, config=paper_cost_cfg
                    )
                    ml_check = run_micro_loss_checks(
                        sl_distance=_ml_sl_dist,
                        tp_distance=_ml_tp_dist,
                        spread=_ml_spread,
                        atr=_ml_atr,
                        roundtrip_cost_points=_ml_rtc,
                        config=ml_defense_cfg,
                    )
                    if not ml_check.passed:
                        for reason in ml_check.rejection_reasons:
                            if reason not in intent.outcome.reason_codes:
                                intent.outcome.reason_codes.append(reason)
                        intent.outcome.reason_codes.append("MICRO_LOSS_DEFENSE")
                        intent.outcome.payload["micro_loss_check"] = ml_check.details
                        final_decision[intent.symbol] = "NO_SIGNAL"
                        LOGGER.info(
                            "Micro-loss defense blocked %s: reasons=%s sl=%.4f min=%.4f edge=%.2f",
                            intent.symbol,
                            ml_check.rejection_reasons,
                            ml_check.actual_sl,
                            ml_check.min_sl_required,
                            ml_check.edge_ratio,
                        )
                        continue

                    # Store roundtrip cost estimate in metadata for later PnL accounting
                    intent.outcome.payload["estimated_roundtrip_cost_points"] = _ml_rtc
                    intent.outcome.payload["micro_loss_check"] = ml_check.details

                key = (
                    f"{intent.symbol}:{intent.signal.side}:{intent.signal.entry_price:.5f}:"
                    f"{intent.signal.expires_at.isoformat()}"
                )
                # Embed cost metadata into signal for fill simulation
                _order_spread = float(spread_now) if spread_now is not None else 0.0
                _rtc_size = estimate_roundtrip_cost(
                    spread=_order_spread,
                    atr=_ml_atr if ml_defense_cfg.enabled else None,
                    size=size,
                    config=paper_cost_cfg,
                ) if paper_cost_cfg.enabled else None
                signal_meta = dict(intent.signal.metadata)
                signal_meta["spread_at_entry"] = _order_spread
                signal_meta["be_buffer_ticks"] = ml_defense_cfg.be_buffer_ticks if ml_defense_cfg.enabled else 0.0
                if _rtc_size is not None:
                    signal_meta["estimated_roundtrip_cost"] = _rtc_size.total
                    signal_meta["cost_breakdown"] = _rtc_size.to_dict()
                intent.signal.metadata = signal_meta
                order = order_executor.place_limit_order(
                    intent.signal,
                    size=size,
                    epic=intent.symbol,
                    currency=intent.state.asset.currency,
                    idempotency_key=key,
                )
                intent.outcome.evaluation.gates.setdefault("RiskGate", True)
                portfolio_supervisor.register_entry(
                    trading_day=day,
                    proposal=EntryProposal(
                        symbol=intent.symbol,
                        strategy_name=intent.outcome.strategy_name,
                        priority=intent.route_priority,
                        score_total=intent.outcome.evaluation.score_total,
                        rank_score=intent.rank_score,
                        risk_r=intent.risk_multiplier,
                        cooldown_seconds=intent.cooldown_seconds,
                    ),
                    now=now,
                )
                journal.increment_daily_trades(day, epic=intent.symbol)
                journal.increment_daily_trades(day, epic="GLOBAL")
                intent.outcome.reason_codes.append("ORDER_PLACED")
                intent.state.entry_state = "ORDER_PLACED"
                LOGGER.info(
                    "Placed LIMIT %s %s order id=%s size=%.4f strategy=%s score=%s",
                    intent.symbol,
                    intent.signal.side,
                    order.order_id,
                    order.size,
                    intent.outcome.strategy_name,
                    f"{intent.outcome.evaluation.score_total:.2f}" if intent.outcome.evaluation.score_total is not None else "-",
                )
                final_decision[intent.symbol] = "PLACE_LIMIT"
                # ── Mark re-entry if applicable ──────────
                if adaptive_cfg is not None:
                    _re_s = states.get(intent.symbol)
                    if _re_s is not None and _re_s.reentry.last_close_side == intent.signal.side:
                        _re_s.reentry.mark_reentry(now)
                        LOGGER.info(
                            "Re-entry #%d for %s %s",
                            _re_s.reentry.reentries_this_leg,
                            intent.symbol, intent.signal.side,
                        )
                outcome = state.pending_outcome
                if outcome is None:
                    continue
                reason_codes = map_reason_codes(outcome.reason_codes)
                state.last_reason_codes = reason_codes
                journal.log_decision(
                    create_decision_record_from_outcome(
                        outcome=StrategyOutcome(
                            symbol=outcome.symbol,
                            strategy_name=outcome.strategy_name,
                            bias=outcome.bias,
                            candidate=outcome.candidate,
                            evaluation=outcome.evaluation,
                            order_request=outcome.order_request,
                            reason_codes=reason_codes,
                            payload=outcome.payload,
                        ),
                        news_blocked=news_blocked,
                    )
                )
                if final_decision.get(epic, "NO_SIGNAL") != "PLACE_LIMIT":
                    for blocker in reason_codes:
                        daily_summary.blockers[blocker] += 1
                h1_new, h1_closed, m15_new, m15_closed, m5_new, m5_closed = close_meta.get(
                    epic,
                    (False, None, False, None, False, None),
                )
                trace = _build_trace(
                    state=state,
                    now=now,
                    h1_last_closed=h1_closed,
                    h1_new_close=h1_new,
                    m15_last_closed=m15_closed,
                    m15_new_close=m15_new,
                    m5_last_closed=m5_closed,
                    m5_new_close=m5_new,
                    strategy_name=outcome.strategy_name,
                    evaluation=outcome.evaluation,
                    final_decision=final_decision.get(epic, "NO_SIGNAL"),
                    reasons=reason_codes,
                )
                signature = _trace_signature(trace)
                should_log = h1_new or m15_new or m5_new or (signature != state.last_trace_signature)
                if should_log and config.monitoring.log_decision_reasons:
                    if args.state_log == "json":
                        LOGGER.info("%s", trace_to_json(trace, config.timezone))
                    else:
                        LOGGER.info("%s", format_trace_text(trace, config.timezone))
                state.last_trace_signature = signature

            mono = time.monotonic()
            if (mono - last_heartbeat) >= config.execution.heartbeat_seconds:
                pending = len(order_executor.get_pending_orders())
                opened = len(position_manager.get_open_positions())
                pnl = journal.get_daily_stats(day, epic="GLOBAL").pnl
                retries = 0
                http_429 = 0
                requests_total = 0
                miss_rates = "-"
                scalp_plugin = strategy_plugins.get("SCALP_ICT_PA")
                if isinstance(scalp_plugin, ScalpIctPriceActionStrategy):
                    parts: list[str] = []
                    for epic, state in states.items():
                        if state.strategy_name != "SCALP_ICT_PA":
                            continue
                        rate = scalp_plugin.missed_opportunity_rate(epic)
                        if rate is None:
                            continue
                        parts.append(f"{epic}:{rate:.2%}")
                    if parts:
                        miss_rates = ",".join(parts)
                if client is not None:
                    metrics = client.metrics_snapshot()
                    retries = metrics.get("total_retries", 0) - daily_summary.api_retries_start
                    http_429 = metrics.get("http_429_count", 0) - daily_summary.api_429_start
                    requests_total = metrics.get("total_requests", 0) - daily_summary.api_requests_start
                LOGGER.info(
                    "Heartbeat cycle=%d open_positions=%d pending_orders=%d daily_pnl=%.2f top_blockers=%s miss_rate=%s api_requests=%d retries=%d http429=%d",
                    cycle,
                    opened,
                    pending,
                    pnl,
                    daily_summary.top_blockers(),
                    miss_rates,
                    requests_total,
                    retries,
                    http_429,
                )
                last_heartbeat = mono
                # Periodic signal candidate aggregation
                if sc_aggregator is not None:
                    sc_aggregator.maybe_log_summary(session_start_utc, now)

            if (mono - last_dashboard) >= config.monitoring.dashboard_interval_seconds:
                dashboard_writer.write(
                    {
                        "mode": "multi-strategy",
                        "trading_day": day,
                        "global_daily_pnl": round(journal.get_daily_stats(day, epic="GLOBAL").pnl, 4),
                        "open_positions": len(position_manager.get_open_positions()),
                        "pending_orders": len(order_executor.get_pending_orders()),
                        "assets": {
                            epic: {
                                "strategy": st.strategy_name,
                                "trade_enabled": st.asset.trade_enabled,
                                "last_reasons": st.last_reason_codes,
                                "score_total": st.last_evaluation.score_total if st.last_evaluation is not None else None,
                            }
                            for epic, st in states.items()
                        },
                    }
                )
                last_dashboard = mono

        except CapitalAPIError as exc:
            LOGGER.error("Capital API error: %s", exc)
            alerts.send(event="CAPITAL_API_ERROR", level="error", message=str(exc), dedupe_key=f"api-{type(exc).__name__}")
        except Exception:
            LOGGER.exception("Unhandled cycle error")
            alerts.send(event="UNHANDLED_RUNTIME_ERROR", level="error", message="Unhandled exception in main loop", dedupe_key="runtime-unhandled")

        stop_event.wait(config.execution.loop_seconds)

    _log_daily_summary(summary=daily_summary, client=client)
    LOGGER.info("Bot stopped.")


def _run_live_body(
    args: argparse.Namespace,
    config: "AppConfig",
    assets: list,
    root: Path,
    paper_mode: bool,
    mode_dry_run: bool,
    conn: object,
    db_path: object,
) -> None:
    """Inner body of the live/paper/dry-run mode, extracted so that *run()*
    can wrap the SQLite *conn* in a ``try / finally`` for guaranteed cleanup."""
    init_db(conn)
    init_signal_candidates_table(conn)
    journal = Journal(conn)
    sc_logger = SignalCandidateLogger(conn)
    sc_aggregator = SignalCandidateAggregator(conn, interval_seconds=600)
    micro_loss_metrics = MicroLossMetrics()
    LOGGER.info("SQLite state path: %s", db_path)

    client = build_client(config, paper_mode)
    market_data = MarketDataService(client=client, config=config, journal=journal)
    news_provider = build_news_provider(config, root)
    risk_engine = RiskEngine(config.risk)
    strategy_router = StrategyRouter(config)
    strategy_plugins: dict[str, StrategyPlugin] = {
        "INDEX_EXISTING": IndexExistingStrategy(config),
        "SCALP_ICT_PA": ScalpIctPriceActionStrategy(config),
        "ORB_H4_RETEST": OrbH4RetestStrategy(config),
        "TREND_PULLBACK_M15": TrendPullbackM15Strategy(config),
    }
    orderflow_provider: OrderflowProvider = CompositeOrderflowProvider(
        default_mode=config.orderflow.default_mode,
        symbol_modes={symbol: "FULL" for symbol in config.orderflow.full_symbols},
    )
    portfolio_supervisor = PortfolioSupervisor(config.portfolio)
    order_executor = OrderExecutor(client=client, journal=journal, dry_run=mode_dry_run, default_epic=assets[0].epic, default_currency=assets[0].currency)
    position_manager = PositionManager(
        client=client, journal=journal, dry_run=mode_dry_run,
        multi_tp=build_multi_tp_profile(config.multi_tp),
    )
    dashboard_writer = DashboardWriter(os.getenv("DASHBOARD_PATH", config.monitoring.dashboard_path))
    alerts = build_alert_dispatcher(config)

    LOGGER.info("Starting bot | mode=%s | primary=%s | timezone=%s", "paper" if paper_mode else "dry-run", assets[0].epic, config.timezone)

    if args.test_order:
        place_single_test_order(order_executor, market_data, assets, config, mode_dry_run, args.test_side, args.test_size, args.test_epic)
        LOGGER.info("Test-order mode completed. Exiting.")
        return

    tf_history = _timeframe_history(config)
    close_grace_seconds = int(
        os.getenv("CANDLE_CLOSE_GRACE_SECONDS", str(config.execution.candle_close_grace_seconds))
    )
    candle_retry_seconds = int(
        os.getenv("CANDLE_RETRY_SECONDS", str(config.execution.candle_retry_seconds))
    )
    sync_pending_seconds = int(
        os.getenv("SYNC_PENDING_SECONDS", str(config.execution.sync_pending_seconds))
    )
    sync_positions_seconds = int(
        os.getenv("SYNC_POSITIONS_SECONDS", str(config.execution.sync_positions_seconds))
    )
    states = {
        a.epic: AssetRuntimeState(
            asset=a,
            strategy_name=strategy_router.route_for(a.epic).strategy,
            last_processed_closed_ts={
                config.timeframes.h1: None,
                config.timeframes.m15: None,
                config.timeframes.m5: None,
            },
            last_poll_target_ts={
                config.timeframes.h1: None,
                config.timeframes.m15: None,
                config.timeframes.m5: None,
            },
            last_poll_attempt_at={
                config.timeframes.h1: None,
                config.timeframes.m15: None,
                config.timeframes.m5: None,
            },
        )
        for a in assets
    }

    run_multi_strategy_loop(
        args=args,
        config=config,
        journal=journal,
        states=states,
        client=client,
        market_data=market_data,
        news_provider=news_provider,
        risk_engine=risk_engine,
        strategy_router=strategy_router,
        strategy_plugins=strategy_plugins,
        orderflow_provider=orderflow_provider,
        portfolio_supervisor=portfolio_supervisor,
        order_executor=order_executor,
        position_manager=position_manager,
        dashboard_writer=dashboard_writer,
        alerts=alerts,
        close_grace_seconds=close_grace_seconds,
        candle_retry_seconds=candle_retry_seconds,
        sync_pending_seconds=sync_pending_seconds,
        sync_positions_seconds=sync_positions_seconds,
        tf_history=tf_history,
        sc_logger=sc_logger,
        sc_aggregator=sc_aggregator,
        micro_loss_metrics=micro_loss_metrics,
    )

    # Export diagnostics on shutdown if requested
    if args.diagnostics_export:
        export_diagnostics(
            conn,
            Path(args.diagnostics_export),
            fmt=args.diagnostics_format,
        )

    # Stop Monte Carlo viewer if running
    _maybe_stop_mc_viewer()


def run() -> None:
    args = parse_args()
    mode_dry_run = True if (not args.paper and not args.dry_run) else args.dry_run
    paper_mode = args.paper
    load_dotenv()
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))

    root = Path(__file__).resolve().parent
    config_path = _resolve_config_path(root, str(args.config))
    config = load_config(config_path)
    _apply_cli_overrides(args, config)

    if bool(args.ops_healthcheck):
        payload = run_ops_healthcheck(root=root, config=config)
        LOGGER.info("Ops healthcheck: %s", json.dumps(payload, ensure_ascii=True))
        if payload.get("status") != "ok":
            raise RuntimeError("Ops healthcheck failed")
        return

    if bool(args.ops_backup_now):
        payload = run_backup_now(root=root, config=config)
        LOGGER.info("Ops backup-now: %s", json.dumps(payload, ensure_ascii=True))
        if payload.get("status") != "ok":
            raise RuntimeError("Ops backup-now failed")
        return

    if args.ops_restore_verify:
        backup_dir = Path(str(args.ops_restore_verify)).expanduser()
        if not backup_dir.is_absolute():
            backup_dir = (root / backup_dir).resolve()
        payload = run_restore_verify(backup_dir=backup_dir)
        LOGGER.info("Ops restore-verify: %s", json.dumps(payload, ensure_ascii=True))
        if payload.get("status") != "ok":
            raise RuntimeError("Ops restore-verify failed")
        return

    assets = build_asset_universe(config)
    LOGGER.info("Assets configured: %s", ",".join(f"{a.epic}{'' if a.trade_enabled else '(observe)'}" for a in assets))
    LOGGER.info(
        "Market hours config | default=%s overrides=%s",
        config.market_hours.default_profile,
        ",".join(f"{symbol}:{profile}" for symbol, profile in sorted(config.market_hours.symbol_profiles.items())) or "none",
    )
    LOGGER.info(
        "Ops config | heartbeat_stale=%ds watchdog_interval=%ds alert_cooldown=%ds backup_retention=%dd verify_on_create=%s required_services=%s",
        int(config.ops.heartbeat_stale_seconds),
        int(config.ops.watchdog_interval_seconds),
        int(config.ops.alert_cooldown_seconds),
        int(config.ops.backup_retention_days),
        bool(config.ops.backup_verify_on_create),
        ",".join(config.ops.required_services) if config.ops.required_services else "none",
    )
    LOGGER.info(
        "Currency config | account=%s fx_fee_rate=%.6f fx_mode=%s fx_source=%s fx_apply_to=%s reporting=%s",
        config.account_currency,
        float(config.fx_conversion_fee_rate),
        config.fx_fee_mode,
        config.fx_rate_source,
        ",".join(config.fx_apply_to),
        config.reporting_currency,
    )
    LOGGER.info(
        "Research config | objective=%s dd_cap_pct=%.2f dd_cap_basis=%s min_trades_oos=%d workers=%d deterministic=%s",
        config.research.objective_mode,
        float(config.research.dd_cap_pct),
        config.research.dd_cap_basis,
        int(config.research.min_trades_oos),
        int(config.research.max_workers),
        bool(config.backtest_runtime.deterministic),
    )
    LOGGER.info(
        "Research optimize config | enabled=%s budget=%s split_ratio_is=%.2f workers=%d objective=%s dd_cap_basis=%s",
        bool(config.research.optimize.enabled),
        config.research.optimize.runtime_budget,
        float(config.research.optimize.split_ratio_is),
        int(config.research.optimize.max_workers),
        config.research.optimize.objective_mode,
        config.research.optimize.dd_cap_basis,
    )

    if args.batch_worker:
        run_batch_worker_mode(args, config, assets, root)
        return

    if args.batch_backtest:
        run_batch_backtest_mode(args, config, root)
        return

    if args.research_run and args.research_optimize:
        raise RuntimeError("Use either --research-run or --research-optimize, not both.")

    if args.research_run:
        run_research_mode(args, config, assets, root)
        return

    if args.research_optimize:
        run_research_optimize_mode(args, config, assets, root)
        return

    if args.backtest:
        run_backtest_mode(args, config, assets, root)
        return

    # Start Monte Carlo live viewer for live/paper/dry-run modes
    _maybe_start_mc_viewer(config, root, cli_override=getattr(args, "mc_viewer", None))

    db_path = resolve_db_path(root, paper_mode=paper_mode)
    conn = get_connection(db_path)
    try:  # try/finally guarantees conn.close() even on crash
        _run_live_body(args, config, assets, root, paper_mode, mode_dry_run, conn, db_path)
    finally:
        try:
            conn.close()
            LOGGER.debug("SQLite connection closed.")
        except Exception:
            LOGGER.debug("SQLite connection already closed or failed to close.")


if __name__ == "__main__":
    run()
