from __future__ import annotations

import argparse
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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
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
from bot.backtest.runner import BacktestRunner
from bot.clock import (
    is_trading_weekday,
    should_poll_closed_candle,
    trading_day,
    utc_now,
)
from bot.config import AppConfig, AssetConfig, load_config
from bot.data.capital_client import CapitalAPIError, CapitalClient
from bot.data.market_data import MarketDataService
from bot.execution.orders import OrderExecutor
from bot.execution.position_manager import PositionManager
from bot.execution.sizing import position_size_from_risk
from bot.monitoring.alerts import AlertConfig, AlertDispatcher
from bot.monitoring.dashboard import DashboardWriter
from bot.news.calendar_provider import CalendarProvider, build_calendar_provider
from bot.news.gate import is_blocked, should_cancel_pending
from bot.reporting.backtest_reporter import BacktestMeta, BacktestReporter, BacktestRun
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup-days", type=int, default=60)
    parser.add_argument("--out-root", default="runs/batch")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--data-root", default=None, help="Batch mode alias for --backtest-data-root")

    parser.add_argument("--state-log", choices=["text", "json"], default="text")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
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


def _asset_from_template(epic: str, template: AssetConfig, trade_enabled: bool) -> AssetConfig:
    return AssetConfig(
        epic=epic,
        currency=template.currency,
        point_size=template.point_size,
        minimal_tick_buffer=template.minimal_tick_buffer,
        min_size=template.min_size,
        size_step=template.size_step,
        trade_enabled=trade_enabled,
    )


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


def _tp2_r_for_target_total_r(
    *,
    target_total_r: float,
    tp1_trigger_r: float,
    tp1_fraction: float,
    mode: str = "strict_tp_price",
) -> float:
    total_r = max(0.1, float(target_total_r))
    mode_norm = str(mode).strip().lower()
    if mode_norm == "strict_tp_price":
        return total_r
    frac = max(0.0, min(0.99, float(tp1_fraction)))
    trigger_r = max(0.0, float(tp1_trigger_r))
    if frac <= 0.0:
        return total_r
    tp2_r = (total_r - (frac * trigger_r)) / max(1e-9, 1.0 - frac)
    return max(total_r, tp2_r)


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
    except Exception:
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
        report = run_backtest_multi_strategy(
            config=config,
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
        )
        reports.append(report)

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
    return merged, segment_meta


def run_backtest_mode(args: argparse.Namespace, config: AppConfig, assets: list[AssetConfig], root: Path) -> None:
    report_formats = _parse_report_formats(args.report_formats)
    reporter = BacktestReporter(_resolve_runtime_path(root, str(args.report_dir))) if args.report else None
    generated_report_dirs: list[Path] = []

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
            extra={"source_report": payload},
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
        if args.walk_forward:
            report = run_walk_forward_from_csv(
                config=config,
                asset=selected,
                csv_path=csv_path,
                wf_splits=args.wf_splits,
                assumed_spread=args.backtest_spread,
                slippage_points=args.backtest_slippage_points,
                slippage_atr_multiplier=args.backtest_slippage_atr_multiplier,
            )
            LOGGER.info("Walk-forward report: %s", json.dumps(report.to_dict(), indent=2, ensure_ascii=True))
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
                variant_code=_first_variant_code(args.backtest_variants),
                mode="walk-forward",
                trades=list(aggregate_report.trade_log),
                payload=report.to_dict(),
                data_root_value=str(csv_path),
            )
        else:
            report = run_backtest_from_csv(
                config=config,
                asset=selected,
                csv_path=csv_path,
                assumed_spread=args.backtest_spread,
                slippage_points=args.backtest_slippage_points,
                slippage_atr_multiplier=args.backtest_slippage_atr_multiplier,
            )
            LOGGER.info("Backtest report: %s", json.dumps(report.to_dict(), indent=2, ensure_ascii=True))
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
                variant_code=_first_variant_code(args.backtest_variants),
                mode="backtest",
                trades=list(report.trade_log),
                payload=report.to_dict(),
                data_root_value=str(csv_path),
            )
        _maybe_open_reports()
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

    def _run_auto() -> dict[str, dict[str, object]]:
        payloads: dict[str, dict[str, object]] = {}
        for variant in variants:
            reports: dict[str, dict[str, object]] = {}
            for symbol in symbols:
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

                debug_file = reports_dir / f"{variant.code}_debug_exec_{symbol}.jsonl"
                no_price_debug_file = reports_dir / f"{variant.code}_debug_no_price_{symbol}.jsonl"
                reaction_timeout_debug_file = reports_dir / f"{variant.code}_debug_reaction_timeout_{symbol}.jsonl"
                data_context = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "price_mode_requested": args.backtest_price,
                    "spread_mode": spread_mode,
                    "assumed_spread_used": float(assumed_spread),
                    **(loaded.diagnostics if isinstance(getattr(loaded, "diagnostics", None), dict) else {}),
                }
                if args.walk_forward:
                    report = run_walk_forward_multi_strategy(
                        config=config,
                        asset=asset_map[symbol],
                        candles_m5=BacktestRunner._frame_to_candles(frame),
                        wf_splits=args.wf_splits,
                        assumed_spread=assumed_spread,
                        slippage_points=args.backtest_slippage_points,
                        slippage_atr_multiplier=args.backtest_slippage_atr_multiplier,
                        variant=variant,
                        execution_debug_path=debug_file,
                        no_price_debug_path=no_price_debug_file,
                        reaction_timeout_debug_path=reaction_timeout_debug_file,
                        data_context=data_context,
                    )
                    report_dict = report.to_dict()
                    report_trades = list(report.aggregate.trade_log)
                else:
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
                    )
                    report_dict = report.to_dict()
                    report_dict["segment_health"] = segment_meta
                    report_trades = list(report.trade_log)
                report_dict["data_health"] = loaded.diagnostics.get("data_health", {})
                report_dict["price_diagnostics"] = {
                    "price_mode_requested": loaded.diagnostics.get("price_mode_requested", args.backtest_price),
                    "source_datasets": loaded.diagnostics.get("source_datasets", []),
                    "source_files_count": len(loaded.diagnostics.get("source_files", [])),
                    "fallback_counters": loaded.diagnostics.get("fallback_counters", {}),
                    "gap_segments": loaded.diagnostics.get("gap_segments", {}),
                    "spread_mode": spread_mode,
                    "assumed_spread_used": float(report_dict.get("assumed_spread_used", assumed_spread)),
                }
                reports[symbol] = report_dict
                _emit_detailed_report(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_raw=str(args.backtest_start),
                    end_raw=str(args.backtest_end),
                    variant_code=variant.code,
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
                target = reports_dir / filename
                target.write_text(json.dumps(report_dict, indent=2, ensure_ascii=True), encoding="utf-8")

            payloads[variant.code] = {
                "variant": variant.code,
                "mode": "walk-forward" if args.walk_forward else "backtest",
                "symbols": symbols,
                "timeframe": timeframe,
                "price": args.backtest_price,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "data_root": str(data_root),
                "reports": reports,
            }
        return payloads

    attempt = 0
    while True:
        try:
            variant_payloads = _run_auto()
            if len(variant_payloads) == 1:
                payload = next(iter(variant_payloads.values()))
                LOGGER.info("Backtest auto-data report: %s", json.dumps(payload, indent=2, ensure_ascii=True))
            else:
                LOGGER.info("Backtest auto-data variants: %s", ",".join(variant_payloads.keys()))
                _log_variant_comparison(variant_payloads=variant_payloads, symbols=symbols)
            _maybe_open_reports()
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
                "r_multiple": float(trade.r_multiple) if trade.r_multiple is not None else None,
                "score": float(trade.score) if getattr(trade, "score", None) is not None else None,
                "forced_exit": bool(getattr(trade, "forced_exit", False)),
                "reason_open": str(getattr(trade, "reason_open", "SIGNAL") or "SIGNAL"),
                "reason_close": str(getattr(trade, "reason_close", "") or getattr(trade, "reason", "")),
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
        workers=max(1, int(args.workers)),
        warmup_days=max(0, int(args.warmup_days)),
        out_root=out_root,
        initial_equity=float(args.initial_equity),
        continue_on_error=bool(args.continue_on_error),
    )
    LOGGER.info("Batch backtest summary: %s", json.dumps(summary, indent=2, ensure_ascii=True))


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
    return StrategyEvaluation(
        action=DecisionAction.OBSERVE,
        score_total=0.0,
        reasons_blocking=[reason],
        would_enter_if=["VALID_CANDIDATE"],
        snapshot={"symbol": symbol},
    )


def _pick_best_candidate(
    *,
    strategy: StrategyPlugin,
    symbol: str,
    candidates: list[SetupCandidate],
    data: StrategyDataBundle,
) -> tuple[SetupCandidate | None, StrategyEvaluation]:
    if not candidates:
        return None, _default_observe_evaluation(symbol=symbol, reason="NO_CANDIDATE")
    best_candidate = candidates[0]
    best_eval = strategy.evaluate_candidate(symbol, best_candidate, data)
    for candidate in candidates[1:]:
        current = strategy.evaluate_candidate(symbol, candidate, data)
        current_score = current.score_total if current.score_total is not None else -1.0
        best_score = best_eval.score_total if best_eval.score_total is not None else -1.0
        if current.action == DecisionAction.TRADE and best_eval.action != DecisionAction.TRADE:
            best_candidate, best_eval = candidate, current
            continue
        if current.action == best_eval.action and current_score > best_score:
            best_candidate, best_eval = candidate, current
    return best_candidate, best_eval


def _normalize_action_for_score(*, evaluation: StrategyEvaluation, config: AppConfig) -> StrategyEvaluation:
    if evaluation.score_total is None:
        return evaluation
    if evaluation.reasons_blocking:
        evaluation.action = DecisionAction.OBSERVE
        return evaluation
    score = float(evaluation.score_total)
    if score >= config.decision_policy.trade_score_threshold:
        evaluation.action = DecisionAction.TRADE
    elif config.decision_policy.small_score_min <= score <= config.decision_policy.small_score_max:
        evaluation.action = DecisionAction.SMALL
    else:
        evaluation.action = DecisionAction.OBSERVE
        if "SCORE_BELOW_MIN" not in evaluation.reasons_blocking:
            evaluation.reasons_blocking.append("SCORE_BELOW_MIN")
    return evaluation


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _resolve_orderflow_mode(*, symbol: str, route_params: dict[str, object], default_mode: str, full_symbols: set[str]) -> str:
    params = route_params.get("orderflow")
    if isinstance(params, dict):
        mode = str(params.get("mode", "")).strip().upper()
        if mode in {"LITE", "FULL"}:
            return mode
    if symbol.strip().upper() in full_symbols:
        return "FULL"
    mode = default_mode.strip().upper()
    return mode if mode in {"LITE", "FULL"} else "LITE"


def _orderflow_param(
    *,
    route_params: dict[str, object],
    settings: dict[str, float] | None,
    key: str,
    default: float,
) -> float:
    params = route_params.get("orderflow")
    if isinstance(params, dict):
        try:
            if key in params:
                return float(params[key])
        except (TypeError, ValueError):
            pass
    if settings is not None and key in settings:
        try:
            return float(settings[key])
        except (TypeError, ValueError):
            return default
    return default


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
    raw = dict(evaluation.score_breakdown)
    evaluation.metadata["raw_score_breakdown"] = raw

    bias_raw = float(max(raw.get("bias", 0.0), raw.get("trend_strength", 0.0), raw.get("breakout_quality", 0.0)))
    sweep_raw = float(max(raw.get("sweep", 0.0), raw.get("liquidity_setup", 0.0), raw.get("retest_quality", 0.0)))
    mss_raw = float(max(raw.get("mss", 0.0), raw.get("confirmation_strength", 0.0), raw.get("trigger_quality", 0.0)))
    displacement_raw = float(max(raw.get("displacement", 0.0), raw.get("trigger_quality", 0.0), raw.get("confirmation_strength", 0.0)))
    fvg_raw = float(max(raw.get("fvg", 0.0), raw.get("mitigation_quality", 0.0), raw.get("retest_quality", 0.0)))

    bias_regime = _clamp((bias_raw / 20.0) * 15.0, 0.0, 15.0)
    location_score = 10.0
    pd_eq = evaluation.metadata.get("h1_pd_eq", evaluation.snapshot.get("h1_pd_eq"))
    h1_close = evaluation.metadata.get("h1_close", evaluation.snapshot.get("h1_close"))
    side = str(setup_side or evaluation.metadata.get("side", "")).upper()
    if pd_eq is not None and h1_close is not None and side in {"LONG", "SHORT"}:
        try:
            eq_float = float(pd_eq)
            close_float = float(h1_close)
            if side == "LONG":
                location_score = 15.0 if close_float < eq_float else 6.0
            else:
                location_score = 15.0 if close_float > eq_float else 6.0
        except (TypeError, ValueError):
            location_score = 10.0
    if strategy_name == "INDEX_EXISTING":
        location_score = 12.0 if not evaluation.reasons_blocking else 6.0
    liquidity_score = _clamp((sweep_raw / 20.0) * 15.0, 0.0, 15.0)
    edge_score = _clamp(bias_regime + location_score + liquidity_score, 0.0, 45.0)

    mitigation_quality = _clamp((fvg_raw / 15.0) * 15.0, 0.0, 15.0)
    trigger_confirmations = int(evaluation.metadata.get("trigger_confirmations", 0))
    reaction_confirmed = _clamp((trigger_confirmations / 3.0) * 15.0, 0.0, 15.0)
    trigger_clean = _clamp(((mss_raw / 20.0) * 5.0) + ((displacement_raw / 20.0) * 5.0), 0.0, 10.0)
    trigger_score = _clamp(mitigation_quality + reaction_confirmed + trigger_clean, 0.0, 40.0)

    atr_value = evaluation.metadata.get("atr_m5", evaluation.snapshot.get("atr_m5"))
    spread_value = evaluation.snapshot.get("spread")
    spread_ratio = None
    if atr_value is not None and spread_value is not None:
        try:
            atr_float = float(atr_value)
            spread_float = float(spread_value)
            if atr_float > 0:
                spread_ratio = spread_float / atr_float
        except (TypeError, ValueError):
            spread_ratio = None
    max_spread_ratio = 0.15
    gates_cfg = route_params.get("quality_gates")
    if isinstance(gates_cfg, dict):
        max_spread_ratio = float(gates_cfg.get("spread_ratio_max", 0.15))
    if spread_ratio is None:
        spread_ratio_score = 3.0
        slippage_risk_score = 2.0
    elif spread_ratio <= max_spread_ratio:
        spread_ratio_score = 8.0
        slippage_risk_score = 4.0
    elif spread_ratio <= max_spread_ratio * 1.25:
        spread_ratio_score = 4.0
        slippage_risk_score = 2.0
    else:
        spread_ratio_score = 0.0
        slippage_risk_score = 0.0
    market_state_score = 3.0 if schedule_open else 0.0
    execution_score = _clamp(spread_ratio_score + slippage_risk_score + market_state_score, 0.0, 15.0)

    of_trigger_bonus = 0.0
    of_execution_bonus = 0.0
    of_divergence_penalty = 0.0
    if orderflow_snapshot is not None:
        of_dict = orderflow_snapshot.to_dict()
        evaluation.metadata["orderflow_snapshot"] = of_dict
        evaluation.snapshot["orderflow"] = of_dict

        confidence = float(orderflow_snapshot.confidence)
        chop_score = float(orderflow_snapshot.metrics.chop_score)
        of_spread_ratio = float(orderflow_snapshot.metrics.spread_ratio)
        of_pressure = float(orderflow_snapshot.pressure)
        of_direction = str(orderflow_snapshot.direction).upper()

        trigger_bonus_cap = _orderflow_param(
            route_params=route_params,
            settings=orderflow_settings,
            key="trigger_bonus_max",
            default=10.0,
        )
        execution_bonus_cap = _orderflow_param(
            route_params=route_params,
            settings=orderflow_settings,
            key="execution_bonus_max",
            default=5.0,
        )
        divergence_penalty_min = _orderflow_param(
            route_params=route_params,
            settings=orderflow_settings,
            key="divergence_penalty_min",
            default=6.0,
        )
        divergence_penalty_max = _orderflow_param(
            route_params=route_params,
            settings=orderflow_settings,
            key="divergence_penalty_max",
            default=10.0,
        )
        if divergence_penalty_min > divergence_penalty_max:
            divergence_penalty_min, divergence_penalty_max = divergence_penalty_max, divergence_penalty_min

        flow_alignment = _clamp(abs(of_pressure), 0.0, 1.0)
        of_trigger_bonus = _clamp(
            confidence * (1.0 - chop_score) * (0.5 + (0.5 * flow_alignment)) * trigger_bonus_cap,
            0.0,
            trigger_bonus_cap,
        )
        execution_quality = _clamp(1.0 - (of_spread_ratio / max(max_spread_ratio, 1e-9)), 0.0, 1.0)
        of_execution_bonus = _clamp(confidence * execution_quality * execution_bonus_cap, 0.0, execution_bonus_cap)

        if side in {"LONG", "SHORT"} and of_direction in {"LONG", "SHORT"} and of_direction != side:
            of_divergence_penalty = _clamp(
                divergence_penalty_min + ((divergence_penalty_max - divergence_penalty_min) * flow_alignment),
                divergence_penalty_min,
                divergence_penalty_max,
            )
        evaluation.metadata["orderflow_influence"] = {
            "trigger_bonus": round(of_trigger_bonus, 4),
            "execution_bonus": round(of_execution_bonus, 4),
            "divergence_penalty": round(of_divergence_penalty, 4),
            "direction": of_direction,
            "pressure": round(of_pressure, 4),
        }

    penalties: dict[str, float] = {}
    if bias.direction == "NEUTRAL":
        penalties["NEUTRAL_BIAS"] = 5.0
    if bool(evaluation.metadata.get("near_adr_exhausted")):
        penalties["NEAR_ADR_EXHAUSTED"] = 6.0
    if bool(evaluation.metadata.get("news_medium_window")):
        penalties["NEWS_MEDIUM_WINDOW"] = 8.0
    if bool(evaluation.metadata.get("correlation_exposure")):
        penalties["CORRELATION_EXPOSURE"] = 6.0
    if bool(evaluation.metadata.get("late_retest")):
        penalties["LATE_RETEST"] = 5.0
    if news_blocked:
        penalties["NEWS_MEDIUM_WINDOW"] = max(penalties.get("NEWS_MEDIUM_WINDOW", 0.0), 8.0)
    if of_divergence_penalty > 0:
        penalties["OF_DIVERGENCE"] = max(penalties.get("OF_DIVERGENCE", 0.0), of_divergence_penalty)

    for key, value in raw.items():
        if key.startswith("penalty_") and value < 0:
            mapped_key = key.replace("penalty_", "").upper()
            penalties[mapped_key] = max(penalties.get(mapped_key, 0.0), abs(float(value)))

    penalty_total = sum(penalties.values())
    score_pre_penalty = edge_score + trigger_score + execution_score + of_trigger_bonus + of_execution_bonus
    score_total = _clamp(score_pre_penalty - penalty_total, 0.0, 100.0)

    evaluation.score_layers = {
        "edge": round(edge_score, 2),
        "trigger": round(trigger_score, 2),
        "execution": round(execution_score, 2),
        "orderflow": round(of_trigger_bonus + of_execution_bonus, 2),
    }
    evaluation.penalties = {key: round(value, 2) for key, value in penalties.items()}
    evaluation.score_total = round(score_total, 2)
    evaluation.score_breakdown = {
        "edge.bias_regime": round(bias_regime, 2),
        "edge.location": round(location_score, 2),
        "edge.liquidity_setup": round(liquidity_score, 2),
        "trigger.mitigation_quality": round(mitigation_quality, 2),
        "trigger.reaction_confirmed": round(reaction_confirmed, 2),
        "trigger.cleanliness": round(trigger_clean, 2),
        "execution.spread_ratio_score": round(spread_ratio_score, 2),
        "execution.slippage_risk_score": round(slippage_risk_score, 2),
        "execution.market_state_score": round(market_state_score, 2),
        "orderflow.trigger_bonus": round(of_trigger_bonus, 2),
        "orderflow.execution_bonus": round(of_execution_bonus, 2),
        "orderflow.divergence_penalty": round(of_divergence_penalty, 2),
        "edge_total": round(edge_score, 2),
        "trigger_total": round(trigger_score, 2),
        "execution_total": round(execution_score, 2),
        "penalty_total": round(penalty_total, 2),
        "score_pre_penalty": round(score_pre_penalty, 2),
        "score_total": round(score_total, 2),
    }
    if spread_ratio is not None:
        evaluation.metadata["spread_ratio"] = spread_ratio
    return evaluation


def _evaluate_hard_gates(
    *,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    now: datetime,
    timezone_name: str,
) -> tuple[dict[str, bool], list[str]]:
    gates = {
        "ExecutionGate": True,
        "ScheduleGate": True,
        "ReactionGate": True,
        "RiskGate": True,
    }
    reasons: list[str] = []

    schedule_cfg = route_params.get("schedule")
    schedule_open = True
    if isinstance(schedule_cfg, dict):
        schedule_open = is_schedule_open(now, schedule_cfg, timezone_name)
    if not schedule_open:
        gates["ScheduleGate"] = False
        reasons.append("EXEC_FAIL_MARKET_CLOSED")

    gates_cfg = route_params.get("quality_gates")
    if isinstance(gates_cfg, dict):
        max_spread_ratio = float(gates_cfg.get("spread_ratio_max", 0.15))
    else:
        max_spread_ratio = 0.15
    spread_ratio = evaluation.metadata.get("spread_ratio")
    atr_value = evaluation.metadata.get("atr_m5", evaluation.snapshot.get("atr_m5"))
    spread_value = evaluation.snapshot.get("spread")
    if atr_value is None:
        gates["ExecutionGate"] = False
        reasons.append("EXEC_FAIL_MISSING_FEATURES")
    else:
        try:
            if float(atr_value) <= 0:
                gates["ExecutionGate"] = False
                reasons.append("EXEC_FAIL_INVALID_ATR")
        except (TypeError, ValueError):
            gates["ExecutionGate"] = False
            reasons.append("EXEC_FAIL_INVALID_ATR")
    if spread_ratio is None:
        if atr_value is not None and spread_value is not None:
            try:
                atr_float = float(atr_value)
                spread_float = float(spread_value)
                if atr_float > 0:
                    spread_ratio = spread_float / atr_float
                    evaluation.metadata["spread_ratio"] = spread_ratio
            except (TypeError, ValueError):
                spread_ratio = None
    if spread_ratio is None:
        if spread_value is None:
            gates["ExecutionGate"] = False
            reasons.append("EXEC_FAIL_NO_PRICE")
        elif atr_value is not None and (isinstance(atr_value, (float, int)) and float(atr_value) > 0):
            gates["ExecutionGate"] = False
            reasons.append("EXEC_FAIL_MISSING_FEATURES")
    elif float(spread_ratio) > max_spread_ratio:
        gates["ExecutionGate"] = False
        reasons.append("EXEC_FAIL_SPREAD_TOO_HIGH")

    setup_state = str(evaluation.metadata.get("setup_state", "READY")).upper()
    if setup_state in {"WAIT_MITIGATION", "WAIT_REACTION"}:
        gates["ReactionGate"] = False
        reasons.append(f"GATE_REACTION_{setup_state}")

    evaluation.gates = gates
    if reasons:
        for key, value in gates.items():
            if not value:
                evaluation.gate_blocked = key
                break
    return gates, reasons


def _quality_gate_reasons(
    *,
    symbol: str,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    now: datetime,
    timezone_name: str,
) -> list[str]:
    del symbol
    _, reasons = _evaluate_hard_gates(
        route_params=route_params,
        evaluation=evaluation,
        now=now,
        timezone_name=timezone_name,
    )
    return reasons


def _apply_orderflow_small_soft_gate(
    *,
    route_params: dict[str, object],
    evaluation: StrategyEvaluation,
    orderflow_settings: dict[str, float] | None,
) -> StrategyEvaluation:
    if evaluation.action != DecisionAction.SMALL:
        return evaluation
    snapshot_raw = evaluation.metadata.get("orderflow_snapshot")
    if not isinstance(snapshot_raw, dict):
        return evaluation
    metrics = snapshot_raw.get("metrics")
    if not isinstance(metrics, dict):
        return evaluation

    try:
        confidence = float(snapshot_raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        chop_score = float(metrics.get("chop_score", 0.0))
    except (TypeError, ValueError):
        chop_score = 0.0

    conf_threshold = _orderflow_param(
        route_params=route_params,
        settings=orderflow_settings,
        key="small_soft_gate_confidence",
        default=0.75,
    )
    chop_threshold = _orderflow_param(
        route_params=route_params,
        settings=orderflow_settings,
        key="small_soft_gate_chop",
        default=0.75,
    )
    if confidence >= conf_threshold and chop_score >= chop_threshold:
        evaluation.action = DecisionAction.OBSERVE
        if "OF_SOFT_GATE_CHOP" not in evaluation.reasons_blocking:
            evaluation.reasons_blocking.append("OF_SOFT_GATE_CHOP")
        if "ORDERFLOW_CHOP_CLEARED" not in evaluation.would_enter_if:
            evaluation.would_enter_if.append("ORDERFLOW_CHOP_CLEARED")
        evaluation.gates.setdefault("OrderflowSoftGate", True)
        evaluation.gates["OrderflowSoftGate"] = False
        if evaluation.gate_blocked is None:
            evaluation.gate_blocked = "OrderflowSoftGate"
    return evaluation


def _risk_multiplier_for(
    *,
    evaluation: StrategyEvaluation,
    route_risk: dict[str, object],
    config: AppConfig,
) -> float:
    signal_override = evaluation.metadata.get("risk_multiplier_override")
    if signal_override is not None:
        return max(0.01, min(1.0, float(signal_override)))
    if evaluation.action == DecisionAction.SMALL:
        value = float(route_risk.get("small_risk_multiplier", config.decision_policy.small_risk_multiplier_default))
        return max(0.01, min(1.0, value))
    value = float(route_risk.get("trade_risk_multiplier", config.decision_policy.trade_risk_multiplier))
    return max(0.01, min(1.0, value))


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
        try:
            if not is_trading_weekday(now, config.timezone):
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

            pending_intents: list[PendingOrderIntent] = []
            close_meta: dict[str, tuple[bool, datetime | None, bool, datetime | None, bool, datetime | None]] = {}
            final_decision: dict[str, str] = {}

            for epic, state in states.items():
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
                        _ = _normalize_action_for_score(evaluation=evaluation, config=config)
                        gate_reasons = _quality_gate_reasons(
                            symbol=epic,
                            route_params=route.params,
                            evaluation=evaluation,
                            now=now,
                            timezone_name=config.timezone,
                        )
                        if gate_reasons:
                            evaluation.action = DecisionAction.OBSERVE
                            for code in gate_reasons:
                                if code not in evaluation.reasons_blocking:
                                    evaluation.reasons_blocking.append(code)
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

                if best_outcome.order_request is None:
                    continue
                daily_summary.signal_candidates += 1
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
                risk_check = risk_engine.can_open_new_trade_multi(
                    now=now,
                    asset_epic=intent.symbol,
                    asset_stats=intent.asset_stats_snapshot,
                    global_stats=global_stats,
                    asset_open_positions=open_asset_now,
                    all_open_positions=open_all_now,
                    new_trade_risk_amount=risk_engine.per_trade_risk_amount() * intent.risk_multiplier,
                    cooldown_until=cooldown,
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

                size = position_size_from_risk(
                    equity=config.risk.equity,
                    risk_per_trade=config.risk.risk_per_trade * intent.risk_multiplier,
                    entry_price=intent.signal.entry_price,
                    stop_price=intent.signal.stop_price,
                    min_size=intent.state.asset.min_size,
                    size_step=intent.state.asset.size_step,
                )
                if size <= 0:
                    intent.outcome.reason_codes.append("SIZE_INVALID")
                    intent.outcome.reason_codes.append("RISK_GATE_SIZE_INVALID")
                    intent.outcome.evaluation.gates.setdefault("RiskGate", True)
                    intent.outcome.evaluation.gates["RiskGate"] = False
                    if intent.outcome.evaluation.gate_blocked is None:
                        intent.outcome.evaluation.gate_blocked = "RiskGate"
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

                key = (
                    f"{intent.symbol}:{intent.signal.side}:{intent.signal.entry_price:.5f}:"
                    f"{intent.signal.expires_at.isoformat()}"
                )
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

            for epic, state in states.items():
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


def run() -> None:
    args = parse_args()
    mode_dry_run = True if (not args.paper and not args.dry_run) else args.dry_run
    paper_mode = args.paper
    load_dotenv()
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))

    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_config(config_path)
    assets = build_asset_universe(config)
    LOGGER.info("Assets configured: %s", ",".join(f"{a.epic}{'' if a.trade_enabled else '(observe)'}" for a in assets))

    if args.batch_worker:
        run_batch_worker_mode(args, config, assets, root)
        return

    if args.batch_backtest:
        run_batch_backtest_mode(args, config, root)
        return

    if args.backtest:
        run_backtest_mode(args, config, assets, root)
        return

    db_path = resolve_db_path(root, paper_mode=paper_mode)
    conn = get_connection(db_path)
    init_db(conn)
    journal = Journal(conn)
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
    position_manager = PositionManager(client=client, journal=journal, dry_run=mode_dry_run)
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
    )
    return

    metrics_start = client.metrics_snapshot() if client is not None else {}
    daily_summary = DailyRuntimeSummary(
        trading_day=trading_day(utc_now(), config.timezone).isoformat(),
        api_requests_start=metrics_start.get("total_requests", 0),
        api_retries_start=metrics_start.get("total_retries", 0),
        api_429_start=metrics_start.get("http_429_count", 0),
    )

    stop_event = threading.Event()
    last_heartbeat = time.monotonic()
    last_dashboard = time.monotonic()
    last_pending_sync_at: datetime | None = None
    last_positions_sync_at: datetime | None = None

    def _stop(signum: int, _frame: object) -> None:
        LOGGER.info("Received signal %s, shutting down.", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)

    cycle = 0
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
        try:
            if not is_trading_weekday(now, config.timezone):
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

            if should_refresh_quote(
                now=now,
                last_fetch_at=last_pending_sync_at,
                interval_seconds=sync_pending_seconds,
            ):
                order_executor.sync_remote_pending_orders()
                last_pending_sync_at = now

            closed_sync: list[ClosedPositionEvent] = []
            if should_refresh_quote(
                now=now,
                last_fetch_at=last_positions_sync_at,
                interval_seconds=sync_positions_seconds,
            ):
                closed_sync = position_manager.sync_positions_from_api()
                last_positions_sync_at = now

            quotes: dict[str, tuple[float, float, float]] = {}
            for epic, state in states.items():
                quote_interval = _quote_refresh_interval_seconds(
                    config=config,
                    trade_enabled=state.asset.trade_enabled,
                )
                if should_refresh_quote(
                    now=now,
                    last_fetch_at=state.quote_last_fetch_at,
                    interval_seconds=quote_interval,
                ):
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

            for epic, state in states.items():
                asset_stats = journal.get_daily_stats(day, epic=epic)
                if risk_engine.should_turn_off_for_day(asset_stats.pnl) and asset_stats.status != "OFF":
                    journal.set_daily_status(day, "OFF", epic=epic)
                    alerts.send(
                        event="DAILY_STOP_ASSET",
                        level="warning",
                        message=f"{epic} daily stop triggered pnl={asset_stats.pnl:.2f}",
                        dedupe_key=f"daily-stop-{epic}-{day}",
                    )

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

                if h1_new and config.timeframes.h1 in state.cache:
                    h1_closed_candles = closed_candles(state.cache[config.timeframes.h1])
                    if h1_closed_candles:
                        state.h1_snapshot = strategy_engine.evaluate_h1(h1_closed_candles)

                if m15_new and config.timeframes.m15 in state.cache:
                    m15_closed_candles = closed_candles(state.cache[config.timeframes.m15])
                    if m15_closed_candles:
                        if state.h1_snapshot is None and config.timeframes.h1 in state.cache:
                            h1_closed_candles = closed_candles(state.cache[config.timeframes.h1])
                            if h1_closed_candles:
                                state.h1_snapshot = strategy_engine.evaluate_h1(h1_closed_candles)
                        if state.h1_snapshot is not None:
                            state.m15_snapshot = strategy_engine.evaluate_m15(
                                candles_m15=m15_closed_candles,
                                h1=state.h1_snapshot,
                                minimal_tick_buffer=state.asset.minimal_tick_buffer,
                                now=now,
                                previous=state.m15_snapshot,
                            )

                open_asset = position_manager.get_open_positions(epic=epic)
                pending_asset = order_executor.get_pending_orders(epic=epic)
                state.entry_state = derive_entry_state(
                    state.entry_state,
                    has_open=bool(open_asset),
                    has_pending=bool(pending_asset),
                )
                if state.m5_snapshot is None:
                    state.m5_snapshot = M5Snapshot(
                        entry_state=state.entry_state,
                        mss_ok=False,
                        displacement_ok=False,
                        fvg_ok=False,
                        fvg_range=None,
                        fvg_mid=None,
                        limit_price=None,
                        reason_codes=["M5_WAIT_NEW_CLOSE"],
                    )
                else:
                    state.m5_snapshot.entry_state = state.entry_state

                extra_reasons: list[str] = []
                final_decision = "MANAGE" if state.entry_state == "FILLED" else "NO_SIGNAL"
                decision: StrategyDecision | None = None

                if m5_closed is None:
                    state.stale_data = False
                    extra_reasons.append("M5_NO_CLOSED_BAR")
                elif (now - m5_closed).total_seconds() > config.execution.max_data_stale_seconds:
                    state.stale_data = True
                    extra_reasons.append("DATA_STALE")
                else:
                    state.stale_data = False

                q = quotes.get(epic)
                spread = q[2] if q is not None else None
                if q is None:
                    extra_reasons.append("QUOTE_MISSING")

                if m5_new and not state.stale_data and q is not None:
                    if state.h1_snapshot is None and config.timeframes.h1 in state.cache:
                        h1_closed_candles = closed_candles(state.cache[config.timeframes.h1])
                        if h1_closed_candles:
                            state.h1_snapshot = strategy_engine.evaluate_h1(h1_closed_candles)
                    if state.m15_snapshot is None and config.timeframes.m15 in state.cache and state.h1_snapshot is not None:
                        m15_closed_candles = closed_candles(state.cache[config.timeframes.m15])
                        if m15_closed_candles:
                            state.m15_snapshot = strategy_engine.evaluate_m15(
                                candles_m15=m15_closed_candles,
                                h1=state.h1_snapshot,
                                minimal_tick_buffer=state.asset.minimal_tick_buffer,
                                now=now,
                                previous=state.m15_snapshot,
                            )

                    m5_closed_candles = closed_candles(state.cache.get(config.timeframes.m5, []))
                    if state.h1_snapshot is None or state.m15_snapshot is None or not m5_closed_candles:
                        extra_reasons.append("PIPELINE_INSUFFICIENT_DATA")
                    else:
                        decision, m5_snapshot = strategy_engine.evaluate_m5(
                            epic=epic,
                            candles_m5=m5_closed_candles,
                            current_spread=spread,
                            spread_history=market_data.spread_history(epic),
                            news_blocked=news_blocked,
                            h1=state.h1_snapshot,
                            m15=state.m15_snapshot,
                            entry_state=state.entry_state,
                        )
                        state.m5_snapshot = m5_snapshot
                        state.m5_snapshot.entry_state = state.entry_state

                        if decision.signal is None:
                            final_decision = "MANAGE" if state.entry_state == "FILLED" else "NO_SIGNAL"
                            journal.log_decision(create_decision_record(decision, epic, None, news_blocked))
                        else:
                            final_decision = "PLACE_LIMIT_PENDING_CHECKS"
                            if not state.asset.trade_enabled:
                                decision.reason_codes.append("OBSERVE_ONLY_ASSET")
                                journal.log_decision(create_decision_record(decision, epic, decision.signal.side, news_blocked))
                                final_decision = "NO_SIGNAL"
                            elif pending_asset:
                                decision.reason_codes.append("PENDING_EXISTS")
                                journal.log_decision(create_decision_record(decision, epic, decision.signal.side, news_blocked))
                                final_decision = "WAIT_LIMIT_FILL"
                            else:
                                safe_mode_multiplier = (
                                    config.strategy_runtime.neutral_bias_risk_multiplier
                                    if state.h1_snapshot is not None and state.h1_snapshot.safe_mode
                                    else 1.0
                                )
                                global_stats = journal.get_daily_stats(day, epic="GLOBAL")
                                cooldown = journal.get_risk_state(f"ASSET:{epic}").cooldown_until
                                open_all = position_manager.get_open_positions()
                                risk_check = risk_engine.can_open_new_trade_multi(
                                    now=now,
                                    asset_epic=epic,
                                    asset_stats=asset_stats,
                                    global_stats=global_stats,
                                    asset_open_positions=open_asset,
                                    all_open_positions=open_all,
                                    new_trade_risk_amount=risk_engine.per_trade_risk_amount() * safe_mode_multiplier,
                                    cooldown_until=cooldown,
                                )
                                if not risk_check.allowed:
                                    decision.reason_codes.extend([c for c in risk_check.reason_codes if c not in decision.reason_codes])
                                    decision.payload["risk"] = risk_check.metadata
                                    journal.log_decision(create_decision_record(decision, epic, decision.signal.side, news_blocked))
                                    final_decision = "NO_SIGNAL"
                                else:
                                    if not _apply_rr_profile_to_signal(
                                        decision.signal,
                                        tp1_trigger_r=float(config.backtest_tuning.tp1_trigger_r),
                                        tp1_fraction=float(config.backtest_tuning.tp1_fraction),
                                        tp_profile_mode=str(config.backtest_tuning.tp_profile_mode),
                                    ):
                                        decision.reason_codes.append("ORDER_INVALID_RISK")
                                        journal.log_decision(create_decision_record(decision, epic, decision.signal.side, news_blocked))
                                        final_decision = "NO_SIGNAL"
                                        continue
                                    size = position_size_from_risk(
                                        equity=config.risk.equity,
                                        risk_per_trade=config.risk.risk_per_trade * safe_mode_multiplier,
                                        entry_price=decision.signal.entry_price,
                                        stop_price=decision.signal.stop_price,
                                        min_size=state.asset.min_size,
                                        size_step=state.asset.size_step,
                                    )
                                    if safe_mode_multiplier < 1.0:
                                        decision.payload["safe_mode_risk_multiplier"] = safe_mode_multiplier
                                    if size <= 0:
                                        decision.reason_codes.append("SIZE_INVALID")
                                        journal.log_decision(create_decision_record(decision, epic, decision.signal.side, news_blocked))
                                        final_decision = "NO_SIGNAL"
                                    else:
                                        key = f"{epic}:{decision.signal.side}:{decision.signal.entry_price:.5f}:{decision.signal.expires_at.isoformat()}"
                                        order = order_executor.place_limit_order(decision.signal, size=size, epic=epic, currency=state.asset.currency, idempotency_key=key)
                                        journal.increment_daily_trades(day, epic=epic)
                                        journal.increment_daily_trades(day, epic="GLOBAL")
                                        decision.reason_codes.append("ORDER_PLACED")
                                        journal.log_decision(create_decision_record(decision, epic, decision.signal.side, news_blocked))
                                        state.entry_state = "ORDER_PLACED"
                                        if state.m5_snapshot is not None:
                                            state.m5_snapshot.entry_state = "ORDER_PLACED"
                                        LOGGER.info("Placed LIMIT %s %s order id=%s size=%.4f", epic, decision.signal.side, order.order_id, order.size)
                                        alerts.send(
                                            event="ORDER_PLACED",
                                            level="info",
                                            message=f"{epic} {decision.signal.side} id={order.order_id} size={order.size:.4f}",
                                            dedupe_key=f"placed-{order.order_id}",
                                        )
                                        final_decision = "PLACE_LIMIT"
                elif not m5_new:
                    extra_reasons.append("M5_WAIT_NEW_CLOSE")
                    if state.entry_state == "ORDER_PLACED":
                        final_decision = "WAIT_LIMIT_FILL"

                reason_codes: list[str] = []
                if decision is not None:
                    reason_codes.extend(decision.reason_codes)
                else:
                    if state.h1_snapshot is not None:
                        reason_codes.extend(state.h1_snapshot.reason_codes)
                    else:
                        reason_codes.append("H1_PENDING_INIT")
                    if state.m15_snapshot is not None:
                        reason_codes.extend(state.m15_snapshot.reason_codes)
                    else:
                        reason_codes.append("M15_PENDING_INIT")
                    if state.m5_snapshot is not None:
                        reason_codes.extend(state.m5_snapshot.reason_codes)
                reason_codes.extend(extra_reasons)
                reason_codes = map_reason_codes(reason_codes)

                trace = _build_trace(
                    state=state,
                    now=now,
                    h1_last_closed=h1_closed,
                    h1_new_close=h1_new,
                    m15_last_closed=m15_closed,
                    m15_new_close=m15_new,
                    m5_last_closed=m5_closed,
                    m5_new_close=m5_new,
                    final_decision=final_decision,
                    reasons=reason_codes,
                )
                if decision is not None and decision.signal is not None:
                    daily_summary.signal_candidates += 1
                if trace.final_decision != "PLACE_LIMIT":
                    for blocker in trace.reasons:
                        daily_summary.blockers[blocker] += 1
                state.last_reason_codes = trace.reasons
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
                if client is not None:
                    metrics = client.metrics_snapshot()
                    retries = metrics.get("total_retries", 0) - daily_summary.api_retries_start
                    http_429 = metrics.get("http_429_count", 0) - daily_summary.api_429_start
                    requests_total = metrics.get("total_requests", 0) - daily_summary.api_requests_start
                LOGGER.info(
                    "Heartbeat cycle=%d open_positions=%d pending_orders=%d daily_pnl=%.2f top_blockers=%s api_requests=%d retries=%d http429=%d",
                    cycle,
                    opened,
                    pending,
                    pnl,
                    daily_summary.top_blockers(),
                    requests_total,
                    retries,
                    http_429,
                )
                last_heartbeat = mono

            if (mono - last_dashboard) >= config.monitoring.dashboard_interval_seconds:
                dashboard_writer.write(
                    {
                        "mode": "paper" if paper_mode else "dry-run",
                        "trading_day": day,
                        "global_daily_pnl": round(journal.get_daily_stats(day, epic="GLOBAL").pnl, 4),
                        "open_positions": len(position_manager.get_open_positions()),
                        "pending_orders": len(order_executor.get_pending_orders()),
                        "assets": {
                            epic: {
                                "trade_enabled": st.asset.trade_enabled,
                                "stale_data": st.stale_data,
                                "last_reasons": st.last_reason_codes,
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


if __name__ == "__main__":
    run()
