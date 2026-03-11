from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Imports from extracted app submodules
# ---------------------------------------------------------------------------
from bot.app.config_helpers import (
    _apply_cli_overrides,
    _daily_gate_mode,
    _resolve_config_path,
    _resolve_runtime_path,
    parse_args,
    setup_logging,
)
from bot.app.decision_records import (
    _apply_rr_profile_to_signal,
    apply_closed_events,
    create_decision_record_from_outcome,
    place_single_test_order,
)
from bot.app.factories import (
    _asset_from_template,
    _build_daily_gate_provider,
    _estimated_open_margin,
    _gate_param_space_for_grid,
    build_alert_dispatcher,
    build_asset_universe,
    build_client,
    build_news_provider,
)
from bot.app.live_helpers import (
    AssetRuntimeState,
    DailyRuntimeSummary,
    PendingOrderIntent,
    _apply_orderflow_small_soft_gate,
    _build_trace,
    _compute_v2_score,
    _default_observe_evaluation,
    _log_daily_summary,
    _normalize_action_for_score,
    _pick_best_candidate,
    _quality_gate_reasons,
    _quote_refresh_interval_seconds,
    _risk_multiplier_for,
    _timeframe_history,
    _trace_signature,
    create_live_score_v3_engine,
    derive_entry_state,
    refresh_timeframe_cache,
    resolve_db_path,
    should_refresh_quote,
)
from bot.app.optimizer import run_research_optimize_mode
from bot.app.report_helpers import (
    _asset_map_for_symbols,
    _augment_report_with_research_fields,
    _autofetch_backtest_data,
    _backtest_symbols,
    _first_variant_code,
    _log_daily_gate_comparison,
    _log_variant_comparison,
    _open_report_html,
    _parse_backtest_datetime,
    _parse_backtest_variants,
    _parse_report_formats,
    _trade_time_bounds,
    _validate_batch_data_root,
    _variant_report_filename,
)
from bot.app.research import run_research_mode
from bot.app.viewer import (
    _maybe_block_dashboard,
    _maybe_start_dashboard,
    _maybe_start_mc_viewer,
    _maybe_stop_mc_viewer,
    _run_monte_carlo_for_batch,
    _run_monte_carlo_for_payloads,
)
from bot.backtest.data_provider import AutoDataLoader, MissingDataError, normalize_timeframe
from bot.backtest.engine import (
    BacktestVariant,
    aggregate_backtest_reports,
    run_backtest_from_csv,
    run_backtest_multi_strategy,
    run_walk_forward_from_csv,
    run_walk_forward_multi_strategy,
)
from bot.backtest.monte_carlo import MCAdaptiveModel
from bot.backtest.runner import BacktestRunner
from bot.batch_backtest import make_trade_id, orchestrate_batch
from bot.capital_ramp import (
    EVENT_STOP_TARGET,
    EVENT_STOP_YEAR_END,
    EVENT_TOPUP,
    START_EQUITY_PLN,
    CapitalRampEvent,
    CapitalRampRuntime,
)
from bot.clock import is_symbol_market_open, trading_day, utc_now
from bot.config import AppConfig, AssetConfig, load_config
from bot.data.candles import Candle
from bot.data.capital_client import CapitalAPIError, CapitalClient
from bot.data.market_data import MarketDataService
from bot.execution.feasibility import validate_order
from bot.execution.micro_loss_defense import (
    MicroLossDefenseConfig,
    MicroLossMetrics,
    run_micro_loss_checks,
)
from bot.execution.order_validation import compute_risk_cash_plan, price_to_points
from bot.execution.orders import OrderExecutor
from bot.execution.paper_costs import (
    PaperCostConfig,
    SlippageModelConfig,
    estimate_roundtrip_cost,
    estimate_roundtrip_cost_points,
)
from bot.execution.position_manager import PositionManager, build_multi_tp_profile
from bot.execution.sizing import (
    compute_compound_equity,
    resolve_score_tier,
    tier_risk_multiplier,
)
from bot.gating.adaptive import (
    AdaptiveThresholdConfig,
    SoftGateResult,
    apply_soft_gates,
    build_adaptive_config,
)
from bot.gating.daily_gate import DailyGateProvider
from bot.monitoring.alerts import AlertDispatcher
from bot.monitoring.dashboard import DashboardWriter
from bot.monitoring.signal_candidates import (
    SignalCandidate,
    SignalCandidateAggregator,
    SignalCandidateLogger,
    export_diagnostics,
    init_signal_candidates_table,
)
from bot.news.calendar_provider import CalendarProvider, Event
from bot.news.gate import is_blocked, should_cancel_pending
from bot.ops_runtime import run_backup_now, run_ops_healthcheck, run_restore_verify
from bot.reporting.backtest_reporter import BacktestMeta, BacktestReporter, BacktestRun
from bot.storage.db import init_db, managed_connection
from bot.storage.journal import Journal
from bot.storage.models import ClosedPositionEvent
from bot.strategy.candidate_queue import CandidateQueue
from bot.strategy.contracts import (
    BiasState,
    DecisionAction,
    SetupCandidate,
    StrategyDataBundle,
    StrategyEvaluation,
    StrategyOutcome,
    StrategyPlugin,
)
from bot.strategy.index_existing import IndexExistingStrategy
from bot.strategy.orb_h4_retest import OrbH4RetestStrategy
from bot.strategy.orderflow import (
    CompositeOrderflowProvider,
    OrderflowProvider,
    OrderflowSnapshot,
)
from bot.strategy.portfolio_supervisor import EntryProposal, PortfolioSupervisor
from bot.strategy.ranker import rank_score
from bot.strategy.risk import RiskEngine
from bot.strategy.route_pipeline_core import (
    RoutePipelineContext,
    RoutePipelineHooks,
    RoutePipelineProfile,
    evaluate_and_finalize_route,
)
from bot.strategy.router import StrategyRouter
from bot.strategy.scalp_ict_pa import ScalpIctPriceActionStrategy
from bot.strategy.session_filter import (
    SessionMatch,
    SessionWindow,
    build_session_windows,
    match_session,
)
from bot.strategy.trace import closed_candles, format_trace_text, map_reason_codes, trace_to_json
from bot.strategy.trend_pullback_m15 import TrendPullbackM15Strategy

LOGGER = logging.getLogger("trading_bot")


def _make_fallback_candle(now: datetime) -> Candle:
    """Create a minimal Candle when no M5 data is available.

    This is only used as a fallback for ``extract_features`` in the rare edge
    case where the M5 cache is completely empty (e.g. first tick of a new
    session).  All OHLC values are set to 0.0 so the candle carries no
    price information — the feature extractor will rely on evaluation
    metadata instead.
    """
    return Candle(timestamp=now, open=0.0, high=0.0, low=0.0, close=0.0)


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
    if bool(config.capital_ramp.enabled):
        segments = [frame]
        segment_info = {
            "segment_count": 1,
            "segment_sizes": [len(frame)],
            "gap_threshold_bars": 3,
            "gap_threshold_minutes": 0.0,
            "soft_gap_minutes": float(config.backtest_tuning.segment_soft_gap_minutes),
            "hard_gap_minutes": float(config.backtest_tuning.segment_hard_gap_minutes),
            "gap_count_over_threshold": 0,
            "gap_count_soft_only": 0,
            "gaps_over_threshold": [],
            "gaps_soft_only": [],
        }
    else:
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
                "segment_sizes": [len(frame)],
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
    rolling_equity = float(START_EQUITY_PLN if config.capital_ramp.enabled else config.risk.equity)
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
        segment_context["segment_input_bars"] = len(segment_input)
        segment_context["segment_trade_bars"] = len(segment)
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
    segment_meta["equity_start"] = float(START_EQUITY_PLN if config.capital_ramp.enabled else config.risk.equity)
    segment_meta["equity_end"] = float(rolling_equity)
    return merged, segment_meta


def run_backtest_mode(args: argparse.Namespace, config: AppConfig, assets: list[AssetConfig], root: Path) -> None:
    # Start the live dashboard BEFORE the backtest so we can watch in real-time
    _maybe_start_dashboard(args, config)

    # Start Monte Carlo live viewer (non-blocking subprocess)
    _maybe_start_mc_viewer(config, root, cli_override=getattr(args, "mc_viewer", None))

    report_formats = _parse_report_formats(args.report_formats)
    reporter = BacktestReporter(_resolve_runtime_path(root, str(args.report_dir))) if args.report else None
    generated_report_dirs: list[Path] = []
    selected_gate_modes = ["off", "trend", "trend_vol_news"] if args.daily_gate_ab else [_daily_gate_mode(config)]

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
            initial_equity=float(START_EQUITY_PLN if config.capital_ramp.enabled else config.risk.equity),
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
                    json.dumps(report_dict, indent=2, ensure_ascii=True)
                    if args.report
                    else "summary-only (--no-report)",
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
                    json.dumps(report_dict, indent=2, ensure_ascii=True)
                    if args.report
                    else "summary-only (--no-report)",
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
        bars_count = (
            int(data_health.get("bars", 0))
            if isinstance(data_health, dict) and data_health.get("bars") is not None
            else 0
        )
        close_bid_nan = int(nan_counts.get("close_bid", bars_count)) if isinstance(nan_counts, dict) else bars_count
        close_ask_nan = int(nan_counts.get("close_ask", bars_count)) if isinstance(nan_counts, dict) else bars_count
        spread_mode = (
            "ASSUMED_OHLC"
            if bars_count > 0 and close_bid_nan >= bars_count and close_ask_nan >= bars_count
            else "REAL_BIDASK"
        )

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

                debug_file = (
                    (reports_dir / f"{variant.code}_{gate_mode}_debug_exec_{symbol}.jsonl") if emit_reports else None
                )
                no_price_debug_file = (
                    (reports_dir / f"{variant.code}_{gate_mode}_debug_no_price_{symbol}.jsonl")
                    if emit_reports
                    else None
                )
                reaction_timeout_debug_file = (
                    (reports_dir / f"{variant.code}_{gate_mode}_debug_reaction_timeout_{symbol}.jsonl")
                    if emit_reports
                    else None
                )
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
                            config.monte_carlo,
                            png_path=_mc_png,
                            json_path=_mc_json,
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
        ranked.sort(
            key=lambda item: (-float(item["total_pnl"]), float(item["max_drawdown"]), -float(item["expectancy"]))
        )
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
            raise RuntimeError(
                "Backtest data missing. Use --backtest-autofetch or provide --backtest-data CSV."
            ) from exc


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
        open_time = trade.entry_time.astimezone(UTC).isoformat()
        close_time = trade.exit_time.astimezone(UTC).isoformat()
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
                config.monte_carlo,
                png_path=_mc_png,
                json_path=_mc_json,
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
    capital_ramp_scope: str = "CAPITAL_RAMP:PAPER",
    capital_ramp_pnl_prefix: str = "PAPER",
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
    orderflow_full_symbols: set[str] = set()  # FULL mode removed
    orderflow_default_mode = "LITE"
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
            "Adaptive threshold ENABLED  base=%.1f  range_adj=%.1f  trend_adj=%.1f  soft_gates=%s  penalty=%.1f",
            adaptive_cfg.base_threshold,
            adaptive_cfg.range_adjust,
            adaptive_cfg.trend_adjust,
            adaptive_cfg.soft_gates_enabled,
            adaptive_cfg.soft_gate_penalty,
        )
    elif config.adaptive_threshold.soft_gates_enabled:
        # Soft gates can be on independently of adaptive threshold
        adaptive_cfg = build_adaptive_config(config.adaptive_threshold)
        LOGGER.info("Soft gates ENABLED (adaptive threshold OFF)")

    # ── ScoreV3 engine (live) ────────────────────────────
    live_score_v3_engine = create_live_score_v3_engine(config)
    if live_score_v3_engine is not None:
        LOGGER.info(
            "ScoreV3 ENABLED (live)  mode=%s  threshold=%.1f  small=[%.1f, %.1f]",
            live_score_v3_engine.config.mode,
            live_score_v3_engine.config.trade_threshold,
            live_score_v3_engine.config.small_min,
            live_score_v3_engine.config.small_max,
        )

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
        for epic in states:
            provider = _build_daily_gate_provider(config=config, mode=daily_gate_mode, events=[])
            if provider is not None:
                daily_gate_services[epic] = provider

    capital_ramp_runtime: CapitalRampRuntime | None = None
    capital_ramp_closed_pnl = 0.0
    if bool(config.capital_ramp.enabled):
        if str(config.account_currency).strip().upper() != "PLN":
            raise RuntimeError("capital_ramp.enabled requires account_currency=PLN")
        scope_value = str(capital_ramp_scope or "CAPITAL_RAMP:PAPER").strip() or "CAPITAL_RAMP:PAPER"
        pnl_prefix = str(capital_ramp_pnl_prefix or "PAPER").strip().upper() or "PAPER"
        capital_ramp_closed_pnl = float(journal.sum_closed_pnl(pnl_prefix))
        persisted_state = journal.get_capital_ramp_state(scope_value)
        if persisted_state is None:
            capital_ramp_runtime = CapitalRampRuntime.initialize(
                scope=scope_value,
                now_utc=utc_now(),
                timezone_name=config.timezone,
                current_closed_pnl=capital_ramp_closed_pnl,
            )
            journal.upsert_capital_ramp_state(capital_ramp_runtime.state)
        else:
            capital_ramp_runtime = CapitalRampRuntime(persisted_state)
        status = capital_ramp_runtime.status(
            now_utc=utc_now(),
            current_closed_pnl=capital_ramp_closed_pnl,
        )
        LOGGER.info(
            "Capital ramp enabled | scope=%s model_equity=%.2f topups_total=%.2f topups_count=%d next_topup=%s stopped_reason=%s",
            scope_value,
            float(status.get("model_equity", 0.0) or 0.0),
            float(status.get("topups_total", 0.0) or 0.0),
            int(status.get("topups_count", 0) or 0),
            status.get("next_topup_date_local"),
            status.get("stopped_reason"),
        )

    def _persist_capital_ramp_event(event: CapitalRampEvent) -> None:
        journal.append_capital_ramp_event(event)
        if capital_ramp_runtime is not None:
            journal.upsert_capital_ramp_state(capital_ramp_runtime.state)
        if event.event_type == EVENT_TOPUP:
            LOGGER.info(
                "Capital ramp topup | scope=%s amount=%.2f model_equity=%.2f local_date=%s",
                event.scope,
                float(event.amount),
                float(event.model_equity),
                event.local_date.isoformat(),
            )
            alerts.send(
                event="CAPITAL_RAMP_TOPUP",
                level="info",
                message=(
                    f"Topup applied amount={event.amount:.2f} model_equity={event.model_equity:.2f} "
                    f"date={event.local_date.isoformat()}"
                ),
                dedupe_key=f"capital-ramp-topup-{event.scope}-{event.local_date.isoformat()}",
            )
            return
        if event.event_type == EVENT_STOP_TARGET:
            LOGGER.info(
                "Capital ramp stopped (target reached) | scope=%s model_equity=%.2f",
                event.scope,
                float(event.model_equity),
            )
            alerts.send(
                event="CAPITAL_RAMP_STOP_TARGET",
                level="warning",
                message=f"Capital ramp stopped: target reached model_equity={event.model_equity:.2f}",
                dedupe_key=f"capital-ramp-stop-target-{event.scope}",
            )
            return
        if event.event_type == EVENT_STOP_YEAR_END:
            LOGGER.info(
                "Capital ramp stopped (year end) | scope=%s model_equity=%.2f",
                event.scope,
                float(event.model_equity),
            )
            alerts.send(
                event="CAPITAL_RAMP_STOP_YEAR_END",
                level="warning",
                message=f"Capital ramp stopped: year end model_equity={event.model_equity:.2f}",
                dedupe_key=f"capital-ramp-stop-year-end-{event.scope}",
            )

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
        capital_ramp_effective_equity = float(config.risk.equity)
        if capital_ramp_runtime is not None:
            topup_event, stop_event_capital = capital_ramp_runtime.maybe_apply_topup(
                now_utc=now,
                current_closed_pnl=capital_ramp_closed_pnl,
            )
            if topup_event is not None:
                _persist_capital_ramp_event(topup_event)
            if stop_event_capital is not None:
                _persist_capital_ramp_event(stop_event_capital)
            journal.upsert_capital_ramp_state(capital_ramp_runtime.state)
            capital_ramp_effective_equity = float(
                capital_ramp_runtime.effective_equity(
                    now_utc=now,
                    current_closed_pnl=capital_ramp_closed_pnl,
                )
            )
        daily_summary.cycles += 1
        # ── Session matching (multi-strategy loop) ────
        session_match: SessionMatch = (
            match_session(
                now,
                session_windows,
                block_outside=bool(config.session_filter.block_outside_sessions),
            )
            if session_windows
            else SessionMatch.no_match()
        )
        try:
            symbol_market_open = {
                epic: is_symbol_market_open(
                    now,
                    symbol=epic,
                    timezone_name=config.timezone,
                    default_profile=config.market_hours.default_profile,
                    symbol_profiles=config.market_hours.symbol_profiles,
                )
                for epic in states
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
            if should_refresh_quote(
                now=now, last_fetch_at=last_positions_sync_at, interval_seconds=sync_positions_seconds
            ):
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
                if should_refresh_quote(
                    now=now, last_fetch_at=state.quote_last_fetch_at, interval_seconds=quote_interval
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
            if daily_gate_services:
                for provider in daily_gate_services.values():
                    provider.set_events(events)
            news_blocked = is_blocked(now, events, block_minutes=config.news_gate.block_minutes)
            if news_blocked:
                for order in order_executor.get_pending_orders():
                    if should_cancel_pending(
                        {"status": order.status}, now, events, block_minutes=config.news_gate.block_minutes
                    ):
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
                closed_delta = apply_closed_events(closed, day, journal, risk_engine, now, alerts)
                if capital_ramp_runtime is not None:
                    capital_ramp_closed_pnl += float(closed_delta)
                    topup_event_after_close, stop_event_after_close = capital_ramp_runtime.maybe_apply_topup(
                        now_utc=now,
                        current_closed_pnl=capital_ramp_closed_pnl,
                    )
                    if topup_event_after_close is not None:
                        _persist_capital_ramp_event(topup_event_after_close)
                    if stop_event_after_close is not None:
                        _persist_capital_ramp_event(stop_event_after_close)
                    journal.upsert_capital_ramp_state(capital_ramp_runtime.state)
                    capital_ramp_effective_equity = float(
                        capital_ramp_runtime.effective_equity(
                            now_utc=now,
                            current_closed_pnl=capital_ramp_closed_pnl,
                        )
                    )
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

                def _main_compute_score(
                    *,
                    context: RoutePipelineContext,
                    bias: BiasState,
                    candidate: SetupCandidate | None,
                    evaluation: StrategyEvaluation,
                    schedule_open: bool,
                    orderflow_snapshot: OrderflowSnapshot | None,
                ) -> StrategyEvaluation:
                    # Always run V2 first — it populates score_layers, penalties,
                    # and orderflow metadata that downstream code may inspect.
                    evaluation = _compute_v2_score(
                        symbol=context.symbol,
                        strategy_name=context.strategy_name,
                        bias=bias,
                        route_params=context.route_params,
                        evaluation=evaluation,
                        news_blocked=context.news_blocked,
                        schedule_open=schedule_open,
                        orderflow_snapshot=orderflow_snapshot,
                        setup_side=candidate.side if candidate is not None else None,
                        orderflow_settings=context.orderflow_settings,
                    )
                    # If V3 is enabled, overwrite score_total and action with V3.
                    if live_score_v3_engine is not None:
                        from bot.strategy.score_v3 import apply_score_v3 as _apply_v3

                        _atr_val = evaluation.metadata.get("atr_m5")
                        _spread_val = evaluation.snapshot.get("spread", evaluation.metadata.get("spread"))
                        evaluation = _apply_v3(
                            live_score_v3_engine,
                            evaluation,
                            bias,
                            candle=context.bundle.candles_m5[-1]
                            if context.bundle.candles_m5
                            else _make_fallback_candle(context.now),
                            atr_m5=float(_atr_val) if _atr_val is not None else None,
                            spread=float(_spread_val) if _spread_val is not None else None,
                        )
                        # Periodically update quantile boundaries
                        if (
                            live_score_v3_engine.score_history_size % 500 == 0
                            and live_score_v3_engine.score_history_size > 0
                        ):
                            live_score_v3_engine.update_quantile_boundaries()
                    return evaluation

                def _main_normalize_and_gate(
                    *,
                    context: RoutePipelineContext,
                    candidate: SetupCandidate | None,
                    evaluation: StrategyEvaluation,
                ) -> StrategyEvaluation:
                    del candidate
                    _v3_active = live_score_v3_engine is not None
                    # When V3 is active, apply_score_v3 already resolved the
                    # action via ScoreV3Engine.resolve_action — skip V2
                    # normalization so V3 thresholds are respected.
                    if not _v3_active:
                        _ = _normalize_action_for_score(
                            evaluation=evaluation,
                            config=config,
                            session_threshold_adjust=session_match.threshold_adjust,
                        )
                    gate_reasons = _quality_gate_reasons(
                        symbol=context.symbol,
                        route_params=context.route_params,
                        evaluation=evaluation,
                        now=context.now,
                        timezone_name=context.timezone_name,
                    )
                    sg_result: SoftGateResult | None = None
                    if adaptive_cfg is not None and adaptive_cfg.soft_gates_enabled and gate_reasons:
                        sg_result = apply_soft_gates(
                            gate_reasons,
                            soft_gates_enabled=True,
                            soft_gate_penalty=adaptive_cfg.soft_gate_penalty,
                        )
                        evaluation.metadata["soft_gate_converted"] = sg_result.converted_gates
                        evaluation.metadata["soft_gate_penalty"] = sg_result.total_penalty
                        if sg_result.hard_reasons:
                            evaluation.action = DecisionAction.OBSERVE
                            for code in sg_result.hard_reasons:
                                if code not in evaluation.reasons_blocking:
                                    evaluation.reasons_blocking.append(code)
                        if not sg_result.hard_reasons:
                            evaluation.reasons_blocking.clear()
                            if _v3_active:
                                # V3 already set the action; only apply soft-gate
                                # penalty metadata without V2 re-normalization.
                                if sg_result.total_penalty:
                                    evaluation.metadata["adaptive_soft_penalty"] = sg_result.total_penalty
                            else:
                                _ = _normalize_action_for_score(
                                    evaluation=evaluation,
                                    config=config,
                                    adaptive_cfg=adaptive_cfg,
                                    soft_gate_result=sg_result,
                                    session_threshold_adjust=session_match.threshold_adjust,
                                )
                    elif gate_reasons:
                        evaluation.action = DecisionAction.OBSERVE
                        for code in gate_reasons:
                            if code not in evaluation.reasons_blocking:
                                evaluation.reasons_blocking.append(code)
                    elif adaptive_cfg is not None and adaptive_cfg.enabled and not _v3_active:
                        evaluation.reasons_blocking.clear()
                        _ = _normalize_action_for_score(
                            evaluation=evaluation,
                            config=config,
                            adaptive_cfg=adaptive_cfg,
                            session_threshold_adjust=session_match.threshold_adjust,
                        )
                    return evaluation

                def _main_apply_orderflow_soft_gate(
                    *,
                    context: RoutePipelineContext,
                    candidate: SetupCandidate | None,
                    evaluation: StrategyEvaluation,
                ) -> StrategyEvaluation:
                    del candidate
                    return _apply_orderflow_small_soft_gate(
                        route_params=context.route_params,
                        evaluation=evaluation,
                        orderflow_settings=context.orderflow_settings,
                    )

                def _main_build_payload(
                    *,
                    context: RoutePipelineContext,
                    candidate: SetupCandidate | None,
                    evaluation: StrategyEvaluation,
                ) -> dict[str, object]:
                    return {
                        "strategy_name": context.strategy_name,
                        "score_total": evaluation.score_total,
                        "score_breakdown": evaluation.score_breakdown,
                        "snapshot": evaluation.snapshot,
                        "candidate_id": candidate.candidate_id if candidate is not None else None,
                        **evaluation.metadata,
                    }

                def _main_on_outcome(context: RoutePipelineContext, outcome: StrategyOutcome) -> None:
                    soft_reasons = outcome.evaluation.metadata.get("soft_reasons")
                    if isinstance(soft_reasons, list):
                        for soft_reason in soft_reasons:
                            code = f"SOFT_REASON_{str(soft_reason).upper()}"
                            if code not in outcome.reason_codes:
                                outcome.reason_codes.append(code)
                    strategy_local = context.strategy
                    if isinstance(strategy_local, IndexExistingStrategy):
                        state.h1_snapshot, state.m15_snapshot, state.m5_snapshot = strategy_local.last_snapshots(
                            context.symbol
                        )
                        legacy = strategy_local.last_legacy_decision(context.symbol)
                        if legacy is not None:
                            for code in legacy.reason_codes:
                                if code not in outcome.reason_codes:
                                    outcome.reason_codes.append(code)

                main_hooks = RoutePipelineHooks(
                    default_observe_evaluation=_default_observe_evaluation,
                    pick_best_candidate=_pick_best_candidate,
                    compute_score=_main_compute_score,
                    normalize_and_gate=_main_normalize_and_gate,
                    apply_orderflow_small_soft_gate=_main_apply_orderflow_soft_gate,
                    build_payload=_main_build_payload,
                    build_reason_codes=lambda _ctx, evaluation: list(evaluation.reasons_blocking),
                    on_outcome=_main_on_outcome,
                    unknown_rank=lambda _ctx, evaluation: rank_score(evaluation),
                    rank=lambda context, evaluation: rank_score(evaluation) + (context.route_priority * 0.01),
                    build_unknown_payload=lambda context: {"strategy_name": context.strategy_name},
                )

                for route in routes:
                    strategy = strategy_plugins.get(route.strategy)
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
                    route_context = RoutePipelineContext(
                        profile=RoutePipelineProfile.MAIN,
                        symbol=epic,
                        now=now,
                        timezone_name=config.timezone,
                        timeframe=config.timeframes.m5,
                        strategy_name=route.strategy,
                        route_priority=route.priority,
                        route_params=route.params,
                        route_risk=route.risk,
                        strategy=strategy,
                        bundle=bundle,
                        news_blocked=news_blocked,
                        spread=spread,
                        quote=q,
                        orderflow_provider=orderflow_provider,
                        orderflow_default_mode=orderflow_default_mode,
                        orderflow_default_window=orderflow_default_window,
                        orderflow_full_symbols=orderflow_full_symbols,
                        orderflow_settings=orderflow_settings,
                    )
                    route_result = evaluate_and_finalize_route(
                        context=route_context,
                        candidate_queue=candidate_queue,
                        hooks=main_hooks,
                    )
                    outcome = route_result.outcome
                    rank_value = route_result.rank

                    # --- SignalCandidate logging (telemetry) ---
                    if sc_logger is not None:
                        _atr_val = outcome.evaluation.metadata.get("atr_m5", outcome.evaluation.snapshot.get("atr_m5"))
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
                            _sl_dist = (
                                abs(sig.entry_price - sig.stop_price) if sig.entry_price and sig.stop_price else None
                            )
                            _tp_dist = (
                                abs(sig.take_profit - sig.entry_price) if sig.take_profit and sig.entry_price else None
                            )
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
                        sc_logger.log(
                            SignalCandidate(
                                timestamp=now,
                                symbol=epic,
                                timeframe=config.timeframes.m5,
                                strategy_name=outcome.strategy_name,
                                setup_name=_setup_name,
                                side=outcome.order_request.side
                                if outcome.order_request
                                else (outcome.candidate.side if outcome.candidate else None),
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
                            )
                        )

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
                _v3_tier_raw = (
                    intent.outcome.evaluation.metadata.get("tier") if live_score_v3_engine is not None else None
                )
                if _v3_tier_raw is not None:
                    # V3 already resolved the tier — normalise to uppercase
                    # format expected by tier_risk_multiplier (A_plus → A_PLUS).
                    _tier_label = str(_v3_tier_raw).upper()
                else:
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
                _sizing_equity = (
                    float(capital_ramp_effective_equity)
                    if capital_ramp_runtime is not None
                    else float(config.risk.equity)
                )
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
                    "tier": _tier_label,
                    "tier_mult": _tier_mult,
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
                    reject_code = (
                        feasibility.reason.value if feasibility.reason is not None else "ORDER_FEASIBILITY_REJECT"
                    )
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
                                intent.symbol,
                                _re_side,
                                _re_reason,
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
                    _ml_rtc = estimate_roundtrip_cost_points(spread=_ml_spread, atr=_ml_atr, config=paper_cost_cfg)
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
                _rtc_size = (
                    estimate_roundtrip_cost(
                        spread=_order_spread,
                        atr=_ml_atr if ml_defense_cfg.enabled else None,
                        size=size,
                        config=paper_cost_cfg,
                    )
                    if paper_cost_cfg.enabled
                    else None
                )
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
                    f"{intent.outcome.evaluation.score_total:.2f}"
                    if intent.outcome.evaluation.score_total is not None
                    else "-",
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
                            intent.symbol,
                            intent.signal.side,
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
                dashboard_payload = {
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
                if capital_ramp_runtime is not None:
                    dashboard_payload["capital_ramp"] = capital_ramp_runtime.status(
                        now_utc=now,
                        current_closed_pnl=capital_ramp_closed_pnl,
                    )
                dashboard_writer.write(dashboard_payload)
                last_dashboard = mono

        except CapitalAPIError as exc:
            LOGGER.error("Capital API error: %s", exc)
            alerts.send(
                event="CAPITAL_API_ERROR", level="error", message=str(exc), dedupe_key=f"api-{type(exc).__name__}"
            )
        except Exception:
            LOGGER.exception("Unhandled cycle error")
            alerts.send(
                event="UNHANDLED_RUNTIME_ERROR",
                level="error",
                message="Unhandled exception in main loop",
                dedupe_key="runtime-unhandled",
            )

        stop_event.wait(config.execution.loop_seconds)

    _log_daily_summary(summary=daily_summary, client=client)
    LOGGER.info("Bot stopped.")


def _run_live_body(
    args: argparse.Namespace,
    config: AppConfig,
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
    orderflow_provider: OrderflowProvider = CompositeOrderflowProvider()
    portfolio_supervisor = PortfolioSupervisor(config.portfolio)
    order_executor = OrderExecutor(
        client=client,
        journal=journal,
        dry_run=mode_dry_run,
        default_epic=assets[0].epic,
        default_currency=assets[0].currency,
    )
    position_manager = PositionManager(
        client=client,
        journal=journal,
        dry_run=mode_dry_run,
        multi_tp=build_multi_tp_profile(config.multi_tp),
    )
    dashboard_writer = DashboardWriter(os.getenv("DASHBOARD_PATH", config.monitoring.dashboard_path))
    alerts = build_alert_dispatcher(config)

    LOGGER.info(
        "Starting bot | mode=%s | primary=%s | timezone=%s",
        "paper" if paper_mode else "dry-run",
        assets[0].epic,
        config.timezone,
    )

    if args.test_order:
        place_single_test_order(
            order_executor, market_data, assets, config, mode_dry_run, args.test_side, args.test_size, args.test_epic
        )
        LOGGER.info("Test-order mode completed. Exiting.")
        return

    tf_history = _timeframe_history(config)
    close_grace_seconds = int(os.getenv("CANDLE_CLOSE_GRACE_SECONDS", str(config.execution.candle_close_grace_seconds)))
    candle_retry_seconds = int(os.getenv("CANDLE_RETRY_SECONDS", str(config.execution.candle_retry_seconds)))
    sync_pending_seconds = int(os.getenv("SYNC_PENDING_SECONDS", str(config.execution.sync_pending_seconds)))
    sync_positions_seconds = int(os.getenv("SYNC_POSITIONS_SECONDS", str(config.execution.sync_positions_seconds)))
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
    capital_ramp_scope = "CAPITAL_RAMP:PAPER" if paper_mode else "CAPITAL_RAMP:LIVE"
    capital_ramp_pnl_prefix = str(position_manager.mode_prefix).strip().upper()

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
        capital_ramp_scope=capital_ramp_scope,
        capital_ramp_pnl_prefix=capital_ramp_pnl_prefix,
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

    # --- P1: Validate critical environment variables early ---
    _missing_env: list[str] = []
    for _var in ("CAPITAL_API_KEY", "CAPITAL_IDENTIFIER"):
        if not os.getenv(_var):
            _missing_env.append(_var)
    if not (os.getenv("CAPITAL_API_PASSWORD") or os.getenv("CAPITAL_PASSWORD")):
        _missing_env.append("CAPITAL_API_PASSWORD")
    if _missing_env and (args.paper or not args.dry_run):
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(_missing_env)}. "
            "Set them in your .env file or environment before running."
        )

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    from bot.runners import backtest as backtest_runner
    from bot.runners import batch as batch_runner
    from bot.runners import live as live_runner
    from bot.runners import ops as ops_runner
    from bot.runners import research as research_runner

    root = Path(__file__).resolve().parent
    config_path = _resolve_config_path(root, str(args.config))
    config = load_config(config_path)
    _apply_cli_overrides(args, config)

    if bool(args.ops_healthcheck):

        def _ops_healthcheck_handler(_args: argparse.Namespace, _config: AppConfig, _root: Path) -> None:
            payload = run_ops_healthcheck(root=_root, config=_config)
            LOGGER.info("Ops healthcheck: %s", json.dumps(payload, ensure_ascii=True))
            if payload.get("status") != "ok":
                raise RuntimeError("Ops healthcheck failed")

        ops_runner.run(args, config, root, handler=_ops_healthcheck_handler)
        return

    if bool(args.ops_backup_now):

        def _ops_backup_handler(_args: argparse.Namespace, _config: AppConfig, _root: Path) -> None:
            payload = run_backup_now(root=_root, config=_config)
            LOGGER.info("Ops backup-now: %s", json.dumps(payload, ensure_ascii=True))
            if payload.get("status") != "ok":
                raise RuntimeError("Ops backup-now failed")

        ops_runner.run(args, config, root, handler=_ops_backup_handler)
        return

    if args.ops_restore_verify:

        def _ops_restore_handler(_args: argparse.Namespace, _config: AppConfig, _root: Path) -> None:
            backup_dir = Path(str(_args.ops_restore_verify)).expanduser()
            if not backup_dir.is_absolute():
                backup_dir = (_root / backup_dir).resolve()
            payload = run_restore_verify(backup_dir=backup_dir)
            LOGGER.info("Ops restore-verify: %s", json.dumps(payload, ensure_ascii=True))
            if payload.get("status") != "ok":
                raise RuntimeError("Ops restore-verify failed")

        ops_runner.run(args, config, root, handler=_ops_restore_handler)
        return

    assets = build_asset_universe(config)
    LOGGER.info("Assets configured: %s", ",".join(f"{a.epic}{'' if a.trade_enabled else '(observe)'}" for a in assets))
    LOGGER.info(
        "Market hours config | default=%s overrides=%s",
        config.market_hours.default_profile,
        ",".join(f"{symbol}:{profile}" for symbol, profile in sorted(config.market_hours.symbol_profiles.items()))
        or "none",
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
        batch_runner.run_worker(args, config, assets, root, handler=run_batch_worker_mode)
        return

    if args.batch_backtest:
        batch_runner.run_backtest(args, config, root, handler=run_batch_backtest_mode)
        return

    if args.research_run and args.research_optimize:
        raise RuntimeError("Use either --research-run or --research-optimize, not both.")

    if args.research_run:
        research_runner.run(args, config, assets, root, handler=run_research_mode)
        return

    if args.research_optimize:
        research_runner.run(args, config, assets, root, handler=run_research_optimize_mode)
        return

    if args.backtest:
        backtest_runner.run(args, config, assets, root, handler=run_backtest_mode)
        return

    # Start Monte Carlo live viewer for live/paper/dry-run modes
    _maybe_start_mc_viewer(config, root, cli_override=getattr(args, "mc_viewer", None))

    db_path = resolve_db_path(root, paper_mode=paper_mode)
    with managed_connection(db_path) as conn:
        live_runner.run(
            args,
            config,
            assets,
            root,
            handler=lambda a, c, aset, r: _run_live_body(a, c, aset, r, paper_mode, mode_dry_run, conn, db_path),
        )
    LOGGER.debug("SQLite connection closed.")


if __name__ == "__main__":
    run()
