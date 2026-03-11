from __future__ import annotations

import logging
import math
from collections import Counter, deque
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bot.capital_ramp import (
    START_EQUITY_PLN,
    CapitalRampEvent,
    CapitalRampRuntime,
)
from bot.config import AppConfig, AssetConfig
from bot.data.candles import Candle
from bot.diagnostics.decision_trace import DecisionTraceWriter
from bot.execution.feasibility import RejectReason, validate_order
from bot.execution.fx import FxConverter
from bot.execution.order_validation import (
    compute_risk_cash_plan,
    expected_move_too_small,
    in_rollover_entry_block_window,
    minutes_to_next_rollover,
    price_to_points,
)
from bot.gating.daily_gate import DailyGateProvider
from bot.reporting.decision_funnel import DecisionFunnel
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
from bot.strategy.indicators import atr
from bot.strategy.orb_h4_retest import OrbH4RetestStrategy
from bot.strategy.orderflow import CompositeOrderflowProvider, OrderflowSnapshot
from bot.strategy.ranker import rank_score
from bot.strategy.risk import RiskEngine
from bot.strategy.risk_budget import PortfolioRiskBudget
from bot.strategy.route_pipeline_core import (
    RoutePipelineContext,
    RoutePipelineHooks,
    RoutePipelineProfile,
    evaluate_and_finalize_route,
)
from bot.strategy.router import StrategyRoute, StrategyRouter
from bot.strategy.scalp_ict_pa import ScalpIctPriceActionStrategy
from bot.strategy.score_tiers import resolve_tier_from_config
from bot.strategy.score_v3 import ScoreV3Config, ScoreV3Engine, apply_score_v3
from bot.strategy.shadow_observer import ShadowCandidate, ShadowObserver, classify_session
from bot.strategy.state_machine import StrategyEngine
from bot.strategy.tp_profile import tp2_r_for_target_total_r as _tp2_r_for_target_total_r
from bot.strategy.trend_pullback_m15 import TrendPullbackM15Strategy
from bot.strategy.utils import as_float_or_none, as_int

if TYPE_CHECKING:
    from bot.backtest.monte_carlo import MCAdaptiveModel

# ---------------------------------------------------------------------------
# Extracted sub-modules (backward-compatible re-exports)
# ---------------------------------------------------------------------------
from bot.backtest.candle_io import (  # noqa: F401
    _bucket_time,
    _parse_dt,
    aggregate_candles,
    load_candles_csv,
)
from bot.backtest.costs import (
    _apply_overnight_swap_if_due,
    _compute_trade_pnl_fields,
    _convert_cash_to_account,
    _next_rollover_timestamp,
    _parse_swap_time_utc,
    _trade_r_multiple,
)
from bot.backtest.diagnostics import (
    _build_decision_trace_record,
    _collect_execution_fail_sample,
    _collect_no_price_sample,
    _emit_decision,
    _emit_fill,
    _write_decision_trace,
    _write_execution_fail_debug,
    _write_no_price_debug,
    _write_reaction_timeout_debug,
)
from bot.backtest.gating import (
    _apply_reaction_gate_with_timeout,
    _apply_wait_timeout_soft_mode,
)
from bot.backtest.math_utils import (  # noqa: F401
    _action_priority,
    _clamp,
    _quantile,
    _quantile_from_sorted,
    _spread_point_stats,
)
from bot.backtest.models import (
    BacktestReport,
    BacktestTrade,
    BacktestVariant,
    WalkForwardReport,
    _ExecutionFailSample,
    _NoPriceSample,
    _OpenPosition,
    _PendingOrder,
    _ReactionTimeoutSample,
    _WaitGateState,
)
from bot.backtest.position_logic import (
    _append_live_placeholder,
    _calc_exit,
    _estimate_structure_target,
    _expected_rr,
    _live_placeholder_from,
    _manage_open_position,
    _normalize_tp_by_r,
    _resolve_tp_target_r,
)
from bot.backtest.reporting import (  # noqa: F401
    _capital_ramp_event_to_dict,
    _capital_ramp_summary,
    _exit_reason_distribution,
    _gate_counts_from_blockers,
    _is_better_outcome,
    _merge_wait_metrics,
    _net_quality_metrics,
    _per_bias_trade_metrics,
    _trade_quality_metrics,
    _trade_r_quality_metrics,
    aggregate_backtest_reports,
)
from bot.backtest.scoring_wrappers import (  # noqa: F401
    _adjust_thresholds_dynamic,
    _apply_orderflow_small_soft_gate,
    _apply_soft_reason_penalties,
    _compute_v2_score,
    _default_observe_evaluation,
    _evaluate_hard_gates,
    _missing_execution_features,
    _normalize_action_for_score,
    _orderflow_param,
    _pick_best_candidate,
    _quality_gate_reasons,
    _resolve_orderflow_mode,
    _risk_multiplier_for,
    _soft_reason_penalty_map,
    _thresholds_for_variant,
)
from bot.backtest.spread import (
    _build_dynamic_assumed_spread_series,
    _resolve_dynamic_spread_bounds,
)

LOGGER = logging.getLogger(__name__)


def run_backtest_multi_strategy(
    *,
    config: AppConfig,
    asset: AssetConfig,
    candles_m5: list[Candle],
    assumed_spread: float = 0.2,
    slippage_points: float = 0.0,
    slippage_atr_multiplier: float = 0.0,
    variant: BacktestVariant | None = None,
    execution_debug_path: str | Path | None = None,
    no_price_debug_path: str | Path | None = None,
    reaction_timeout_debug_path: str | Path | None = None,
    decision_trace_path: str | Path | None = None,
    data_context: dict[str, Any] | None = None,
    trade_start_utc: datetime | None = None,
    flatten_at_chunk_end: bool = False,
    daily_gate: DailyGateProvider | None = None,
    daily_gate_prepared: bool = False,
    mc_model: MCAdaptiveModel | None = None,
) -> BacktestReport:
    variant_cfg = variant or BacktestVariant()
    debug_path = Path(execution_debug_path) if execution_debug_path is not None else None
    no_price_path = Path(no_price_debug_path) if no_price_debug_path is not None else None
    reaction_timeout_path = Path(reaction_timeout_debug_path) if reaction_timeout_debug_path is not None else None
    _decision_trace_file = Path(decision_trace_path) if decision_trace_path is not None else None
    _decision_trace_records: list[dict[str, Any]] = []
    # --- live-flushing JSONL writer (new schema) ---
    _diag_trace_path = decision_trace_path
    if _diag_trace_path is None and hasattr(config, "diagnostics") and config.diagnostics.decision_trace_enabled:
        _diag_trace_path = config.diagnostics.decision_trace_path
    _dtw = DecisionTraceWriter(
        _diag_trace_path,
        enabled=(_diag_trace_path is not None),
    )
    _dtw.open()
    backtest_context: dict[str, Any] = dict(data_context or {})
    trade_start = trade_start_utc.astimezone(UTC) if trade_start_utc is not None else None
    segment_id = str(backtest_context.get("segment_index", "1"))
    segment_start_index = 0
    segment_start_raw = backtest_context.get("segment_start_utc")
    if isinstance(segment_start_raw, str) and segment_start_raw:
        try:
            seg_ts = _parse_dt(segment_start_raw)
            for idx, candle in enumerate(candles_m5):
                if candle.timestamp >= seg_ts:
                    segment_start_index = idx
                    break
        except ValueError:
            segment_start_index = 0
    spread_mode = str(backtest_context.get("spread_mode", "REAL_BIDASK")).upper()
    assumed_spread_used = float(backtest_context.get("assumed_spread_used", assumed_spread))
    risk_engine = RiskEngine(config.risk)
    router = StrategyRouter(config)
    strategy_plugins: dict[str, StrategyPlugin] = {
        "INDEX_EXISTING": IndexExistingStrategy(config),
        "SCALP_ICT_PA": ScalpIctPriceActionStrategy(config),
        "ORB_H4_RETEST": OrbH4RetestStrategy(config),
        "TREND_PULLBACK_M15": TrendPullbackM15Strategy(config),
    }
    orderflow_provider = CompositeOrderflowProvider()
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
    candidate_queue = CandidateQueue()
    daily_gate_mode = str(daily_gate.mode).lower() if daily_gate is not None else "off"
    if daily_gate is not None and daily_gate.enabled and not daily_gate_prepared:
        daily_gate.refresh_from_candles(candles_m5)

    candles_m15 = aggregate_candles(candles_m5, 15)
    candles_h1 = aggregate_candles(candles_m5, 60)
    atr_values = atr(candles_m5, config.indicators.atr_period)
    spread_bounds = _resolve_dynamic_spread_bounds(
        config=config,
        symbol=asset.epic,
        fallback_spread=assumed_spread_used,
    )
    dynamic_assumed_spread: list[float] | None = None
    if spread_mode == "ASSUMED_OHLC" and spread_bounds is not None:
        dynamic_assumed_spread = _build_dynamic_assumed_spread_series(
            candles_m5=candles_m5,
            atr_values=atr_values,
            min_spread=spread_bounds[0],
            max_spread=spread_bounds[1],
        )
        if dynamic_assumed_spread:
            assumed_spread_used = float(sum(dynamic_assumed_spread) / len(dynamic_assumed_spread))

    capital_ramp_runtime: CapitalRampRuntime | None = None
    capital_ramp_events: list[CapitalRampEvent] = []
    capital_ramp_closed_pnl = 0.0
    if bool(config.capital_ramp.enabled):
        if trade_start is not None:
            ramp_start_ts = trade_start
        elif candles_m5:
            ramp_start_ts = candles_m5[0].timestamp
        else:
            ramp_start_ts = datetime.now(UTC)
        capital_ramp_runtime = CapitalRampRuntime.initialize(
            scope=f"CAPITAL_RAMP:BACKTEST:{asset.epic}",
            now_utc=ramp_start_ts,
            timezone_name=config.timezone,
            current_closed_pnl=0.0,
        )
        equity = float(START_EQUITY_PLN)
    else:
        equity = float(config.risk.equity)
    peak_equity = equity
    max_drawdown = 0.0
    _dd_halt_pct = float(config.backtest_tuning.max_drawdown_halt_pct)
    _dd_halt_abs = (equity * _dd_halt_pct / 100.0) if _dd_halt_pct > 0 else 0.0
    equity_start_value = float(equity)
    _dd_halted = False
    trades: list[BacktestTrade] = []
    pending: _PendingOrder | None = None
    open_pos: _OpenPosition | None = None
    time_in_market_bars = 0
    count_be_moves = 0
    count_tp1_hits = 0
    decision_counts: Counter[str] = Counter()
    blockers: Counter[str] = Counter()
    signal_candidates = 0
    timeout_resets: Counter[str] = Counter()
    wait_states: dict[str, _WaitGateState] = {}
    wait_reset_block_bar: dict[str, int] = {}
    wait_durations: dict[str, list[int]] = {"REACTION": [], "MITIGATION": []}
    reaction_timeout_samples: list[_ReactionTimeoutSample] = []
    execution_fail_breakdown: Counter[str] = Counter()
    missing_feature_counts: Counter[str] = Counter()
    execution_fail_samples: list[_ExecutionFailSample] = []
    no_price_samples: list[_NoPriceSample] = []
    spread_gate_adjustments: Counter[str] = Counter()
    score_values: list[float] = []
    score_bins: Counter[str] = Counter()
    daily_gate_bias_bars: Counter[str] = Counter()
    daily_gate_bias_days: Counter[str] = Counter()
    seen_day_bias: dict[date, str] = {}
    blocked_by_gate_reasons: Counter[str] = Counter()
    blocked_by_gate = 0
    missing_feature_debug_logged = 0
    feature_warmup_bars = max(3, int(config.indicators.atr_period) + 2)
    trade_thr_base, small_min_base, small_max_base = _thresholds_for_variant(config, variant_cfg)
    timeouts_enabled = bool(config.backtest_tuning.reaction_timeout_force_enable or variant_cfg.reaction_timeout_reset)

    daily_trades: dict[str, int] = {}
    daily_pnl: dict[str, float] = {}

    m15_ptr = -1
    h1_ptr = -1
    last_h1_closed_ts: datetime | None = None
    last_m15_closed_ts: datetime | None = None
    swap_hour, swap_minute = _parse_swap_time_utc(config.backtest_tuning.overnight_swap_time_utc)
    long_swap_pct = float(config.backtest_tuning.overnight_swap_long_pct)
    short_swap_pct = float(config.backtest_tuning.overnight_swap_short_pct)
    account_currency = str(config.account_currency).strip().upper()
    instrument_currency = str(asset.instrument_currency or asset.currency).strip().upper()
    fx_apply_to = {str(item).strip().lower() for item in config.fx_apply_to}
    fx_converter: FxConverter | None = None
    if instrument_currency != account_currency:
        fx_converter = FxConverter(
            fee_rate=float(config.fx_conversion_fee_rate),
            fee_mode=str(config.fx_fee_mode),
            rate_source=str(config.fx_rate_source),
            static_rates=config.fx_static_rates,
        )
    rejected_by_reason: Counter[str] = Counter()
    orders_submitted = 0
    trades_filled = 0
    spread_points_series: list[float] = []
    forced_closes_count = 0
    min_size_overrides_count = 0
    margin_capped_count = 0

    # ── ScoreV3 engine + shadow observer ──
    score_v3_engine: ScoreV3Engine | None = None
    shadow_observer: ShadowObserver | None = None
    score_v3_bins: Counter[str] = Counter()
    if config.score_v3.enabled:
        _v3cfg = config.score_v3
        v3_dc = ScoreV3Config(
            enabled=True,
            mode=str(_v3cfg.mode),
            model_path=str(_v3cfg.model_path),
            trade_threshold=float(_v3cfg.trade_threshold),
            small_min=float(_v3cfg.small_min),
            small_max=float(_v3cfg.small_max),
            tier_enabled=bool(_v3cfg.tier_enabled),
            tier_a_plus_pct=float(_v3cfg.tier_a_plus_pct),
            tier_a_pct=float(_v3cfg.tier_a_pct),
            tier_b_pct=float(_v3cfg.tier_b_pct),
            shadow_enabled=bool(_v3cfg.shadow_enabled),
            shadow_output_path=str(_v3cfg.shadow_output_path),
            shadow_simulate_outcomes=bool(_v3cfg.shadow_simulate_outcomes),
            fill_prob_weight=float(_v3cfg.fill_prob_weight),
        )
        score_v3_engine = ScoreV3Engine(v3_dc)
        if v3_dc.shadow_enabled:
            _shadow_path = Path(v3_dc.shadow_output_path)
            shadow_observer = ShadowObserver(_shadow_path)

    def _capture_capital_ramp_event(event: CapitalRampEvent | None) -> None:
        if event is not None:
            capital_ramp_events.append(event)

    def _refresh_capital_ramp_for_timestamp(ts: datetime) -> None:
        nonlocal equity
        if capital_ramp_runtime is None:
            return
        topup_event, stop_event = capital_ramp_runtime.maybe_apply_topup(
            now_utc=ts,
            current_closed_pnl=capital_ramp_closed_pnl,
        )
        _capture_capital_ramp_event(topup_event)
        _capture_capital_ramp_event(stop_event)
        equity = float(
            capital_ramp_runtime.effective_equity(
                now_utc=ts,
                current_closed_pnl=capital_ramp_closed_pnl,
            )
        )

    def _apply_closed_pnl(pnl_delta: float, ts: datetime) -> None:
        nonlocal equity, capital_ramp_closed_pnl
        equity += float(pnl_delta)
        if capital_ramp_runtime is None:
            return
        capital_ramp_closed_pnl += float(pnl_delta)
        topup_event, stop_event = capital_ramp_runtime.maybe_apply_topup(
            now_utc=ts,
            current_closed_pnl=capital_ramp_closed_pnl,
        )
        _capture_capital_ramp_event(topup_event)
        _capture_capital_ramp_event(stop_event)
        equity = float(
            capital_ramp_runtime.effective_equity(
                now_utc=ts,
                current_closed_pnl=capital_ramp_closed_pnl,
            )
        )

    def _spread_for(index: int) -> float:
        candle = candles_m5[index]
        if candle.bid is not None and candle.ask is not None:
            return max(0.0, candle.ask - candle.bid)
        if dynamic_assumed_spread is not None and index < len(dynamic_assumed_spread):
            return max(0.0, float(dynamic_assumed_spread[index]))
        return max(0.0, assumed_spread)

    def _slippage_for(index: int) -> float:
        atr_term = 0.0
        if 0 <= index < len(atr_values):
            atr_val = atr_values[index]
            if atr_val is not None:
                atr_term = max(0.0, slippage_atr_multiplier * atr_val)
        return max(0.0, slippage_points) + atr_term

    def _spread_points_for(spread_price: float) -> float:
        return max(0.0, price_to_points(float(spread_price), point_size=float(asset.point_size)))

    def _cap_size_by_margin(entry_price: float, requested_size: float) -> float:
        if entry_price <= 0:
            return 0.0
        margin_requirement_pct = float(config.backtest_tuning.broker_margin_requirement_pct)
        leverage = float(config.backtest_tuning.broker_leverage)
        caps: list[float] = []
        if margin_requirement_pct > 0:
            caps.append((equity / (margin_requirement_pct / 100.0)) / entry_price)
        if leverage > 0:
            caps.append((equity * leverage) / entry_price)
        if not caps:
            return max(0.0, requested_size)
        max_size = max(0.0, min(caps))
        step = asset.size_step if asset.size_step > 0 else 0.01
        max_size = math.floor(max_size / step) * step
        if max_size < asset.min_size:
            return 0.0
        return min(max(0.0, requested_size), max_size)

    start_idx = max(config.indicators.ema_period_h1 + 10, 250)
    spread_window = max(1, int(config.spread_filter.window)) + 1
    spread_history_window: deque[float] = deque(maxlen=spread_window)
    # Slice trimming limits (same as run_backtest)
    _SLICE_KEEP_M5 = max(config.indicators.ema_period_h1 * 12, start_idx * 3, 2000)
    _SLICE_KEEP_M15 = max(config.indicators.ema_period_h1 * 4, 800)
    _SLICE_KEEP_H1 = max(config.indicators.ema_period_h1 + 100, 500)
    slice_m5_live = _append_live_placeholder(candles_m5[: start_idx + 1], 5)
    slice_m15_live: list[Candle] = []
    slice_h1_live: list[Candle] = []
    if capital_ramp_runtime is not None:
        for _pre in candles_m5[: min(start_idx, len(candles_m5))]:
            _refresh_capital_ramp_for_timestamp(_pre.timestamp)

    for i in range(start_idx, len(candles_m5)):
        candle = candles_m5[i]
        _refresh_capital_ramp_for_timestamp(candle.timestamp)
        if i > start_idx:
            slice_m5_live[-1] = candle
            slice_m5_live.append(_live_placeholder_from(candle, 5))
            if len(slice_m5_live) > _SLICE_KEEP_M5 * 2:
                del slice_m5_live[:-_SLICE_KEEP_M5]
        spread_now = _spread_for(i)
        slippage_now = _slippage_for(i)
        spread_points_now = _spread_points_for(spread_now)
        spread_points_series.append(spread_points_now)
        spread_history_window.append(spread_now)
        spread_history = list(spread_history_window)
        day_key = candle.timestamp.date().isoformat()
        daily_trades.setdefault(day_key, 0)
        daily_pnl.setdefault(day_key, 0.0)
        gate_result = None
        if daily_gate is not None and daily_gate.enabled:
            gate_result = daily_gate.evaluate(
                ts=candle.timestamp,
                symbol=asset.epic,
                spread=spread_now,
            )
            gate_bias = str(gate_result.bias).upper()
            daily_gate_bias_bars[gate_bias] += 1
            gate_day = candle.timestamp.astimezone(UTC).date()
            if gate_day not in seen_day_bias:
                seen_day_bias[gate_day] = gate_bias
                daily_gate_bias_days[gate_bias] += 1

        if timeouts_enabled and wait_states:
            for key, state in list(wait_states.items()):
                timeout_bars = (
                    int(config.backtest_tuning.wait_reaction_timeout_bars)
                    if state.wait_type == "REACTION"
                    else int(config.backtest_tuning.wait_mitigation_timeout_bars)
                )
                elapsed = max(0, i - state.enter_bar_index)
                if state.timed_out_soft:
                    soft_decay_bars = max(0, int(config.backtest_tuning.wait_timeout_soft_grace_bars))
                    if elapsed > (timeout_bars + soft_decay_bars):
                        wait_states.pop(key, None)
                        if soft_decay_bars > 0:
                            wait_reset_block_bar[key] = i + soft_decay_bars
                    continue
                if elapsed <= timeout_bars:
                    continue
                state.timed_out_soft = True
                wait_states[key] = state
                wait_durations.setdefault(state.wait_type, []).append(elapsed)
                timeout_resets[state.wait_type] = int(timeout_resets.get(state.wait_type, 0)) + 1
                timeout_resets["REACTION_TIMEOUT_RESET"] = int(timeout_resets.get("REACTION_TIMEOUT_RESET", 0)) + 1
                reason = (
                    "REACTION_TIMEOUT_SOFT_REACTION"
                    if state.wait_type == "REACTION"
                    else "REACTION_TIMEOUT_SOFT_MITIGATION"
                )
                if len(reaction_timeout_samples) < 50:
                    symbol, strategy = key.split(":", 1) if ":" in key else (key, "UNKNOWN")
                    reaction_timeout_samples.append(
                        _ReactionTimeoutSample(
                            ts_utc=candle.timestamp.isoformat(),
                            symbol=symbol,
                            strategy=strategy,
                            state=state.wait_type,
                            waited_bars=int(elapsed),
                            reason=reason,
                        )
                    )

        if trade_start is not None and candle.timestamp < trade_start:
            continue

        if pending is not None and i > pending.expiry_index:
            pending = None

        if pending is not None:
            if bool(config.backtest_tuning.no_overnight) and in_rollover_entry_block_window(
                ts=candle.timestamp,
                swap_hour=swap_hour,
                swap_minute=swap_minute,
                cfg=config.backtest_tuning,
            ):
                pending = None
        if pending is not None:
            touched = pending.entry >= candle.low and pending.entry <= candle.high
            if touched:
                slippage = slippage_now
                # Limit order fill: use the limit price + half spread, not bar close/ask.
                if pending.side == "LONG":
                    base_entry = pending.entry + (spread_now * 0.5)
                    entry_fill = base_entry + slippage
                else:
                    base_entry = pending.entry - (spread_now * 0.5)
                    entry_fill = base_entry - slippage
                initial_risk = abs(pending.entry - pending.stop)
                entry_spread_cost = max(0.0, float(spread_now)) * 0.5 * float(pending.size)
                entry_slippage_cost = abs(float(slippage)) * float(pending.size)
                open_pos = _OpenPosition(
                    side=pending.side,
                    entry=entry_fill,
                    stop=pending.stop,
                    tp=pending.tp,
                    size=pending.size,
                    opened_at=candle.timestamp,
                    initial_stop=pending.stop,
                    initial_risk=initial_risk,
                    initial_size=pending.size,
                    max_loss_r_cap=float(config.backtest_tuning.max_loss_r_cap),
                    tp1_trigger_r=float(config.backtest_tuning.tp1_trigger_r),
                    tp1_fraction=float(config.backtest_tuning.tp1_fraction),
                    be_offset_r=float(config.backtest_tuning.be_offset_r),
                    be_delay_bars_after_tp1=int(config.backtest_tuning.be_delay_bars_after_tp1),
                    trailing_after_tp1=bool(config.backtest_tuning.trailing_after_tp1),
                    trailing_window_bars=int(config.backtest_tuning.trailing_swing_window_bars),
                    trailing_buffer_r=float(config.backtest_tuning.trailing_buffer_r),
                    next_swap_ts=_next_rollover_timestamp(
                        candle.timestamp,
                        hour=swap_hour,
                        minute=swap_minute,
                    ),
                    realized_partial=0.0,
                    spread_cost_total=entry_spread_cost,
                    slippage_cost_total=entry_slippage_cost,
                    reason_open=pending.reason_open,
                    score=pending.score,
                    gate_bias=pending.gate_bias,
                    margin_capped=pending.margin_capped,
                )
                trades_filled += 1
                _emit_fill(
                    _dtw,
                    ts=candle.timestamp,
                    symbol=asset.epic,
                    side=open_pos.side,
                    pnl=0.0,
                    equity_after=equity,
                    reason_close="OPEN",
                    spread_cost=entry_spread_cost,
                    swap_cost=0.0,
                    extra={"entry_price": round(entry_fill, 4), "size": float(open_pos.size)},
                )
                pending = None

        if open_pos is not None:
            # Guard: skip management + exit on the bar the position was filled.
            # Processing starts on the next bar to avoid same-bar entry+exit artifacts.
            if open_pos._skip_first_bar:
                open_pos._skip_first_bar = False
                continue
            time_in_market_bars += 1
            if bool(config.backtest_tuning.no_overnight):
                mins_to_roll = minutes_to_next_rollover(
                    candle.timestamp,
                    hour=swap_hour,
                    minute=swap_minute,
                )
                force_close_before = int(config.backtest_tuning.force_close_before_rollover_minutes)
                if 0 <= mins_to_roll <= float(force_close_before):
                    close_size = float(open_pos.size)
                    open_pos.spread_cost_total += max(0.0, float(spread_now)) * 0.5 * close_size
                    open_pos.slippage_cost_total += abs(float(slippage_now)) * close_size
                    if open_pos.side == "LONG":
                        forced_exit_price = (
                            candle.bid if candle.bid is not None else (candle.close - (spread_now * 0.5))
                        )
                        remaining_pnl_instr = (forced_exit_price - open_pos.entry) * close_size
                    else:
                        forced_exit_price = (
                            candle.ask if candle.ask is not None else (candle.close + (spread_now * 0.5))
                        )
                        remaining_pnl_instr = (open_pos.entry - forced_exit_price) * close_size
                    remaining_pnl, close_fx_cost = _convert_cash_to_account(
                        amount=remaining_pnl_instr,
                        category="pnl",
                        fx_converter=fx_converter,
                        instrument_currency=instrument_currency,
                        account_currency=account_currency,
                        fx_apply_to=fx_apply_to,
                    )
                    open_pos.fx_conversion_total += close_fx_cost
                    open_pos.fx_cost_total += close_fx_cost
                    total_pnl = open_pos.realized_partial + remaining_pnl
                    _pnl_g, _pnl_n, _fees, _pnl_eq = _compute_trade_pnl_fields(
                        total_pnl=total_pnl,
                        swap_total=open_pos.swap_total,
                        swap_cost_total=open_pos.swap_cost_total,
                        spread_cost=open_pos.spread_cost_total,
                        slippage_cost=open_pos.slippage_cost_total,
                        commission_cost=open_pos.commission_total,
                        fx_cost=open_pos.fx_cost_total,
                    )
                    _eq_before = equity
                    _apply_closed_pnl(_pnl_eq, candle.timestamp)
                    daily_pnl[day_key] += _pnl_eq
                    r_mult = _trade_r_multiple(
                        total_pnl=total_pnl,
                        position=open_pos,
                        fx_converter=fx_converter,
                        instrument_currency=instrument_currency,
                        account_currency=account_currency,
                        fx_apply_to=fx_apply_to,
                    )
                    trades.append(
                        BacktestTrade(
                            epic=asset.epic,
                            side=open_pos.side,
                            entry_time=open_pos.opened_at,
                            exit_time=candle.timestamp,
                            entry_price=open_pos.entry,
                            exit_price=forced_exit_price,
                            size=open_pos.size,
                            pnl=total_pnl,
                            fees=_fees,
                            r_multiple=r_mult,
                            reason="FORCED_ROLLOVER",
                            forced_exit=True,
                            score=open_pos.score,
                            reason_open=open_pos.reason_open,
                            reason_close="FORCED_ROLLOVER",
                            gate_bias=open_pos.gate_bias,
                            spread_cost=open_pos.spread_cost_total,
                            slippage_cost=open_pos.slippage_cost_total,
                            commission_cost=open_pos.commission_total,
                            swap_cost=open_pos.swap_cost_total,
                            fx_cost=open_pos.fx_cost_total,
                            pnl_gross=_pnl_g,
                            pnl_net=_pnl_n,
                            equity_before=_eq_before,
                            equity_after=equity,
                            margin_capped=open_pos.margin_capped,
                        )
                    )
                    forced_closes_count += 1
                    _holding = (candle.timestamp - open_pos.opened_at).total_seconds() / 60.0
                    _emit_fill(
                        _dtw,
                        ts=candle.timestamp,
                        symbol=asset.epic,
                        side=open_pos.side,
                        pnl=round(total_pnl, 4),
                        equity_after=round(equity, 2),
                        reason_close="FORCED_ROLLOVER",
                        holding_min=round(_holding, 1),
                        spread_cost=round(open_pos.spread_cost_total, 4),
                        swap_cost=round(open_pos.swap_cost_total, 4),
                    )
                    if mc_model is not None:
                        mc_model.add_trade(_pnl_eq)
                        mc_model.update(equity)
                    open_pos = None
                    peak_equity = max(peak_equity, equity)
                    drawdown = peak_equity - equity
                    max_drawdown = max(max_drawdown, drawdown)
                    if _dd_halt_abs > 0 and max_drawdown >= _dd_halt_abs:
                        _dd_halted = True
                        break
                    continue
            _apply_overnight_swap_if_due(
                position=open_pos,
                candle_ts=candle.timestamp,
                swap_hour=swap_hour,
                swap_minute=swap_minute,
                long_swap_pct=long_swap_pct,
                short_swap_pct=short_swap_pct,
                fx_converter=fx_converter,
                instrument_currency=instrument_currency,
                account_currency=account_currency,
                fx_apply_to=fx_apply_to,
            )
            tp1_hit, be_moved_now = _manage_open_position(
                position=open_pos,
                candle=candle,
                candles_m5=candles_m5,
                index=i,
                spread=spread_now,
                slippage=slippage_now,
                fx_converter=fx_converter,
                instrument_currency=instrument_currency,
                account_currency=account_currency,
                fx_apply_to=fx_apply_to,
            )
            if tp1_hit:
                count_tp1_hits += 1
            if be_moved_now:
                count_be_moves += 1

            should_close, exit_price, reason = _calc_exit(
                open_pos,
                candle,
                assumed_spread=spread_now,
                slippage=slippage_now,
            )
            if should_close:
                close_size = float(open_pos.size)
                open_pos.spread_cost_total += max(0.0, float(spread_now)) * 0.5 * close_size
                open_pos.slippage_cost_total += abs(float(slippage_now)) * close_size
                if open_pos.side == "LONG":
                    remaining_pnl_instr = (exit_price - open_pos.entry) * close_size
                else:
                    remaining_pnl_instr = (open_pos.entry - exit_price) * close_size
                remaining_pnl, close_fx_cost = _convert_cash_to_account(
                    amount=remaining_pnl_instr,
                    category="pnl",
                    fx_converter=fx_converter,
                    instrument_currency=instrument_currency,
                    account_currency=account_currency,
                    fx_apply_to=fx_apply_to,
                )
                open_pos.fx_conversion_total += close_fx_cost
                open_pos.fx_cost_total += close_fx_cost
                total_pnl = open_pos.realized_partial + remaining_pnl
                _pnl_g, _pnl_n, _fees, _pnl_eq = _compute_trade_pnl_fields(
                    total_pnl=total_pnl,
                    swap_total=open_pos.swap_total,
                    swap_cost_total=open_pos.swap_cost_total,
                    spread_cost=open_pos.spread_cost_total,
                    slippage_cost=open_pos.slippage_cost_total,
                    commission_cost=open_pos.commission_total,
                    fx_cost=open_pos.fx_cost_total,
                )
                _eq_before = equity
                _apply_closed_pnl(_pnl_eq, candle.timestamp)
                daily_pnl[day_key] += _pnl_eq
                r_mult = _trade_r_multiple(
                    total_pnl=total_pnl,
                    position=open_pos,
                    fx_converter=fx_converter,
                    instrument_currency=instrument_currency,
                    account_currency=account_currency,
                    fx_apply_to=fx_apply_to,
                )
                trades.append(
                    BacktestTrade(
                        epic=asset.epic,
                        side=open_pos.side,
                        entry_time=open_pos.opened_at,
                        exit_time=candle.timestamp,
                        entry_price=open_pos.entry,
                        exit_price=exit_price,
                        size=open_pos.size,
                        pnl=total_pnl,
                        fees=_fees,
                        r_multiple=r_mult,
                        reason=reason,
                        score=open_pos.score,
                        reason_open=open_pos.reason_open,
                        reason_close=reason,
                        gate_bias=open_pos.gate_bias,
                        spread_cost=open_pos.spread_cost_total,
                        slippage_cost=open_pos.slippage_cost_total,
                        commission_cost=open_pos.commission_total,
                        swap_cost=open_pos.swap_cost_total,
                        fx_cost=open_pos.fx_cost_total,
                        pnl_gross=_pnl_g,
                        pnl_net=_pnl_n,
                        equity_before=_eq_before,
                        equity_after=equity,
                        margin_capped=open_pos.margin_capped,
                    )
                )
                _holding = (candle.timestamp - open_pos.opened_at).total_seconds() / 60.0
                _emit_fill(
                    _dtw,
                    ts=candle.timestamp,
                    symbol=asset.epic,
                    side=open_pos.side,
                    pnl=round(total_pnl, 4),
                    equity_after=round(equity, 2),
                    reason_close=reason,
                    holding_min=round(_holding, 1),
                    spread_cost=round(open_pos.spread_cost_total, 4),
                    swap_cost=round(open_pos.swap_cost_total, 4),
                )
                if mc_model is not None:
                    mc_model.add_trade(_pnl_eq)
                    mc_model.update(equity)
                open_pos = None
                peak_equity = max(peak_equity, equity)
                drawdown = peak_equity - equity
                max_drawdown = max(max_drawdown, drawdown)
                if _dd_halt_abs > 0 and max_drawdown >= _dd_halt_abs:
                    _dd_halted = True
                    break

        if _dd_halted:
            break

        if open_pos is not None or pending is not None:
            continue

        if daily_trades[day_key] >= risk_engine.effective_max_trades_per_day(equity=equity):
            blockers["RISK_MAX_TRADES_DAY"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue
        if risk_engine.should_turn_off_for_day(daily_pnl[day_key], equity=equity):
            blockers["RISK_DAILY_STOP"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        if bool(config.backtest_tuning.no_overnight) and in_rollover_entry_block_window(
            ts=candle.timestamp,
            swap_hour=swap_hour,
            swap_minute=swap_minute,
            cfg=config.backtest_tuning,
        ):
            rejected_by_reason[RejectReason.SESSION_BLOCK.value] += 1
            blockers[RejectReason.SESSION_BLOCK.value] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        t = candle.timestamp
        prev_m15_ptr = m15_ptr
        prev_h1_ptr = h1_ptr
        while (m15_ptr + 1) < len(candles_m15) and candles_m15[m15_ptr + 1].timestamp <= t:
            m15_ptr += 1
        while (h1_ptr + 1) < len(candles_h1) and candles_h1[h1_ptr + 1].timestamp <= t:
            h1_ptr += 1
        if m15_ptr < 20 or h1_ptr < 50:
            blockers["PIPELINE_WARMUP"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        if m15_ptr != prev_m15_ptr:
            if not slice_m15_live:
                slice_m15_live = _append_live_placeholder(candles_m15[: m15_ptr + 1], 15)
            else:
                for idx_m15 in range(prev_m15_ptr + 1, m15_ptr + 1):
                    latest_m15 = candles_m15[idx_m15]
                    slice_m15_live[-1] = latest_m15
                    slice_m15_live.append(_live_placeholder_from(latest_m15, 15))
        if h1_ptr != prev_h1_ptr:
            if not slice_h1_live:
                slice_h1_live = _append_live_placeholder(candles_h1[: h1_ptr + 1], 60)
            else:
                for idx_h1 in range(prev_h1_ptr + 1, h1_ptr + 1):
                    latest_h1 = candles_h1[idx_h1]
                    slice_h1_live[-1] = latest_h1
                    slice_h1_live.append(_live_placeholder_from(latest_h1, 60))

        # Periodic trim to bound memory on multi-year runs
        if len(slice_m15_live) > _SLICE_KEEP_M15 * 2:
            slice_m15_live = slice_m15_live[-_SLICE_KEEP_M15:]
        if len(slice_h1_live) > _SLICE_KEEP_H1 * 2:
            slice_h1_live = slice_h1_live[-_SLICE_KEEP_H1:]

        m15_closed_ts = candles_m15[m15_ptr].timestamp if m15_ptr >= 0 else None
        h1_closed_ts = candles_h1[h1_ptr].timestamp if h1_ptr >= 0 else None
        m15_new_close = m15_closed_ts is not None and m15_closed_ts != last_m15_closed_ts
        h1_new_close = h1_closed_ts is not None and h1_closed_ts != last_h1_closed_ts
        if m15_new_close:
            last_m15_closed_ts = m15_closed_ts
        if h1_new_close:
            last_h1_closed_ts = h1_closed_ts

        slice_m5 = slice_m5_live
        slice_m15 = slice_m15_live
        slice_h1 = slice_h1_live
        quote = None
        if candle.bid is not None and candle.ask is not None:
            quote = (candle.bid, candle.ask, spread_now)
        gate_required_side: str | None = None
        if daily_gate is not None and daily_gate.enabled and gate_result is not None:
            bias_now = str(gate_result.bias).upper()
            if bias_now in {"LONG", "SHORT", "FLAT"}:
                gate_required_side = bias_now

        routes = router.routes_for(asset.epic)
        best_outcome: StrategyOutcome | None = None
        best_route: StrategyRoute | None = None
        best_rank = float("-inf")

        def _backtest_pre_score(
            *,
            context: RoutePipelineContext,
            candidate: SetupCandidate | None,
            evaluation: StrategyEvaluation,
        ) -> StrategyEvaluation:
            del candidate
            evaluation.snapshot.setdefault("spread", spread_now)
            evaluation.snapshot.setdefault("close", candle.close)
            evaluation.metadata["spread"] = spread_now
            evaluation.metadata["close"] = candle.close
            evaluation.metadata["price_mode"] = str(backtest_context.get("price_mode_requested") or "unknown")
            evaluation.metadata["timeframe"] = str(backtest_context.get("timeframe") or config.timeframes.m5)
            evaluation.metadata["spread_mode"] = spread_mode
            evaluation.metadata["assumed_spread_used"] = assumed_spread_used
            evaluation.metadata["data_context"] = backtest_context
            bars_since_segment_start = max(0, i - segment_start_index)
            evaluation.metadata["bars_since_segment_start"] = bars_since_segment_start
            atr_runtime = atr_values[i] if 0 <= i < len(atr_values) else None
            if bars_since_segment_start >= feature_warmup_bars and atr_runtime is not None:
                atr_runtime_f = as_float_or_none(atr_runtime)
                if atr_runtime_f is not None and atr_runtime_f > 0:
                    evaluation.metadata.setdefault("atr_m5", atr_runtime_f)
                    evaluation.snapshot.setdefault("atr_m5", atr_runtime_f)
            trigger_value = evaluation.metadata.get("trigger_confirmations")
            if trigger_value is None:
                alt_trigger = evaluation.metadata.get("confirmations")
                if alt_trigger is None:
                    alt_trigger = evaluation.snapshot.get("trigger_confirmations")
                trigger_int = as_int(alt_trigger, 0)
                evaluation.metadata["trigger_confirmations"] = max(0, trigger_int)
            if quote is not None:
                evaluation.metadata["quote"] = quote
                evaluation.metadata["bid"] = quote[0]
                evaluation.metadata["ask"] = quote[1]
            return _apply_soft_reason_penalties(
                evaluation=evaluation,
                config=config,
                route_params=context.route_params,
                enabled=variant_cfg.soft_reason_penalties,
            )

        def _backtest_compute_score(
            *,
            context: RoutePipelineContext,
            bias: BiasState,
            candidate: SetupCandidate | None,
            evaluation: StrategyEvaluation,
            schedule_open: bool,
            orderflow_snapshot: OrderflowSnapshot | None,
        ) -> StrategyEvaluation:
            return _compute_v2_score(
                strategy_name=context.strategy_name,
                bias=bias,
                route_params=context.route_params,
                config=config,
                evaluation=evaluation,
                news_blocked=context.news_blocked,
                schedule_open=schedule_open,
                orderflow_snapshot=orderflow_snapshot,
                setup_side=candidate.side if candidate is not None else None,
                orderflow_settings=context.orderflow_settings,
            )

        def _backtest_normalize_and_gate(
            *,
            context: RoutePipelineContext,
            candidate: SetupCandidate | None,
            evaluation: StrategyEvaluation,
        ) -> StrategyEvaluation:
            nonlocal missing_feature_debug_logged
            if score_v3_engine is not None:
                atr_hist_slice = atr_values[max(0, i - 499) : i + 1] if atr_values else None
                evaluation = apply_score_v3(
                    score_v3_engine,
                    evaluation,
                    context.runtime.get("bias"),
                    candle=candle,
                    atr_m5=evaluation.metadata.get("atr_m5"),
                    atr_history=atr_hist_slice,
                    spread=spread_now,
                    assumed_spread=float(assumed_spread_used),
                )
                if score_v3_engine.score_history_size % 2000 == 0 and score_v3_engine.score_history_size > 0:
                    score_v3_engine.update_quantile_boundaries()
            else:
                trade_thr, small_min, small_max, dynamic_reasons = _adjust_thresholds_dynamic(
                    trade_threshold=trade_thr_base,
                    small_min=small_min_base,
                    small_max=small_max_base,
                    route_params=context.route_params,
                    evaluation=evaluation,
                    config=config,
                    enabled=variant_cfg.dynamic_threshold_bump,
                )
                if dynamic_reasons:
                    evaluation.metadata["dynamic_threshold_reasons"] = dynamic_reasons
                _ = _normalize_action_for_score(
                    evaluation=evaluation,
                    config=config,
                    trade_threshold=trade_thr,
                    small_min=small_min,
                    small_max=small_max,
                )
            bars_since_segment_start = max(0, i - segment_start_index)
            if candidate is None or bars_since_segment_start < feature_warmup_bars:
                missing_features: list[str] = []
                evaluation.metadata["is_ready"] = True
            else:
                missing_features = _missing_execution_features(
                    route_params=context.route_params,
                    evaluation=evaluation,
                )
                if missing_features and missing_feature_debug_logged < 10:
                    missing_feature_debug_logged += 1
                    LOGGER.info(
                        "Missing features | ts=%s segment=%s bars_since_segment_start=%d missing=%s",
                        t.isoformat(),
                        segment_id,
                        bars_since_segment_start,
                        ",".join(str(item) for item in missing_features),
                    )
            if missing_features:
                gate_reasons = ["PIPELINE_NOT_READY_MISSING_FEATURES"]
            else:
                gate_reasons = _quality_gate_reasons(
                    route_params=context.route_params,
                    evaluation=evaluation,
                    now=t,
                    timezone_name=config.timezone,
                )
            reaction_reasons = _apply_reaction_gate_with_timeout(
                strategy_key=f"{asset.epic}:{context.strategy_name}",
                bar_index=i,
                now=t,
                evaluation=evaluation,
                wait_states=wait_states,
                variant=variant_cfg,
                config=config,
                timeout_resets=timeout_resets,
                wait_durations=wait_durations,
                reset_block_bar=wait_reset_block_bar,
                timeout_samples=reaction_timeout_samples,
            )
            gate_reasons.extend(reaction_reasons)
            evaluation = _apply_wait_timeout_soft_mode(
                evaluation=evaluation,
                config=config,
            )
            if gate_reasons:
                for code in gate_reasons:
                    if code not in evaluation.reasons_blocking:
                        evaluation.reasons_blocking.append(code)
                evaluation.action = DecisionAction.OBSERVE
            return evaluation

        def _backtest_apply_orderflow_soft_gate(
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

        def _backtest_build_payload(
            *,
            context: RoutePipelineContext,
            candidate: SetupCandidate | None,
            evaluation: StrategyEvaluation,
        ) -> dict[str, object]:
            del candidate
            return {
                "score_total": evaluation.score_total,
                "route_priority": context.route_priority,
            }

        def _backtest_on_outcome(context: RoutePipelineContext, outcome: StrategyOutcome) -> None:
            soft_reasons = outcome.evaluation.metadata.get("soft_reasons")
            if isinstance(soft_reasons, list):
                for soft_reason in soft_reasons:
                    code = f"SOFT_REASON_{str(soft_reason).upper()}"
                    if code not in outcome.reason_codes:
                        outcome.reason_codes.append(code)

        backtest_hooks = RoutePipelineHooks(
            default_observe_evaluation=_default_observe_evaluation,
            pick_best_candidate=_pick_best_candidate,
            compute_score=_backtest_compute_score,
            normalize_and_gate=_backtest_normalize_and_gate,
            apply_orderflow_small_soft_gate=_backtest_apply_orderflow_soft_gate,
            build_payload=_backtest_build_payload,
            pre_score=_backtest_pre_score,
            build_reason_codes=lambda _ctx, evaluation: list(dict.fromkeys(evaluation.reasons_blocking)),
            on_outcome=_backtest_on_outcome,
            unknown_rank=lambda _ctx, _evaluation: -1000.0,
            rank=lambda context, evaluation: rank_score(evaluation) + (context.route_priority * 0.01),
            build_unknown_payload=lambda _context: {},
        )

        for route in routes:
            strategy = strategy_plugins.get(route.strategy)
            bundle = StrategyDataBundle(
                symbol=asset.epic,
                now=t,
                candles_h1=slice_h1,
                candles_m15=slice_m15,
                candles_m5=slice_m5,
                spread=spread_now,
                spread_history=spread_history,
                news_blocked=False,
                entry_state="WAIT",
                h1_new_close=h1_new_close,
                m15_new_close=m15_new_close,
                m5_new_close=True,
                quote=quote,
                extra={
                    "minimal_tick_buffer": asset.minimal_tick_buffer,
                    "strategy_params": route.params,
                    "strategy_risk": route.risk,
                    "origin_strategy": route.strategy,
                    "suppress_missed_opportunity_logs": True,
                },
            )
            bias_for_gate: BiasState | None = None
            preprocessed = False
            if strategy is not None:
                strategy.preprocess(asset.epic, bundle)
                bias_for_gate = strategy.compute_bias(asset.epic, bundle)
                preprocessed = True
            if gate_required_side == "FLAT" and bias_for_gate is not None:
                gate_reason = "DAILY_GATE_FLAT"
                gated_eval = _default_observe_evaluation(symbol=asset.epic, reason=gate_reason)
                gated_outcome = StrategyOutcome(
                    symbol=asset.epic,
                    strategy_name=route.strategy,
                    bias=bias_for_gate,
                    candidate=None,
                    evaluation=gated_eval,
                    order_request=None,
                    reason_codes=[gate_reason],
                    payload={"score_total": gated_eval.score_total, "route_priority": route.priority},
                )
                rank = -850.0 + (route.priority * 0.01)
                if _is_better_outcome(current=gated_outcome, current_rank=rank, best=best_outcome, best_rank=best_rank):
                    best_outcome = gated_outcome
                    best_route = route
                    best_rank = rank
                continue
            if gate_required_side in {"LONG", "SHORT"} and bias_for_gate is not None:
                bias_dir = str(getattr(bias_for_gate, "direction", "NEUTRAL")).upper()
                side_mismatch = (gate_required_side == "LONG" and bias_dir == "SHORT") or (
                    gate_required_side == "SHORT" and bias_dir == "LONG"
                )
                if side_mismatch:
                    gate_reason = "DAILY_GATE_LONG_ONLY" if gate_required_side == "LONG" else "DAILY_GATE_SHORT_ONLY"
                    gated_eval = _default_observe_evaluation(symbol=asset.epic, reason=gate_reason)
                    gated_outcome = StrategyOutcome(
                        symbol=asset.epic,
                        strategy_name=route.strategy,
                        bias=bias_for_gate,
                        candidate=None,
                        evaluation=gated_eval,
                        order_request=None,
                        reason_codes=[gate_reason],
                        payload={"score_total": gated_eval.score_total, "route_priority": route.priority},
                    )
                    rank = -840.0 + (route.priority * 0.01)
                    if _is_better_outcome(
                        current=gated_outcome, current_rank=rank, best=best_outcome, best_rank=best_rank
                    ):
                        best_outcome = gated_outcome
                        best_route = route
                        best_rank = rank
                    continue
            route_context = RoutePipelineContext(
                profile=RoutePipelineProfile.BACKTEST,
                symbol=asset.epic,
                now=t,
                timezone_name=config.timezone,
                timeframe=config.timeframes.m5,
                strategy_name=route.strategy,
                route_priority=route.priority,
                route_params=route.params,
                route_risk=route.risk,
                strategy=strategy,
                bundle=bundle,
                news_blocked=False,
                spread=spread_now,
                quote=quote,
                orderflow_provider=orderflow_provider,
                orderflow_default_mode=orderflow_default_mode,
                orderflow_default_window=orderflow_default_window,
                orderflow_full_symbols=orderflow_full_symbols,
                orderflow_settings=orderflow_settings,
                runtime={
                    "preprocessed": preprocessed,
                    "precomputed_bias": bias_for_gate,
                    "bias": bias_for_gate,
                },
            )
            route_result = evaluate_and_finalize_route(
                context=route_context,
                candidate_queue=candidate_queue,
                hooks=backtest_hooks,
            )
            outcome = route_result.outcome
            rank = route_result.rank
            if _is_better_outcome(current=outcome, current_rank=rank, best=best_outcome, best_rank=best_rank):
                best_outcome = outcome
                best_route = route
                best_rank = rank

        if best_outcome is None or best_route is None:
            blockers["NO_ROUTE"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        if best_outcome.evaluation.score_total is not None:
            score_now = float(best_outcome.evaluation.score_total)
            score_values.append(score_now)
            if score_now >= trade_thr_base:
                score_bins["trade_bin"] += 1
            elif small_min_base <= score_now <= small_max_base:
                score_bins["small_bin"] += 1
            else:
                score_bins["observe_bin"] += 1

        # ── ScoreV3 bins + shadow candidate recording ──
        if score_v3_engine is not None and best_outcome.evaluation.score_total is not None:
            _v3_score = float(best_outcome.evaluation.metadata.get("score_v3", best_outcome.evaluation.score_total))
            _v3_thr = score_v3_engine.config.trade_threshold
            _v3_smin = score_v3_engine.config.small_min
            _v3_smax = score_v3_engine.config.small_max
            if _v3_score >= _v3_thr:
                score_v3_bins["trade_bin"] += 1
            elif _v3_smin <= _v3_score <= _v3_smax:
                score_v3_bins["small_bin"] += 1
            else:
                score_v3_bins["observe_bin"] += 1

        if shadow_observer is not None and best_outcome.evaluation.score_total is not None:
            _eval = best_outcome.evaluation
            # Only record non-TRADE candidates (TRADE outcomes get real tracking)
            _shadow_action = str(_eval.action.value) if _eval.action else "OBSERVE"
            if _shadow_action != "TRADE":
                _meta = _eval.metadata if isinstance(_eval.metadata, dict) else {}
                _layers = _eval.score_layers or {}
                _penalties_map = _eval.penalties or {}
                _sc = ShadowCandidate(
                    timestamp=candle.timestamp.isoformat(),
                    symbol=asset.epic,
                    side=str(_meta.get("side", "")),
                    action=_shadow_action,
                    tier=str(_meta.get("tier", "NONE")),
                    score_v2=float(_meta.get("score_v2", _eval.score_total or 0)),
                    score_v3=float(_meta.get("score_v3")) if _meta.get("score_v3") is not None else None,
                    h1_bias_direction=str(best_outcome.bias.direction) if best_outcome.bias else "NEUTRAL",
                    trigger_confirmations=int(_meta.get("trigger_confirmations", 0)),
                    atr_m5=float(_meta.get("atr_m5", 0)),
                    spread=float(spread_now),
                    hour_utc=candle.timestamp.hour,
                    day_of_week=candle.timestamp.weekday(),
                    session=classify_session(candle.timestamp.hour),
                    entry_price=float(_meta.get("entry_price", _meta.get("fvg_mid", candle.close))),
                    stop_price=float(_meta.get("stop_price", _meta.get("sweep_level", candle.close))),
                    tp_price=float(_meta.get("tp_price", candle.close)),
                    edge_score=float(_layers.get("edge", 0)),
                    trigger_score=float(_layers.get("trigger", 0)),
                    execution_score=float(_layers.get("execution", 0)),
                    penalty_total=float(sum(_penalties_map.values())) if _penalties_map else 0,
                    gate_reasons=list(best_outcome.reason_codes or []),
                    raw_score_breakdown=dict(_eval.score_breakdown) if _eval.score_breakdown else {},
                    raw_penalties=dict(_penalties_map),
                )
                shadow_observer.record(_sc)

        meta = best_outcome.evaluation.metadata if isinstance(best_outcome.evaluation.metadata, dict) else {}
        if bool(meta.get("spread_gate_soft_penalty_applied")):
            spread_gate_adjustments["ASSUMED_OHLC_SOFT_PENALTY_APPLIED"] += 1
        if bool(meta.get("spread_gate_ohlc_hard_skipped")):
            spread_gate_adjustments["ASSUMED_OHLC_HARD_GATE_SKIPPED"] += 1

        if best_outcome.order_request is None:
            reasons = best_outcome.reason_codes or ["NO_SIGNAL"]
            if "PIPELINE_NOT_READY_MISSING_FEATURES" in reasons:
                missing_items = best_outcome.evaluation.metadata.get("missing_features")
                if isinstance(missing_items, list) and missing_items:
                    for item in missing_items:
                        key = str(item).strip() or "UNKNOWN"
                        missing_feature_counts[key] += 1
                else:
                    missing_feature_counts["UNKNOWN"] += 1
            for reason in reasons:
                blockers[reason] += 1
                if reason.startswith("EXEC_FAIL_"):
                    execution_fail_breakdown[reason] += 1
                    _collect_execution_fail_sample(
                        samples=execution_fail_samples,
                        max_samples=200,
                        ts=t,
                        symbol=asset.epic,
                        strategy=best_outcome.strategy_name,
                        reason=reason,
                        evaluation=best_outcome.evaluation,
                    )
                    if reason == "EXEC_FAIL_NO_PRICE":
                        _collect_no_price_sample(
                            samples=no_price_samples,
                            max_samples=50,
                            ts=t,
                            symbol=asset.epic,
                            strategy=best_outcome.strategy_name,
                            evaluation=best_outcome.evaluation,
                            data_context=backtest_context,
                        )
            decision_counts["NO_SIGNAL"] += 1
            _emit_decision(
                _dtw,
                ts=t,
                symbol=asset.epic,
                candidates=1,
                signal=best_outcome.strategy_name,
                features_ok=("PIPELINE_NOT_READY_MISSING_FEATURES" not in reasons),
                missing=(
                    list(best_outcome.evaluation.metadata.get("missing_features", []))
                    if isinstance(best_outcome.evaluation.metadata, dict)
                    else []
                ),
                score=float(best_outcome.evaluation.score_total)
                if best_outcome.evaluation.score_total is not None
                else None,
                threshold=trade_thr_base,
                evaluation=best_outcome.evaluation,
                reject_reason=reasons[0] if reasons else "NO_SIGNAL",
                spread_points=spread_points_now,
                session_ok=True,
                min_lot=float(asset.min_size),
                lot_step=float(asset.size_step),
            )
            continue

        order_request = best_outcome.order_request
        if daily_gate is not None and daily_gate.enabled and gate_result is not None:
            gate_reasons: list[str] = list(gate_result.reasons)
            gate_bias = str(gate_result.bias).upper()
            if gate_bias == "FLAT":
                gate_reasons.append("DAILY_GATE_FLAT")
            elif gate_bias == "LONG" and str(order_request.side).upper() != "LONG":
                gate_reasons.append("DAILY_GATE_LONG_ONLY")
            elif gate_bias == "SHORT" and str(order_request.side).upper() != "SHORT":
                gate_reasons.append("DAILY_GATE_SHORT_ONLY")
            if gate_result.allowed_strategies:
                allowed = {str(item).upper() for item in gate_result.allowed_strategies}
                if str(best_route.strategy).upper() not in allowed:
                    gate_reasons.append("DAILY_GATE_STRATEGY_BLOCKED")
            if gate_reasons:
                blocked_by_gate += 1
                for reason in list(dict.fromkeys(gate_reasons)):
                    blockers[reason] += 1
                    blocked_by_gate_reasons[reason] += 1
                if _decision_trace_file is not None:
                    _decision_trace_records.append(
                        _build_decision_trace_record(
                            ts=t,
                            symbol=asset.epic,
                            strategy=best_outcome.strategy_name,
                            evaluation=best_outcome.evaluation,
                            order_request=order_request,
                            spread_points=spread_points_now,
                            fate="DAILY_GATE",
                            detail={"gate_reasons": gate_reasons},
                        )
                    )
                _emit_decision(
                    _dtw,
                    ts=t,
                    symbol=asset.epic,
                    candidates=1,
                    signal=best_outcome.strategy_name,
                    score=float(best_outcome.evaluation.score_total)
                    if best_outcome.evaluation.score_total is not None
                    else None,
                    threshold=trade_thr_base,
                    evaluation=best_outcome.evaluation,
                    reject_reason="DAILY_GATE",
                    spread_points=spread_points_now,
                    session_ok=True,
                    min_lot=float(asset.min_size),
                    lot_step=float(asset.size_step),
                )
                decision_counts["NO_SIGNAL"] += 1
                continue

        risk_dist = abs(float(order_request.entry_price) - float(order_request.stop_price))
        if risk_dist <= 0:
            if _decision_trace_file is not None:
                _decision_trace_records.append(
                    _build_decision_trace_record(
                        ts=t,
                        symbol=asset.epic,
                        strategy=best_outcome.strategy_name,
                        evaluation=best_outcome.evaluation,
                        order_request=order_request,
                        spread_points=spread_points_now,
                        fate="ORDER_INVALID_RISK",
                    )
                )
            _emit_decision(
                _dtw,
                ts=t,
                symbol=asset.epic,
                candidates=1,
                signal=best_outcome.strategy_name,
                score=float(best_outcome.evaluation.score_total)
                if best_outcome.evaluation.score_total is not None
                else None,
                threshold=trade_thr_base,
                evaluation=best_outcome.evaluation,
                reject_reason="ORDER_INVALID_RISK",
                spread_points=spread_points_now,
                min_lot=float(asset.min_size),
                lot_step=float(asset.size_step),
            )
            blockers["ORDER_INVALID_RISK"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        expected_rr_min = float(config.backtest_tuning.expected_rr_min)
        expected_rr_lookback = max(10, int(config.backtest_tuning.expected_rr_lookback_bars))
        structure_target = _estimate_structure_target(
            side=order_request.side,
            entry=float(order_request.entry_price),
            candles=candles_m5[max(0, i - expected_rr_lookback + 1) : i + 1],
            lookback_bars=expected_rr_lookback,
        )
        requested_tp = float(order_request.take_profit)
        if structure_target is None:
            rr_target = requested_tp
        elif order_request.side == "LONG":
            rr_target = max(requested_tp, float(structure_target))
        else:
            rr_target = min(requested_tp, float(structure_target))
        expected_rr_value = _expected_rr(
            side=order_request.side,
            entry=float(order_request.entry_price),
            stop=float(order_request.stop_price),
            target=rr_target,
        )
        signal_rr = getattr(order_request, "rr", None)
        try:
            if signal_rr is not None:
                expected_rr_value = max(expected_rr_value, float(signal_rr))
        except (TypeError, ValueError):
            pass
        best_outcome.evaluation.metadata["expected_rr"] = round(expected_rr_value, 4)
        if structure_target is not None:
            best_outcome.evaluation.metadata["expected_rr_target_structure"] = float(structure_target)
        if expected_rr_value < expected_rr_min:
            if _decision_trace_file is not None:
                _decision_trace_records.append(
                    _build_decision_trace_record(
                        ts=t,
                        symbol=asset.epic,
                        strategy=best_outcome.strategy_name,
                        evaluation=best_outcome.evaluation,
                        order_request=order_request,
                        spread_points=spread_points_now,
                        fate="EXPECTED_RR_TOO_LOW",
                        detail={"expected_rr": round(expected_rr_value, 4), "min": expected_rr_min},
                    )
                )
            _emit_decision(
                _dtw,
                ts=t,
                symbol=asset.epic,
                candidates=1,
                signal=best_outcome.strategy_name,
                score=float(best_outcome.evaluation.score_total)
                if best_outcome.evaluation.score_total is not None
                else None,
                threshold=trade_thr_base,
                evaluation=best_outcome.evaluation,
                reject_reason="EXPECTED_RR_TOO_LOW",
                spread_points=spread_points_now,
                min_lot=float(asset.min_size),
                lot_step=float(asset.size_step),
            )
            blockers["EXPECTED_RR_TOO_LOW"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        is_a_plus = bool(getattr(order_request, "a_plus", False))
        _gradient_score: float | None = None
        try:
            _gradient_score = (
                float(best_outcome.evaluation.score_total) if best_outcome.evaluation.score_total is not None else None
            )
        except (TypeError, ValueError, AttributeError):
            pass
        target_total_r, _tp_profile = _resolve_tp_target_r(
            is_a_plus=is_a_plus,
            score=_gradient_score,
            tuning=config.backtest_tuning,
        )
        best_outcome.evaluation.metadata["tp_target_profile"] = _tp_profile
        target_min_r = _tp2_r_for_target_total_r(
            target_total_r=target_total_r,
            tp1_trigger_r=float(config.backtest_tuning.tp1_trigger_r),
            tp1_fraction=float(config.backtest_tuning.tp1_fraction),
            mode=str(config.backtest_tuning.tp_profile_mode),
        )
        target_max_r = target_min_r
        best_outcome.evaluation.metadata["target_r_profile_total"] = round(target_total_r, 4)
        best_outcome.evaluation.metadata["target_r_tp2"] = round(target_min_r, 4)
        tp_source = rr_target
        normalized_tp, normalized_r = _normalize_tp_by_r(
            side=order_request.side,
            entry=float(order_request.entry_price),
            stop=float(order_request.stop_price),
            requested_tp=tp_source,
            min_r=target_min_r,
            max_r=target_max_r,
        )
        order_request.take_profit = normalized_tp
        best_outcome.evaluation.metadata["target_r_normalized"] = round(normalized_r, 4)

        spread_limit_points = config.backtest_tuning.spread_limit_points
        if spread_limit_points is not None and spread_points_now > float(spread_limit_points):
            if _decision_trace_file is not None:
                _decision_trace_records.append(
                    _build_decision_trace_record(
                        ts=t,
                        symbol=asset.epic,
                        strategy=best_outcome.strategy_name,
                        evaluation=best_outcome.evaluation,
                        order_request=order_request,
                        spread_points=spread_points_now,
                        fate="SPREAD_TOO_WIDE",
                        detail={"spread_pts": round(spread_points_now, 2), "limit_pts": float(spread_limit_points)},
                    )
                )
            _emit_decision(
                _dtw,
                ts=t,
                symbol=asset.epic,
                candidates=1,
                signal=best_outcome.strategy_name,
                score=float(best_outcome.evaluation.score_total)
                if best_outcome.evaluation.score_total is not None
                else None,
                threshold=trade_thr_base,
                evaluation=best_outcome.evaluation,
                reject_reason="SPREAD_TOO_WIDE",
                spread_points=spread_points_now,
                min_lot=float(asset.min_size),
                lot_step=float(asset.size_step),
            )
            rejected_by_reason[RejectReason.SPREAD_TOO_WIDE.value] += 1
            blockers[RejectReason.SPREAD_TOO_WIDE.value] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        expected_move_points = abs(
            price_to_points(
                float(order_request.take_profit) - float(order_request.entry_price),
                point_size=float(asset.point_size),
            )
        )
        if expected_move_too_small(
            expected_move_points=expected_move_points,
            spread_points=spread_points_now,
            min_edge_to_cost_ratio=float(config.backtest_tuning.min_edge_to_cost_ratio),
        ):
            if _decision_trace_file is not None:
                _decision_trace_records.append(
                    _build_decision_trace_record(
                        ts=t,
                        symbol=asset.epic,
                        strategy=best_outcome.strategy_name,
                        evaluation=best_outcome.evaluation,
                        order_request=order_request,
                        spread_points=spread_points_now,
                        fate="EDGE_TOO_SMALL",
                        detail={
                            "move_pts": round(expected_move_points, 2),
                            "required_pts": round(
                                float(config.backtest_tuning.min_edge_to_cost_ratio) * spread_points_now, 2
                            ),
                        },
                    )
                )
            _emit_decision(
                _dtw,
                ts=t,
                symbol=asset.epic,
                candidates=1,
                signal=best_outcome.strategy_name,
                score=float(best_outcome.evaluation.score_total)
                if best_outcome.evaluation.score_total is not None
                else None,
                threshold=trade_thr_base,
                evaluation=best_outcome.evaluation,
                reject_reason="EDGE_TOO_SMALL",
                spread_points=spread_points_now,
                min_lot=float(asset.min_size),
                lot_step=float(asset.size_step),
            )
            rejected_by_reason[RejectReason.EDGE_TOO_SMALL.value] += 1
            blockers[RejectReason.EDGE_TOO_SMALL.value] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        signal_candidates += 1
        risk_multiplier = _risk_multiplier_for(
            evaluation=best_outcome.evaluation,
            route_risk=best_route.risk,
            config=config,
        )
        if mc_model is not None:
            risk_multiplier *= mc_model.risk_multiplier
        effective_risk_per_trade = risk_engine.effective_risk_per_trade(
            risk_multiplier=risk_multiplier,
            equity=equity,
        )
        risk_distance = abs(float(order_request.entry_price) - float(order_request.stop_price))
        if risk_distance <= 0:
            blockers["M5_INVALID_RISK_DISTANCE"] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue
        risk_cash_plan = compute_risk_cash_plan(
            risk=config.risk,
            equity=equity,
            effective_risk_per_trade=effective_risk_per_trade,
        )
        max_risk_cash = float(risk_cash_plan.max_risk_cash)
        raw_size = float(risk_cash_plan.target_risk_cash) / risk_distance if risk_distance > 0 else 0.0
        feasibility = validate_order(
            raw_size=raw_size,
            entry_price=float(order_request.entry_price),
            stop_price=float(order_request.stop_price),
            take_profit=float(order_request.take_profit),
            min_size=float(asset.min_size),
            size_step=float(asset.size_step),
            max_risk_cash=max_risk_cash,
            equity=float(equity),
            open_positions_count=0,
            max_positions=int(config.risk.max_positions),
            spread=float(spread_points_now),
            spread_limit=(float(spread_limit_points) if spread_limit_points is not None else None),
            min_stop_distance=float(asset.minimal_tick_buffer),
            free_margin=float(equity),
            margin_requirement_pct=float(config.backtest_tuning.broker_margin_requirement_pct),
            max_leverage=float(config.backtest_tuning.broker_leverage),
            margin_safety_factor=1.0,
            allow_min_size_override_if_within_risk=bool(config.risk.allow_min_size_override_if_within_risk),
        )
        if not feasibility.ok:
            reject = feasibility.reason.value if feasibility.reason is not None else "UNKNOWN_REJECT"
            if _decision_trace_file is not None:
                _decision_trace_records.append(
                    _build_decision_trace_record(
                        ts=t,
                        symbol=asset.epic,
                        strategy=best_outcome.strategy_name,
                        evaluation=best_outcome.evaluation,
                        order_request=order_request,
                        spread_points=spread_points_now,
                        fate=reject,
                        detail={
                            k: v
                            for k, v in feasibility.details.items()
                            if k
                            in (
                                "raw_size",
                                "rounded_size",
                                "min_size",
                                "risk_cash_rounded",
                                "max_risk_cash",
                                "required_margin",
                                "free_margin",
                            )
                        },
                    )
                )
            _emit_decision(
                _dtw,
                ts=t,
                symbol=asset.epic,
                candidates=1,
                signal=best_outcome.strategy_name,
                score=float(best_outcome.evaluation.score_total)
                if best_outcome.evaluation.score_total is not None
                else None,
                threshold=trade_thr_base,
                evaluation=best_outcome.evaluation,
                reject_reason=reject,
                spread_points=spread_points_now,
                size_raw=raw_size,
                size_final=float(feasibility.details.get("rounded_size", 0)),
                min_lot=float(asset.min_size),
                lot_step=float(asset.size_step),
                margin_capped=bool(feasibility.details.get("margin_capped", False)),
            )
            rejected_by_reason[reject] += 1
            blockers[reject] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue
        size = float(feasibility.details.get("rounded_size", 0.0))
        if bool(feasibility.details.get("min_size_override_used", False)):
            min_size_overrides_count += 1
        if bool(feasibility.details.get("margin_capped", False)):
            margin_capped_count += 1
        if size <= 0:
            if _decision_trace_file is not None:
                _decision_trace_records.append(
                    _build_decision_trace_record(
                        ts=t,
                        symbol=asset.epic,
                        strategy=best_outcome.strategy_name,
                        evaluation=best_outcome.evaluation,
                        order_request=order_request,
                        spread_points=spread_points_now,
                        fate="SIZE_INVALID",
                    )
                )
            _emit_decision(
                _dtw,
                ts=t,
                symbol=asset.epic,
                candidates=1,
                signal=best_outcome.strategy_name,
                score=float(best_outcome.evaluation.score_total)
                if best_outcome.evaluation.score_total is not None
                else None,
                threshold=trade_thr_base,
                evaluation=best_outcome.evaluation,
                reject_reason="SIZE_INVALID",
                spread_points=spread_points_now,
                size_raw=raw_size,
                size_final=0.0,
                min_lot=float(asset.min_size),
                lot_step=float(asset.size_step),
            )
            blockers["SIZE_MARGIN_LIMIT"] += 1
            blockers["SIZE_INVALID"] += 1
            blockers["INSUFFICIENT_EQUITY"] += 1
            rejected_by_reason[RejectReason.SIZE_TOO_SMALL.value] += 1
            decision_counts["NO_SIGNAL"] += 1
            continue

        if _decision_trace_file is not None:
            _decision_trace_records.append(
                _build_decision_trace_record(
                    ts=t,
                    symbol=asset.epic,
                    strategy=best_outcome.strategy_name,
                    evaluation=best_outcome.evaluation,
                    order_request=order_request,
                    spread_points=spread_points_now,
                    fate="ACCEPTED",
                    detail={"size": size, "risk_cash": round(risk_distance * size, 2)},
                )
            )

        _emit_decision(
            _dtw,
            ts=t,
            symbol=asset.epic,
            candidates=1,
            signal=best_outcome.strategy_name,
            score=float(best_outcome.evaluation.score_total)
            if best_outcome.evaluation.score_total is not None
            else None,
            threshold=trade_thr_base,
            evaluation=best_outcome.evaluation,
            reject_reason=None,
            spread_points=spread_points_now,
            size_raw=raw_size,
            size_final=size,
            min_lot=float(asset.min_size),
            lot_step=float(asset.size_step),
            margin_capped=bool(feasibility.details.get("margin_capped", False)),
        )

        pending = _PendingOrder(
            side=order_request.side,
            entry=order_request.entry_price,
            stop=order_request.stop_price,
            tp=order_request.take_profit,
            size=size,
            expiry_index=i + config.execution.limit_ttl_bars,
            created_at=t,
            reason_open=",".join(best_outcome.reason_codes) if best_outcome.reason_codes else "SIGNAL",
            score=best_outcome.evaluation.score_total,
            gate_bias=(str(gate_result.bias).upper() if gate_result is not None else None),
            margin_capped=bool(feasibility.details.get("margin_capped", False)),
        )
        orders_submitted += 1
        daily_trades[day_key] += 1
        decision_counts[best_outcome.evaluation.action.value] += 1

    if candles_m5:
        last_index = len(candles_m5) - 1
        for state in wait_states.values():
            if state.timed_out_soft:
                continue
            wait_durations.setdefault(state.wait_type, []).append(max(0, last_index - state.enter_bar_index))

    _write_execution_fail_debug(debug_path, execution_fail_samples)
    _write_no_price_debug(no_price_path, no_price_samples)
    _write_reaction_timeout_debug(reaction_timeout_path, reaction_timeout_samples)
    _write_decision_trace(_decision_trace_file, _decision_trace_records)

    wait_metrics: dict[str, float] = {}
    for wait_type, durations in wait_durations.items():
        if not durations:
            wait_metrics[f"{wait_type.lower()}_avg_bars"] = 0.0
            wait_metrics[f"{wait_type.lower()}_max_bars"] = 0.0
            continue
        wait_metrics[f"{wait_type.lower()}_avg_bars"] = round(sum(durations) / len(durations), 3)
        wait_metrics[f"{wait_type.lower()}_max_bars"] = float(max(durations))

    avg_score = round(sum(score_values) / len(score_values), 4) if score_values else None

    if flatten_at_chunk_end and open_pos is not None and candles_m5:
        last_candle = candles_m5[-1]
        spread_now = _spread_for(len(candles_m5) - 1)
        close_size = float(open_pos.size)
        open_pos.spread_cost_total += max(0.0, float(spread_now)) * 0.5 * close_size
        open_pos.slippage_cost_total += abs(float(_slippage_for(len(candles_m5) - 1))) * close_size
        if open_pos.side == "LONG":
            exit_price = last_candle.bid if last_candle.bid is not None else (last_candle.close - (spread_now * 0.5))
            remaining_pnl_instr = (exit_price - open_pos.entry) * close_size
        else:
            exit_price = last_candle.ask if last_candle.ask is not None else (last_candle.close + (spread_now * 0.5))
            remaining_pnl_instr = (open_pos.entry - exit_price) * close_size
        remaining_pnl, close_fx_cost = _convert_cash_to_account(
            amount=remaining_pnl_instr,
            category="pnl",
            fx_converter=fx_converter,
            instrument_currency=instrument_currency,
            account_currency=account_currency,
            fx_apply_to=fx_apply_to,
        )
        open_pos.fx_conversion_total += close_fx_cost
        open_pos.fx_cost_total += close_fx_cost
        total_pnl = open_pos.realized_partial + remaining_pnl
        _pnl_g, _pnl_n, _fees, _pnl_eq = _compute_trade_pnl_fields(
            total_pnl=total_pnl,
            swap_total=open_pos.swap_total,
            swap_cost_total=open_pos.swap_cost_total,
            spread_cost=open_pos.spread_cost_total,
            slippage_cost=open_pos.slippage_cost_total,
            commission_cost=open_pos.commission_total,
            fx_cost=open_pos.fx_cost_total,
        )
        _eq_before = equity
        _apply_closed_pnl(_pnl_eq, last_candle.timestamp)
        r_mult = _trade_r_multiple(
            total_pnl=total_pnl,
            position=open_pos,
            fx_converter=fx_converter,
            instrument_currency=instrument_currency,
            account_currency=account_currency,
            fx_apply_to=fx_apply_to,
        )
        trades.append(
            BacktestTrade(
                epic=asset.epic,
                side=open_pos.side,
                entry_time=open_pos.opened_at,
                exit_time=last_candle.timestamp,
                entry_price=open_pos.entry,
                exit_price=exit_price,
                size=close_size,
                pnl=total_pnl,
                fees=_fees,
                r_multiple=r_mult,
                reason="FORCED_CHUNK_END",
                forced_exit=True,
                score=open_pos.score,
                reason_open=open_pos.reason_open,
                reason_close="FORCED_CHUNK_END",
                gate_bias=open_pos.gate_bias,
                spread_cost=open_pos.spread_cost_total,
                slippage_cost=open_pos.slippage_cost_total,
                commission_cost=open_pos.commission_total,
                swap_cost=open_pos.swap_cost_total,
                fx_cost=open_pos.fx_cost_total,
                pnl_gross=_pnl_g,
                pnl_net=_pnl_n,
                equity_before=_eq_before,
                equity_after=equity,
                margin_capped=open_pos.margin_capped,
            )
        )
        peak_equity = max(peak_equity, equity)
        drawdown = peak_equity - equity
        max_drawdown = max(max_drawdown, drawdown)
        _holding_chunk = (last_candle.timestamp - open_pos.opened_at).total_seconds() / 60.0
        _emit_fill(
            _dtw,
            ts=last_candle.timestamp,
            symbol=asset.epic,
            side=open_pos.side,
            pnl=round(total_pnl, 4),
            equity_after=round(equity, 2),
            reason_close="FORCED_CHUNK_END",
            holding_min=round(_holding_chunk, 1),
            spread_cost=round(open_pos.spread_cost_total, 4),
            swap_cost=round(open_pos.swap_cost_total, 4),
        )
        if mc_model is not None:
            mc_model.add_trade(_pnl_eq)
            mc_model.update(equity)
        open_pos = None

    # Close the live-flush writer
    _dtw.close()

    wins = sum(1 for trade in trades if trade.pnl > 0)
    losses = sum(1 for trade in trades if trade.pnl <= 0)
    total_pnl = sum(trade.pnl for trade in trades)
    trade_count = len(trades)
    expectancy = (total_pnl / trade_count) if trade_count else 0.0
    avg_r = (sum(trade.r_multiple for trade in trades) / trade_count) if trade_count else 0.0
    win_rate = (wins / trade_count) if trade_count else 0.0
    avg_win, avg_loss, payoff_ratio, profit_factor = _trade_quality_metrics(trades)
    avg_win_r, avg_loss_r, payoff_r = _trade_r_quality_metrics(trades)
    exit_reason_distribution = _exit_reason_distribution(trades)
    spread_cost_sum = sum(float(trade.spread_cost) for trade in trades)
    slippage_cost_sum = sum(float(trade.slippage_cost) for trade in trades)
    commission_cost_sum = sum(float(trade.commission_cost) for trade in trades)
    swap_cost_sum = sum(float(trade.swap_cost) for trade in trades)
    fx_cost_sum = sum(float(trade.fx_cost) for trade in trades)
    avg_spread_points, median_spread_points, p90_spread_points = _spread_point_stats(spread_points_series)
    expectancy_net, profit_factor_net, max_drawdown_net = _net_quality_metrics(trades, equity_start_value)

    # ── ScoreV3 / shadow observer finalization ──
    _score_v3_summary: dict[str, Any] = {}
    if score_v3_engine is not None:
        score_v3_engine.update_quantile_boundaries()
        _score_v3_summary = {
            "enabled": True,
            "mode": score_v3_engine.config.mode,
            "trade_threshold": score_v3_engine.config.trade_threshold,
            "small_min": score_v3_engine.config.small_min,
            "small_max": score_v3_engine.config.small_max,
            "score_v3_bins": dict(score_v3_bins),
            "quantile_boundaries": score_v3_engine.quantile_boundaries,
            "total_scored": score_v3_engine.score_history_size,
        }
    _shadow_summary: dict[str, Any] = {}
    if shadow_observer is not None:
        shadow_observer.flush()
        shadow_observer.close()
        _shadow_summary = shadow_observer.summary()
        _shadow_out = Path(config.score_v3.shadow_output_path).with_suffix(".summary.json")
        shadow_observer.save_summary(_shadow_out)

    (
        capital_ramp_enabled,
        capital_ramp_topups_total,
        capital_ramp_topups_count,
        capital_ramp_stopped_reason,
        capital_ramp_events_payload,
    ) = _capital_ramp_summary(capital_ramp_runtime, capital_ramp_events)

    return BacktestReport(
        epic=asset.epic,
        trades=trade_count,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        expectancy=expectancy,
        avg_r=avg_r,
        max_drawdown=max_drawdown,
        time_in_market_bars=time_in_market_bars,
        equity_end=equity,
        trade_log=trades,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff_ratio,
        profit_factor=profit_factor,
        avg_win_r=avg_win_r,
        avg_loss_r=avg_loss_r,
        payoff_r=payoff_r,
        count_be_moves=count_be_moves,
        count_tp1_hits=count_tp1_hits,
        exit_reason_distribution=exit_reason_distribution,
        top_blockers=dict(blockers.most_common(10)),
        gate_block_counts=_gate_counts_from_blockers(blockers),
        missing_feature_counts=dict(missing_feature_counts),
        decision_counts=dict(decision_counts),
        signal_candidates=signal_candidates,
        wait_timeout_resets={
            "reaction": int(timeout_resets.get("REACTION", 0)),
            "mitigation": int(timeout_resets.get("MITIGATION", 0)),
            "total": int(timeout_resets.get("REACTION_TIMEOUT_RESET", 0)),
        },
        wait_metrics=wait_metrics,
        execution_fail_breakdown=dict(execution_fail_breakdown),
        avg_score=avg_score,
        score_bins=dict(score_bins),
        spread_mode=spread_mode,
        assumed_spread_used=float(assumed_spread_used),
        spread_gate_adjustments=dict(spread_gate_adjustments),
        fx_conversion_pct_used=float(config.backtest_tuning.fx_conversion_pct),
        daily_gate_mode=daily_gate_mode,
        daily_gate_bias_bars=dict(daily_gate_bias_bars),
        daily_gate_bias_days=dict(daily_gate_bias_days),
        blocked_by_gate=blocked_by_gate,
        blocked_by_gate_reasons=dict(blocked_by_gate_reasons),
        per_bias_trade_metrics=_per_bias_trade_metrics(trades),
        orders_submitted=orders_submitted,
        trades_filled=trades_filled,
        rejected_by_reason=dict(rejected_by_reason),
        spread_cost_sum=spread_cost_sum,
        slippage_cost_sum=slippage_cost_sum,
        commission_cost_sum=commission_cost_sum,
        swap_cost_sum=swap_cost_sum,
        fx_cost_sum=fx_cost_sum,
        total_pnl_net=sum(float(t.pnl_net) for t in trades) if trades else total_pnl,
        total_pnl_gross=sum(float(t.pnl_gross) for t in trades) if trades else total_pnl,
        expectancy_net=expectancy_net,
        profit_factor_net=profit_factor_net,
        max_drawdown_net=max_drawdown_net,
        account_currency=account_currency,
        instrument_currency=instrument_currency,
        fx_conversion_fee_rate_used=float(config.fx_conversion_fee_rate),
        avg_spread_points=avg_spread_points,
        median_spread_points=median_spread_points,
        p90_spread_points=p90_spread_points,
        forced_closes_count=forced_closes_count,
        min_size_overrides_count=min_size_overrides_count,
        margin_capped_count=margin_capped_count,
        score_v3_summary=_score_v3_summary,
        shadow_summary=_shadow_summary,
        capital_ramp_enabled=capital_ramp_enabled,
        capital_ramp_topups_total=capital_ramp_topups_total,
        capital_ramp_topups_count=capital_ramp_topups_count,
        capital_ramp_stopped_reason=capital_ramp_stopped_reason,
        capital_ramp_events=capital_ramp_events_payload,
    )


def run_backtest(
    *,
    config: AppConfig,
    asset: AssetConfig,
    candles_m5: list[Candle],
    assumed_spread: float = 0.2,
    slippage_points: float = 0.0,
    slippage_atr_multiplier: float = 0.0,
    daily_gate: DailyGateProvider | None = None,
    mc_model: MCAdaptiveModel | None = None,
) -> BacktestReport:
    strategy_engine = StrategyEngine(config)
    risk_engine = RiskEngine(config.risk)
    candles_m15 = aggregate_candles(candles_m5, 15)
    candles_h1 = aggregate_candles(candles_m5, 60)
    atr_values = atr(candles_m5, config.indicators.atr_period)
    daily_gate_mode = str(daily_gate.mode).lower() if daily_gate is not None else "off"
    if daily_gate is not None and daily_gate.enabled:
        daily_gate.refresh_from_candles(candles_m5)
    spread_mode = "REAL_BIDASK" if any(c.bid is not None and c.ask is not None for c in candles_m5) else "ASSUMED_OHLC"
    assumed_spread_used = float(max(0.0, assumed_spread))
    spread_bounds = _resolve_dynamic_spread_bounds(
        config=config,
        symbol=asset.epic,
        fallback_spread=assumed_spread_used,
    )
    dynamic_assumed_spread: list[float] | None = None
    if spread_mode == "ASSUMED_OHLC" and spread_bounds is not None:
        dynamic_assumed_spread = _build_dynamic_assumed_spread_series(
            candles_m5=candles_m5,
            atr_values=atr_values,
            min_spread=spread_bounds[0],
            max_spread=spread_bounds[1],
        )
        if dynamic_assumed_spread:
            assumed_spread_used = float(sum(dynamic_assumed_spread) / len(dynamic_assumed_spread))

    capital_ramp_runtime: CapitalRampRuntime | None = None
    capital_ramp_events: list[CapitalRampEvent] = []
    capital_ramp_closed_pnl = 0.0
    if bool(config.capital_ramp.enabled):
        if candles_m5:
            ramp_start_ts = candles_m5[0].timestamp
        else:
            ramp_start_ts = datetime.now(UTC)
        capital_ramp_runtime = CapitalRampRuntime.initialize(
            scope=f"CAPITAL_RAMP:BACKTEST:{asset.epic}",
            now_utc=ramp_start_ts,
            timezone_name=config.timezone,
            current_closed_pnl=0.0,
        )
        equity = float(START_EQUITY_PLN)
    else:
        equity = float(config.risk.equity)
    peak_equity = equity
    max_drawdown = 0.0
    _dd_halt_pct = float(config.backtest_tuning.max_drawdown_halt_pct)
    _dd_halt_abs = (equity * _dd_halt_pct / 100.0) if _dd_halt_pct > 0 else 0.0
    equity_start_value = float(equity)
    _dd_halted = False
    trades: list[BacktestTrade] = []
    pending_orders: list[_PendingOrder] = []
    open_positions: list[_OpenPosition] = []
    time_in_market_bars = 0
    count_be_moves = 0
    count_tp1_hits = 0
    daily_gate_bias_bars: Counter[str] = Counter()
    daily_gate_bias_days: Counter[str] = Counter()
    seen_day_bias: dict[date, str] = {}
    blocked_by_gate_reasons: Counter[str] = Counter()
    blocked_by_gate = 0

    daily_trades: dict[str, int] = {}
    daily_pnl: dict[str, float] = {}
    swap_hour, swap_minute = _parse_swap_time_utc(config.backtest_tuning.overnight_swap_time_utc)
    long_swap_pct = float(config.backtest_tuning.overnight_swap_long_pct)
    short_swap_pct = float(config.backtest_tuning.overnight_swap_short_pct)
    account_currency = str(config.account_currency).strip().upper()
    instrument_currency = str(asset.instrument_currency or asset.currency).strip().upper()
    fx_apply_to = {str(item).strip().lower() for item in config.fx_apply_to}
    fx_converter: FxConverter | None = None
    if instrument_currency != account_currency:
        fx_converter = FxConverter(
            fee_rate=float(config.fx_conversion_fee_rate),
            fee_mode=str(config.fx_fee_mode),
            rate_source=str(config.fx_rate_source),
            static_rates=config.fx_static_rates,
        )
    rejected_by_reason: Counter[str] = Counter()
    orders_submitted = 0
    trades_filled = 0
    spread_points_series: list[float] = []
    forced_closes_count = 0
    min_size_overrides_count = 0
    margin_capped_count = 0
    max_local_positions = max(1, config.portfolio.max_per_symbol)
    risk_budget_inst = PortfolioRiskBudget(
        enabled=config.risk_budget.enabled,
        risk_per_trade_pct=config.risk_budget.risk_per_trade_pct,
        max_open_risk_pct=config.risk_budget.max_open_risk_pct,
        daily_loss_limit_pct=config.risk_budget.daily_loss_limit_pct,
        daily_profit_lock_pct=config.risk_budget.daily_profit_lock_pct,
    )
    funnel = DecisionFunnel()
    m15_ptr = -1
    h1_ptr = -1

    def _capture_capital_ramp_event(event: CapitalRampEvent | None) -> None:
        if event is not None:
            capital_ramp_events.append(event)

    def _refresh_capital_ramp_for_timestamp(ts: datetime) -> None:
        nonlocal equity
        if capital_ramp_runtime is None:
            return
        topup_event, stop_event = capital_ramp_runtime.maybe_apply_topup(
            now_utc=ts,
            current_closed_pnl=capital_ramp_closed_pnl,
        )
        _capture_capital_ramp_event(topup_event)
        _capture_capital_ramp_event(stop_event)
        equity = float(
            capital_ramp_runtime.effective_equity(
                now_utc=ts,
                current_closed_pnl=capital_ramp_closed_pnl,
            )
        )

    def _apply_closed_pnl(pnl_delta: float, ts: datetime) -> None:
        nonlocal equity, capital_ramp_closed_pnl
        equity += float(pnl_delta)
        if capital_ramp_runtime is None:
            return
        capital_ramp_closed_pnl += float(pnl_delta)
        topup_event, stop_event = capital_ramp_runtime.maybe_apply_topup(
            now_utc=ts,
            current_closed_pnl=capital_ramp_closed_pnl,
        )
        _capture_capital_ramp_event(topup_event)
        _capture_capital_ramp_event(stop_event)
        equity = float(
            capital_ramp_runtime.effective_equity(
                now_utc=ts,
                current_closed_pnl=capital_ramp_closed_pnl,
            )
        )

    def _spread_for(index: int) -> float:
        candle = candles_m5[index]
        if candle.bid is not None and candle.ask is not None:
            return max(0.0, candle.ask - candle.bid)
        if dynamic_assumed_spread is not None and index < len(dynamic_assumed_spread):
            return max(0.0, float(dynamic_assumed_spread[index]))
        return max(0.0, assumed_spread)

    def _slippage_for(index: int) -> float:
        atr_term = 0.0
        if 0 <= index < len(atr_values):
            atr_val = atr_values[index]
            if atr_val is not None:
                atr_term = max(0.0, slippage_atr_multiplier * atr_val)
        return max(0.0, slippage_points) + atr_term

    def _spread_points_for(spread_price: float) -> float:
        return max(0.0, price_to_points(float(spread_price), point_size=float(asset.point_size)))

    def _cap_size_by_margin(entry_price: float, requested_size: float) -> float:
        if entry_price <= 0:
            return 0.0
        margin_requirement_pct = float(config.backtest_tuning.broker_margin_requirement_pct)
        leverage = float(config.backtest_tuning.broker_leverage)
        caps: list[float] = []
        if margin_requirement_pct > 0:
            caps.append((equity / (margin_requirement_pct / 100.0)) / entry_price)
        if leverage > 0:
            caps.append((equity * leverage) / entry_price)
        if not caps:
            return max(0.0, requested_size)
        max_size = max(0.0, min(caps))
        step = asset.size_step if asset.size_step > 0 else 0.01
        max_size = math.floor(max_size / step) * step
        if max_size < asset.min_size:
            return 0.0
        return min(max(0.0, requested_size), max_size)

    start_idx = max(config.indicators.ema_period_h1 + 10, 250)
    spread_window = max(1, int(config.spread_filter.window)) + 1
    spread_history_window: deque[float] = deque(maxlen=spread_window)
    # Keep enough history for the longest indicator window + generous buffer.
    # Periodically trim the front of slices to cap memory on multi-year runs.
    _SLICE_KEEP_M5 = max(config.indicators.ema_period_h1 * 12, start_idx * 3, 2000)
    _SLICE_KEEP_M15 = max(config.indicators.ema_period_h1 * 4, 800)
    _SLICE_KEEP_H1 = max(config.indicators.ema_period_h1 + 100, 500)
    slice_m5: list[Candle] = list(candles_m5[:start_idx])
    slice_m15: list[Candle] = []
    slice_h1: list[Candle] = []
    if capital_ramp_runtime is not None:
        for _pre in candles_m5[: min(start_idx, len(candles_m5))]:
            _refresh_capital_ramp_for_timestamp(_pre.timestamp)
    for i in range(start_idx, len(candles_m5)):
        candle = candles_m5[i]
        _refresh_capital_ramp_for_timestamp(candle.timestamp)
        spread_now = _spread_for(i)
        slippage_now = _slippage_for(i)
        spread_points_now = _spread_points_for(spread_now)
        spread_points_series.append(spread_points_now)
        spread_history_window.append(spread_now)
        spread_history = list(spread_history_window)
        slice_m5.append(candle)
        if len(slice_m5) > _SLICE_KEEP_M5 * 2:
            del slice_m5[:-_SLICE_KEEP_M5]
        day_key = candle.timestamp.date().isoformat()
        daily_trades.setdefault(day_key, 0)
        daily_pnl.setdefault(day_key, 0.0)
        gate_result = None
        if daily_gate is not None and daily_gate.enabled:
            gate_result = daily_gate.evaluate(
                ts=candle.timestamp,
                symbol=asset.epic,
                spread=spread_now,
            )
            gate_bias = str(gate_result.bias).upper()
            daily_gate_bias_bars[gate_bias] += 1
            gate_day = candle.timestamp.astimezone(UTC).date()
            if gate_day not in seen_day_bias:
                seen_day_bias[gate_day] = gate_bias
                daily_gate_bias_days[gate_bias] += 1

        # ---- process pending orders ----
        _pend_remove: list[int] = []
        for _qi, _pend in enumerate(pending_orders):
            if i > _pend.expiry_index:
                funnel.orders_expired_ttl += 1
                _pend_remove.append(_qi)
                continue
            if bool(config.backtest_tuning.no_overnight) and in_rollover_entry_block_window(
                ts=candle.timestamp,
                swap_hour=swap_hour,
                swap_minute=swap_minute,
                cfg=config.backtest_tuning,
            ):
                _pend_remove.append(_qi)
                continue
            touched = _pend.entry >= candle.low and _pend.entry <= candle.high
            if not touched:
                continue
            slippage = slippage_now
            # Limit order fill: use the limit price + half spread, not bar close/ask.
            if _pend.side == "LONG":
                base_entry = _pend.entry + (spread_now * 0.5)
                entry_fill = base_entry + slippage
            else:
                base_entry = _pend.entry - (spread_now * 0.5)
                entry_fill = base_entry - slippage
            initial_risk = abs(_pend.entry - _pend.stop)
            entry_spread_cost = max(0.0, float(spread_now)) * 0.5 * float(_pend.size)
            entry_slippage_cost = abs(float(slippage)) * float(_pend.size)
            open_positions.append(
                _OpenPosition(
                    side=_pend.side,
                    entry=entry_fill,
                    stop=_pend.stop,
                    tp=_pend.tp,
                    size=_pend.size,
                    opened_at=candle.timestamp,
                    initial_stop=_pend.stop,
                    initial_risk=initial_risk,
                    initial_size=_pend.size,
                    max_loss_r_cap=float(config.backtest_tuning.max_loss_r_cap),
                    tp1_trigger_r=float(config.backtest_tuning.tp1_trigger_r),
                    tp1_fraction=float(config.backtest_tuning.tp1_fraction),
                    be_offset_r=float(config.backtest_tuning.be_offset_r),
                    be_delay_bars_after_tp1=int(config.backtest_tuning.be_delay_bars_after_tp1),
                    trailing_after_tp1=bool(config.backtest_tuning.trailing_after_tp1),
                    trailing_window_bars=int(config.backtest_tuning.trailing_swing_window_bars),
                    trailing_buffer_r=float(config.backtest_tuning.trailing_buffer_r),
                    next_swap_ts=_next_rollover_timestamp(
                        candle.timestamp,
                        hour=swap_hour,
                        minute=swap_minute,
                    ),
                    realized_partial=0.0,
                    spread_cost_total=entry_spread_cost,
                    slippage_cost_total=entry_slippage_cost,
                    reason_open=_pend.reason_open,
                    score=_pend.score,
                    gate_bias=_pend.gate_bias,
                    margin_capped=_pend.margin_capped,
                )
            )
            trades_filled += 1
            funnel.filled_orders += 1
            funnel.trades_opened += 1
            _pend_remove.append(_qi)
        for _ri in reversed(_pend_remove):
            pending_orders.pop(_ri)

        # ---- process open positions ----
        _pos_remove: list[int] = []
        for _pi, _pos in enumerate(open_positions):
            # Guard: skip management + exit on the bar the position was filled.
            if _pos._skip_first_bar:
                _pos._skip_first_bar = False
                continue
            time_in_market_bars += 1
            if bool(config.backtest_tuning.no_overnight):
                mins_to_roll = minutes_to_next_rollover(
                    candle.timestamp,
                    hour=swap_hour,
                    minute=swap_minute,
                )
                force_close_before = int(config.backtest_tuning.force_close_before_rollover_minutes)
                if 0 <= mins_to_roll <= float(force_close_before):
                    close_size = float(_pos.size)
                    _pos.spread_cost_total += max(0.0, float(spread_now)) * 0.5 * close_size
                    _pos.slippage_cost_total += abs(float(slippage_now)) * close_size
                    if _pos.side == "LONG":
                        forced_exit_price = (
                            candle.bid if candle.bid is not None else (candle.close - (spread_now * 0.5))
                        )
                        remaining_pnl_instr = (forced_exit_price - _pos.entry) * close_size
                    else:
                        forced_exit_price = (
                            candle.ask if candle.ask is not None else (candle.close + (spread_now * 0.5))
                        )
                        remaining_pnl_instr = (_pos.entry - forced_exit_price) * close_size
                    remaining_pnl, close_fx_cost = _convert_cash_to_account(
                        amount=remaining_pnl_instr,
                        category="pnl",
                        fx_converter=fx_converter,
                        instrument_currency=instrument_currency,
                        account_currency=account_currency,
                        fx_apply_to=fx_apply_to,
                    )
                    _pos.fx_conversion_total += close_fx_cost
                    _pos.fx_cost_total += close_fx_cost
                    total_pnl = _pos.realized_partial + remaining_pnl
                    _pnl_g, _pnl_n, _fees, _pnl_eq = _compute_trade_pnl_fields(
                        total_pnl=total_pnl,
                        swap_total=_pos.swap_total,
                        swap_cost_total=_pos.swap_cost_total,
                        spread_cost=_pos.spread_cost_total,
                        slippage_cost=_pos.slippage_cost_total,
                        commission_cost=_pos.commission_total,
                        fx_cost=_pos.fx_cost_total,
                    )
                    _eq_before = equity
                    _apply_closed_pnl(_pnl_eq, candle.timestamp)
                    daily_pnl[day_key] += _pnl_eq
                    if risk_budget_inst.enabled:
                        risk_budget_inst.record_trade_pnl(_pnl_eq)
                    r_mult = _trade_r_multiple(
                        total_pnl=total_pnl,
                        position=_pos,
                        fx_converter=fx_converter,
                        instrument_currency=instrument_currency,
                        account_currency=account_currency,
                        fx_apply_to=fx_apply_to,
                    )
                    trades.append(
                        BacktestTrade(
                            epic=asset.epic,
                            side=_pos.side,
                            entry_time=_pos.opened_at,
                            exit_time=candle.timestamp,
                            entry_price=_pos.entry,
                            exit_price=forced_exit_price,
                            size=_pos.size,
                            pnl=total_pnl,
                            fees=_fees,
                            r_multiple=r_mult,
                            reason="FORCED_ROLLOVER",
                            forced_exit=True,
                            score=_pos.score,
                            reason_open=_pos.reason_open,
                            reason_close="FORCED_ROLLOVER",
                            gate_bias=_pos.gate_bias,
                            spread_cost=_pos.spread_cost_total,
                            slippage_cost=_pos.slippage_cost_total,
                            commission_cost=_pos.commission_total,
                            swap_cost=_pos.swap_cost_total,
                            fx_cost=_pos.fx_cost_total,
                            pnl_gross=_pnl_g,
                            pnl_net=_pnl_n,
                            equity_before=_eq_before,
                            equity_after=equity,
                            margin_capped=_pos.margin_capped,
                        )
                    )
                    forced_closes_count += 1
                    funnel.trades_closed += 1
                    if mc_model is not None:
                        mc_model.add_trade(_pnl_eq)
                        mc_model.update(equity)
                    _pos_remove.append(_pi)
                    continue
            _apply_overnight_swap_if_due(
                position=_pos,
                candle_ts=candle.timestamp,
                swap_hour=swap_hour,
                swap_minute=swap_minute,
                long_swap_pct=long_swap_pct,
                short_swap_pct=short_swap_pct,
                fx_converter=fx_converter,
                instrument_currency=instrument_currency,
                account_currency=account_currency,
                fx_apply_to=fx_apply_to,
            )

            # TP1 / BE / trailing management — MUST run BEFORE _calc_exit
            # so that stop is updated (BE / trailing) before the exit check.
            _tp1_hit, _be_moved = _manage_open_position(
                position=_pos,
                candle=candle,
                candles_m5=candles_m5,
                index=i,
                spread=spread_now,
                slippage=slippage_now,
                fx_converter=fx_converter,
                instrument_currency=instrument_currency,
                account_currency=account_currency,
                fx_apply_to=fx_apply_to,
            )
            if _tp1_hit:
                count_tp1_hits += 1
            if _be_moved:
                count_be_moves += 1

            should_close, exit_price, reason = _calc_exit(
                _pos,
                candle,
                assumed_spread=spread_now,
                slippage=slippage_now,
            )
            if should_close:
                close_size = float(_pos.size)
                _pos.spread_cost_total += max(0.0, float(spread_now)) * 0.5 * close_size
                _pos.slippage_cost_total += abs(float(slippage_now)) * close_size
                if _pos.side == "LONG":
                    remaining_pnl_instr = (exit_price - _pos.entry) * close_size
                else:
                    remaining_pnl_instr = (_pos.entry - exit_price) * close_size
                remaining_pnl, close_fx_cost = _convert_cash_to_account(
                    amount=remaining_pnl_instr,
                    category="pnl",
                    fx_converter=fx_converter,
                    instrument_currency=instrument_currency,
                    account_currency=account_currency,
                    fx_apply_to=fx_apply_to,
                )
                _pos.fx_conversion_total += close_fx_cost
                _pos.fx_cost_total += close_fx_cost
                total_pnl = _pos.realized_partial + remaining_pnl
                _pnl_g, _pnl_n, _fees, _pnl_eq = _compute_trade_pnl_fields(
                    total_pnl=total_pnl,
                    swap_total=_pos.swap_total,
                    swap_cost_total=_pos.swap_cost_total,
                    spread_cost=_pos.spread_cost_total,
                    slippage_cost=_pos.slippage_cost_total,
                    commission_cost=_pos.commission_total,
                    fx_cost=_pos.fx_cost_total,
                )
                _eq_before = equity
                _apply_closed_pnl(_pnl_eq, candle.timestamp)
                daily_pnl[day_key] += _pnl_eq
                if risk_budget_inst.enabled:
                    risk_budget_inst.record_trade_pnl(_pnl_eq)
                r_mult = _trade_r_multiple(
                    total_pnl=total_pnl,
                    position=_pos,
                    fx_converter=fx_converter,
                    instrument_currency=instrument_currency,
                    account_currency=account_currency,
                    fx_apply_to=fx_apply_to,
                )
                trades.append(
                    BacktestTrade(
                        epic=asset.epic,
                        side=_pos.side,
                        entry_time=_pos.opened_at,
                        exit_time=candle.timestamp,
                        entry_price=_pos.entry,
                        exit_price=exit_price,
                        size=_pos.size,
                        pnl=total_pnl,
                        fees=_fees,
                        r_multiple=r_mult,
                        reason=reason,
                        score=_pos.score,
                        reason_open=_pos.reason_open,
                        reason_close=reason,
                        gate_bias=_pos.gate_bias,
                        spread_cost=_pos.spread_cost_total,
                        slippage_cost=_pos.slippage_cost_total,
                        commission_cost=_pos.commission_total,
                        swap_cost=_pos.swap_cost_total,
                        fx_cost=_pos.fx_cost_total,
                        pnl_gross=_pnl_g,
                        pnl_net=_pnl_n,
                        equity_before=_eq_before,
                        equity_after=equity,
                        margin_capped=_pos.margin_capped,
                    )
                )
                funnel.trades_closed += 1
                if mc_model is not None:
                    mc_model.add_trade(_pnl_eq)
                    mc_model.update(equity)
                _pos_remove.append(_pi)
                continue
        for _ri in reversed(_pos_remove):
            open_positions.pop(_ri)
        peak_equity = max(peak_equity, equity)
        drawdown = peak_equity - equity
        max_drawdown = max(max_drawdown, drawdown)
        if _dd_halt_abs > 0 and max_drawdown >= _dd_halt_abs:
            _dd_halted = True
            break
        funnel.sample_concurrent(len(open_positions))

        # ---- check entry slots ----
        _n_active = len(open_positions) + len(pending_orders)
        if _n_active >= max_local_positions:
            continue
        # second-position rule: only allow 2nd pos if first is at BE or >= 0.7R
        if open_positions:
            _first = open_positions[0]
            _first_at_be = _first.be_moved
            _first_r = 0.0
            if _first.initial_risk > 0:
                if _first.side == "LONG":
                    _first_r = (candle.close - _first.entry) / _first.initial_risk
                else:
                    _first_r = (_first.entry - candle.close) / _first.initial_risk
            _second_ok = (
                _first_at_be
                or _first_r >= config.correlation_v2.allow_second_same_symbol_only_if.or_profit_r_greater_equal
            )
            if not _second_ok:
                funnel.record_block("SECOND_POS_RULE_BLOCKED")
                continue

        risk_budget_inst.reset_day(day_key)
        if risk_budget_inst.is_killed:
            funnel.record_block("KILL_SWITCH_DAILY_LOSS")
            continue

        if daily_trades[day_key] >= risk_engine.effective_max_trades_per_day(equity=equity):
            funnel.record_block("RISK_MAX_TRADES_DAY")
            continue
        if risk_engine.should_turn_off_for_day(daily_pnl[day_key], equity=equity):
            funnel.record_block("RISK_DAILY_STOP")
            continue
        if bool(config.backtest_tuning.no_overnight) and in_rollover_entry_block_window(
            ts=candle.timestamp,
            swap_hour=swap_hour,
            swap_minute=swap_minute,
            cfg=config.backtest_tuning,
        ):
            rejected_by_reason[RejectReason.SESSION_BLOCK.value] += 1
            continue

        t = candle.timestamp
        while (m15_ptr + 1) < len(candles_m15) and candles_m15[m15_ptr + 1].timestamp <= t:
            m15_ptr += 1
            slice_m15.append(candles_m15[m15_ptr])
        if len(slice_m15) > _SLICE_KEEP_M15 * 2:
            slice_m15 = slice_m15[-_SLICE_KEEP_M15:]
        while (h1_ptr + 1) < len(candles_h1) and candles_h1[h1_ptr + 1].timestamp <= t:
            h1_ptr += 1
            slice_h1.append(candles_h1[h1_ptr])
        if len(slice_h1) > _SLICE_KEEP_H1 * 2:
            slice_h1 = slice_h1[-_SLICE_KEEP_H1:]
        if m15_ptr <= 20 or h1_ptr <= 50:
            continue

        decision = strategy_engine.evaluate(
            epic=asset.epic,
            minimal_tick_buffer=asset.minimal_tick_buffer,
            candles_h1=slice_h1,
            candles_m15=slice_m15,
            candles_m5=slice_m5,
            current_spread=spread_now,
            spread_history=spread_history,
            news_blocked=False,
        )
        if decision.signal is None:
            continue
        signal = decision.signal
        if daily_gate is not None and daily_gate.enabled and gate_result is not None:
            gate_reasons: list[str] = list(gate_result.reasons)
            gate_bias = str(gate_result.bias).upper()
            if gate_bias == "FLAT":
                gate_reasons.append("DAILY_GATE_FLAT")
            elif gate_bias == "LONG" and str(signal.side).upper() != "LONG":
                gate_reasons.append("DAILY_GATE_LONG_ONLY")
            elif gate_bias == "SHORT" and str(signal.side).upper() != "SHORT":
                gate_reasons.append("DAILY_GATE_SHORT_ONLY")
            if gate_reasons:
                blocked_by_gate += 1
                for reason in list(dict.fromkeys(gate_reasons)):
                    blocked_by_gate_reasons[reason] += 1
                continue
        risk_dist = abs(float(signal.entry_price) - float(signal.stop_price))
        if risk_dist <= 0:
            continue

        expected_rr_lookback = max(10, int(config.backtest_tuning.expected_rr_lookback_bars))
        structure_target = _estimate_structure_target(
            side=signal.side,
            entry=float(signal.entry_price),
            candles=candles_m5[max(0, i - expected_rr_lookback + 1) : i + 1],
            lookback_bars=expected_rr_lookback,
        )
        requested_tp = float(signal.take_profit)
        if structure_target is None:
            rr_target = requested_tp
        elif signal.side == "LONG":
            rr_target = max(requested_tp, float(structure_target))
        else:
            rr_target = min(requested_tp, float(structure_target))
        expected_rr_value = _expected_rr(
            side=signal.side,
            entry=float(signal.entry_price),
            stop=float(signal.stop_price),
            target=rr_target,
        )
        try:
            expected_rr_value = max(expected_rr_value, float(signal.rr))
        except (TypeError, ValueError):
            pass
        if expected_rr_value < float(config.backtest_tuning.expected_rr_min):
            continue
        is_a_plus = bool(getattr(signal, "a_plus", False))
        target_total_r, _ = _resolve_tp_target_r(
            is_a_plus=is_a_plus,
            score=None,
            tuning=config.backtest_tuning,
        )
        target_min_r = _tp2_r_for_target_total_r(
            target_total_r=target_total_r,
            tp1_trigger_r=float(config.backtest_tuning.tp1_trigger_r),
            tp1_fraction=float(config.backtest_tuning.tp1_fraction),
            mode=str(config.backtest_tuning.tp_profile_mode),
        )
        target_max_r = target_min_r
        tp_source = rr_target
        normalized_tp, _ = _normalize_tp_by_r(
            side=signal.side,
            entry=float(signal.entry_price),
            stop=float(signal.stop_price),
            requested_tp=tp_source,
            min_r=target_min_r,
            max_r=target_max_r,
        )
        signal.take_profit = normalized_tp

        spread_limit_points = config.backtest_tuning.spread_limit_points
        if spread_limit_points is not None and spread_points_now > float(spread_limit_points):
            rejected_by_reason[RejectReason.SPREAD_TOO_WIDE.value] += 1
            continue

        expected_move_points = abs(
            price_to_points(
                float(signal.take_profit) - float(signal.entry_price),
                point_size=float(asset.point_size),
            )
        )
        if expected_move_too_small(
            expected_move_points=expected_move_points,
            spread_points=spread_points_now,
            min_edge_to_cost_ratio=float(config.backtest_tuning.min_edge_to_cost_ratio),
        ):
            rejected_by_reason[RejectReason.EDGE_TOO_SMALL.value] += 1
            continue

        funnel.signal_candidates += 1
        # ---- tier-based size scaling ----
        _signal_score = float(getattr(signal, "score", 0) or 0)
        _tier = resolve_tier_from_config(_signal_score, config.score_tiers)
        if _tier.name == "OBSERVE" and config.score_tiers.enabled:
            funnel.record_block("TIER_OBSERVE")
            continue
        _tier_mult = max(0.01, _tier.size_mult)
        if mc_model is not None:
            _tier_mult *= mc_model.risk_multiplier
        effective_risk_per_trade = risk_engine.effective_risk_per_trade(
            risk_multiplier=_tier_mult,
            equity=equity,
        )
        risk_cash_plan = compute_risk_cash_plan(
            risk=config.risk,
            equity=equity,
            effective_risk_per_trade=effective_risk_per_trade,
        )
        max_risk_cash = float(risk_cash_plan.max_risk_cash)
        raw_size = float(risk_cash_plan.target_risk_cash) / risk_dist if risk_dist > 0 else 0.0
        # ---- risk budget gate ----
        _open_risk = sum(abs(p.entry - p.initial_stop) * p.size for p in open_positions)
        _new_risk = risk_dist * raw_size
        _budget_check = risk_budget_inst.check_can_open(
            equity=equity,
            new_trade_risk=_new_risk,
            open_positions_risk=_open_risk,
        )
        if not _budget_check.allowed:
            for _br in _budget_check.reasons:
                funnel.record_block(_br)
            funnel.blocked_by_risk_budget += 1
            continue
        funnel.proposals_created += 1
        feasibility = validate_order(
            raw_size=raw_size,
            entry_price=float(signal.entry_price),
            stop_price=float(signal.stop_price),
            take_profit=float(signal.take_profit),
            min_size=float(asset.min_size),
            size_step=float(asset.size_step),
            max_risk_cash=max_risk_cash,
            equity=float(equity),
            open_positions_count=len(open_positions),
            max_positions=max_local_positions,
            spread=float(spread_points_now),
            spread_limit=(float(spread_limit_points) if spread_limit_points is not None else None),
            min_stop_distance=float(asset.minimal_tick_buffer),
            free_margin=float(equity),
            margin_requirement_pct=float(config.backtest_tuning.broker_margin_requirement_pct),
            max_leverage=float(config.backtest_tuning.broker_leverage),
            margin_safety_factor=1.0,
            allow_min_size_override_if_within_risk=bool(config.risk.allow_min_size_override_if_within_risk),
        )
        if not feasibility.ok:
            reject = feasibility.reason.value if feasibility.reason is not None else "UNKNOWN_REJECT"
            rejected_by_reason[reject] += 1
            continue
        size = float(feasibility.details.get("rounded_size", 0.0))
        if bool(feasibility.details.get("min_size_override_used", False)):
            min_size_overrides_count += 1
        if bool(feasibility.details.get("margin_capped", False)):
            margin_capped_count += 1
        if size <= 0:
            rejected_by_reason[RejectReason.SIZE_TOO_SMALL.value] += 1
            continue

        pending_orders.append(
            _PendingOrder(
                side=signal.side,
                entry=signal.entry_price,
                stop=signal.stop_price,
                tp=signal.take_profit,
                size=size,
                expiry_index=i + config.execution.limit_ttl_bars,
                created_at=t,
                reason_open=",".join(decision.reason_codes) if decision.reason_codes else "SIGNAL",
                score=_signal_score if _signal_score else None,
                gate_bias=(str(gate_result.bias).upper() if gate_result is not None else None),
                margin_capped=bool(feasibility.details.get("margin_capped", False)),
            )
        )
        orders_submitted += 1
        funnel.orders_placed += 1
        daily_trades[day_key] += 1

    wins = sum(1 for trade in trades if trade.pnl > 0)
    losses = sum(1 for trade in trades if trade.pnl <= 0)
    total_pnl = sum(trade.pnl for trade in trades)
    trade_count = len(trades)
    expectancy = (total_pnl / trade_count) if trade_count else 0.0
    avg_r = (sum(trade.r_multiple for trade in trades) / trade_count) if trade_count else 0.0
    win_rate = (wins / trade_count) if trade_count else 0.0
    avg_win, avg_loss, payoff_ratio, profit_factor = _trade_quality_metrics(trades)
    avg_win_r, avg_loss_r, payoff_r = _trade_r_quality_metrics(trades)
    exit_reason_distribution = _exit_reason_distribution(trades)
    spread_cost_sum = sum(float(trade.spread_cost) for trade in trades)
    slippage_cost_sum = sum(float(trade.slippage_cost) for trade in trades)
    commission_cost_sum = sum(float(trade.commission_cost) for trade in trades)
    swap_cost_sum = sum(float(trade.swap_cost) for trade in trades)
    fx_cost_sum = sum(float(trade.fx_cost) for trade in trades)
    avg_spread_points, median_spread_points, p90_spread_points = _spread_point_stats(spread_points_series)
    expectancy_net, profit_factor_net, max_drawdown_net = _net_quality_metrics(trades, equity_start_value)
    (
        capital_ramp_enabled,
        capital_ramp_topups_total,
        capital_ramp_topups_count,
        capital_ramp_stopped_reason,
        capital_ramp_events_payload,
    ) = _capital_ramp_summary(capital_ramp_runtime, capital_ramp_events)

    return BacktestReport(
        epic=asset.epic,
        trades=trade_count,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        expectancy=expectancy,
        avg_r=avg_r,
        max_drawdown=max_drawdown,
        time_in_market_bars=time_in_market_bars,
        equity_end=equity,
        trade_log=trades,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff_ratio,
        profit_factor=profit_factor,
        avg_win_r=avg_win_r,
        avg_loss_r=avg_loss_r,
        payoff_r=payoff_r,
        count_be_moves=count_be_moves,
        count_tp1_hits=count_tp1_hits,
        exit_reason_distribution=exit_reason_distribution,
        spread_mode=spread_mode,
        assumed_spread_used=float(assumed_spread_used),
        fx_conversion_pct_used=float(config.backtest_tuning.fx_conversion_pct),
        daily_gate_mode=daily_gate_mode,
        daily_gate_bias_bars=dict(daily_gate_bias_bars),
        daily_gate_bias_days=dict(daily_gate_bias_days),
        blocked_by_gate=blocked_by_gate,
        blocked_by_gate_reasons=dict(blocked_by_gate_reasons),
        per_bias_trade_metrics=_per_bias_trade_metrics(trades),
        orders_submitted=orders_submitted,
        trades_filled=trades_filled,
        rejected_by_reason=dict(rejected_by_reason),
        spread_cost_sum=spread_cost_sum,
        slippage_cost_sum=slippage_cost_sum,
        commission_cost_sum=commission_cost_sum,
        swap_cost_sum=swap_cost_sum,
        fx_cost_sum=fx_cost_sum,
        total_pnl_net=sum(float(t.pnl_net) for t in trades) if trades else total_pnl,
        total_pnl_gross=sum(float(t.pnl_gross) for t in trades) if trades else total_pnl,
        expectancy_net=expectancy_net,
        profit_factor_net=profit_factor_net,
        max_drawdown_net=max_drawdown_net,
        account_currency=account_currency,
        instrument_currency=instrument_currency,
        fx_conversion_fee_rate_used=float(config.fx_conversion_fee_rate),
        avg_spread_points=avg_spread_points,
        median_spread_points=median_spread_points,
        p90_spread_points=p90_spread_points,
        forced_closes_count=forced_closes_count,
        min_size_overrides_count=min_size_overrides_count,
        margin_capped_count=margin_capped_count,
        capital_ramp_enabled=capital_ramp_enabled,
        capital_ramp_topups_total=capital_ramp_topups_total,
        capital_ramp_topups_count=capital_ramp_topups_count,
        capital_ramp_stopped_reason=capital_ramp_stopped_reason,
        capital_ramp_events=capital_ramp_events_payload,
        decision_funnel=funnel.to_dict(
            total_pnl=total_pnl,
            expectancy=expectancy,
            max_drawdown=max_drawdown,
            avg_r=avg_r,
            win_rate=win_rate,
        ),
    )


def run_backtest_from_csv(
    *,
    config: AppConfig,
    asset: AssetConfig,
    csv_path: str | Path,
    assumed_spread: float = 0.2,
    slippage_points: float = 0.0,
    slippage_atr_multiplier: float = 0.0,
    daily_gate: DailyGateProvider | None = None,
) -> BacktestReport:
    candles = load_candles_csv(csv_path)
    return run_backtest(
        config=config,
        asset=asset,
        candles_m5=candles,
        assumed_spread=assumed_spread,
        slippage_points=slippage_points,
        slippage_atr_multiplier=slippage_atr_multiplier,
        daily_gate=daily_gate,
    )


def run_walk_forward_from_csv(
    *,
    config: AppConfig,
    asset: AssetConfig,
    csv_path: str | Path,
    wf_splits: int = 4,
    assumed_spread: float = 0.2,
    slippage_points: float = 0.0,
    slippage_atr_multiplier: float = 0.0,
    daily_gate: DailyGateProvider | None = None,
) -> WalkForwardReport:
    candles = load_candles_csv(csv_path)
    return run_walk_forward(
        config=config,
        asset=asset,
        candles_m5=candles,
        wf_splits=wf_splits,
        assumed_spread=assumed_spread,
        slippage_points=slippage_points,
        slippage_atr_multiplier=slippage_atr_multiplier,
        daily_gate=daily_gate,
    )


def run_walk_forward(
    *,
    config: AppConfig,
    asset: AssetConfig,
    candles_m5: list[Candle],
    wf_splits: int = 4,
    assumed_spread: float = 0.2,
    slippage_points: float = 0.0,
    slippage_atr_multiplier: float = 0.0,
    daily_gate: DailyGateProvider | None = None,
) -> WalkForwardReport:
    if wf_splits < 2:
        wf_splits = 2
    chunk = len(candles_m5) // wf_splits
    if chunk < 260:
        raise ValueError("Not enough candles for walk-forward splits")

    roll_equity = bool(config.backtest_tuning.wf_roll_equity_forward)
    split_config = config
    reports: list[BacktestReport] = []
    for split in range(wf_splits):
        start = split * chunk
        end = (split + 1) * chunk if split < (wf_splits - 1) else len(candles_m5)
        part = candles_m5[start:end]
        if len(part) < 260:
            continue
        rpt = run_backtest(
            config=split_config,
            asset=asset,
            candles_m5=part,
            assumed_spread=assumed_spread,
            slippage_points=slippage_points,
            slippage_atr_multiplier=slippage_atr_multiplier,
            daily_gate=daily_gate,
        )
        reports.append(rpt)
        if roll_equity and rpt.equity_end > 0:
            split_config = split_config.model_copy(
                update={"risk": split_config.risk.model_copy(update={"equity": rpt.equity_end})},
            )
    if not reports:
        raise ValueError("No valid walk-forward splits produced")

    aggregate = aggregate_backtest_reports(
        config=config,
        asset=asset,
        reports=reports,
    )
    return WalkForwardReport(epic=asset.epic, splits=reports, aggregate=aggregate)


def run_walk_forward_multi_strategy(
    *,
    config: AppConfig,
    asset: AssetConfig,
    candles_m5: list[Candle],
    wf_splits: int = 4,
    assumed_spread: float = 0.2,
    slippage_points: float = 0.0,
    slippage_atr_multiplier: float = 0.0,
    variant: BacktestVariant | None = None,
    execution_debug_path: str | Path | None = None,
    no_price_debug_path: str | Path | None = None,
    reaction_timeout_debug_path: str | Path | None = None,
    data_context: dict[str, Any] | None = None,
    daily_gate: DailyGateProvider | None = None,
) -> WalkForwardReport:
    if wf_splits < 2:
        wf_splits = 2
    chunk = len(candles_m5) // wf_splits
    if chunk < 260:
        raise ValueError("Not enough candles for walk-forward splits")

    roll_equity = bool(config.backtest_tuning.wf_roll_equity_forward)
    split_config = config
    reports: list[BacktestReport] = []
    for split in range(wf_splits):
        start = split * chunk
        end = (split + 1) * chunk if split < (wf_splits - 1) else len(candles_m5)
        part = candles_m5[start:end]
        if len(part) < 260:
            continue
        rpt = run_backtest_multi_strategy(
            config=split_config,
            asset=asset,
            candles_m5=part,
            assumed_spread=assumed_spread,
            slippage_points=slippage_points,
            slippage_atr_multiplier=slippage_atr_multiplier,
            variant=variant,
            execution_debug_path=execution_debug_path,
            no_price_debug_path=no_price_debug_path,
            reaction_timeout_debug_path=reaction_timeout_debug_path,
            data_context=data_context,
            daily_gate=daily_gate,
        )
        reports.append(rpt)
        if roll_equity and rpt.equity_end > 0:
            split_config = split_config.model_copy(
                update={"risk": split_config.risk.model_copy(update={"equity": rpt.equity_end})},
            )
    if not reports:
        raise ValueError("No valid walk-forward splits produced")

    aggregate = aggregate_backtest_reports(
        config=config,
        asset=asset,
        reports=reports,
    )
    return WalkForwardReport(epic=asset.epic, splits=reports, aggregate=aggregate)
