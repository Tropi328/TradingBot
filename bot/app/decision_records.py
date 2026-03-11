"""Decision record creation and order placement — extracted from app_main.py."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from bot.clock import utc_now
from bot.config import AppConfig, AssetConfig
from bot.data.market_data import MarketDataService
from bot.execution.orders import OrderExecutor
from bot.monitoring.alerts import AlertDispatcher
from bot.storage.journal import Journal
from bot.storage.models import ClosedPositionEvent, StrategyDecisionRecord
from bot.strategy.contracts import StrategyOutcome
from bot.strategy.risk import RiskEngine
from bot.strategy.state_machine import StrategyDecision, StrategySignal
from bot.strategy.tp_profile import tp2_r_for_target_total_r as _tp2_r_for_target_total_r

LOGGER = logging.getLogger("trading_bot")


def create_decision_record(
    decision: StrategyDecision, epic: str, side: str | None, news_blocked: bool
) -> StrategyDecisionRecord:
    return StrategyDecisionRecord(
        created_at=datetime.now(UTC),
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
    side = (
        outcome.order_request.side
        if outcome.order_request is not None
        else outcome.candidate.side
        if outcome.candidate is not None
        else None
    )
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

    has_sweep = bool(payload.get("sweep") or payload.get("sweep_level") or payload.get("m15_setup_state") == "ARMED")
    has_mss = bool(payload.get("mss") or payload.get("mss_index"))
    has_disp = bool(payload.get("displacement") or payload.get("displacement_ratio"))
    has_fvg = bool(payload.get("fvg") or payload.get("fvg_mid"))
    spread_ok = "SCALP_SPREAD_ELEVATED" not in reason_codes and "M5_SPREAD_FAIL" not in reason_codes
    pd_state = str(payload.get("pd_state") or payload.get("h1_pd_state") or "UNKNOWN")

    return StrategyDecisionRecord(
        created_at=datetime.now(UTC),
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


def apply_closed_events(
    events: list[ClosedPositionEvent],
    trading_day_str: str,
    journal: Journal,
    risk_engine: RiskEngine,
    now: datetime,
    alerts: AlertDispatcher,
) -> float:
    total_closed_pnl = 0.0
    for event in events:
        total_closed_pnl += float(event.pnl)
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
        alerts.send(
            event="POSITION_CLOSED",
            message=f"{event.epic} deal={event.deal_id} pnl={event.pnl:.2f}",
            dedupe_key=f"close-{event.deal_id}",
        )
    return total_closed_pnl


def place_single_test_order(
    order_executor: OrderExecutor,
    market_data: MarketDataService,
    assets: list[AssetConfig],
    config: AppConfig,
    dry_run: bool,
    side: str,
    test_size: float | None,
    test_epic: str | None,
) -> None:
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
    order = order_executor.place_limit_order(
        signal,
        size=size,
        epic=asset.epic,
        currency=asset.currency,
        idempotency_key=f"TEST-{asset.epic}-{int(now.timestamp())}",
    )
    LOGGER.info(
        "Test LIMIT order placed: id=%s epic=%s side=%s size=%.4f", order.order_id, order.epic, order.side, order.size
    )
    if dry_run:
        filled = order_executor.process_pending_fills(quotes_by_epic={asset.epic: (bid, ask, ask - bid)}, now=now)
        LOGGER.info("Dry-run test fill=%s", bool(filled))
