"""Trade cost computation (PnL, swap, FX) — extracted from engine.py."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from bot.backtest.models import _OpenPosition
from bot.execution.fx import FxConverter

LOGGER = logging.getLogger(__name__)


def _compute_trade_pnl_fields(
    *,
    total_pnl: float,
    swap_total: float,
    swap_cost_total: float,
    spread_cost: float,
    slippage_cost: float,
    commission_cost: float,
    fx_cost: float,
) -> tuple[float, float, float, float]:
    """Return (pnl_gross, pnl_net, fees, pnl_for_equity).

    Definitions
    -----------
    pnl_gross : price-movement P&L *without* swap
                (swap is removed from ``total_pnl`` which contains it
                via ``realized_partial``).
    fees      : == ``-swap_cost``.  By convention the broker "fee" on a
                CFD position IS the overnight swap.  ``swap_cost_total``
                is stored as a positive number for costs, so
                ``fees = -swap_cost_total`` = ``swap_total`` (the actual
                cash-flow, negative when you pay).
    pnl_net   : ``pnl_gross + fees``
                = ``pnl_gross - swap_cost``
                = ``total_pnl - commission_cost``
                (spread / slippage / fx are already embedded in
                ``total_pnl`` through bid/ask pricing and FX conversion).
    pnl_for_equity : the amount to add to equity.  Currently equals
                ``total_pnl`` (keeping backward-compatible equity
                tracking).

    IMPORTANT -- ``fees == -swap_cost``: never subtract BOTH ``fees``
    and ``swap_cost`` from the same base -- that would double-count.
    """
    pnl_gross = total_pnl - swap_total
    fees = -swap_cost_total  # fees == -swap_cost
    pnl_net = pnl_gross + fees  # = total_pnl - commission_cost (commission~=0)
    pnl_for_equity = total_pnl  # backward compat: equity tracks via total_pnl
    return pnl_gross, pnl_net, fees, pnl_for_equity


def _convert_cash_to_account(
    *,
    amount: float,
    category: str,
    fx_converter: FxConverter | None,
    instrument_currency: str,
    account_currency: str,
    fx_apply_to: set[str],
) -> tuple[float, float]:
    if fx_converter is None or str(instrument_currency).upper() == str(account_currency).upper():
        return float(amount), 0.0
    apply_fee = str(category).strip().lower() in fx_apply_to
    converted = fx_converter.convert(
        amount=float(amount),
        from_currency=instrument_currency,
        to_currency=account_currency,
        apply_fee=apply_fee,
    )
    return float(converted.converted_amount), float(converted.fx_cost)


def _trade_r_multiple(
    *,
    total_pnl: float,
    position: _OpenPosition,
    fx_converter: FxConverter | None,
    instrument_currency: str,
    account_currency: str,
    fx_apply_to: set[str] | None,
) -> float:
    """Compute R-multiple using the trade's own initial risk (entry-stop x size)."""
    r_denom_instr = float(position.initial_risk) * float(position.initial_size)
    if r_denom_instr <= 0:
        LOGGER.warning(
            "R-multiple denominator <= 0 (risk=%.6f, size=%.6f) -- returning 0.0",
            position.initial_risk,
            position.initial_size,
        )
        return 0.0
    r_denom, _ = _convert_cash_to_account(
        amount=r_denom_instr,
        category="pnl",
        fx_converter=fx_converter,
        instrument_currency=instrument_currency,
        account_currency=account_currency,
        fx_apply_to=fx_apply_to,
    )
    if r_denom <= 0:
        LOGGER.warning(
            "R-multiple FX-converted denominator <= 0 (instr=%.6f, converted=%.6f) -- returning 0.0",
            r_denom_instr,
            r_denom,
        )
        return 0.0
    return total_pnl / r_denom


def _parse_swap_time_utc(value: str) -> tuple[int, int]:
    raw = str(value).strip()
    parts = raw.split(":")
    if len(parts) != 2:
        return 23, 0
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return 23, 0
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return 23, 0
    return hour, minute


def _next_rollover_timestamp(ts: datetime, *, hour: int, minute: int) -> datetime:
    ts_utc = ts.astimezone(UTC)
    rollover = ts_utc.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if ts_utc >= rollover:
        rollover += timedelta(days=1)
    return rollover


def _apply_overnight_swap_if_due(
    *,
    position: _OpenPosition,
    candle_ts: datetime,
    swap_hour: int,
    swap_minute: int,
    long_swap_pct: float,
    short_swap_pct: float,
    fx_converter: FxConverter | None = None,
    instrument_currency: str = "USD",
    account_currency: str = "USD",
    fx_apply_to: set[str] | None = None,
) -> float:
    if position.next_swap_ts is None:
        position.next_swap_ts = _next_rollover_timestamp(candle_ts, hour=swap_hour, minute=swap_minute)
        return 0.0

    ts_utc = candle_ts.astimezone(UTC)
    applied = 0.0
    while ts_utc >= position.next_swap_ts:
        rate_pct = float(long_swap_pct) if position.side == "LONG" else float(short_swap_pct)
        swap_instr = float(position.entry) * float(position.size) * (rate_pct / 100.0)
        fx_apply = fx_apply_to or {"pnl", "swap", "commission"}
        swap_account, swap_fx_cost = _convert_cash_to_account(
            amount=swap_instr,
            category="swap",
            fx_converter=fx_converter,
            instrument_currency=instrument_currency,
            account_currency=account_currency,
            fx_apply_to=fx_apply,
        )
        position.realized_partial += swap_account
        position.swap_total += swap_account
        position.swap_cost_total += -swap_account
        position.fx_conversion_total += swap_fx_cost
        position.fx_cost_total += swap_fx_cost
        applied += swap_account
        position.next_swap_ts = position.next_swap_ts + timedelta(days=1)
    return applied
