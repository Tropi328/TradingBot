"""Position sizing module with multiple sizing modes.

Modes
-----
fixed_qty        : Always use a fixed lot size (``qty``).
fixed_notional   : Size so that notional = ``notional_value`` (in instrument ccy).
risk_pct_equity  : Size so that ``SL-distance × qty ≈ equity × risk_pct``.

All modes respect ``min_qty``, ``max_qty``, and ``size_step``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class SizingMode(str, Enum):
    FIXED_QTY = "fixed_qty"
    FIXED_NOTIONAL = "fixed_notional"
    RISK_PCT_EQUITY = "risk_pct_equity"


@dataclass(slots=True)
class SizingRequest:
    """All inputs needed to compute a position size."""

    mode: SizingMode
    equity: float
    entry_price: float
    sl_price: float
    # mode-specific ---------------------------------------------------------
    qty: float = 0.0               # fixed_qty: desired quantity
    notional_value: float = 0.0    # fixed_notional: target notional
    risk_pct: float = 0.005        # risk_pct_equity: fraction of equity
    value_per_point: float = 1.0   # multiplier (used for index CFDs etc.)
    # constraints -----------------------------------------------------------
    min_qty: float = 0.01
    max_qty: float = 100.0
    size_step: float = 0.01
    max_leverage: float = 0.0      # 0 = unlimited


@dataclass(slots=True)
class SizingResult:
    """Output of the sizer."""

    qty: float
    risk_cash: float               # estimated risk in account currency
    notional: float                # entry_price × qty
    mode_used: str
    clamped: bool = False          # True if qty was clamped by min/max


class GoldCfdSizingStatus(str, Enum):
    OK = "OK"
    BLOCKED_INVALID_SL = "BLOCKED_INVALID_SL"
    BLOCKED_MIN_QTY = "BLOCKED_MIN_QTY"
    BLOCKED_RISK_TOO_HIGH = "BLOCKED_RISK_TOO_HIGH"


@dataclass(slots=True)
class GoldCfdSizingResult:
    status: str
    raw_qty_oz: float
    rounded_qty_oz: float
    risk_pln: float
    real_risk_pln: float
    sl_pts: float
    point_size_pln: float
    debug_reason: str


def _floor_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step


def compute_gold_cfd_position_size(
    *,
    equity_pln: float,
    risk_pct: float,
    entry_price_pln_per_oz: float,
    stop_loss_price_pln_per_oz: float,
    max_risk_multiplier: float = 1.05,
    min_qty_oz: float = 0.01,
    qty_step_oz: float = 0.01,
    point_size_pln: float = 0.01,
    symbol: str = "XAUUSD",
) -> GoldCfdSizingResult:
    if point_size_pln <= 0:
        raise ValueError("point_size_pln must be > 0")
    if qty_step_oz <= 0:
        raise ValueError("qty_step_oz must be > 0")
    if min_qty_oz <= 0:
        raise ValueError("min_qty_oz must be > 0")

    risk_pln = float(equity_pln) * float(risk_pct)
    sl_pts = abs(float(entry_price_pln_per_oz) - float(stop_loss_price_pln_per_oz)) / float(point_size_pln)

    if sl_pts <= 0:
        return GoldCfdSizingResult(
            status=GoldCfdSizingStatus.BLOCKED_INVALID_SL.value,
            raw_qty_oz=0.0,
            rounded_qty_oz=0.0,
            risk_pln=round(risk_pln, 8),
            real_risk_pln=0.0,
            sl_pts=round(sl_pts, 8),
            point_size_pln=round(float(point_size_pln), 8),
            debug_reason=(
                f"{symbol}: invalid SL distance. "
                f"entry={entry_price_pln_per_oz:.8f}, stop={stop_loss_price_pln_per_oz:.8f}, sl_pts={sl_pts:.8f}"
            ),
        )

    value_per_point_pln_for_1oz = float(point_size_pln)
    raw_qty_oz = risk_pln / (sl_pts * value_per_point_pln_for_1oz)
    rounded_qty_oz = _floor_step(raw_qty_oz, float(qty_step_oz))

    if rounded_qty_oz < float(min_qty_oz):
        real_risk_pln = rounded_qty_oz * sl_pts * float(point_size_pln)
        return GoldCfdSizingResult(
            status=GoldCfdSizingStatus.BLOCKED_MIN_QTY.value,
            raw_qty_oz=round(raw_qty_oz, 8),
            rounded_qty_oz=round(rounded_qty_oz, 8),
            risk_pln=round(risk_pln, 8),
            real_risk_pln=round(real_risk_pln, 8),
            sl_pts=round(sl_pts, 8),
            point_size_pln=round(float(point_size_pln), 8),
            debug_reason=(
                f"{symbol}: rounded qty below min. "
                f"raw_qty={raw_qty_oz:.8f}, rounded_qty={rounded_qty_oz:.8f}, min_qty={min_qty_oz:.8f}"
            ),
        )

    real_risk_pln = rounded_qty_oz * sl_pts * float(point_size_pln)
    if real_risk_pln > (risk_pln * float(max_risk_multiplier)):
        return GoldCfdSizingResult(
            status=GoldCfdSizingStatus.BLOCKED_RISK_TOO_HIGH.value,
            raw_qty_oz=round(raw_qty_oz, 8),
            rounded_qty_oz=round(rounded_qty_oz, 8),
            risk_pln=round(risk_pln, 8),
            real_risk_pln=round(real_risk_pln, 8),
            sl_pts=round(sl_pts, 8),
            point_size_pln=round(float(point_size_pln), 8),
            debug_reason=(
                f"{symbol}: real risk above cap. "
                f"real_risk={real_risk_pln:.8f}, cap={(risk_pln * float(max_risk_multiplier)):.8f}"
            ),
        )

    return GoldCfdSizingResult(
        status=GoldCfdSizingStatus.OK.value,
        raw_qty_oz=round(raw_qty_oz, 8),
        rounded_qty_oz=round(rounded_qty_oz, 8),
        risk_pln=round(risk_pln, 8),
        real_risk_pln=round(real_risk_pln, 8),
        sl_pts=round(sl_pts, 8),
        point_size_pln=round(float(point_size_pln), 8),
        debug_reason=(
            f"{symbol}: sizing ok. "
            f"risk={risk_pln:.8f}, sl_pts={sl_pts:.8f}, raw_qty={raw_qty_oz:.8f}, rounded_qty={rounded_qty_oz:.8f}"
        ),
    )


def compute_position_size(req: SizingRequest) -> SizingResult:
    """Return the position size for *req*.

    Raises ``ValueError`` for invalid inputs.
    """
    entry = float(req.entry_price)
    sl = float(req.sl_price)
    sl_distance = abs(entry - sl)
    equity = max(0.0, float(req.equity))
    step = max(1e-12, float(req.size_step))
    min_q = max(0.0, float(req.min_qty))
    max_q = max(min_q, float(req.max_qty)) if req.max_qty > 0 else 1e18

    # ---- raw qty per mode ------------------------------------------------
    mode = SizingMode(req.mode)

    if mode is SizingMode.FIXED_QTY:
        raw = float(req.qty)

    elif mode is SizingMode.FIXED_NOTIONAL:
        if entry <= 0:
            raise ValueError("entry_price must be > 0 for fixed_notional")
        raw = float(req.notional_value) / entry

    elif mode is SizingMode.RISK_PCT_EQUITY:
        if sl_distance <= 0:
            raise ValueError("SL distance must be > 0 for risk_pct_equity")
        risk_cash = equity * max(0.0, float(req.risk_pct))
        vpp = max(1e-12, float(req.value_per_point))
        raw = risk_cash / (sl_distance * vpp)

    else:
        raise ValueError(f"Unknown sizing mode: {req.mode}")

    # ---- floor to step & clamp -------------------------------------------
    sized = _floor_step(max(0.0, raw), step)
    clamped = False
    if sized < min_q:
        sized = min_q
        clamped = True
    if sized > max_q:
        sized = max_q
        clamped = True
    sized = round(sized, 8)

    # ---- leverage check --------------------------------------------------
    if req.max_leverage > 0 and entry > 0:
        max_notional = equity * float(req.max_leverage)
        max_qty_lev = _floor_step(max_notional / entry, step)
        if sized > max_qty_lev:
            sized = max(min_q, round(max_qty_lev, 8))
            clamped = True

    notional = entry * sized
    risk_cash = sl_distance * sized * max(1e-12, float(req.value_per_point))

    return SizingResult(
        qty=sized,
        risk_cash=risk_cash,
        notional=notional,
        mode_used=mode.value,
        clamped=clamped,
    )
