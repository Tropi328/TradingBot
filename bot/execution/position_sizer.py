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


def _floor_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step


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
