from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from bot.execution.sizing import ceil_to_step, floor_to_step


class RejectReason(str, Enum):
    SIZE_TOO_SMALL = "SIZE_TOO_SMALL"
    RISK_AFTER_ROUNDING_TOO_HIGH = "RISK_AFTER_ROUNDING_TOO_HIGH"
    INSUFFICIENT_MARGIN = "INSUFFICIENT_MARGIN"
    SL_TOO_CLOSE = "SL_TOO_CLOSE"
    TP_TOO_CLOSE = "TP_TOO_CLOSE"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    EDGE_TOO_SMALL = "EDGE_TOO_SMALL"
    MAX_POSITIONS = "MAX_POSITIONS"
    COOLDOWN = "COOLDOWN"
    SESSION_BLOCK = "SESSION_BLOCK"
    NEWS_BLOCK = "NEWS_BLOCK"


@dataclass(slots=True)
class FeasibilityResult:
    ok: bool
    reason: RejectReason | None = None
    details: dict[str, Any] = field(default_factory=dict)


def estimate_required_margin(
    *,
    entry_price: float,
    size: float,
    margin_requirement_pct: float | None = None,
    max_leverage: float | None = None,
) -> float:
    notional = max(0.0, float(entry_price) * float(size))
    if notional <= 0:
        return 0.0
    candidates: list[float] = []
    if margin_requirement_pct is not None:
        mr = float(margin_requirement_pct)
        if mr > 0:
            candidates.append(notional * (mr / 100.0))
    if max_leverage is not None:
        lev = float(max_leverage)
        if lev > 0:
            candidates.append(notional / lev)
    if not candidates:
        return 0.0
    return max(candidates)


def _result(
    *,
    ok: bool,
    reason: RejectReason | None,
    details: dict[str, Any],
) -> FeasibilityResult:
    return FeasibilityResult(ok=ok, reason=reason, details=details)


def validate_order(
    *,
    raw_size: float,
    entry_price: float,
    stop_price: float,
    take_profit: float,
    min_size: float,
    size_step: float,
    max_risk_cash: float,
    equity: float,
    open_positions_count: int,
    max_positions: int,
    spread: float | None = None,
    spread_limit: float | None = None,
    min_stop_distance: float = 0.0,
    free_margin: float | None = None,
    margin_requirement_pct: float | None = None,
    max_leverage: float | None = None,
    margin_safety_factor: float = 1.0,
    allow_min_size_override_if_within_risk: bool = False,
    cooldown_blocked: bool = False,
    session_blocked: bool = False,
    news_blocked: bool = False,
) -> FeasibilityResult:
    details: dict[str, Any] = {
        "raw_size": float(raw_size),
        "entry_price": float(entry_price),
        "stop_price": float(stop_price),
        "take_profit": float(take_profit),
    }

    if session_blocked:
        return _result(ok=False, reason=RejectReason.SESSION_BLOCK, details=details)
    if news_blocked:
        return _result(ok=False, reason=RejectReason.NEWS_BLOCK, details=details)
    if cooldown_blocked:
        return _result(ok=False, reason=RejectReason.COOLDOWN, details=details)
    if max_positions > 0 and open_positions_count >= max_positions:
        details["open_positions_count"] = int(open_positions_count)
        details["max_positions"] = int(max_positions)
        return _result(ok=False, reason=RejectReason.MAX_POSITIONS, details=details)

    spread_now = float(spread) if spread is not None else None
    if spread_now is not None and spread_limit is not None and spread_now > float(spread_limit):
        details["spread"] = spread_now
        details["spread_limit"] = float(spread_limit)
        return _result(ok=False, reason=RejectReason.SPREAD_TOO_WIDE, details=details)

    stop_distance = abs(float(entry_price) - float(stop_price))
    tp_distance = abs(float(take_profit) - float(entry_price))
    min_stop = max(0.0, float(min_stop_distance))
    details["stop_distance"] = stop_distance
    details["tp_distance"] = tp_distance
    details["min_stop_distance"] = min_stop
    if stop_distance < min_stop:
        return _result(ok=False, reason=RejectReason.SL_TOO_CLOSE, details=details)
    if tp_distance < min_stop:
        return _result(ok=False, reason=RejectReason.TP_TOO_CLOSE, details=details)

    if size_step <= 0:
        raise ValueError("size_step must be > 0")
    rounded_size = floor_to_step(max(0.0, float(raw_size)), float(size_step))
    rounded_size = round(max(0.0, rounded_size), 8)
    details["rounded_size"] = rounded_size
    details["min_size"] = float(min_size)
    if rounded_size < float(min_size):
        if allow_min_size_override_if_within_risk:
            min_size_rounded = ceil_to_step(float(min_size), float(size_step))
            min_size_rounded = round(max(0.0, min_size_rounded), 8)
            risk_cash_at_min_size = stop_distance * min_size_rounded
            details["min_size_override_size"] = min_size_rounded
            details["risk_cash_at_min_size"] = risk_cash_at_min_size
            if risk_cash_at_min_size <= float(max_risk_cash) + 1e-12:
                rounded_size = min_size_rounded
                details["rounded_size"] = rounded_size
                details["min_size_override_used"] = True
            else:
                details["min_size_override_used"] = False
                return _result(ok=False, reason=RejectReason.SIZE_TOO_SMALL, details=details)
        else:
            details["min_size_override_used"] = False
            return _result(ok=False, reason=RejectReason.SIZE_TOO_SMALL, details=details)

    risk_cash_rounded = stop_distance * rounded_size
    details["risk_cash_rounded"] = risk_cash_rounded
    details["max_risk_cash"] = float(max_risk_cash)
    if risk_cash_rounded > float(max_risk_cash) + 1e-12:
        return _result(ok=False, reason=RejectReason.RISK_AFTER_ROUNDING_TOO_HIGH, details=details)

    required_margin = estimate_required_margin(
        entry_price=entry_price,
        size=rounded_size,
        margin_requirement_pct=margin_requirement_pct,
        max_leverage=max_leverage,
    )
    open_margin = estimate_required_margin(
        entry_price=entry_price,
        size=0.0,
        margin_requirement_pct=margin_requirement_pct,
        max_leverage=max_leverage,
    )
    if free_margin is None:
        free_margin_eff = float(equity) - float(open_margin)
    else:
        free_margin_eff = float(free_margin)
    free_margin_eff = max(0.0, free_margin_eff)
    safety = max(0.0, float(margin_safety_factor))
    details["required_margin"] = required_margin
    details["free_margin"] = free_margin_eff
    details["margin_safety_factor"] = safety
    if required_margin > (free_margin_eff * safety) + 1e-12:
        # --- margin-aware size cap: shrink position to fit margin ---
        margin_per_unit = estimate_required_margin(
            entry_price=entry_price,
            size=1.0,
            margin_requirement_pct=margin_requirement_pct,
            max_leverage=max_leverage,
        )
        if margin_per_unit > 0 and free_margin_eff > 0 and safety > 0:
            max_size_for_margin = (free_margin_eff * safety) / margin_per_unit
            capped = floor_to_step(max_size_for_margin, float(size_step))
            capped = round(max(0.0, capped), 8)
            if capped >= float(min_size):
                rounded_size = capped
                details["rounded_size"] = rounded_size
                details["margin_capped"] = True
                risk_cash_rounded = stop_distance * rounded_size
                details["risk_cash_rounded"] = risk_cash_rounded
                required_margin = estimate_required_margin(
                    entry_price=entry_price,
                    size=rounded_size,
                    margin_requirement_pct=margin_requirement_pct,
                    max_leverage=max_leverage,
                )
                details["required_margin"] = required_margin
            else:
                return _result(ok=False, reason=RejectReason.INSUFFICIENT_MARGIN, details=details)
        else:
            return _result(ok=False, reason=RejectReason.INSUFFICIENT_MARGIN, details=details)

    details["notional"] = max(0.0, float(entry_price) * rounded_size)
    details["equity"] = float(equity)
    return _result(ok=True, reason=None, details=details)

