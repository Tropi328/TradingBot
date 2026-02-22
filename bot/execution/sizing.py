from __future__ import annotations

import math
from typing import Any


def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        raise ValueError("step must be > 0")
    return math.floor(value / step) * step


def ceil_to_step(value: float, step: float) -> float:
    if step <= 0:
        raise ValueError("step must be > 0")
    return math.ceil(value / step) * step


def position_size_from_risk(
    *,
    equity: float,
    risk_per_trade: float,
    entry_price: float,
    stop_price: float,
    min_size: float,
    size_step: float,
    allow_min_size_fallback: bool = False,
    min_size_fallback_max_risk_pct: float = 0.0,
) -> float:
    risk_distance = abs(entry_price - stop_price)
    if risk_distance <= 0:
        return 0.0
    risk_amount = equity * risk_per_trade
    raw_size = risk_amount / risk_distance
    sized = floor_to_step(raw_size, size_step)
    if sized < min_size:
        if allow_min_size_fallback and min_size_fallback_max_risk_pct > 0 and equity > 0:
            fallback_size = ceil_to_step(min_size, size_step)
            fallback_risk = fallback_size * risk_distance
            fallback_risk_cap = equity * min_size_fallback_max_risk_pct
            if fallback_risk <= fallback_risk_cap:
                return round(fallback_size, 8)
        return 0.0
    return round(sized, 8)


def r_multiple(side: str, entry_price: float, stop_price: float, current_price: float) -> float:
    risk = abs(entry_price - stop_price)
    if risk == 0:
        return 0.0
    if side == "LONG":
        return (current_price - entry_price) / risk
    return (entry_price - current_price) / risk


# ---------------------------------------------------------------------------
# Tier-based sizing
# ---------------------------------------------------------------------------

def resolve_score_tier(
    score: float | None,
    *,
    tier_cfg: Any | None = None,
) -> str:
    """Map a numeric score to a tier label: A_PLUS, A, B, OBSERVE.

    Uses tier_cfg (TierSizingConfig) score boundaries.
    Falls back to sensible defaults if not provided.
    """
    if score is None:
        return "OBSERVE"
    s = float(score)
    a_plus_min = getattr(tier_cfg, "a_plus_min_score", 80.0) if tier_cfg else 80.0
    a_min = getattr(tier_cfg, "a_min_score", 65.0) if tier_cfg else 65.0
    b_min = getattr(tier_cfg, "b_min_score", 55.0) if tier_cfg else 55.0
    if s >= a_plus_min:
        return "A_PLUS"
    if s >= a_min:
        return "A"
    if s >= b_min:
        return "B"
    return "OBSERVE"


def tier_risk_multiplier(
    tier: str,
    *,
    tier_cfg: Any | None = None,
) -> float:
    """Return the risk multiplier for a given tier.

    A_PLUS → 1.5×, A → 1.0×, B → 0.6×, OBSERVE → 0.0×
    """
    a_plus = getattr(tier_cfg, "a_plus_mult", 1.5) if tier_cfg else 1.5
    a = getattr(tier_cfg, "a_mult", 1.0) if tier_cfg else 1.0
    b = getattr(tier_cfg, "b_mult", 0.6) if tier_cfg else 0.6
    obs = getattr(tier_cfg, "observe_mult", 0.0) if tier_cfg else 0.0
    return {
        "A_PLUS": a_plus,
        "A": a,
        "B": b,
        "OBSERVE": obs,
    }.get(tier, 1.0)


# ---------------------------------------------------------------------------
# Compound equity sizing
# ---------------------------------------------------------------------------

def compute_compound_equity(
    raw_equity: float,
    *,
    floor_equity: float = 50.0,
    cap_equity: float = 0.0,
) -> float:
    """Return the equity value to use for sizing in compound mode.

    - Never goes below floor_equity.
    - If cap_equity > 0, caps the sizing equity (still grows, but
      size stops increasing beyond cap).
    """
    eq = max(floor_equity, raw_equity)
    if cap_equity > 0:
        eq = min(eq, cap_equity)
    return eq


def position_size_tiered(
    *,
    equity: float,
    risk_per_trade: float,
    entry_price: float,
    stop_price: float,
    min_size: float,
    size_step: float,
    score: float | None = None,
    tier_cfg: Any | None = None,
    compound_cfg: Any | None = None,
    session_risk_mult: float = 1.0,
    allow_min_size_fallback: bool = False,
    min_size_fallback_max_risk_pct: float = 0.0,
) -> tuple[float, str, dict[str, float]]:
    """Full-featured sizing: tier × compound × session multipliers.

    Returns (size, tier_label, metadata_dict).
    """
    # 1. Compound equity
    sizing_equity = equity
    if compound_cfg is not None and getattr(compound_cfg, "enabled", False):
        sizing_equity = compute_compound_equity(
            equity,
            floor_equity=getattr(compound_cfg, "floor_equity", 50.0),
            cap_equity=getattr(compound_cfg, "cap_equity", 0.0),
        )

    # 2. Tier multiplier
    tier = "A"
    tier_mult = 1.0
    if tier_cfg is not None and getattr(tier_cfg, "enabled", False):
        tier = resolve_score_tier(score, tier_cfg=tier_cfg)
        tier_mult = tier_risk_multiplier(tier, tier_cfg=tier_cfg)
        if tier_mult <= 0:
            return 0.0, tier, {"tier_mult": 0.0, "session_mult": session_risk_mult}

    # 3. Combined risk
    effective_risk = risk_per_trade * tier_mult * session_risk_mult

    # 4. Size
    size = position_size_from_risk(
        equity=sizing_equity,
        risk_per_trade=effective_risk,
        entry_price=entry_price,
        stop_price=stop_price,
        min_size=min_size,
        size_step=size_step,
        allow_min_size_fallback=allow_min_size_fallback,
        min_size_fallback_max_risk_pct=min_size_fallback_max_risk_pct,
    )

    meta = {
        "sizing_equity": sizing_equity,
        "tier": tier,
        "tier_mult": tier_mult,
        "session_mult": session_risk_mult,
        "effective_risk_pct": effective_risk,
    }
    return size, tier, meta
