"""Score-based tier resolution and size scaling.

Maps a V2 composite score to one of four tiers:

    A+  →  full size, market-fallback eligible
    A   →  0.8 × size
    B   →  0.6 × size
    OBSERVE → no entry

The tier thresholds and multipliers are fully configurable via
``ScoreTiersConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TierResult:
    """Resolved tier information for a signal."""

    name: str  # "A_plus", "A", "B", "OBSERVE"
    size_mult: float
    allow_market_after_ttl: bool


def resolve_tier(
    score: float,
    *,
    enabled: bool = False,
    a_plus_min: float = 72.0,
    a_plus_mult: float = 1.0,
    a_plus_market: bool = True,
    a_min: float = 68.0,
    a_mult: float = 0.8,
    a_market: bool = False,
    b_min: float = 62.0,
    b_mult: float = 0.6,
    b_market: bool = False,
) -> TierResult:
    """Return the tier for *score* based on thresholds.

    When ``enabled`` is False the function returns a transparent "NONE" tier
    with ``size_mult=1.0`` so that the caller can always apply the multiplier
    without branching.
    """
    if not enabled:
        return TierResult(name="NONE", size_mult=1.0, allow_market_after_ttl=False)

    if score >= a_plus_min:
        return TierResult(name="A_plus", size_mult=a_plus_mult, allow_market_after_ttl=a_plus_market)
    if score >= a_min:
        return TierResult(name="A", size_mult=a_mult, allow_market_after_ttl=a_market)
    if score >= b_min:
        return TierResult(name="B", size_mult=b_mult, allow_market_after_ttl=b_market)
    return TierResult(name="OBSERVE", size_mult=0.0, allow_market_after_ttl=False)


def resolve_tier_from_config(score: float, config: object) -> TierResult:
    """Convenience wrapper that reads fields from a ``ScoreTiersConfig``-like
    object (Pydantic model or plain namespace).

    Expected attributes on *config*:
        enabled, A_plus, A, B   (each having min_score, size_mult,
        allow_market_after_ttl).
    """
    enabled = getattr(config, "enabled", False)
    if not enabled:
        return TierResult(name="NONE", size_mult=1.0, allow_market_after_ttl=False)

    ap = getattr(config, "A_plus", None)
    a = getattr(config, "A", None)
    b = getattr(config, "B", None)

    return resolve_tier(
        score,
        enabled=True,
        a_plus_min=float(getattr(ap, "min_score", 72.0)) if ap else 72.0,
        a_plus_mult=float(getattr(ap, "size_mult", 1.0)) if ap else 1.0,
        a_plus_market=bool(getattr(ap, "allow_market_after_ttl", True)) if ap else True,
        a_min=float(getattr(a, "min_score", 68.0)) if a else 68.0,
        a_mult=float(getattr(a, "size_mult", 0.8)) if a else 0.8,
        a_market=bool(getattr(a, "allow_market_after_ttl", False)) if a else False,
        b_min=float(getattr(b, "min_score", 62.0)) if b else 62.0,
        b_mult=float(getattr(b, "size_mult", 0.6)) if b else 0.6,
        b_market=bool(getattr(b, "allow_market_after_ttl", False)) if b else False,
    )
