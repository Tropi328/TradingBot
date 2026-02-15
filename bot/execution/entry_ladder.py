"""FVG entry-ladder generator.

Instead of a single LIMIT order at the FVG midpoint, the ladder places
multiple orders at different FVG depth levels.  Each level carries a fraction
of the total size.

When all ladder orders expire (TTL), an optional MARKET fallback is generated
for ``A_plus`` signals at reduced size.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class LadderLevel:
    """A single rung on the entry ladder."""

    name: str
    entry_price: float
    size: float
    is_market: bool = False


@dataclass(slots=True)
class LadderResult:
    """Complete set of orders produced by the ladder generator."""

    levels: list[LadderLevel] = field(default_factory=list)
    ttl_bars: int = 8
    market_fallback_eligible: bool = False


def compute_ladder(
    *,
    side: str,
    fvg_high: float,
    fvg_low: float,
    base_entry: float,
    total_size: float,
    enabled: bool = False,
    level_specs: list[dict[str, float]] | None = None,
    ttl_bars: int = 8,
    market_fallback_eligible: bool = False,
) -> LadderResult:
    """Build entry-ladder orders.

    Parameters
    ----------
    side : "LONG" | "SHORT"
    fvg_high / fvg_low : FVG boundaries.
    base_entry : Original single-order entry price (FVG midpoint).
    total_size : Total position size *before* splitting.
    enabled : When False, returns a single order at *base_entry*.
    level_specs : List of dicts ``{"fvg_fraction": …, "size_fraction": …, "name": …}``.
    ttl_bars : Bars before ladder orders expire.
    market_fallback_eligible : Whether this signal qualifies for MARKET fallback
        after TTL (typically only A+ tier).
    """
    if not enabled or not level_specs:
        return LadderResult(
            levels=[LadderLevel(name="single", entry_price=base_entry, size=total_size)],
            ttl_bars=ttl_bars,
            market_fallback_eligible=market_fallback_eligible,
        )

    fvg_range = abs(fvg_high - fvg_low)
    if fvg_range <= 0:
        return LadderResult(
            levels=[LadderLevel(name="single", entry_price=base_entry, size=total_size)],
            ttl_bars=ttl_bars,
            market_fallback_eligible=market_fallback_eligible,
        )

    levels: list[LadderLevel] = []
    for spec in level_specs:
        frac = float(spec.get("fvg_fraction", 0.5))
        size_frac = float(spec.get("size_fraction", 0.5))
        name = str(spec.get("name", f"L{frac:.0%}"))

        if side == "LONG":
            entry = fvg_high - (fvg_range * frac)
        else:
            entry = fvg_low + (fvg_range * frac)

        level_size = total_size * size_frac
        if level_size > 0:
            levels.append(LadderLevel(name=name, entry_price=round(entry, 6), size=round(level_size, 6)))

    if not levels:
        levels.append(LadderLevel(name="single", entry_price=base_entry, size=total_size))

    return LadderResult(
        levels=levels,
        ttl_bars=ttl_bars,
        market_fallback_eligible=market_fallback_eligible,
    )


def make_market_fallback(
    *,
    side: str,
    current_price: float,
    stop: float,
    tp: float,
    base_size: float,
    size_mult: float = 0.5,
) -> LadderLevel:
    """Create a MARKET fallback order at *current_price* with reduced size."""
    return LadderLevel(
        name="market_fallback",
        entry_price=current_price,
        size=round(base_size * size_mult, 6),
        is_market=True,
    )
