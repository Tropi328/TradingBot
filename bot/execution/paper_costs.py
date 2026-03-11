"""Paper-mode realistic cost model.

Provides:
- Bid/Ask fill pricing (entry/exit on correct side of spread)
- Configurable slippage model per order type
- Round-trip cost estimation
- Cost breakdown for PnL reporting

All costs are in *price units* (points), not cash.
Multiply by ``size`` to get cash impact.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class SlippageModelConfig(BaseModel):
    """Slippage = base_ticks + beta_spread * spread + beta_atr * ATR"""

    base_ticks: float = 0.02
    beta_spread: float = 0.15
    beta_atr: float = 0.005

    @model_validator(mode="after")
    def validate(self) -> SlippageModelConfig:
        if self.base_ticks < 0:
            raise ValueError("base_ticks must be >= 0")
        if self.beta_spread < 0:
            raise ValueError("beta_spread must be >= 0")
        if self.beta_atr < 0:
            raise ValueError("beta_atr must be >= 0")
        return self


class PaperCostConfig(BaseModel):
    """Top-level cost config for PAPER mode."""

    enabled: bool = True
    slippage_market: SlippageModelConfig = Field(
        default_factory=lambda: SlippageModelConfig(base_ticks=0.03, beta_spread=0.20, beta_atr=0.008)
    )
    slippage_stop: SlippageModelConfig = Field(
        default_factory=lambda: SlippageModelConfig(base_ticks=0.05, beta_spread=0.25, beta_atr=0.010)
    )
    slippage_limit: SlippageModelConfig = Field(
        default_factory=lambda: SlippageModelConfig(base_ticks=0.01, beta_spread=0.10, beta_atr=0.003)
    )
    commission_per_side: float = 0.0  # in account currency per lot
    swap_per_day: float = 0.0  # TODO: implement per-asset swap cost model (currently always 0)
    use_bid_ask_fills: bool = True

    @model_validator(mode="after")
    def validate(self) -> PaperCostConfig:
        if self.commission_per_side < 0:
            raise ValueError("commission_per_side must be >= 0")
        return self


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------
def compute_slippage(
    model: SlippageModelConfig,
    spread: float,
    atr: float | None,
) -> float:
    """Compute one-side slippage in price units."""
    s = model.base_ticks + model.beta_spread * max(0.0, spread)
    if atr is not None and atr > 0:
        s += model.beta_atr * atr
    return max(0.0, s)


@dataclass(slots=True)
class FillPrices:
    """Realistic fill prices for an order."""

    entry_fill: float
    exit_sl_fill: float
    exit_tp_fill: float
    entry_slippage: float
    exit_sl_slippage: float
    exit_tp_slippage: float
    spread_at_entry: float


def compute_fill_prices(
    *,
    side: str,
    entry_price: float,
    stop_price: float,
    take_profit: float,
    spread: float,
    atr: float | None,
    config: PaperCostConfig,
) -> FillPrices:
    """Compute realistic fill prices for PAPER mode.

    LONG:  entry on ASK, SL exit on BID, TP exit on BID
    SHORT: entry on BID, SL exit on ASK, TP exit on ASK

    Slippage is always *adverse*: makes entry worse and exit worse.
    """
    half_spread = spread / 2.0

    # Entry: limit order
    entry_slip = compute_slippage(config.slippage_limit, spread, atr)
    # SL exit: stop order
    sl_slip = compute_slippage(config.slippage_stop, spread, atr)
    # TP exit: limit order
    tp_slip = compute_slippage(config.slippage_limit, spread, atr)

    if side == "LONG":
        # Entry at ASK + slippage (adverse = higher)
        entry_fill = entry_price + half_spread + entry_slip
        # SL exit at BID - slippage (adverse = lower)
        exit_sl_fill = stop_price - half_spread - sl_slip
        # TP exit at BID - slippage (adverse = lower)
        exit_tp_fill = take_profit - half_spread - tp_slip
    else:
        # SHORT: entry at BID - slippage (adverse = lower for shorts)
        entry_fill = entry_price - half_spread - entry_slip
        # SL exit at ASK + slippage (adverse = higher)
        exit_sl_fill = stop_price + half_spread + sl_slip
        # TP exit at ASK + slippage (adverse = higher)
        exit_tp_fill = take_profit + half_spread + tp_slip

    return FillPrices(
        entry_fill=entry_fill,
        exit_sl_fill=exit_sl_fill,
        exit_tp_fill=exit_tp_fill,
        entry_slippage=entry_slip,
        exit_sl_slippage=sl_slip,
        exit_tp_slippage=tp_slip,
        spread_at_entry=spread,
    )


@dataclass(slots=True)
class RoundtripCost:
    """Estimated round-trip cost breakdown."""

    spread_cost: float  # full spread (entry + exit)
    slippage_entry: float  # slippage at entry
    slippage_exit: float  # slippage at exit (worst-case: stop)
    commission: float  # both sides
    swap: float  # daily swap estimate
    total: float  # sum of all

    def to_dict(self) -> dict[str, float]:
        return {
            "spread_cost": round(self.spread_cost, 6),
            "slippage_entry": round(self.slippage_entry, 6),
            "slippage_exit": round(self.slippage_exit, 6),
            "commission": round(self.commission, 6),
            "swap": round(self.swap, 6),
            "total": round(self.total, 6),
        }


def estimate_roundtrip_cost(
    *,
    spread: float,
    atr: float | None,
    size: float,
    config: PaperCostConfig,
) -> RoundtripCost:
    """Estimate total round-trip cost in price units per lot, then × size.

    spread_cost   = full spread (applies at entry + exit)
    slippage_cost = entry slip + worst-case exit slip (stop)
    """
    entry_slip = compute_slippage(config.slippage_limit, spread, atr)
    exit_slip = compute_slippage(config.slippage_stop, spread, atr)  # worst-case
    spread_cost = spread  # one full spread for the round-trip
    commission = config.commission_per_side * 2  # both sides

    total_per_unit = spread_cost + entry_slip + exit_slip + commission
    return RoundtripCost(
        spread_cost=spread_cost * size,
        slippage_entry=entry_slip * size,
        slippage_exit=exit_slip * size,
        commission=commission * size,
        swap=config.swap_per_day * size,
        total=total_per_unit * size + config.swap_per_day * size,
    )


def estimate_roundtrip_cost_points(
    *,
    spread: float,
    atr: float | None,
    config: PaperCostConfig,
) -> float:
    """Quick helper: total cost per unit in points (for edge-filter)."""
    entry_slip = compute_slippage(config.slippage_limit, spread, atr)
    exit_slip = compute_slippage(config.slippage_stop, spread, atr)
    return spread + entry_slip + exit_slip + config.commission_per_side * 2


# ---------------------------------------------------------------------------
# BE offset computation
# ---------------------------------------------------------------------------
def compute_be_offset(
    *,
    side: str,
    entry_price: float,
    spread: float,
    atr: float | None,
    config: PaperCostConfig,
    buffer_ticks: float = 0.05,
) -> float:
    """Compute break-even price that covers costs.

    BE_long  = entry + spread + slippage + buffer
    BE_short = entry - spread - slippage - buffer
    """
    entry_slip = compute_slippage(config.slippage_limit, spread, atr)
    exit_slip = compute_slippage(config.slippage_limit, spread, atr)
    offset = spread + entry_slip + exit_slip + buffer_ticks
    if side == "LONG":
        return entry_price + offset
    else:
        return entry_price - offset
