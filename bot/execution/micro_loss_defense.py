"""Micro-loss defense — pre-trade filters that prevent death-by-spread.

A *micro-loss* is defined as:
    net_pnl < 0  AND  |net_pnl| <= K × estimated_roundtrip_cost

Defenses (all configurable):
1. Minimum SL distance — SL must be ≥ max(N×spread, M×ATR)
2. Minimum edge filter — expected_move_to_target ≥ EDGE_MULT × cost
3. Metrics tracking — micro-loss stats for diagnostics
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, model_validator

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class MicroLossDefenseConfig(BaseModel):
    """All micro-loss defense knobs."""

    enabled: bool = True

    # Micro-loss definition: |pnl| <= K * roundtrip_cost
    micro_loss_k: float = 1.5

    # Minimum SL distance
    min_stop_spread_mult: float = 5.0       # SL >= 5× spread
    min_stop_atr_mult: float = 0.15         # SL >= 0.15× ATR

    # Minimum edge filter
    edge_mult: float = 3.0                  # expected_move / cost ≥ 3.0

    # BE buffer (in ATR fraction)
    be_buffer_atr_frac: float = 0.05
    be_buffer_ticks: float = 0.05

    @model_validator(mode="after")
    def validate(self) -> "MicroLossDefenseConfig":
        if self.micro_loss_k <= 0:
            raise ValueError("micro_loss_k must be > 0")
        if self.min_stop_spread_mult < 0:
            raise ValueError("min_stop_spread_mult must be >= 0")
        if self.min_stop_atr_mult < 0:
            raise ValueError("min_stop_atr_mult must be >= 0")
        if self.edge_mult <= 0:
            raise ValueError("edge_mult must be > 0")
        return self


# ---------------------------------------------------------------------------
# Pre-trade checks
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class MicroLossCheckResult:
    """Result of micro-loss pre-trade validation."""

    passed: bool
    rejection_reasons: list[str] = field(default_factory=list)
    min_sl_required: float = 0.0
    actual_sl: float = 0.0
    edge_ratio: float = 0.0
    roundtrip_cost_points: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


def check_min_sl_distance(
    *,
    sl_distance: float,
    spread: float,
    atr: float | None,
    config: MicroLossDefenseConfig,
) -> tuple[bool, float, list[str]]:
    """Check that SL distance is large enough to survive spread.

    Returns (passed, min_required, rejection_reasons).
    """
    reasons: list[str] = []
    spread_min = config.min_stop_spread_mult * max(0.0, spread)
    atr_min = config.min_stop_atr_mult * (atr if atr and atr > 0 else 0.0)
    min_required = max(spread_min, atr_min)

    if sl_distance < min_required:
        reasons.append("SL_TOO_TIGHT")
        return False, min_required, reasons
    return True, min_required, reasons


def check_edge_filter(
    *,
    expected_move: float,
    roundtrip_cost_points: float,
    config: MicroLossDefenseConfig,
) -> tuple[bool, float, list[str]]:
    """Check that edge exceeds cost by sufficient margin.

    expected_move / roundtrip_cost ≥ edge_mult

    Returns (passed, edge_ratio, rejection_reasons).
    """
    reasons: list[str] = []
    if roundtrip_cost_points <= 0:
        return True, float("inf"), reasons

    edge_ratio = expected_move / roundtrip_cost_points
    if edge_ratio < config.edge_mult:
        reasons.append("EDGE_TOO_LOW")
        return False, edge_ratio, reasons
    return True, edge_ratio, reasons


def run_micro_loss_checks(
    *,
    sl_distance: float,
    tp_distance: float,
    spread: float,
    atr: float | None,
    roundtrip_cost_points: float,
    config: MicroLossDefenseConfig,
) -> MicroLossCheckResult:
    """Run all micro-loss pre-trade checks.

    ``tp_distance`` is used as ``expected_move`` to nearest target.
    """
    if not config.enabled:
        return MicroLossCheckResult(
            passed=True,
            actual_sl=sl_distance,
            roundtrip_cost_points=roundtrip_cost_points,
        )

    all_reasons: list[str] = []

    sl_ok, min_sl, sl_reasons = check_min_sl_distance(
        sl_distance=sl_distance,
        spread=spread,
        atr=atr,
        config=config,
    )
    all_reasons.extend(sl_reasons)

    edge_ok, edge_ratio, edge_reasons = check_edge_filter(
        expected_move=tp_distance,
        roundtrip_cost_points=roundtrip_cost_points,
        config=config,
    )
    all_reasons.extend(edge_reasons)

    return MicroLossCheckResult(
        passed=sl_ok and edge_ok,
        rejection_reasons=all_reasons,
        min_sl_required=min_sl,
        actual_sl=sl_distance,
        edge_ratio=edge_ratio,
        roundtrip_cost_points=roundtrip_cost_points,
        details={
            "sl_ok": sl_ok,
            "edge_ok": edge_ok,
            "spread": spread,
            "atr": atr,
            "min_sl_spread": config.min_stop_spread_mult * max(0.0, spread),
            "min_sl_atr": config.min_stop_atr_mult * (atr if atr and atr > 0 else 0.0),
        },
    )


# ---------------------------------------------------------------------------
# Post-trade classification
# ---------------------------------------------------------------------------
def is_micro_loss(
    net_pnl: float,
    roundtrip_cost: float,
    k: float = 1.5,
) -> bool:
    """Classify whether a closed trade is a micro-loss."""
    if net_pnl >= 0:
        return False
    return abs(net_pnl) <= k * abs(roundtrip_cost)


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------
@dataclass
class MicroLossMetrics:
    """Accumulates micro-loss stats during a session."""

    total_closed: int = 0
    micro_loss_count: int = 0
    micro_loss_total_pnl: float = 0.0
    causes: Counter = field(default_factory=Counter)
    by_setup: Counter = field(default_factory=Counter)

    @property
    def micro_loss_rate(self) -> float:
        if self.total_closed == 0:
            return 0.0
        return self.micro_loss_count / self.total_closed

    def record_close(
        self,
        net_pnl: float,
        roundtrip_cost: float,
        k: float,
        *,
        cause: str = "UNKNOWN",
        setup_name: str = "UNKNOWN",
    ) -> bool:
        """Record a closed trade. Returns True if it was a micro-loss."""
        self.total_closed += 1
        if is_micro_loss(net_pnl, roundtrip_cost, k):
            self.micro_loss_count += 1
            self.micro_loss_total_pnl += net_pnl
            self.causes[cause] += 1
            self.by_setup[setup_name] += 1
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_closed": self.total_closed,
            "micro_loss_count": self.micro_loss_count,
            "micro_loss_rate": round(self.micro_loss_rate, 4),
            "micro_loss_total_pnl": round(self.micro_loss_total_pnl, 4),
            "top_causes": self.causes.most_common(5),
            "top_setups": self.by_setup.most_common(5),
        }
