"""Adaptive threshold & soft-gate system for PAPER/LIVE mode.

Converts selected hard gates to score penalties and adjusts the
TRADE/SMALL threshold based on market regime (trend/range, volatility).

Design goals
────────────
• Increase trade frequency 2-4×  without blowing DD
• Soft gates: ReactionGate + ExecutionGate(spread_too_high) become
  penalties (-N pts) instead of hard blocks
• Adaptive threshold: base ± regime_adj ± vol_adj → dynamic cutoff
• Re-entry: after a profitable close (BE/TP), allow one re-entry
  per directional leg with cooldown
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

LOGGER = logging.getLogger("trading_bot")


# ────────────────────────────────────────────────────────────
# Soft-gate conversion
# ────────────────────────────────────────────────────────────
# Gates that can be "softened" — instead of fully blocking the trade
# we subtract a score penalty and let the scoring decide.

SOFT_GATE_ELIGIBLE = frozenset({
    "GATE_REACTION_WAIT_MITIGATION",
    "GATE_REACTION_WAIT_REACTION",
    "EXEC_FAIL_SPREAD_TOO_HIGH",
    "EDGE_TOO_SMALL",
})

# Gates that should NEVER become soft (missing data / market closed)
HARD_GATE_ALWAYS = frozenset({
    "EXEC_FAIL_MISSING_FEATURES",
    "EXEC_FAIL_INVALID_ATR",
    "EXEC_FAIL_NO_PRICE",
    "EXEC_FAIL_MARKET_CLOSED",
})


@dataclass(slots=True)
class SoftGateResult:
    """Outcome of soft-gate conversion for a single evaluation."""
    hard_reasons: list[str]
    soft_reasons: list[str]
    total_penalty: float
    converted_gates: list[str]

    @property
    def blocked(self) -> bool:
        """True if any *truly hard* gate still blocks."""
        return len(self.hard_reasons) > 0


def apply_soft_gates(
    gate_reasons: list[str],
    *,
    soft_gates_enabled: bool,
    soft_gate_penalty: float = 4.0,
) -> SoftGateResult:
    """Split gate_reasons into hard blocks and soft penalties.

    Parameters
    ----------
    gate_reasons : list[str]
        Original list from ``_quality_gate_reasons``.
    soft_gates_enabled : bool
        Master switch from ``adaptive_threshold.soft_gates_enabled``.
    soft_gate_penalty : float
        Points subtracted per soft-gate hit.

    Returns
    -------
    SoftGateResult
        Remaining hard reasons, soft reasons, total penalty, converted names.
    """
    if not soft_gates_enabled or not gate_reasons:
        return SoftGateResult(
            hard_reasons=list(gate_reasons),
            soft_reasons=[],
            total_penalty=0.0,
            converted_gates=[],
        )

    hard: list[str] = []
    soft: list[str] = []
    converted: list[str] = []

    for reason in gate_reasons:
        if reason in SOFT_GATE_ELIGIBLE:
            soft.append(reason)
            converted.append(reason)
        else:
            hard.append(reason)

    penalty = soft_gate_penalty * len(soft)
    return SoftGateResult(
        hard_reasons=hard,
        soft_reasons=soft,
        total_penalty=penalty,
        converted_gates=converted,
    )


# ────────────────────────────────────────────────────────────
# Adaptive threshold
# ────────────────────────────────────────────────────────────

class RegimeType(str, Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    UNKNOWN = "UNKNOWN"


class VolRegime(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"
    NORMAL = "NORMAL"


@dataclass(slots=True, frozen=True)
class AdaptiveThresholdConfig:
    """Runtime config — mirrored from Pydantic config for fast access."""
    enabled: bool = False
    base_threshold: float = 62.0
    range_adjust: float = -6.0
    trend_adjust: float = -2.0
    high_vol_adjust: float = 2.0
    low_vol_adjust: float = -3.0
    soft_gates_enabled: bool = True
    soft_gate_penalty: float = 4.0

    # Derived thresholds
    min_threshold: float = 48.0
    max_threshold: float = 75.0


def compute_adaptive_threshold(
    *,
    config: AdaptiveThresholdConfig,
    trend_regime: str | None = None,
    vol_regime: str | None = None,
    soft_gate_penalty: float = 0.0,
) -> float:
    """Compute dynamic trade/small threshold.

    threshold = base + regime_adj + vol_adj - soft_penalty

    The result is clamped to [min_threshold, max_threshold].
    """
    if not config.enabled:
        return config.base_threshold

    threshold = config.base_threshold

    # Regime adjustment
    tr = (trend_regime or "UNKNOWN").upper()
    if tr in {"RANGING", "RANGE", "FLAT"}:
        threshold += config.range_adjust      # e.g. −6 → lower bar
    elif tr in {"TRENDING", "STRONG_TREND"}:
        threshold += config.trend_adjust      # e.g. −2

    # Volatility adjustment
    vr = (vol_regime or "NORMAL").upper()
    if vr in {"HIGH", "EXTREME"}:
        threshold += config.high_vol_adjust   # e.g. +2 → raise bar
    elif vr in {"LOW", "QUIET"}:
        threshold += config.low_vol_adjust    # e.g. −3

    # Clamp
    threshold = max(config.min_threshold, min(config.max_threshold, threshold))
    return threshold


def normalize_action_adaptive(
    *,
    score: float,
    threshold: float,
    small_band: float = 5.0,
) -> str:
    """Map score to action using the adaptive threshold.

    Returns one of: "TRADE", "SMALL", "OBSERVE"
    """
    if score >= threshold:
        return "TRADE"
    elif score >= (threshold - small_band):
        return "SMALL"
    else:
        return "OBSERVE"


# ────────────────────────────────────────────────────────────
# Re-entry tracking
# ────────────────────────────────────────────────────────────

@dataclass(slots=True)
class ReentryState:
    """Per-asset re-entry tracking."""
    # Last closed position info
    last_close_side: str | None = None          # "LONG" / "SHORT"
    last_close_exit_type: str | None = None     # "TP" / "TP1" / "BE" / "SL"
    last_close_pnl: float = 0.0
    last_close_at: datetime | None = None

    # Re-entry counters (reset when direction flips)
    reentries_this_leg: int = 0
    max_reentries_per_leg: int = 1
    reentry_cooldown_seconds: int = 120         # 2 min between close → re-entry

    # Re-entry result
    last_reentry_at: datetime | None = None

    def record_close(
        self,
        side: str,
        exit_type: str,
        pnl: float,
        closed_at: datetime,
    ) -> None:
        """Call when a position closes."""
        # Direction flip → reset
        if self.last_close_side is not None and side != self.last_close_side:
            self.reentries_this_leg = 0

        self.last_close_side = side.upper()
        self.last_close_exit_type = exit_type.upper() if exit_type else "UNKNOWN"
        self.last_close_pnl = pnl
        self.last_close_at = closed_at

    def can_reenter(
        self,
        side: str,
        now: datetime,
    ) -> tuple[bool, str]:
        """Check if a re-entry is allowed.

        Returns (allowed, reason_if_blocked).

        Rules:
        1. Same direction as last close
        2. Last close was profitable (BE/TP/TP1, pnl >= 0)
        3. At most max_reentries_per_leg already used
        4. Cooldown elapsed since last close
        """
        if self.last_close_side is None:
            return True, ""  # No history → fresh entry, always ok

        # Different direction → fresh leg, always ok
        if side.upper() != self.last_close_side:
            return True, ""

        # Same direction → check re-entry rules
        if self.last_close_exit_type in {"SL", "STOP", "LIQUIDATION", "UNKNOWN"}:
            return False, "REENTRY_AFTER_STOP"

        if self.last_close_pnl < 0:
            return False, "REENTRY_AFTER_LOSS"

        if self.reentries_this_leg >= self.max_reentries_per_leg:
            return False, "REENTRY_MAX_REACHED"

        if self.last_close_at is not None:
            elapsed = (now - self.last_close_at).total_seconds()
            if elapsed < self.reentry_cooldown_seconds:
                return False, "REENTRY_COOLDOWN"

        return True, ""

    def mark_reentry(self, now: datetime) -> None:
        """Call when a re-entry order is placed."""
        self.reentries_this_leg += 1
        self.last_reentry_at = now

    def reset_leg(self) -> None:
        """Reset counters for a fresh directional leg."""
        self.reentries_this_leg = 0
        self.last_close_side = None
        self.last_close_exit_type = None
        self.last_close_pnl = 0.0
        self.last_close_at = None
        self.last_reentry_at = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_close_side": self.last_close_side,
            "last_close_exit_type": self.last_close_exit_type,
            "last_close_pnl": self.last_close_pnl,
            "last_close_at": self.last_close_at.isoformat() if self.last_close_at else None,
            "reentries_this_leg": self.reentries_this_leg,
            "max_reentries_per_leg": self.max_reentries_per_leg,
            "reentry_cooldown_seconds": self.reentry_cooldown_seconds,
            "last_reentry_at": self.last_reentry_at.isoformat() if self.last_reentry_at else None,
        }


# ────────────────────────────────────────────────────────────
# Helper: build config from Pydantic model
# ────────────────────────────────────────────────────────────

def build_adaptive_config(pydantic_cfg: Any) -> AdaptiveThresholdConfig:
    """Convert PydanticConfig → frozen dataclass."""
    return AdaptiveThresholdConfig(
        enabled=getattr(pydantic_cfg, "enabled", False),
        base_threshold=getattr(pydantic_cfg, "base_threshold", 62.0),
        range_adjust=getattr(pydantic_cfg, "range_adjust", -6.0),
        trend_adjust=getattr(pydantic_cfg, "trend_adjust", -2.0),
        high_vol_adjust=getattr(pydantic_cfg, "high_vol_adjust", 2.0),
        low_vol_adjust=getattr(pydantic_cfg, "low_vol_adjust", -3.0),
        soft_gates_enabled=getattr(pydantic_cfg, "soft_gates_enabled", True),
        soft_gate_penalty=getattr(pydantic_cfg, "soft_gate_penalty", 4.0),
    )
