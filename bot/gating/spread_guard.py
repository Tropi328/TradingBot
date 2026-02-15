"""SpreadGuard – block entries when spread is too wide.

Two gating modes
----------------
* **absolute** : reject when ``spread_cost > spread_cost_max``
* **percentile** : reject when the current spread exceeds the
  ``percentile_limit``-th percentile of recent ``window`` observations.

Both modes can be combined; if either triggers, entry is blocked.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class SpreadGuardConfig:
    """Configuration for :class:`SpreadGuard`.

    Parameters
    ----------
    enabled : bool
        Master switch.  When *False* the guard never blocks.
    spread_cost_max : float
        Absolute spread threshold (in price units).
        Set to ``0`` to disable the absolute check.
    percentile_limit : float
        Percentile (0–100) threshold for the rolling window.
        Set to ``0`` to disable.
    window : int
        Rolling-window size for percentile computation.
    """

    enabled: bool = False
    spread_cost_max: float = 0.0
    percentile_limit: float = 0.0
    window: int = 100


@dataclass(slots=True)
class SpreadGuardResult:
    """Outcome of a spread check."""

    blocked: bool
    reason: str = ""
    spread: float = 0.0
    threshold: float = 0.0


class SpreadGuard:
    """Rolling spread gate usable in both backtest and live execution."""

    def __init__(self, cfg: SpreadGuardConfig) -> None:
        self._cfg = cfg
        self._history: deque[float] = deque(maxlen=max(1, cfg.window))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def observe(self, spread: float) -> None:
        """Record a spread observation (call once per bar / tick)."""
        self._history.append(max(0.0, float(spread)))

    def check(self, spread: float) -> SpreadGuardResult:
        """Return whether *spread* should block entry."""
        if not self._cfg.enabled:
            return SpreadGuardResult(blocked=False, spread=spread)

        spread_val = max(0.0, float(spread))

        # --- absolute gate ------------------------------------------------
        if self._cfg.spread_cost_max > 0 and spread_val > self._cfg.spread_cost_max:
            return SpreadGuardResult(
                blocked=True,
                reason="SPREAD_EXCEEDS_MAX",
                spread=spread_val,
                threshold=self._cfg.spread_cost_max,
            )

        # --- percentile gate ----------------------------------------------
        if self._cfg.percentile_limit > 0 and len(self._history) >= 10:
            pct_threshold = _percentile(list(self._history), self._cfg.percentile_limit)
            if spread_val > pct_threshold:
                return SpreadGuardResult(
                    blocked=True,
                    reason="SPREAD_EXCEEDS_PERCENTILE",
                    spread=spread_val,
                    threshold=pct_threshold,
                )

        return SpreadGuardResult(blocked=False, spread=spread_val)

    @property
    def config(self) -> SpreadGuardConfig:
        return self._cfg


# ---------- helpers -------------------------------------------------------

def _percentile(values: list[float], pct: float) -> float:
    """Interpolated percentile (0–100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    pos = (pct / 100.0) * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    if lo == hi:
        return ordered[lo]
    w = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * w
