"""Portfolio-level risk budget with daily kill-switch and open-risk cap.

This module is the *master safety layer*.  When ``RiskBudgetConfig.enabled``
is ``True``, the checks here take precedence over (and supplement) the legacy
``risk.*`` parameters.  When disabled the module is a transparent no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RiskBudgetCheck:
    """Result of a single risk-budget evaluation."""

    allowed: bool
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, float] = field(default_factory=dict)


class PortfolioRiskBudget:
    """Tracks daily P&L and open-position risk against hard budget limits.

    Typical usage (inside the backtest bar-loop)::

        budget.reset_day(day_key)
        check = budget.check_can_open(equity=..., new_trade_risk=..., open_risk=...)
        if not check.allowed:
            log(check.reasons)
            continue
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        risk_per_trade_pct: float = 0.0035,
        max_open_risk_pct: float = 0.02,
        daily_loss_limit_pct: float = 0.025,
        daily_profit_lock_pct: float = 0.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.max_open_risk_pct = float(max_open_risk_pct)
        self.daily_loss_limit_pct = float(daily_loss_limit_pct)
        self.daily_profit_lock_pct = float(daily_profit_lock_pct)
        self._daily_pnl: float = 0.0
        self._day_key: str = ""
        self._killed: bool = False

    # ------------------------------------------------------------------
    # Day lifecycle
    # ------------------------------------------------------------------
    def reset_day(self, day_key: str) -> None:
        """Call at the start of each trading day (or when ``day_key`` changes)."""
        if day_key != self._day_key:
            self._daily_pnl = 0.0
            self._day_key = day_key
            self._killed = False

    def record_trade_pnl(self, pnl: float) -> None:
        """Accumulate realised P&L for the current day."""
        self._daily_pnl += pnl

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def is_killed(self) -> bool:
        return self._killed

    # ------------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------------
    def check_can_open(
        self,
        *,
        equity: float,
        new_trade_risk: float,
        open_positions_risk: float,
    ) -> RiskBudgetCheck:
        """Return whether a new trade is allowed under the budget.

        Parameters
        ----------
        equity:
            Current account equity.
        new_trade_risk:
            Dollar-risk of the proposed new trade (positive value).
        open_positions_risk:
            Summed dollar-risk of all currently open positions (positive).
        """
        if not self.enabled:
            return RiskBudgetCheck(allowed=True)

        reasons: list[str] = []
        meta: dict[str, float] = {}

        eq = max(equity, 1e-9)

        # ---- daily loss kill-switch ----
        loss_limit = eq * self.daily_loss_limit_pct
        meta["daily_pnl"] = round(self._daily_pnl, 4)
        meta["daily_loss_limit"] = round(-loss_limit, 4)
        if self._daily_pnl <= -loss_limit:
            reasons.append("KILL_SWITCH_DAILY_LOSS")
            self._killed = True

        # ---- daily profit lock ----
        if self.daily_profit_lock_pct > 0:
            profit_lock = eq * self.daily_profit_lock_pct
            meta["daily_profit_lock"] = round(profit_lock, 4)
            if self._daily_pnl >= profit_lock:
                reasons.append("DAILY_PROFIT_LOCKED")

        # ---- per-trade risk cap ----
        max_risk_per_trade = eq * self.risk_per_trade_pct
        meta["new_trade_risk"] = round(new_trade_risk, 4)
        meta["max_risk_per_trade"] = round(max_risk_per_trade, 4)
        if new_trade_risk > max_risk_per_trade * 1.01:  # 1% tolerance
            reasons.append("PER_TRADE_RISK_TOO_HIGH")

        # ---- open-risk cap ----
        projected_risk = open_positions_risk + new_trade_risk
        max_open = eq * self.max_open_risk_pct
        meta["open_positions_risk"] = round(open_positions_risk, 4)
        meta["projected_risk"] = round(projected_risk, 4)
        meta["max_open_risk"] = round(max_open, 4)
        if projected_risk > max_open * 1.01:
            reasons.append("OPEN_RISK_CAP_EXCEEDED")

        return RiskBudgetCheck(
            allowed=len(reasons) == 0,
            reasons=reasons,
            metadata=meta,
        )
