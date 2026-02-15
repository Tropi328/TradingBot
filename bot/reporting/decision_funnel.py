"""Decision-funnel tracker for backtest and paper-trading.

Collects counts at every stage of the entry pipeline so that after a run you
can see exactly where candidates were lost:

    signals → proposals → supervisor OK → orders placed → filled → trades
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DecisionFunnel:
    """Accumulates pipeline-stage counters throughout a backtest run."""

    # -- pipeline stages --
    signal_candidates: int = 0
    proposals_created: int = 0
    blocked_by_risk_budget: int = 0
    blocked_by_supervisor: int = 0
    orders_placed: int = 0
    orders_expired_ttl: int = 0
    market_fallbacks_triggered: int = 0
    filled_orders: int = 0
    trades_opened: int = 0
    trades_closed: int = 0

    # -- rejection breakdown --
    blocked_reasons: dict[str, int] = field(default_factory=dict)

    # -- concurrent-position samples (one per bar) --
    _concurrent_samples: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------
    def record_block(self, reason: str) -> None:
        self.blocked_reasons[reason] = self.blocked_reasons.get(reason, 0) + 1

    def sample_concurrent(self, n_open: int) -> None:
        self._concurrent_samples.append(n_open)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(
        self,
        *,
        total_pnl: float = 0.0,
        expectancy: float = 0.0,
        max_drawdown: float = 0.0,
        avg_r: float = 0.0,
        win_rate: float = 0.0,
    ) -> dict:
        samples = self._concurrent_samples
        avg_concurrent = (sum(samples) / len(samples)) if samples else 0.0
        fill_rate = (self.filled_orders / self.orders_placed * 100.0) if self.orders_placed > 0 else 0.0
        top_reasons = sorted(self.blocked_reasons.items(), key=lambda x: -x[1])[:10]

        return {
            "signal_candidates": self.signal_candidates,
            "proposals_created": self.proposals_created,
            "blocked_by_risk_budget": self.blocked_by_risk_budget,
            "blocked_by_supervisor": self.blocked_by_supervisor,
            "orders_placed": self.orders_placed,
            "orders_expired_ttl": self.orders_expired_ttl,
            "market_fallbacks_triggered": self.market_fallbacks_triggered,
            "filled_orders": self.filled_orders,
            "fill_rate_pct": round(fill_rate, 2),
            "trades_opened": self.trades_opened,
            "trades_closed": self.trades_closed,
            "avg_concurrent_positions": round(avg_concurrent, 2),
            # -- performance summary --
            "total_pnl": round(total_pnl, 2),
            "expectancy": round(expectancy, 4),
            "max_drawdown": round(max_drawdown, 2),
            "avg_r": round(avg_r, 4),
            "win_rate": round(win_rate, 4),
            # -- rejection breakdown --
            "top_10_rejection_reasons": dict(top_reasons),
        }

    def save_json(self, path: str | Path, **extra_metrics: float) -> None:
        out = self.to_dict(**extra_metrics)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
