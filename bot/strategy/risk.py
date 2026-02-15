from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from bot.config import RiskConfig
from bot.storage.models import DailyStats, PositionRecord


@dataclass(slots=True)
class RiskCheck:
    allowed: bool
    reason_codes: list[str]
    metadata: dict[str, float | int | str] = field(default_factory=dict)


class RiskEngine:
    def __init__(self, risk: RiskConfig):
        self.risk = risk

    def _resolved_equity(self, equity: float | None = None) -> float:
        return equity if equity is not None else self.risk.equity

    def is_low_equity_mode(self, equity: float | None = None) -> bool:
        if not self.risk.low_equity_mode_enabled:
            return False
        eq = self._resolved_equity(equity)
        return eq <= self.risk.low_equity_threshold

    def effective_risk_per_trade(self, *, risk_multiplier: float = 1.0, equity: float | None = None) -> float:
        base = max(0.0, float(self.risk.risk_per_trade) * max(0.0, float(risk_multiplier)))
        if self.is_low_equity_mode(equity):
            base *= self.risk.low_equity_risk_multiplier
            base = min(base, self.risk.low_equity_risk_per_trade_cap)
        return min(base, 1.0)

    def effective_max_trades_per_day(self, equity: float | None = None) -> int:
        if self.is_low_equity_mode(equity):
            return min(int(self.risk.max_trades_per_day), int(self.risk.low_equity_max_trades_per_day))
        return int(self.risk.max_trades_per_day)

    def effective_daily_stop_pct(self, equity: float | None = None) -> float:
        base = float(self.risk.daily_stop_pct)
        if self.is_low_equity_mode(equity):
            return min(base, float(self.risk.low_equity_daily_stop_pct))
        return base

    def min_size_fallback_allowed(self, equity: float | None = None) -> bool:
        return self.is_low_equity_mode(equity) and bool(self.risk.low_equity_min_size_fallback_enabled)

    def min_size_fallback_max_risk_pct(self) -> float:
        return float(self.risk.low_equity_min_size_fallback_max_risk_pct)

    def max_daily_loss_value(self, equity: float | None = None) -> float:
        eq = self._resolved_equity(equity)
        stop_pct = self.effective_daily_stop_pct(eq)
        return -(eq * stop_pct)

    def per_trade_risk_amount(self, equity: float | None = None) -> float:
        eq = self._resolved_equity(equity)
        return eq * self.effective_risk_per_trade(equity=eq)

    def should_turn_off_for_day(self, daily_pnl: float, equity: float | None = None) -> bool:
        return daily_pnl <= self.max_daily_loss_value(equity=equity)

    def position_risk_amount(self, position: PositionRecord) -> float:
        return abs(position.entry_price - position.stop_price) * position.size

    def total_open_risk_pct(self, open_positions: list[PositionRecord], equity: float | None = None) -> float:
        eq = self._resolved_equity(equity)
        if eq <= 0:
            return 0.0
        total_risk = sum(self.position_risk_amount(position) for position in open_positions)
        return total_risk / eq

    def can_open_new_trade(self, stats: DailyStats, open_positions_count: int, equity: float | None = None) -> RiskCheck:
        reasons: list[str] = []
        eq = self._resolved_equity(equity)
        if stats.status == "OFF":
            reasons.append("BOT_OFF_DAILY_STOP")
        if self.should_turn_off_for_day(stats.pnl, equity=eq):
            reasons.append("DAILY_STOP_HIT")
        if stats.trades_count >= self.effective_max_trades_per_day(eq):
            reasons.append("MAX_TRADES_REACHED")
        if open_positions_count >= self.risk.max_positions:
            reasons.append("MAX_POSITIONS_REACHED")
        return RiskCheck(allowed=len(reasons) == 0, reason_codes=reasons)

    def can_open_new_trade_multi(
        self,
        *,
        now: datetime,
        asset_epic: str,
        asset_stats: DailyStats,
        global_stats: DailyStats,
        asset_open_positions: list[PositionRecord],
        all_open_positions: list[PositionRecord],
        new_trade_risk_amount: float,
        cooldown_until: datetime | None,
        equity: float | None = None,
    ) -> RiskCheck:
        reasons: list[str] = []
        metadata: dict[str, float | int | str] = {}
        epic = asset_epic.strip().upper()
        eq = self._resolved_equity(equity)

        if asset_stats.status == "OFF":
            reasons.append("ASSET_BOT_OFF_DAILY_STOP")
        if global_stats.status == "OFF":
            reasons.append("GLOBAL_BOT_OFF_DAILY_STOP")
        if self.should_turn_off_for_day(asset_stats.pnl, equity=eq):
            reasons.append("ASSET_DAILY_STOP_HIT")
        if self.should_turn_off_for_day(global_stats.pnl, equity=eq):
            reasons.append("GLOBAL_DAILY_STOP_HIT")
        if asset_stats.trades_count >= self.effective_max_trades_per_day(eq):
            reasons.append("ASSET_MAX_TRADES_REACHED")
        if len(asset_open_positions) >= self.risk.max_positions:
            reasons.append("ASSET_MAX_POSITIONS_REACHED")
        if len(all_open_positions) >= self.risk.global_max_positions:
            reasons.append("GLOBAL_MAX_POSITIONS_REACHED")

        if cooldown_until is not None and now < cooldown_until:
            reasons.append("COOLDOWN_ACTIVE")
            metadata["cooldown_until"] = cooldown_until.isoformat()

        current_risk_pct = self.total_open_risk_pct(all_open_positions, equity=eq)
        metadata["current_open_risk_pct"] = round(current_risk_pct, 6)
        new_risk_pct = new_trade_risk_amount / eq if eq > 0 else 0.0
        metadata["new_trade_risk_pct"] = round(new_risk_pct, 6)
        projected_risk_pct = current_risk_pct + new_risk_pct
        metadata["projected_open_risk_pct"] = round(projected_risk_pct, 6)
        if projected_risk_pct > self.risk.max_total_risk_pct:
            reasons.append("GLOBAL_MAX_TOTAL_RISK_REACHED")

        for group in self.risk.correlation_groups:
            if epic not in group.epics:
                continue
            open_in_group = sum(
                1 for position in all_open_positions if position.epic.strip().upper() in group.epics
            )
            if open_in_group >= group.max_open_positions:
                reasons.append(f"CORRELATION_LIMIT_{group.name}")

        return RiskCheck(allowed=len(reasons) == 0, reason_codes=reasons, metadata=metadata)

    @staticmethod
    def is_valid_size(size: float, min_size: float) -> bool:
        return size >= min_size
