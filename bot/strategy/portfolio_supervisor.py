from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from bot.config import PortfolioSupervisorConfig
from bot.storage.models import PositionRecord


@dataclass(slots=True)
class EntryProposal:
    symbol: str
    strategy_name: str
    priority: int
    score_total: float | None
    rank_score: float | None
    risk_r: float
    cooldown_seconds: int
    payload: dict = field(default_factory=dict)


@dataclass(slots=True)
class SupervisorResult:
    accepted_symbols: list[str]
    blocked: dict[str, list[str]]
    selected: list[EntryProposal]


class PortfolioSupervisor:
    def __init__(self, config: PortfolioSupervisorConfig):
        self.config = config
        self._last_entry_at: dict[str, datetime] = {}
        self._daily_r_used_by_day: dict[str, float] = {}

    def daily_r_used(self, trading_day: str) -> float:
        return self._daily_r_used_by_day.get(trading_day, 0.0)

    def register_entry(self, trading_day: str, proposal: EntryProposal, now: datetime) -> None:
        self._last_entry_at[proposal.symbol] = now
        self._daily_r_used_by_day[trading_day] = self.daily_r_used(trading_day) + proposal.risk_r

    def evaluate_entries(
        self,
        *,
        now: datetime,
        trading_day: str,
        proposals: list[EntryProposal],
        open_positions: list[PositionRecord],
    ) -> SupervisorResult:
        blocked: dict[str, list[str]] = {}
        selected: list[EntryProposal] = []
        accepted_symbols: list[str] = []
        symbol_open_count: dict[str, int] = {}
        for position in open_positions:
            symbol_open_count[position.epic] = symbol_open_count.get(position.epic, 0) + 1

        slots_available = max(0, self.config.max_open_positions_total - len(open_positions))
        if slots_available <= 0:
            for proposal in proposals:
                blocked.setdefault(proposal.symbol, []).append("SUPERVISOR_MAX_OPEN_TOTAL")
            return SupervisorResult(accepted_symbols=[], blocked=blocked, selected=[])

        sorted_proposals = sorted(
            proposals,
            key=self._proposal_sort_key,
            reverse=True,
        )
        daily_r = self.daily_r_used(trading_day)
        for proposal in sorted_proposals:
            reasons: list[str] = []
            open_for_symbol = symbol_open_count.get(proposal.symbol, 0)
            if open_for_symbol >= self.config.max_per_symbol:
                reasons.append("SUPERVISOR_MAX_PER_SYMBOL")

            cooldown_seconds = max(
                proposal.cooldown_seconds,
                self.config.default_cooldown_seconds,
            )
            last_entry = self._last_entry_at.get(proposal.symbol)
            if last_entry is not None and now < (last_entry + timedelta(seconds=cooldown_seconds)):
                reasons.append("SUPERVISOR_COOLDOWN")

            projected_r = daily_r + proposal.risk_r
            if projected_r > self.config.daily_risk_r:
                reasons.append("SUPERVISOR_DAILY_R_LIMIT")

            if reasons:
                blocked.setdefault(proposal.symbol, []).extend(reasons)
                continue

            selected.append(proposal)
            accepted_symbols.append(proposal.symbol)
            slots_available -= 1
            daily_r = projected_r
            symbol_open_count[proposal.symbol] = open_for_symbol + 1
            if slots_available <= 0 or len(selected) >= self.config.max_entries_per_cycle:
                break

        return SupervisorResult(
            accepted_symbols=accepted_symbols,
            blocked=blocked,
            selected=selected,
        )

    @staticmethod
    def _proposal_sort_key(proposal: EntryProposal) -> tuple[float, float, float]:
        rank = proposal.rank_score if proposal.rank_score is not None else proposal.score_total or -1.0
        if proposal.strategy_name == "INDEX_EXISTING":
            # For indexes use explicit priority first.
            return (2.0, float(proposal.priority), rank)
        # For scalp score dominates.
        return (1.0, rank, float(proposal.priority))
