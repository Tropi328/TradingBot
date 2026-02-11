from __future__ import annotations

from datetime import datetime, timezone

from bot.config import PortfolioSupervisorConfig
from bot.storage.models import PositionRecord
from bot.strategy.portfolio_supervisor import EntryProposal, PortfolioSupervisor


def _open_position(symbol: str) -> PositionRecord:
    now = datetime(2026, 2, 11, tzinfo=timezone.utc)
    return PositionRecord(
        deal_id=f"PAPER-{symbol}-1",
        epic=symbol,
        side="LONG",
        size=1.0,
        entry_price=100.0,
        stop_price=90.0,
        take_profit=120.0,
        status="OPEN",
        opened_at=now,
    )


def test_portfolio_supervisor_blocks_limits() -> None:
    supervisor = PortfolioSupervisor(
        PortfolioSupervisorConfig(
            max_open_positions_total=2,
            max_per_symbol=1,
            daily_risk_r=2.0,
            default_cooldown_seconds=0,
            max_entries_per_cycle=2,
        )
    )
    now = datetime(2026, 2, 11, 10, 0, tzinfo=timezone.utc)
    proposals = [
        EntryProposal(symbol="US500", strategy_name="INDEX_EXISTING", priority=95, score_total=None, rank_score=96.0, risk_r=1.0, cooldown_seconds=0),
        EntryProposal(symbol="XAUUSD", strategy_name="SCALP_ICT_PA", priority=70, score_total=82.0, rank_score=82.5, risk_r=1.0, cooldown_seconds=0),
        EntryProposal(symbol="BTCUSD", strategy_name="SCALP_ICT_PA", priority=65, score_total=60.0, rank_score=61.0, risk_r=1.0, cooldown_seconds=0),
    ]
    result = supervisor.evaluate_entries(
        now=now,
        trading_day="2026-02-11",
        proposals=proposals,
        open_positions=[],
    )
    assert len(result.selected) == 2
    assert result.selected[0].symbol == "US500"
    assert {item.symbol for item in result.selected} == {"US500", "XAUUSD"}


def test_portfolio_supervisor_daily_r_limit_and_per_symbol() -> None:
    supervisor = PortfolioSupervisor(
        PortfolioSupervisorConfig(
            max_open_positions_total=3,
            max_per_symbol=1,
            daily_risk_r=1.2,
            default_cooldown_seconds=0,
            max_entries_per_cycle=3,
        )
    )
    now = datetime(2026, 2, 11, 10, 0, tzinfo=timezone.utc)
    proposals = [
        EntryProposal(symbol="EURUSD", strategy_name="SCALP_ICT_PA", priority=70, score_total=88.0, rank_score=88.0, risk_r=0.4, cooldown_seconds=0),
        EntryProposal(symbol="EURUSD", strategy_name="SCALP_ICT_PA", priority=70, score_total=80.0, rank_score=80.0, risk_r=0.4, cooldown_seconds=0),
        EntryProposal(symbol="BTCUSD", strategy_name="SCALP_ICT_PA", priority=70, score_total=75.0, rank_score=75.0, risk_r=1.0, cooldown_seconds=0),
    ]
    result = supervisor.evaluate_entries(
        now=now,
        trading_day="2026-02-11",
        proposals=proposals,
        open_positions=[_open_position("US100")],
    )
    assert any(item.symbol == "EURUSD" for item in result.selected)
    assert "EURUSD" in result.blocked
    assert "SUPERVISOR_MAX_PER_SYMBOL" in result.blocked["EURUSD"]
