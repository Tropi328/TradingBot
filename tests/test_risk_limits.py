from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bot.config import RiskConfig
from bot.execution.sizing import position_size_from_risk
from bot.news.calendar_provider import Event
from bot.news.gate import is_blocked, should_cancel_pending
from bot.storage.models import DailyStats, PositionRecord
from bot.strategy.risk import RiskEngine


def test_max_trades_per_day_limit() -> None:
    engine = RiskEngine(
        RiskConfig(
            equity=10000,
            risk_per_trade=0.005,
            max_trades_per_day=3,
            daily_stop_pct=0.015,
            max_positions=1,
        )
    )
    stats = DailyStats(trading_day="2026-01-01", pnl=0.0, trades_count=3, status="ON")
    result = engine.can_open_new_trade(stats, open_positions_count=0)

    assert result.allowed is False
    assert "MAX_TRADES_REACHED" in result.reason_codes


def test_daily_stop_limit() -> None:
    engine = RiskEngine(
        RiskConfig(
            equity=10000,
            risk_per_trade=0.005,
            max_trades_per_day=3,
            daily_stop_pct=0.015,
            max_positions=1,
        )
    )
    stats = DailyStats(trading_day="2026-01-01", pnl=-151.0, trades_count=1, status="ON")
    result = engine.can_open_new_trade(stats, open_positions_count=0)

    assert result.allowed is False
    assert "DAILY_STOP_HIT" in result.reason_codes
    assert engine.should_turn_off_for_day(stats.pnl) is True


def test_position_size_from_risk() -> None:
    # Risk amount = 10000 * 0.005 = 50, distance=10 -> size=5
    size = position_size_from_risk(
        equity=10000,
        risk_per_trade=0.005,
        entry_price=2000,
        stop_price=1990,
        min_size=0.01,
        size_step=0.01,
    )
    assert size == 5.0


def test_news_gate_block_and_cancel_window_60_minutes() -> None:
    now = datetime(2026, 2, 10, 12, 0, tzinfo=timezone.utc)
    events = [
        Event(
            event_id="1",
            title="US CPI",
            currency="USD",
            impact="HIGH",
            time=now + timedelta(minutes=30),
            category="macro",
            source="test",
        )
    ]

    assert is_blocked(now, events, block_minutes=60) is True
    assert is_blocked(now, events, block_minutes=20) is False

    order = {"status": "PENDING"}
    assert should_cancel_pending(order, now, events, block_minutes=60) is True


def test_global_risk_limit_blocks_new_trade() -> None:
    engine = RiskEngine(
        RiskConfig(
            equity=10000,
            risk_per_trade=0.005,
            max_trades_per_day=3,
            daily_stop_pct=0.015,
            max_positions=1,
            global_max_positions=3,
            max_total_risk_pct=0.01,
        )
    )
    now = datetime(2026, 2, 10, 12, 0, tzinfo=timezone.utc)
    stats_asset = DailyStats(trading_day="2026-02-10", epic="GOLD", pnl=0.0, trades_count=0, status="ON")
    stats_global = DailyStats(trading_day="2026-02-10", epic="GLOBAL", pnl=0.0, trades_count=0, status="ON")
    open_positions = [
        PositionRecord(
            deal_id="1",
            epic="GOLD",
            side="LONG",
            size=1.0,
            entry_price=2000.0,
            stop_price=1960.0,
            take_profit=2080.0,
            status="OPEN",
            opened_at=now,
        ),
        PositionRecord(
            deal_id="2",
            epic="US100",
            side="LONG",
            size=1.0,
            entry_price=25000.0,
            stop_price=24940.0,
            take_profit=25120.0,
            status="OPEN",
            opened_at=now,
        ),
    ]
    result = engine.can_open_new_trade_multi(
        now=now,
        asset_epic="US500",
        asset_stats=stats_asset,
        global_stats=stats_global,
        asset_open_positions=[],
        all_open_positions=open_positions,
        new_trade_risk_amount=50.0,
        cooldown_until=None,
    )
    assert result.allowed is False
    assert "GLOBAL_MAX_TOTAL_RISK_REACHED" in result.reason_codes


def test_correlation_limit_blocks_us_indices() -> None:
    engine = RiskEngine(RiskConfig())
    now = datetime(2026, 2, 10, 12, 0, tzinfo=timezone.utc)
    stats_asset = DailyStats(trading_day="2026-02-10", epic="US500", pnl=0.0, trades_count=0, status="ON")
    stats_global = DailyStats(trading_day="2026-02-10", epic="GLOBAL", pnl=0.0, trades_count=0, status="ON")
    open_positions = [
        PositionRecord(
            deal_id="idx-1",
            epic="US100",
            side="LONG",
            size=1.0,
            entry_price=25000.0,
            stop_price=24950.0,
            take_profit=25100.0,
            status="OPEN",
            opened_at=now,
        )
    ]
    result = engine.can_open_new_trade_multi(
        now=now,
        asset_epic="US500",
        asset_stats=stats_asset,
        global_stats=stats_global,
        asset_open_positions=[],
        all_open_positions=open_positions,
        new_trade_risk_amount=10.0,
        cooldown_until=None,
    )
    assert result.allowed is False
    assert "CORRELATION_LIMIT_US_INDICES" in result.reason_codes


def test_cooldown_blocks_new_trade() -> None:
    engine = RiskEngine(RiskConfig())
    now = datetime(2026, 2, 10, 12, 0, tzinfo=timezone.utc)
    stats_asset = DailyStats(trading_day="2026-02-10", epic="GOLD", pnl=0.0, trades_count=0, status="ON")
    stats_global = DailyStats(trading_day="2026-02-10", epic="GLOBAL", pnl=0.0, trades_count=0, status="ON")
    result = engine.can_open_new_trade_multi(
        now=now,
        asset_epic="GOLD",
        asset_stats=stats_asset,
        global_stats=stats_global,
        asset_open_positions=[],
        all_open_positions=[],
        new_trade_risk_amount=10.0,
        cooldown_until=now + timedelta(minutes=5),
    )
    assert result.allowed is False
    assert "COOLDOWN_ACTIVE" in result.reason_codes
