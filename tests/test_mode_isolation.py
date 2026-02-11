from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bot.execution.orders import OrderExecutor
from bot.execution.position_manager import PositionManager
from bot.storage.db import get_connection, init_db
from bot.storage.journal import Journal
from bot.storage.models import OrderRecord, PositionRecord


def _order(order_id: str, now: datetime) -> OrderRecord:
    return OrderRecord(
        order_id=order_id,
        deal_reference=None,
        request_id=f"REQ-{order_id}",
        epic="GOLD",
        side="LONG",
        size=0.01,
        entry_price=2000.0,
        stop_price=1990.0,
        take_profit=2020.0,
        status="PENDING",
        remote_status=None,
        filled_size=0.0,
        expires_at=now + timedelta(minutes=30),
        created_at=now,
        updated_at=now,
    )


def _position(deal_id: str, now: datetime) -> PositionRecord:
    return PositionRecord(
        deal_id=deal_id,
        epic="GOLD",
        side="LONG",
        size=0.01,
        entry_price=2000.0,
        stop_price=1990.0,
        take_profit=2020.0,
        status="OPEN",
        opened_at=now,
    )


def test_order_and_position_mode_isolation(tmp_path) -> None:
    db_path = tmp_path / "mode_isolation.db"
    conn = get_connection(db_path)
    init_db(conn)
    journal = Journal(conn)
    now = datetime(2026, 2, 10, tzinfo=timezone.utc)

    journal.upsert_order(_order("DRY-111", now))
    journal.upsert_order(_order("PAPER-222", now))
    journal.upsert_position(_position("DRY-aaa", now))
    journal.upsert_position(_position("PAPER-bbb", now))

    dry_orders = OrderExecutor(
        client=None,
        journal=journal,
        dry_run=True,
        default_epic="GOLD",
        default_currency="USD",
    ).get_pending_orders()
    paper_orders = OrderExecutor(
        client=None,
        journal=journal,
        dry_run=False,
        default_epic="GOLD",
        default_currency="USD",
    ).get_pending_orders()

    dry_positions = PositionManager(client=None, journal=journal, dry_run=True).get_open_positions()
    paper_positions = PositionManager(client=None, journal=journal, dry_run=False).get_open_positions()

    assert [o.order_id for o in dry_orders] == ["DRY-111"]
    assert [o.order_id for o in paper_orders] == ["PAPER-222"]
    assert [p.deal_id for p in dry_positions] == ["DRY-aaa"]
    assert [p.deal_id for p in paper_positions] == ["PAPER-bbb"]
