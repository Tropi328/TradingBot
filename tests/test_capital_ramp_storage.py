from __future__ import annotations

from datetime import datetime, timezone

from bot.capital_ramp import CapitalRampEvent, CapitalRampRuntime
from bot.storage.db import get_connection, init_db
from bot.storage.journal import Journal
from bot.storage.models import PositionRecord


def test_capital_ramp_state_and_events_persist_across_restart(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    conn = get_connection(db_path)
    init_db(conn)
    journal = Journal(conn)

    runtime = CapitalRampRuntime.initialize(
        scope="CAPITAL_RAMP:PAPER",
        now_utc=datetime(2026, 2, 23, 12, 0, tzinfo=timezone.utc),
        timezone_name="Europe/Warsaw",
        current_closed_pnl=0.0,
    )
    journal.upsert_capital_ramp_state(runtime.state)

    topup_event, stop_event = runtime.maybe_apply_topup(
        now_utc=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        current_closed_pnl=0.0,
    )
    assert topup_event is not None
    assert stop_event is None
    journal.append_capital_ramp_event(topup_event)
    journal.upsert_capital_ramp_state(runtime.state)

    persisted = journal.get_capital_ramp_state("CAPITAL_RAMP:PAPER")
    assert persisted is not None
    assert persisted.topups_count == 1
    assert persisted.topups_total == 100.0
    assert persisted.next_topup_date_local is not None

    # Simulate process restart.
    conn.close()
    conn2 = get_connection(db_path)
    init_db(conn2)
    journal2 = Journal(conn2)
    restored = journal2.get_capital_ramp_state("CAPITAL_RAMP:PAPER")
    assert restored is not None
    assert restored.topups_count == 1
    events = journal2.list_capital_ramp_events("CAPITAL_RAMP:PAPER")
    assert len(events) == 1
    assert events[0].event_type == "TOPUP"

    conn2.close()


def test_sum_closed_pnl_uses_prefix_scope(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    conn = get_connection(db_path)
    init_db(conn)
    journal = Journal(conn)

    now = datetime(2026, 2, 23, 12, 0, tzinfo=timezone.utc)
    for deal_id, pnl in (("PAPER-1", 10.0), ("PAPER-2", -3.0), ("DRY-1", 99.0)):
        journal.upsert_position(
            PositionRecord(
                deal_id=deal_id,
                epic="XAUUSD",
                side="LONG",
                size=1.0,
                entry_price=100.0,
                stop_price=99.0,
                take_profit=101.0,
                status="CLOSED",
                opened_at=now,
                closed_at=now,
                pnl=pnl,
                metadata={},
            )
        )

    assert journal.sum_closed_pnl("PAPER") == 7.0
    assert journal.sum_closed_pnl("DRY") == 99.0
    assert journal.sum_closed_pnl("LIVE") == 0.0
    conn.close()


def test_append_and_list_capital_ramp_event_roundtrip(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    conn = get_connection(db_path)
    init_db(conn)
    journal = Journal(conn)

    event = CapitalRampEvent(
        scope="CAPITAL_RAMP:LIVE",
        event_type="TOPUP",
        event_ts_utc=datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc),
        local_date=datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc).date(),
        amount=100.0,
        model_equity=250.0,
        payload={"topups_count": 1},
    )
    journal.append_capital_ramp_event(event)
    loaded = journal.list_capital_ramp_events("CAPITAL_RAMP:LIVE")
    assert len(loaded) == 1
    assert loaded[0].payload.get("topups_count") == 1
    assert loaded[0].model_equity == 250.0
    conn.close()
