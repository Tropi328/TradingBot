from __future__ import annotations

from datetime import datetime, timezone

from bot.capital_ramp import (
    EVENT_STOP_TARGET,
    EVENT_STOP_YEAR_END,
    EVENT_TOPUP,
    MONTHLY_TOPUP_PLN,
    START_EQUITY_PLN,
    STOP_REASON_TARGET,
    STOP_REASON_YEAR_END,
    TARGET_EQUITY_PLN,
    CapitalRampRuntime,
)


def test_first_topup_is_next_month_first_day() -> None:
    runtime = CapitalRampRuntime.initialize(
        scope="CAPITAL_RAMP:PAPER",
        now_utc=datetime(2026, 2, 23, 12, 0, tzinfo=timezone.utc),
        timezone_name="Europe/Warsaw",
        current_closed_pnl=0.0,
    )

    topup_early, stop_early = runtime.maybe_apply_topup(
        now_utc=datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc),
        current_closed_pnl=0.0,
    )
    assert topup_early is None
    assert stop_early is None

    topup_due, stop_due = runtime.maybe_apply_topup(
        now_utc=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        current_closed_pnl=0.0,
    )
    assert topup_due is not None
    assert topup_due.event_type == EVENT_TOPUP
    assert topup_due.amount == MONTHLY_TOPUP_PLN
    assert stop_due is None

    topup_same_month, _ = runtime.maybe_apply_topup(
        now_utc=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        current_closed_pnl=0.0,
    )
    assert topup_same_month is None


def test_stop_when_target_reached_from_realized_pnl() -> None:
    runtime = CapitalRampRuntime.initialize(
        scope="CAPITAL_RAMP:PAPER",
        now_utc=datetime(2026, 2, 23, 12, 0, tzinfo=timezone.utc),
        timezone_name="Europe/Warsaw",
        current_closed_pnl=0.0,
    )
    topup_event, _ = runtime.maybe_apply_topup(
        now_utc=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        current_closed_pnl=0.0,
    )
    assert topup_event is not None

    # 100 start + 100 topup + 1000 realized = 1200 target
    topup_after_target, stop_after_target = runtime.maybe_apply_topup(
        now_utc=datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc),
        current_closed_pnl=1000.0,
    )
    assert topup_after_target is None
    assert stop_after_target is not None
    assert stop_after_target.event_type == EVENT_STOP_TARGET
    assert runtime.state.stopped_reason == STOP_REASON_TARGET
    assert runtime.model_equity(current_closed_pnl=1000.0) == TARGET_EQUITY_PLN


def test_stop_on_year_end_when_target_not_reached() -> None:
    runtime = CapitalRampRuntime.initialize(
        scope="CAPITAL_RAMP:PAPER",
        now_utc=datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc),
        timezone_name="Europe/Warsaw",
        current_closed_pnl=0.0,
    )
    topup_dec, stop_dec = runtime.maybe_apply_topup(
        now_utc=datetime(2026, 12, 1, 12, 0, tzinfo=timezone.utc),
        current_closed_pnl=0.0,
    )
    assert topup_dec is not None
    assert stop_dec is None

    topup_next_year, stop_next_year = runtime.maybe_apply_topup(
        now_utc=datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc),
        current_closed_pnl=0.0,
    )
    assert topup_next_year is None
    assert stop_next_year is not None
    assert stop_next_year.event_type == EVENT_STOP_YEAR_END
    assert runtime.state.stopped_reason == STOP_REASON_YEAR_END


def test_effective_equity_tracks_topups_and_realized_pnl() -> None:
    runtime = CapitalRampRuntime.initialize(
        scope="CAPITAL_RAMP:PAPER",
        now_utc=datetime(2026, 2, 23, 12, 0, tzinfo=timezone.utc),
        timezone_name="Europe/Warsaw",
        current_closed_pnl=0.0,
    )
    runtime.maybe_apply_topup(
        now_utc=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        current_closed_pnl=0.0,
    )
    runtime.maybe_apply_topup(
        now_utc=datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc),
        current_closed_pnl=0.0,
    )
    effective = runtime.effective_equity(
        now_utc=datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc),
        current_closed_pnl=55.0,
    )
    assert effective == START_EQUITY_PLN + (2 * MONTHLY_TOPUP_PLN) + 55.0
