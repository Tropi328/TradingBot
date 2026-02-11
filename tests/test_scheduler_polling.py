from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from bot.clock import should_poll_closed_candle
from main import resolve_db_path


def test_should_poll_closed_candle_avoids_duplicates() -> None:
    now = datetime(2026, 2, 10, 12, 7, 30, tzinfo=timezone.utc)
    processed = datetime(2026, 2, 10, 12, 5, 0, tzinfo=timezone.utc)
    should_poll, target = should_poll_closed_candle(
        now_utc=now,
        timeframe="M5",
        last_processed_closed_ts=processed,
        last_attempt_target_ts=None,
        last_attempt_at=None,
        close_grace_seconds=3,
        retry_seconds=15,
    )
    assert target == datetime(2026, 2, 10, 12, 5, 0, tzinfo=timezone.utc)
    assert should_poll is False


def test_should_poll_closed_candle_retry_guard() -> None:
    now = datetime(2026, 2, 10, 12, 7, 30, tzinfo=timezone.utc)
    should_poll, target = should_poll_closed_candle(
        now_utc=now,
        timeframe="M5",
        last_processed_closed_ts=datetime(2026, 2, 10, 12, 0, 0, tzinfo=timezone.utc),
        last_attempt_target_ts=datetime(2026, 2, 10, 12, 5, 0, tzinfo=timezone.utc),
        last_attempt_at=now - timedelta(seconds=5),
        close_grace_seconds=3,
        retry_seconds=15,
    )
    assert target == datetime(2026, 2, 10, 12, 5, 0, tzinfo=timezone.utc)
    assert should_poll is False


def test_resolve_db_path_isolated_by_mode(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SQLITE_PATH", raising=False)
    monkeypatch.delenv("SQLITE_PATH_TEMPLATE", raising=False)
    root = Path(tmp_path)
    dry = resolve_db_path(root, paper_mode=False)
    paper = resolve_db_path(root, paper_mode=True)
    assert dry.endswith("bot_state_dry.db")
    assert paper.endswith("bot_state_paper.db")
    assert dry != paper


def test_resolve_db_path_suffixes_sqlite_path(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SQLITE_PATH_TEMPLATE", raising=False)
    monkeypatch.setenv("SQLITE_PATH", "bot_state.db")
    root = Path(tmp_path)
    dry = resolve_db_path(root, paper_mode=False)
    paper = resolve_db_path(root, paper_mode=True)
    assert dry.endswith("bot_state_dry.db")
    assert paper.endswith("bot_state_paper.db")
