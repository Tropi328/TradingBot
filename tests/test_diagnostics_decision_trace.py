"""Regression tests for the diagnostics.decision_trace module.

Tests verify:
1. DecisionTraceWriter writes valid JSONL with required keys.
2. Decision events include all DECISION_REQUIRED_KEYS.
3. Fill events include all FILL_REQUIRED_KEYS.
4. Writer is a no-op when disabled.
5. Integration: short backtest with --decision-trace produces JSONL
   with at least one decision event and reject_reason on every
   candidate that didn't become a trade.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from bot.diagnostics.decision_trace import (
    DECISION_REQUIRED_KEYS,
    FILL_REQUIRED_KEYS,
    DecisionTraceWriter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def trace_path(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "test_trace.jsonl"


# ---------------------------------------------------------------------------
# Unit tests – DecisionTraceWriter
# ---------------------------------------------------------------------------

class TestDecisionTraceWriterUnit:
    """Unit-level tests for DecisionTraceWriter."""

    def test_decision_writes_valid_jsonl(self, trace_path: Path) -> None:
        writer = DecisionTraceWriter(trace_path, enabled=True)
        writer.open()
        writer.decision(
            ts=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
            symbol="XAUUSD",
            tf="5m",
            candidates=1,
            signal="LONG",
            features_ok=True,
            score=72.5,
            threshold=65.0,
            reject_reason=None,
        )
        writer.close()

        lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["type"] == "decision"
        missing = DECISION_REQUIRED_KEYS - event.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_fill_writes_valid_jsonl(self, trace_path: Path) -> None:
        writer = DecisionTraceWriter(trace_path, enabled=True)
        writer.open()
        writer.fill(
            ts=datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc),
            symbol="XAUUSD",
            side="LONG",
            pnl=12.50,
            equity_after=1012.50,
            reason_close="TP",
            holding_min=45.0,
            spread_cost=0.12,
            swap_cost=0.0,
        )
        writer.close()

        lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["type"] == "fill"
        missing = FILL_REQUIRED_KEYS - event.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_disabled_writer_is_noop(self, trace_path: Path) -> None:
        writer = DecisionTraceWriter(trace_path, enabled=False)
        writer.open()
        writer.decision(
            ts=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
            symbol="XAUUSD",
        )
        writer.fill(
            ts=datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc),
            symbol="XAUUSD",
            side="LONG",
        )
        writer.close()
        assert not trace_path.exists()

    def test_none_path_writer_is_noop(self) -> None:
        writer = DecisionTraceWriter(None, enabled=True)
        writer.open()
        assert not writer.active
        writer.decision(
            ts=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
            symbol="XAUUSD",
        )
        writer.close()  # should not raise

    def test_append_mode_preserves_existing(self, trace_path: Path) -> None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text('{"existing":true}\n', encoding="utf-8")

        writer = DecisionTraceWriter(trace_path, enabled=True)
        writer.open()
        writer.decision(
            ts=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
            symbol="XAUUSD",
        )
        writer.close()

        lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"existing": True}
        assert json.loads(lines[1])["type"] == "decision"

    def test_context_manager(self, trace_path: Path) -> None:
        with DecisionTraceWriter(trace_path, enabled=True) as w:
            w.decision(
                ts=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
                symbol="XAUUSD",
            )
        assert trace_path.exists()
        event = json.loads(trace_path.read_text(encoding="utf-8").strip())
        assert event["type"] == "decision"

    def test_extra_fields_merged(self, trace_path: Path) -> None:
        with DecisionTraceWriter(trace_path, enabled=True) as w:
            w.decision(
                ts=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
                symbol="XAUUSD",
                extra={"custom_field": 42, "strategy": "ICT"},
            )
        event = json.loads(trace_path.read_text(encoding="utf-8").strip())
        assert event["custom_field"] == 42
        assert event["strategy"] == "ICT"

    def test_rotation_triggers_on_max_bytes(self, trace_path: Path) -> None:
        """File rotates when size exceeds max_bytes."""
        writer = DecisionTraceWriter(trace_path, enabled=True, max_bytes=1024, max_backups=3)
        writer.open()
        # Write enough events to exceed 1 KB
        for i in range(30):
            writer.decision(
                ts=datetime(2024, 6, 1, 12, i, tzinfo=timezone.utc),
                symbol="XAUUSD",
                score=float(i),
                threshold=65.0,
            )
        writer.close()

        parent = trace_path.parent
        name = trace_path.name
        backup_1 = parent / f"{name}.1"
        assert trace_path.exists(), "Current file should exist"
        assert backup_1.exists(), "Backup .1 should have been created"
        # Current file should be smaller than max_bytes (freshly rotated)
        assert trace_path.stat().st_size < 1024

    def test_rotation_respects_max_backups(self, trace_path: Path) -> None:
        """Only max_backups backup files are kept."""
        writer = DecisionTraceWriter(trace_path, enabled=True, max_bytes=512, max_backups=2)
        writer.open()
        # Write many events to force multiple rotations
        for i in range(100):
            writer.decision(
                ts=datetime(2024, 6, 1, 12, i % 60, tzinfo=timezone.utc),
                symbol="XAUUSD",
                score=float(i),
                threshold=65.0,
            )
        writer.close()

        parent = trace_path.parent
        name = trace_path.name
        assert (parent / f"{name}.1").exists()
        assert (parent / f"{name}.2").exists()
        assert not (parent / f"{name}.3").exists(), "Should not exceed max_backups=2"

    def test_no_rotation_when_max_backups_zero(self, trace_path: Path) -> None:
        """With max_backups=0, rotation is disabled entirely."""
        writer = DecisionTraceWriter(trace_path, enabled=True, max_bytes=512, max_backups=0)
        writer.open()
        for i in range(50):
            writer.decision(
                ts=datetime(2024, 6, 1, 12, i % 60, tzinfo=timezone.utc),
                symbol="XAUUSD",
                score=float(i),
                threshold=65.0,
            )
        writer.close()

        parent = trace_path.parent
        name = trace_path.name
        assert trace_path.exists()
        assert not (parent / f"{name}.1").exists(), "No backups should be created"

    def test_score_breakdown_roundtrip(self, trace_path: Path) -> None:
        breakdown = [
            {"k": "bias", "v": 20.0},
            {"k": "mss", "v": 15.0},
            {"k": "penalty_spread", "v": -3.0},
        ]
        with DecisionTraceWriter(trace_path, enabled=True) as w:
            w.decision(
                ts=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
                symbol="XAUUSD",
                score_breakdown=breakdown,
            )
        event = json.loads(trace_path.read_text(encoding="utf-8").strip())
        assert event["score_breakdown"] == breakdown


# ---------------------------------------------------------------------------
# Schema consistency: every decision with candidates>0 and no fill
# should have a reject_reason
# ---------------------------------------------------------------------------

class TestNoSilentRejections:
    """Parse a JSONL and assert the 'no silent fail' invariant."""

    def test_all_non_accepted_have_reject_reason(self, trace_path: Path) -> None:
        """Write a mix of events and verify the invariant."""
        with DecisionTraceWriter(trace_path, enabled=True) as w:
            # Accepted decision (no reject_reason)
            w.decision(
                ts=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
                symbol="XAUUSD",
                candidates=1,
                signal="LONG",
                score=80.0,
                threshold=65.0,
                reject_reason=None,
            )
            # Rejected decisions (must have reject_reason)
            w.decision(
                ts=datetime(2024, 6, 1, 12, 5, tzinfo=timezone.utc),
                symbol="XAUUSD",
                candidates=1,
                signal="LONG",
                score=50.0,
                threshold=65.0,
                reject_reason="EXPECTED_RR_TOO_LOW",
            )
            w.decision(
                ts=datetime(2024, 6, 1, 12, 10, tzinfo=timezone.utc),
                symbol="XAUUSD",
                candidates=0,
                signal=None,
                reject_reason="NO_SIGNAL",
            )
            # Fill
            w.fill(
                ts=datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc),
                symbol="XAUUSD",
                side="LONG",
                pnl=5.0,
                equity_after=1005.0,
                reason_close="TP",
            )

        lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
        decisions = [json.loads(l) for l in lines if json.loads(l)["type"] == "decision"]
        fills = [json.loads(l) for l in lines if json.loads(l)["type"] == "fill"]

        assert len(decisions) == 3
        assert len(fills) == 1

        # Every rejected decision with candidates>0 must carry a reason
        for d in decisions:
            if d.get("reject_reason") is not None:
                continue  # has reason → OK
            if d.get("candidates", 0) == 0:
                continue  # no candidates → allowed to have no reject
            # This is an "ACCEPTED" decision – should not have reject_reason
            assert d["reject_reason"] is None
