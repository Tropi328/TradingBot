"""Tests for ATR cache content-based key, ShadowObserver context-manager,
_trade_r_multiple warning, and _strip_mode_prefix shared util.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bot.data.candles import Candle
from bot.strategy.indicators import _ATR_CACHE, _atr_cache_key, atr
from bot.strategy.shadow_observer import ShadowCandidate, ShadowObserver
from bot.execution.utils import strip_mode_prefix


# ── ATR cache ────────────────────────────────────────────────────────────

def _make_candles(n: int, *, start_ts: int = 1_000_000) -> list[Candle]:
    """Build a list of synthetic candles."""
    candles = []
    for i in range(n):
        ts = datetime.fromtimestamp(start_ts + i * 60, tz=timezone.utc)
        candles.append(
            Candle(timestamp=ts, open=100 + i, high=101 + i, low=99 + i, close=100.5 + i)
        )
    return candles


class TestAtrCacheKey:
    def test_same_content_same_key(self):
        a = _make_candles(20)
        b = _make_candles(20)
        assert _atr_cache_key(a, 14) == _atr_cache_key(b, 14)

    def test_different_length_different_key(self):
        a = _make_candles(20)
        b = _make_candles(25)
        assert _atr_cache_key(a, 14) != _atr_cache_key(b, 14)

    def test_different_period_different_key(self):
        a = _make_candles(20)
        assert _atr_cache_key(a, 14) != _atr_cache_key(a, 20)

    def test_incremental_cache_hit(self):
        """New candles appended => cache hit with same prefix timestamps."""
        _ATR_CACHE.clear()
        candles = _make_candles(30)
        result_1 = atr(candles[:20], 14)
        assert len(result_1) == 20

        result_2 = atr(candles[:20], 14)
        # Should have hit the cache (same list content)
        assert result_2 is result_1  # exact same list object returned

    def test_cache_miss_after_gc_no_id_collision(self):
        """Ensure new list with different content doesn't wrongly hit cache.

        Old bug: ``id(candles)`` could be reused after GC.
        """
        _ATR_CACHE.clear()
        c1 = _make_candles(20, start_ts=1_000_000)
        atr(c1, 14)

        # Create a *different* series that could get the same id() after GC
        c2 = _make_candles(20, start_ts=9_000_000)
        key1 = _atr_cache_key(c1, 14)
        key2 = _atr_cache_key(c2, 14)
        assert key1 != key2


# ── _trade_r_multiple ────────────────────────────────────────────────────

class TestTradeRMultipleWarning:
    """Ensure _trade_r_multiple logs warnings when denominator is zero."""

    def test_warns_on_zero_initial_risk(self, caplog: pytest.LogCaptureFixture):
        from bot.backtest.engine import _trade_r_multiple

        @dataclass
        class _FakePosition:
            initial_risk: float = 0.0
            initial_size: float = 1.0

        with caplog.at_level(logging.WARNING):
            result = _trade_r_multiple(
                total_pnl=100.0,
                position=_FakePosition(),
                fx_converter=None,
                instrument_currency="USD",
                account_currency="USD",
                fx_apply_to=None,
            )
        assert result == 0.0
        assert "denominator" in caplog.text.lower() or "R-multiple" in caplog.text

    def test_warns_on_zero_initial_size(self, caplog: pytest.LogCaptureFixture):
        from bot.backtest.engine import _trade_r_multiple

        @dataclass
        class _FakePosition:
            initial_risk: float = 10.0
            initial_size: float = 0.0

        with caplog.at_level(logging.WARNING):
            result = _trade_r_multiple(
                total_pnl=100.0,
                position=_FakePosition(),
                fx_converter=None,
                instrument_currency="USD",
                account_currency="USD",
                fx_apply_to=None,
            )
        assert result == 0.0
        assert "denominator" in caplog.text.lower() or "R-multiple" in caplog.text


# ── ShadowObserver context manager ──────────────────────────────────────

class TestShadowObserverContextManager:
    def test_context_manager_closes_file(self, tmp_path: Path):
        out = tmp_path / "shadow.jsonl"
        candidate = ShadowCandidate(
            timestamp="2024-01-01T00:00:00Z",
            symbol="XAUUSD",
            side="LONG",
            action="OBSERVE",
            tier="B",
            score_v2=0.5,
        )
        with ShadowObserver(out) as obs:
            obs.record(candidate)
            assert obs._file is not None
        # After exit, file should be closed
        assert obs._file is None
        # Data should be written
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["symbol"] == "XAUUSD"
        assert data["action"] == "OBSERVE"

    def test_flush_after_write(self, tmp_path: Path):
        out = tmp_path / "shadow2.jsonl"
        obs = ShadowObserver(out)
        candidate = ShadowCandidate(
            timestamp="2024-01-01T00:00:00Z",
            symbol="BTCUSD",
            side="SHORT",
            action="TRADE",
            tier="A_plus",
            score_v2=0.9,
        )
        obs.record(candidate)
        # File should exist and be non-empty even before close (flush-after-write)
        assert out.exists()
        assert out.stat().st_size > 0
        obs.close()

    def test_no_output_path_still_records(self):
        obs = ShadowObserver(output_path=None)
        candidate = ShadowCandidate(
            timestamp="2024-01-01T00:00:00Z",
            symbol="XAUUSD",
            side="LONG",
            action="SMALL",
            tier="B",
            score_v2=0.3,
        )
        obs.record(candidate)
        assert len(obs.records) == 1
        obs.close()


# ── strip_mode_prefix ───────────────────────────────────────────────────

class TestStripModePrefix:
    def test_strips_dry(self):
        assert strip_mode_prefix("DRY-abc123") == "abc123"

    def test_strips_paper(self):
        assert strip_mode_prefix("PAPER-order456") == "order456"

    def test_preserves_plain_id(self):
        assert strip_mode_prefix("plain_id") == "plain_id"

    def test_preserves_non_mode_prefix(self):
        assert strip_mode_prefix("LIVE-abc123") == "LIVE-abc123"

    def test_preserves_empty_rest(self):
        # "DRY-" with empty rest should return original
        assert strip_mode_prefix("DRY-") == "DRY-"

    def test_no_hyphen(self):
        assert strip_mode_prefix("nodash") == "nodash"
