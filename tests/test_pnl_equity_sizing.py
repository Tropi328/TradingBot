"""Tests for PnL definitions, equity tracking, position sizing, and SpreadGuard.

Covers:
1. fees == -swap_cost  (no double-counting)
2. equity_after == equity_before + pnl_net  (per-trade equity)
3. pnl_gross excludes swap; pnl_net = pnl_gross + fees
4. Position sizing modes (fixed_qty, fixed_notional, risk_pct_equity)
5. SpreadGuard blocking
"""

from __future__ import annotations

import pytest

# ── engine helpers ──────────────────────────────────────────────────────────
from bot.backtest.engine import BacktestTrade, _compute_trade_pnl_fields

# ── position sizer ─────────────────────────────────────────────────────────
from bot.execution.position_sizer import (
    SizingMode,
    SizingRequest,
    SizingResult,
    compute_position_size,
)

# ── spread guard ───────────────────────────────────────────────────────────
from bot.gating.spread_guard import SpreadGuard, SpreadGuardConfig


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PnL definition – fees == -swap_cost, no double-counting
# ═══════════════════════════════════════════════════════════════════════════


class TestPnlDefinition:
    def test_fees_equals_negative_swap_cost(self) -> None:
        """fees must be -swap_cost_total."""
        _, _, fees, _ = _compute_trade_pnl_fields(
            total_pnl=10.0,
            swap_total=-0.5,       # actual swap cash flow (negative = paid)
            swap_cost_total=0.5,   # positive number = cost
            spread_cost=0.02,
            slippage_cost=0.01,
            commission_cost=0.0,
            fx_cost=0.0,
        )
        assert fees == pytest.approx(-0.5)  # fees == -swap_cost

    def test_pnl_gross_excludes_swap(self) -> None:
        """pnl_gross = total_pnl - swap_total."""
        total_pnl = 5.0
        swap_total = -0.3  # paid 0.3
        pnl_g, _, _, _ = _compute_trade_pnl_fields(
            total_pnl=total_pnl,
            swap_total=swap_total,
            swap_cost_total=0.3,
            spread_cost=0.0,
            slippage_cost=0.0,
            commission_cost=0.0,
            fx_cost=0.0,
        )
        assert pnl_g == pytest.approx(total_pnl - swap_total)  # 5.0 - (-0.3) = 5.3

    def test_pnl_net_equals_pnl_gross_plus_fees(self) -> None:
        """pnl_net = pnl_gross + fees."""
        total_pnl = 5.0
        swap_total = -0.3
        swap_cost = 0.3
        pnl_g, pnl_n, fees, _ = _compute_trade_pnl_fields(
            total_pnl=total_pnl,
            swap_total=swap_total,
            swap_cost_total=swap_cost,
            spread_cost=0.0,
            slippage_cost=0.0,
            commission_cost=0.0,
            fx_cost=0.0,
        )
        assert pnl_n == pytest.approx(pnl_g + fees)

    def test_no_double_counting_swap_and_fees(self) -> None:
        """Subtracting BOTH swap_cost AND fees from pnl_gross would give
        a number lower than pnl_net.  We must NOT do that."""
        total_pnl = 10.0
        swap_total = -1.0
        swap_cost = 1.0
        pnl_g, pnl_n, fees, _ = _compute_trade_pnl_fields(
            total_pnl=total_pnl,
            swap_total=swap_total,
            swap_cost_total=swap_cost,
            spread_cost=0.0,
            slippage_cost=0.0,
            commission_cost=0.0,
            fx_cost=0.0,
        )
        wrong_double_counted = pnl_g - swap_cost + fees  # would subtract swap twice
        assert pnl_n != pytest.approx(wrong_double_counted)
        # correct: pnl_net uses fees only (NOT a separate -swap_cost)
        assert pnl_n == pytest.approx(pnl_g + fees)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Equity tracking: equity_after == equity_before + pnl_for_equity
# ═══════════════════════════════════════════════════════════════════════════


class TestEquityTracking:
    def test_equity_after_equals_before_plus_pnl(self) -> None:
        """equity_after = equity_before + pnl_for_equity (=total_pnl)."""
        total_pnl = 3.5
        _, _, _, pnl_eq = _compute_trade_pnl_fields(
            total_pnl=total_pnl,
            swap_total=-0.1,
            swap_cost_total=0.1,
            spread_cost=0.0,
            slippage_cost=0.0,
            commission_cost=0.0,
            fx_cost=0.0,
        )
        equity_before = 100.0
        equity_after = equity_before + pnl_eq
        assert equity_after == pytest.approx(103.5)

    def test_backtest_trade_equity_fields(self) -> None:
        """BacktestTrade dataclass can store equity_before / after."""
        trade = BacktestTrade(
            epic="XAUUSD",
            side="LONG",
            entry_time=None,  # type: ignore[arg-type]
            exit_time=None,  # type: ignore[arg-type]
            entry_price=2000.0,
            exit_price=2010.0,
            size=0.01,
            pnl=1.0,
            r_multiple=0.5,
            reason="SL",
            equity_before=100.0,
            equity_after=101.0,
            pnl_gross=1.2,
            pnl_net=1.0,
        )
        assert trade.equity_before == 100.0
        assert trade.equity_after == 101.0
        assert trade.pnl_gross == 1.2
        assert trade.pnl_net == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Position sizing modes
# ═══════════════════════════════════════════════════════════════════════════


class TestPositionSizer:
    def test_fixed_qty_mode(self) -> None:
        result = compute_position_size(SizingRequest(
            mode=SizingMode.FIXED_QTY,
            equity=10_000,
            entry_price=2000.0,
            sl_price=1990.0,
            qty=0.05,
        ))
        assert result.qty == pytest.approx(0.05)
        assert result.mode_used == "fixed_qty"

    def test_fixed_notional_mode(self) -> None:
        result = compute_position_size(SizingRequest(
            mode=SizingMode.FIXED_NOTIONAL,
            equity=10_000,
            entry_price=2000.0,
            sl_price=1990.0,
            notional_value=200.0,
        ))
        # 200 / 2000 = 0.10
        assert result.qty == pytest.approx(0.1)
        assert result.notional == pytest.approx(200.0)

    def test_risk_pct_equity_mode(self) -> None:
        result = compute_position_size(SizingRequest(
            mode=SizingMode.RISK_PCT_EQUITY,
            equity=10_000,
            entry_price=2000.0,
            sl_price=1990.0,
            risk_pct=0.01,
        ))
        # risk_cash = 10_000 * 0.01 = 100
        # sl_dist = 10
        # qty = 100 / 10 = 10.0
        assert result.qty == pytest.approx(10.0)
        assert result.risk_cash == pytest.approx(100.0)

    def test_risk_pct_equity_scales_with_equity(self) -> None:
        """Doubling equity should roughly double the position size."""
        small = compute_position_size(SizingRequest(
            mode=SizingMode.RISK_PCT_EQUITY,
            equity=500,
            entry_price=2000.0,
            sl_price=1990.0,
            risk_pct=0.01,
            min_qty=0.01,
        ))
        large = compute_position_size(SizingRequest(
            mode=SizingMode.RISK_PCT_EQUITY,
            equity=1000,
            entry_price=2000.0,
            sl_price=1990.0,
            risk_pct=0.01,
            min_qty=0.01,
        ))
        assert large.qty >= small.qty * 1.9  # roughly 2x

    def test_max_qty_clamp(self) -> None:
        result = compute_position_size(SizingRequest(
            mode=SizingMode.FIXED_QTY,
            equity=10_000,
            entry_price=2000.0,
            sl_price=1990.0,
            qty=999.0,
            max_qty=1.0,
        ))
        assert result.qty == pytest.approx(1.0)
        assert result.clamped is True

    def test_min_qty_clamp(self) -> None:
        result = compute_position_size(SizingRequest(
            mode=SizingMode.RISK_PCT_EQUITY,
            equity=10,
            entry_price=2000.0,
            sl_price=1990.0,
            risk_pct=0.001,
            min_qty=0.01,
        ))
        # risk = 0.01 → qty = 0.001 → clamped to min 0.01
        assert result.qty == pytest.approx(0.01)
        assert result.clamped is True

    def test_max_leverage_limit(self) -> None:
        result = compute_position_size(SizingRequest(
            mode=SizingMode.FIXED_QTY,
            equity=100,
            entry_price=2000.0,
            sl_price=1990.0,
            qty=10.0,
            max_qty=100.0,
            max_leverage=20.0,
        ))
        # max_notional = 100 * 20 = 2000
        # max_qty = 2000 / 2000 = 1.0
        assert result.qty == pytest.approx(1.0)
        assert result.clamped is True


# ═══════════════════════════════════════════════════════════════════════════
# 4.  SpreadGuard
# ═══════════════════════════════════════════════════════════════════════════


class TestSpreadGuard:
    def test_disabled_never_blocks(self) -> None:
        guard = SpreadGuard(SpreadGuardConfig(enabled=False, spread_cost_max=0.1))
        result = guard.check(spread=999.0)
        assert result.blocked is False

    def test_absolute_gate_blocks(self) -> None:
        guard = SpreadGuard(SpreadGuardConfig(enabled=True, spread_cost_max=0.5))
        result = guard.check(spread=0.6)
        assert result.blocked is True
        assert result.reason == "SPREAD_EXCEEDS_MAX"

    def test_absolute_gate_passes(self) -> None:
        guard = SpreadGuard(SpreadGuardConfig(enabled=True, spread_cost_max=0.5))
        result = guard.check(spread=0.4)
        assert result.blocked is False

    def test_percentile_gate_blocks(self) -> None:
        guard = SpreadGuard(SpreadGuardConfig(
            enabled=True,
            spread_cost_max=0,  # disable absolute
            percentile_limit=90.0,
            window=20,
        ))
        # Feed 20 observations: 19 at 0.2, 1 at 0.9
        for _ in range(19):
            guard.observe(0.2)
        guard.observe(0.9)
        # p90 should be around 0.2, so 0.5 should exceed it
        result = guard.check(spread=0.5)
        assert result.blocked is True
        assert result.reason == "SPREAD_EXCEEDS_PERCENTILE"

    def test_percentile_gate_passes(self) -> None:
        guard = SpreadGuard(SpreadGuardConfig(
            enabled=True,
            spread_cost_max=0,
            percentile_limit=90.0,
            window=20,
        ))
        for _ in range(20):
            guard.observe(0.5)
        result = guard.check(spread=0.5)
        assert result.blocked is False


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Reporting metrics – gross / net side-by-side
# ═══════════════════════════════════════════════════════════════════════════


class TestReportingMetrics:
    def test_metrics_include_gross_net(self) -> None:
        from bot.reporting.metrics import compute_metrics

        trades = [
            {"pnl": 5.0, "pnl_gross": 6.0, "pnl_net": 4.5, "swap_cost": 1.0},
            {"pnl": -2.0, "pnl_gross": -1.5, "pnl_net": -2.5, "swap_cost": 0.5},
        ]
        metrics = compute_metrics(trades, [])
        assert "total_pnl_gross" in metrics
        assert "total_pnl_net" in metrics
        assert "profit_factor_net" in metrics
        assert "trades_per_day" in metrics
        assert metrics["total_pnl_gross"] == pytest.approx(4.5)
        assert metrics["total_pnl_net"] == pytest.approx(2.0)

    def test_metrics_backward_compat_without_gross_net(self) -> None:
        """Old trades without pnl_gross/pnl_net should still work."""
        from bot.reporting.metrics import compute_metrics

        trades = [{"pnl": 3.0}, {"pnl": -1.0}]
        metrics = compute_metrics(trades, [])
        # Fallback: pnl_gross = pnl, pnl_net = pnl
        assert metrics["total_pnl_gross"] == pytest.approx(2.0)
        assert metrics["total_pnl_net"] == pytest.approx(2.0)
