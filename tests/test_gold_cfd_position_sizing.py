from __future__ import annotations

import pytest

from bot.execution.position_sizer import (
    GoldCfdSizingStatus,
    compute_gold_cfd_position_size,
)


def test_rounding_to_step_for_raw_qty_examples() -> None:
    # risk=0.019, sl_distance=1.0 -> raw_qty=0.019 -> rounded=0.01
    res_low = compute_gold_cfd_position_size(
        equity_pln=1000.0,
        risk_pct=0.000019,
        entry_price_pln_per_oz=100.0,
        stop_loss_price_pln_per_oz=99.0,
    )
    assert res_low.raw_qty_oz == pytest.approx(0.019)
    assert res_low.rounded_qty_oz == pytest.approx(0.01)

    # risk=0.021, sl_distance=1.0 -> raw_qty=0.021 -> rounded=0.02
    res_high = compute_gold_cfd_position_size(
        equity_pln=1000.0,
        risk_pct=0.000021,
        entry_price_pln_per_oz=100.0,
        stop_loss_price_pln_per_oz=99.0,
    )
    assert res_high.raw_qty_oz == pytest.approx(0.021)
    assert res_high.rounded_qty_oz == pytest.approx(0.02)


def test_blocked_invalid_sl() -> None:
    result = compute_gold_cfd_position_size(
        equity_pln=1000.0,
        risk_pct=0.005,
        entry_price_pln_per_oz=5181.29,
        stop_loss_price_pln_per_oz=5181.29,
    )
    assert result.status == GoldCfdSizingStatus.BLOCKED_INVALID_SL.value
    assert result.sl_pts == pytest.approx(0.0)


def test_blocked_min_qty() -> None:
    # risk=0.05, sl_distance=10.0 -> raw=0.005 -> floor to 0.00 -> min qty block
    result = compute_gold_cfd_position_size(
        equity_pln=1000.0,
        risk_pct=0.00005,
        entry_price_pln_per_oz=5200.0,
        stop_loss_price_pln_per_oz=5190.0,
    )
    assert result.status == GoldCfdSizingStatus.BLOCKED_MIN_QTY.value
    assert result.rounded_qty_oz < 0.01


def test_blocked_risk_too_high_when_multiplier_tightened() -> None:
    # With floor rounding and default multiplier this is typically unreachable.
    # We force a tighter cap to hit this branch deterministically.
    result = compute_gold_cfd_position_size(
        equity_pln=1000.0,
        risk_pct=0.005,  # risk=5
        entry_price_pln_per_oz=5181.29,
        stop_loss_price_pln_per_oz=5171.29,  # sl_distance=10
        max_risk_multiplier=0.5,  # cap=2.5
    )
    assert result.status == GoldCfdSizingStatus.BLOCKED_RISK_TOO_HIGH.value
    assert result.real_risk_pln > result.risk_pln * 0.5


def test_sanity_case_qty_half_oz() -> None:
    # risk=1000*0.005=5 PLN, SL=10 PLN per oz -> qty=0.5 oz
    result = compute_gold_cfd_position_size(
        equity_pln=1000.0,
        risk_pct=0.005,
        entry_price_pln_per_oz=5181.29,
        stop_loss_price_pln_per_oz=5171.29,
    )
    assert result.status == GoldCfdSizingStatus.OK.value
    assert result.rounded_qty_oz == pytest.approx(0.5)

