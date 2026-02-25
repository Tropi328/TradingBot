from __future__ import annotations

from bot.research.optimizer import evaluate_quality_filter


def _summary(
    *,
    trades: int = 150,
    orders_submitted: int = 10,
    trades_filled: int = 5,
    anomaly_flags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "trades": trades,
        "orders_submitted": orders_submitted,
        "trades_filled": trades_filled,
        "anomaly_flags": anomaly_flags or [],
    }


def test_quality_filter_strict_rejects_blocked_anomaly_flags() -> None:
    result = evaluate_quality_filter(
        is_summary=_summary(anomaly_flags=["PF_EXTREME"]),
        oos_summary=_summary(),
        mode="strict",
        apply_windows=["is", "oos"],
        blocked_anomaly_flags=["PF_EXTREME", "PAYOFF_EXTREME"],
    )
    assert result["quality_pass"] is False
    assert any(str(reason).startswith("IS_ANOMALY_FLAGS") for reason in result["quality_reasons"])


def test_quality_filter_strict_rejects_min_trades() -> None:
    result = evaluate_quality_filter(
        is_summary=_summary(trades=20),
        oos_summary=_summary(trades=30),
        mode="strict",
        apply_windows=["is", "oos"],
        min_is_trades=120,
        min_oos_trades=120,
    )
    assert result["quality_pass"] is False
    assert any(str(reason).startswith("IS_TRADES_BELOW_MIN") for reason in result["quality_reasons"])
    assert any(str(reason).startswith("OOS_TRADES_BELOW_MIN") for reason in result["quality_reasons"])


def test_quality_filter_strict_rejects_missing_orders_or_fills() -> None:
    result = evaluate_quality_filter(
        is_summary=_summary(orders_submitted=0, trades_filled=0),
        oos_summary=_summary(orders_submitted=0, trades_filled=0),
        mode="strict",
        apply_windows=["is", "oos"],
        require_orders_submitted=True,
        require_trades_filled=True,
    )
    assert result["quality_pass"] is False
    assert "IS_ORDERS_SUBMITTED_ZERO" in result["quality_reasons"]
    assert "IS_TRADES_FILLED_ZERO" in result["quality_reasons"]
    assert "OOS_ORDERS_SUBMITTED_ZERO" in result["quality_reasons"]
    assert "OOS_TRADES_FILLED_ZERO" in result["quality_reasons"]


def test_quality_filter_strict_passes_for_healthy_metrics() -> None:
    result = evaluate_quality_filter(
        is_summary=_summary(),
        oos_summary=_summary(),
        mode="strict",
        apply_windows=["is", "oos"],
        blocked_anomaly_flags=["PF_EXTREME", "PAYOFF_EXTREME"],
        min_is_trades=120,
        min_oos_trades=120,
        require_orders_submitted=True,
        require_trades_filled=True,
    )
    assert result["quality_pass"] is True
    assert result["quality_reasons"] == []
