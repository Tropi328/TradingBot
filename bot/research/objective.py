from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

OBJECTIVE_FAIL_VALUE = -1.0e18


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(sum(values_list) / len(values_list))


def _normalize_anomaly_flags(values: Any) -> list[str]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, Mapping)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        key = str(item).strip().upper()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _normalize_dd_cap_basis(value: str | None) -> str:
    basis = str(value or "both").strip().lower()
    if basis not in {"initial", "peak", "both"}:
        basis = "both"
    return basis


def build_cost_breakdown_net(report: Mapping[str, Any]) -> dict[str, float]:
    return {
        "spread_cost_sum": _safe_float(report.get("spread_cost_sum")),
        "slippage_cost_sum": _safe_float(report.get("slippage_cost_sum")),
        "commission_cost_sum": _safe_float(report.get("commission_cost_sum")),
        "swap_cost_sum": _safe_float(report.get("swap_cost_sum")),
        "fx_cost_sum": _safe_float(report.get("fx_cost_sum")),
    }


def build_blocked_by_reason(report: Mapping[str, Any]) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for key in ("blocked_by_reason", "rejected_by_reason", "blocked_by_gate_reasons"):
        raw = report.get(key)
        if not isinstance(raw, Mapping):
            continue
        for reason, count in raw.items():
            reason_key = str(reason).strip().upper()
            if not reason_key:
                continue
            merged[reason_key] += max(0, _safe_int(count))
    return dict(merged)


def _drawdown_abs(report: Mapping[str, Any]) -> float:
    return _safe_float(report.get("max_drawdown_net", report.get("max_drawdown", 0.0)), default=0.0)


def compute_max_drawdown_pct_peak(report: Mapping[str, Any], *, initial_equity: float) -> float:
    raw_peak = report.get("max_drawdown_pct_peak")
    if raw_peak is not None:
        return max(0.0, _safe_float(raw_peak, default=0.0))

    # Backward compatibility alias.
    raw_alias = report.get("max_drawdown_pct")
    if raw_alias is not None:
        return max(0.0, _safe_float(raw_alias, default=0.0))

    peak_equity = _safe_float(report.get("equity_peak"), default=0.0)
    drawdown_abs = _drawdown_abs(report)
    if peak_equity > 0:
        return max(0.0, (drawdown_abs / peak_equity) * 100.0)
    # Legacy fallback when only absolute drawdown is available.
    equity_base = _safe_float(report.get("initial_equity", initial_equity), default=0.0)
    if equity_base <= 0.0:
        equity_base = _safe_float(report.get("equity_start", initial_equity), default=0.0)
    if equity_base > 0:
        return max(0.0, (drawdown_abs / equity_base) * 100.0)
    return 0.0


def compute_max_drawdown_pct_initial(report: Mapping[str, Any], *, initial_equity: float) -> float:
    raw_initial = report.get("max_drawdown_pct_initial")
    if raw_initial is not None:
        return max(0.0, _safe_float(raw_initial, default=0.0))

    drawdown_abs = _drawdown_abs(report)
    equity_base = _safe_float(report.get("initial_equity", initial_equity), default=0.0)
    if equity_base <= 0.0:
        equity_base = _safe_float(report.get("equity_start", initial_equity), default=0.0)
    if equity_base <= 0.0:
        return 0.0
    return max(0.0, (drawdown_abs / equity_base) * 100.0)


def compute_max_drawdown_pct(report: Mapping[str, Any], *, initial_equity: float) -> float:
    # Compatibility: alias remains peak-based.
    return compute_max_drawdown_pct_peak(report, initial_equity=initial_equity)


def compute_dd_ref_pct(*, max_drawdown_pct_peak: float, max_drawdown_pct_initial: float) -> float:
    return max(max_drawdown_pct_peak, max_drawdown_pct_initial)


def evaluate_dd_constraints(
    *,
    max_drawdown_pct_peak: float,
    max_drawdown_pct_initial: float,
    dd_cap_pct: float,
    dd_cap_basis: str,
) -> tuple[bool, bool, bool]:
    basis = _normalize_dd_cap_basis(dd_cap_basis)
    if dd_cap_pct <= 0:
        return True, True, True

    cap = float(dd_cap_pct)
    pass_peak = max_drawdown_pct_peak <= cap
    pass_initial = max_drawdown_pct_initial <= cap

    if basis == "peak":
        overall = pass_peak
    elif basis == "initial":
        overall = pass_initial
    else:
        overall = pass_peak and pass_initial
    return overall, pass_peak, pass_initial


def compute_objective_value(
    *,
    objective_mode: str,
    total_pnl_net: float,
    oos_pass: bool,
    constraint_dd_cap_pass: bool,
    dd_ref_pct: float | None = None,
    oos_total_pnl_net: float | None = None,
    oos_dd_ref_pct: float | None = None,
    dd_floor_pct: float = 0.25,
) -> float:
    mode = str(objective_mode or "pnl_dd_cap").strip().lower()
    if mode == "pnl_dd_cap":
        return total_pnl_net if (oos_pass and constraint_dd_cap_pass) else OBJECTIVE_FAIL_VALUE
    if mode == "risk_adjusted_pnl_dd":
        if not (oos_pass and constraint_dd_cap_pass):
            return OBJECTIVE_FAIL_VALUE
        pnl_value = _safe_float(oos_total_pnl_net if oos_total_pnl_net is not None else total_pnl_net, default=0.0)
        dd_value = _safe_float(
            oos_dd_ref_pct if oos_dd_ref_pct is not None else (dd_ref_pct if dd_ref_pct is not None else 0.0),
            default=0.0,
        )
        return pnl_value / max(_safe_float(dd_floor_pct, default=0.25), dd_value, 0.25)
    return total_pnl_net


def augment_report(
    report: Mapping[str, Any],
    *,
    initial_equity: float,
    dd_cap_pct: float,
    dd_cap_basis: str = "both",
    min_trades_oos: int,
    objective_mode: str,
    oos_pass: bool | None = None,
) -> dict[str, Any]:
    out = dict(report)
    trades = _safe_int(out.get("trades", out.get("trades_count", 0)), default=0)
    total_pnl_net = _safe_float(out.get("total_pnl_net", out.get("total_pnl", 0.0)), default=0.0)
    profit_factor_net = _safe_float(out.get("profit_factor_net", out.get("profit_factor", 0.0)), default=0.0)
    payoff_ratio = _safe_float(out.get("payoff_ratio", 0.0), default=0.0)
    anomaly_flags = _normalize_anomaly_flags(out.get("anomaly_flags", []))
    orders_submitted = max(0, _safe_int(out.get("orders_submitted", 0), default=0))
    trades_filled = max(0, _safe_int(out.get("trades_filled", 0), default=0))
    max_drawdown_pct_peak = compute_max_drawdown_pct_peak(out, initial_equity=initial_equity)
    max_drawdown_pct_initial = compute_max_drawdown_pct_initial(out, initial_equity=initial_equity)
    dd_ref_pct = compute_dd_ref_pct(
        max_drawdown_pct_peak=max_drawdown_pct_peak,
        max_drawdown_pct_initial=max_drawdown_pct_initial,
    )

    computed_oos_pass = bool(oos_pass) if oos_pass is not None else (trades >= max(0, int(min_trades_oos)))
    constraint_dd_cap_pass, pass_peak, pass_initial = evaluate_dd_constraints(
        max_drawdown_pct_peak=max_drawdown_pct_peak,
        max_drawdown_pct_initial=max_drawdown_pct_initial,
        dd_cap_pct=dd_cap_pct,
        dd_cap_basis=dd_cap_basis,
    )
    objective_value = compute_objective_value(
        objective_mode=objective_mode,
        total_pnl_net=total_pnl_net,
        oos_pass=computed_oos_pass,
        constraint_dd_cap_pass=constraint_dd_cap_pass,
        dd_ref_pct=dd_ref_pct,
        oos_total_pnl_net=total_pnl_net,
        oos_dd_ref_pct=dd_ref_pct,
    )

    out["max_drawdown_pct_peak"] = max_drawdown_pct_peak
    out["max_drawdown_pct_initial"] = max_drawdown_pct_initial
    out["dd_ref_pct"] = dd_ref_pct
    out["max_drawdown_pct"] = max_drawdown_pct_peak
    out["cost_breakdown_net"] = build_cost_breakdown_net(out)
    out["blocked_by_reason"] = build_blocked_by_reason(out)
    out["anomaly_flags"] = anomaly_flags
    out["orders_submitted"] = orders_submitted
    out["trades_filled"] = trades_filled
    out["profit_factor_net"] = profit_factor_net
    out["payoff_ratio"] = payoff_ratio
    out["oos_pass"] = computed_oos_pass
    out["constraint_dd_cap_pass_peak"] = pass_peak
    out["constraint_dd_cap_pass_initial"] = pass_initial
    out["constraint_dd_cap_pass"] = constraint_dd_cap_pass
    out["objective_value"] = objective_value
    return out


def aggregate_reports(
    reports: Iterable[Mapping[str, Any]],
    *,
    initial_equity: float,
    dd_cap_pct: float,
    dd_cap_basis: str = "both",
    min_trades_oos: int,
    objective_mode: str,
) -> dict[str, Any]:
    normalized_basis = _normalize_dd_cap_basis(dd_cap_basis)
    augmented = [
        augment_report(
            report,
            initial_equity=initial_equity,
            dd_cap_pct=dd_cap_pct,
            dd_cap_basis=normalized_basis,
            min_trades_oos=min_trades_oos,
            objective_mode=objective_mode,
            oos_pass=None,
        )
        for report in reports
        if isinstance(report, Mapping)
    ]

    total_trades = sum(_safe_int(item.get("trades", item.get("trades_count", 0))) for item in augmented)
    wins = sum(_safe_int(item.get("wins", 0)) for item in augmented)
    losses = sum(_safe_int(item.get("losses", 0)) for item in augmented)
    total_pnl_net = sum(_safe_float(item.get("total_pnl_net", item.get("total_pnl", 0.0))) for item in augmented)
    total_pnl = sum(_safe_float(item.get("total_pnl", item.get("total_pnl_net", 0.0))) for item in augmented)
    max_drawdown_pct_peak = max(
        (compute_max_drawdown_pct_peak(item, initial_equity=initial_equity) for item in augmented),
        default=0.0,
    )
    max_drawdown_pct_initial = max(
        (compute_max_drawdown_pct_initial(item, initial_equity=initial_equity) for item in augmented),
        default=0.0,
    )
    dd_ref_pct = compute_dd_ref_pct(
        max_drawdown_pct_peak=max_drawdown_pct_peak,
        max_drawdown_pct_initial=max_drawdown_pct_initial,
    )
    max_drawdown = max((_safe_float(item.get("max_drawdown", 0.0)) for item in augmented), default=0.0)
    expectancy = _mean(_safe_float(item.get("expectancy", 0.0)) for item in augmented)
    expectancy_net = _mean(_safe_float(item.get("expectancy_net", item.get("expectancy", 0.0))) for item in augmented)
    avg_r = _mean(_safe_float(item.get("avg_r", 0.0)) for item in augmented)
    profit_factor_net = _mean(_safe_float(item.get("profit_factor_net", item.get("profit_factor", 0.0))) for item in augmented)
    payoff_ratio = _mean(_safe_float(item.get("payoff_ratio", 0.0)) for item in augmented)

    cost_counter: Counter[str] = Counter()
    blocked_counter: Counter[str] = Counter()
    anomaly_flags_set: set[str] = set()
    orders_submitted = 0
    trades_filled = 0
    for item in augmented:
        for cost_key, value in build_cost_breakdown_net(item).items():
            cost_counter[cost_key] += value
        blocked_counter.update(build_blocked_by_reason(item))
        anomaly_flags_set.update(_normalize_anomaly_flags(item.get("anomaly_flags", [])))
        orders_submitted += max(0, _safe_int(item.get("orders_submitted", 0), default=0))
        trades_filled += max(0, _safe_int(item.get("trades_filled", 0), default=0))

    oos_pass = total_trades >= max(0, int(min_trades_oos))
    constraint_dd_cap_pass, pass_peak, pass_initial = evaluate_dd_constraints(
        max_drawdown_pct_peak=max_drawdown_pct_peak,
        max_drawdown_pct_initial=max_drawdown_pct_initial,
        dd_cap_pct=dd_cap_pct,
        dd_cap_basis=normalized_basis,
    )
    objective_value = compute_objective_value(
        objective_mode=objective_mode,
        total_pnl_net=total_pnl_net,
        oos_pass=oos_pass,
        constraint_dd_cap_pass=constraint_dd_cap_pass,
        dd_ref_pct=dd_ref_pct,
        oos_total_pnl_net=total_pnl_net,
        oos_dd_ref_pct=dd_ref_pct,
    )

    return {
        "reports_count": len(augmented),
        "trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / total_trades) if total_trades > 0 else 0.0,
        "total_pnl": total_pnl,
        "total_pnl_net": total_pnl_net,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct_peak": max_drawdown_pct_peak,
        "max_drawdown_pct_initial": max_drawdown_pct_initial,
        "dd_ref_pct": dd_ref_pct,
        "max_drawdown_pct": max_drawdown_pct_peak,
        "expectancy": expectancy,
        "expectancy_net": expectancy_net,
        "avg_r": avg_r,
        "profit_factor_net": profit_factor_net,
        "payoff_ratio": payoff_ratio,
        "cost_breakdown_net": dict(cost_counter),
        "blocked_by_reason": dict(blocked_counter),
        "anomaly_flags": sorted(anomaly_flags_set),
        "orders_submitted": orders_submitted,
        "trades_filled": trades_filled,
        "oos_pass": oos_pass,
        "constraint_dd_cap_pass_peak": pass_peak,
        "constraint_dd_cap_pass_initial": pass_initial,
        "constraint_dd_cap_pass": constraint_dd_cap_pass,
        "objective_value": objective_value,
    }


def objective_rank_key(summary: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    quality_pass = bool(summary.get("quality_pass", True))
    objective_value = _safe_float(summary.get("objective_value", OBJECTIVE_FAIL_VALUE), default=OBJECTIVE_FAIL_VALUE)
    pnl_value = _safe_float(summary.get("oos_total_pnl_net", summary.get("total_pnl_net", 0.0)), default=0.0)
    dd_value = _safe_float(
        summary.get(
            "oos_dd_ref_pct",
            summary.get("dd_ref_pct", summary.get("max_drawdown_pct_peak", summary.get("max_drawdown_pct", 0.0))),
        ),
        default=0.0,
    )
    expectancy = _safe_float(summary.get("oos_expectancy_net", summary.get("expectancy_net", summary.get("expectancy", 0.0))), default=0.0)
    # quality_pass wins over raw objective value.
    quality_rank = 0.0 if quality_pass else 1.0
    return (quality_rank, -objective_value, -pnl_value, dd_value, -expectancy)
