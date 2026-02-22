from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from bot.research.objective import (
    OBJECTIVE_FAIL_VALUE,
    compute_dd_ref_pct,
    compute_objective_value,
    evaluate_dd_constraints,
    objective_rank_key,
)


DEFAULT_DEEP_BUDGET = "deep"


def normalize_runtime_budget(value: str | None) -> str:
    budget = str(value or DEFAULT_DEEP_BUDGET).strip().lower()
    if budget not in {"quick", "medium", "deep"}:
        return DEFAULT_DEEP_BUDGET
    return budget


def _parse_datetime_utc(value: str, *, end_value: bool = False) -> datetime:
    raw = str(value).strip()
    normalized = raw.replace(" ", "T")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    if end_value and len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        dt += timedelta(days=1)
    return dt


def _date_floor_utc(dt: datetime) -> datetime:
    return datetime.combine(dt.date(), time.min, tzinfo=timezone.utc)


def _cli_end_from_exclusive(end_exclusive: datetime) -> str:
    # CLI end for date-only is inclusive and transformed to exclusive inside main.py.
    return (end_exclusive - timedelta(days=1)).date().isoformat()


@dataclass(frozen=True)
class TimeSplit:
    start_utc: str
    end_utc_exclusive: str
    split_ratio_is: float
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    is_start_utc: str
    is_end_utc_exclusive: str
    oos_start_utc: str
    oos_end_utc_exclusive: str
    days_total: int
    days_is: int
    days_oos: int
    fallback_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_utc": self.start_utc,
            "end_utc_exclusive": self.end_utc_exclusive,
            "split_ratio_is": self.split_ratio_is,
            "is": {
                "start": self.is_start,
                "end": self.is_end,
                "start_utc": self.is_start_utc,
                "end_utc_exclusive": self.is_end_utc_exclusive,
                "days": self.days_is,
            },
            "oos": {
                "start": self.oos_start,
                "end": self.oos_end,
                "start_utc": self.oos_start_utc,
                "end_utc_exclusive": self.oos_end_utc_exclusive,
                "days": self.days_oos,
            },
            "days_total": self.days_total,
            "fallback_applied": self.fallback_applied,
        }


def build_time_split(
    *,
    backtest_start: str,
    backtest_end: str,
    split_ratio_is: float = 0.70,
    min_days_is: int = 30,
    min_days_oos: int = 30,
) -> TimeSplit:
    start_dt = _date_floor_utc(_parse_datetime_utc(backtest_start, end_value=False))
    end_dt_exclusive = _date_floor_utc(_parse_datetime_utc(backtest_end, end_value=True))
    if end_dt_exclusive <= start_dt:
        raise ValueError("Invalid time range: backtest_end must be after backtest_start")

    total_days = int((end_dt_exclusive - start_dt).days)
    if total_days < 2:
        raise ValueError("Optimizer requires at least 2 days to split IS/OOS")

    ratio = float(split_ratio_is)
    ratio = max(0.05, min(0.95, ratio))
    target_is_days = max(1, min(total_days - 1, int(round(total_days * ratio))))

    min_is = max(1, int(min_days_is))
    min_oos = max(1, int(min_days_oos))
    fallback_applied = False

    if (min_is + min_oos) <= total_days:
        is_days = target_is_days
        if is_days < min_is:
            is_days = min_is
            fallback_applied = True
        if (total_days - is_days) < min_oos:
            is_days = total_days - min_oos
            fallback_applied = True
    else:
        # Nearest valid split while preserving at least one day on each side.
        is_days = max(1, min(total_days - 1, target_is_days))
        fallback_applied = True

    oos_days = total_days - is_days
    if oos_days < 1:
        is_days = total_days - 1
        oos_days = 1
        fallback_applied = True

    is_start_dt = start_dt
    is_end_exclusive = is_start_dt + timedelta(days=is_days)
    oos_start_dt = is_end_exclusive
    oos_end_exclusive = end_dt_exclusive

    return TimeSplit(
        start_utc=start_dt.isoformat(),
        end_utc_exclusive=end_dt_exclusive.isoformat(),
        split_ratio_is=ratio,
        is_start=is_start_dt.date().isoformat(),
        is_end=_cli_end_from_exclusive(is_end_exclusive),
        oos_start=oos_start_dt.date().isoformat(),
        oos_end=_cli_end_from_exclusive(oos_end_exclusive),
        is_start_utc=is_start_dt.isoformat(),
        is_end_utc_exclusive=is_end_exclusive.isoformat(),
        oos_start_utc=oos_start_dt.isoformat(),
        oos_end_utc_exclusive=oos_end_exclusive.isoformat(),
        days_total=total_days,
        days_is=is_days,
        days_oos=oos_days,
        fallback_applied=fallback_applied,
    )


def _candidate_id(stage: str, mode: str, params: Mapping[str, Any], risk_name: str | None = None) -> str:
    parts = [stage.upper(), mode.lower()]
    if risk_name:
        parts.append(str(risk_name).strip().upper())
    if params:
        payload = json.dumps(dict(sorted(params.items())), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        parts.append(payload)
    key = "|".join(parts)
    token = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{stage.upper()}_{mode.lower()}_{token}"


def _budget_gate_space(gate_cfg: Mapping[str, Any], runtime_budget: str) -> dict[str, Any]:
    budget = normalize_runtime_budget(runtime_budget)
    trend_thr = list(gate_cfg.get("trend_thr") or [])
    tvn_thr = list(gate_cfg.get("trend_vol_news_thr") or [])
    tvn_pre_post = list(gate_cfg.get("trend_vol_news_pre_post") or [])
    tvn_vol_max = list(gate_cfg.get("trend_vol_news_vol_max") or [])
    tvn_spread = list(gate_cfg.get("trend_vol_news_max_spread_mult") or [])

    if budget == "quick":
        return {
            "trend_thr": trend_thr[:3],
            "tvn_thr": tvn_thr[:1],
            "tvn_pre_post": tvn_pre_post[:2],
            "tvn_vol_max": tvn_vol_max[:2],
            "tvn_spread": tvn_spread[:1],
        }
    if budget == "medium":
        return {
            "trend_thr": trend_thr[:6],
            "tvn_thr": tvn_thr[:2],
            "tvn_pre_post": tvn_pre_post[:3],
            "tvn_vol_max": tvn_vol_max[:2],
            "tvn_spread": tvn_spread[:2],
        }
    return {
        "trend_thr": trend_thr,
        "tvn_thr": tvn_thr,
        "tvn_pre_post": tvn_pre_post,
        "tvn_vol_max": tvn_vol_max,
        "tvn_spread": tvn_spread,
    }


def build_stage_a_gate_candidates(
    *,
    search_space_gate: Mapping[str, Any],
    runtime_budget: str = DEFAULT_DEEP_BUDGET,
) -> list[dict[str, Any]]:
    scoped = _budget_gate_space(search_space_gate, runtime_budget)
    candidates: list[dict[str, Any]] = []

    off_params: dict[str, Any] = {}
    candidates.append(
        {
            "stage": "A",
            "candidate_id": _candidate_id("A", "off", off_params),
            "gate_mode": "off",
            "gate_params": off_params,
        }
    )

    for thr in scoped["trend_thr"]:
        params = {"thr": float(thr)}
        candidates.append(
            {
                "stage": "A",
                "candidate_id": _candidate_id("A", "trend", params),
                "gate_mode": "trend",
                "gate_params": params,
            }
        )

    for thr in scoped["tvn_thr"]:
        for pre_post in scoped["tvn_pre_post"]:
            pre = int(pre_post[0])
            post = int(pre_post[1])
            for vol_max in scoped["tvn_vol_max"]:
                for max_spread_mult in scoped["tvn_spread"]:
                    params = {
                        "thr": float(thr),
                        "pre_minutes": pre,
                        "post_minutes": post,
                        "vol_max": float(vol_max),
                        "max_spread_mult": float(max_spread_mult),
                    }
                    candidates.append(
                        {
                            "stage": "A",
                            "candidate_id": _candidate_id("A", "trend_vol_news", params),
                            "gate_mode": "trend_vol_news",
                            "gate_params": params,
                        }
                    )
    return candidates


def _budget_risk_profiles(risk_profiles: Sequence[Mapping[str, Any]], runtime_budget: str) -> list[dict[str, Any]]:
    budget = normalize_runtime_budget(runtime_budget)
    normalized = [dict(item) for item in risk_profiles if isinstance(item, Mapping)]
    if budget == "quick":
        return normalized[:6]
    if budget == "medium":
        return normalized[:8]
    return normalized


def build_stage_b_candidates(
    *,
    top_gate_candidates: Sequence[Mapping[str, Any]],
    risk_profiles: Sequence[Mapping[str, Any]],
    runtime_budget: str = DEFAULT_DEEP_BUDGET,
) -> list[dict[str, Any]]:
    scoped_risk = _budget_risk_profiles(risk_profiles, runtime_budget)
    stage_b: list[dict[str, Any]] = []
    for gate in top_gate_candidates:
        gate_mode = str(gate.get("gate_mode", "off")).strip().lower()
        gate_params = dict(gate.get("gate_params", {}))
        gate_id = str(gate.get("candidate_id", _candidate_id("A", gate_mode, gate_params)))
        for profile in scoped_risk:
            risk_name = str(profile.get("name", "RISK")).strip().upper() or "RISK"
            candidate_id = _candidate_id("B", gate_mode, gate_params, risk_name=risk_name)
            stage_b.append(
                {
                    "stage": "B",
                    "candidate_id": candidate_id,
                    "gate_candidate_id": gate_id,
                    "gate_mode": gate_mode,
                    "gate_params": gate_params,
                    "risk_profile": dict(profile),
                }
            )
    return stage_b


def build_stage_b_summary(
    *,
    is_summary: Mapping[str, Any],
    oos_summary: Mapping[str, Any],
    dd_cap_pct: float,
    dd_cap_basis: str,
    min_trades_oos: int,
    objective_mode: str,
) -> dict[str, Any]:
    is_total_pnl_net = float(is_summary.get("total_pnl_net", 0.0))
    is_dd_peak = float(is_summary.get("max_drawdown_pct_peak", is_summary.get("max_drawdown_pct", 0.0)))
    is_dd_initial = float(is_summary.get("max_drawdown_pct_initial", 0.0))
    is_dd_ref = compute_dd_ref_pct(max_drawdown_pct_peak=is_dd_peak, max_drawdown_pct_initial=is_dd_initial)

    oos_total_pnl_net = float(oos_summary.get("total_pnl_net", 0.0))
    oos_expectancy_net = float(oos_summary.get("expectancy_net", oos_summary.get("expectancy", 0.0)))
    oos_trades = int(oos_summary.get("trades", 0))
    oos_dd_peak = float(oos_summary.get("max_drawdown_pct_peak", oos_summary.get("max_drawdown_pct", 0.0)))
    oos_dd_initial = float(oos_summary.get("max_drawdown_pct_initial", 0.0))
    oos_dd_ref = compute_dd_ref_pct(max_drawdown_pct_peak=oos_dd_peak, max_drawdown_pct_initial=oos_dd_initial)
    oos_pass = bool(oos_summary.get("oos_pass", oos_trades >= max(0, int(min_trades_oos))))

    constraint_dd_cap_pass, pass_peak, pass_initial = evaluate_dd_constraints(
        max_drawdown_pct_peak=oos_dd_peak,
        max_drawdown_pct_initial=oos_dd_initial,
        dd_cap_pct=float(dd_cap_pct),
        dd_cap_basis=dd_cap_basis,
    )
    objective_value = compute_objective_value(
        objective_mode=objective_mode,
        total_pnl_net=oos_total_pnl_net,
        oos_pass=oos_pass,
        constraint_dd_cap_pass=constraint_dd_cap_pass,
        dd_ref_pct=oos_dd_ref,
        oos_total_pnl_net=oos_total_pnl_net,
        oos_dd_ref_pct=oos_dd_ref,
    )

    return {
        "is_trades": int(is_summary.get("trades", 0)),
        "is_total_pnl_net": is_total_pnl_net,
        "is_expectancy_net": float(is_summary.get("expectancy_net", is_summary.get("expectancy", 0.0))),
        "is_dd_ref_pct": is_dd_ref,
        "is_max_drawdown_pct_peak": is_dd_peak,
        "is_max_drawdown_pct_initial": is_dd_initial,
        "oos_trades": oos_trades,
        "oos_total_pnl_net": oos_total_pnl_net,
        "oos_expectancy_net": oos_expectancy_net,
        "oos_dd_ref_pct": oos_dd_ref,
        "oos_max_drawdown_pct_peak": oos_dd_peak,
        "oos_max_drawdown_pct_initial": oos_dd_initial,
        "oos_pass": oos_pass,
        "constraint_dd_cap_pass_peak": pass_peak,
        "constraint_dd_cap_pass_initial": pass_initial,
        "constraint_dd_cap_pass": constraint_dd_cap_pass,
        "objective_value": float(objective_value),
    }


def optimizer_rank_key(summary: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return objective_rank_key(summary)


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {"stage_a": {}, "stage_b": {}, "metadata": {}}
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"stage_a": {}, "stage_b": {}, "metadata": {}}
    if not isinstance(payload, dict):
        return {"stage_a": {}, "stage_b": {}, "metadata": {}}
    payload.setdefault("stage_a", {})
    payload.setdefault("stage_b", {})
    payload.setdefault("metadata", {})
    return payload


def save_checkpoint(checkpoint_path: Path, payload: Mapping[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def upsert_checkpoint_record(
    checkpoint: dict[str, Any],
    *,
    stage: str,
    candidate_id: str,
    record: Mapping[str, Any],
) -> None:
    key = "stage_a" if stage.upper() == "A" else "stage_b"
    stage_map = checkpoint.setdefault(key, {})
    if not isinstance(stage_map, dict):
        stage_map = {}
        checkpoint[key] = stage_map
    stage_map[str(candidate_id)] = dict(record)


def get_checkpoint_record(
    checkpoint: Mapping[str, Any],
    *,
    stage: str,
    candidate_id: str,
) -> dict[str, Any] | None:
    key = "stage_a" if stage.upper() == "A" else "stage_b"
    stage_map = checkpoint.get(key)
    if not isinstance(stage_map, Mapping):
        return None
    value = stage_map.get(str(candidate_id))
    if not isinstance(value, Mapping):
        return None
    return dict(value)


def build_search_space_payload(
    *,
    runtime_budget: str,
    gate_space: Mapping[str, Any],
    risk_profiles: Sequence[Mapping[str, Any]],
    stage_a_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    budget = normalize_runtime_budget(runtime_budget)
    return {
        "runtime_budget": budget,
        "gate": dict(gate_space),
        "risk_profiles": [dict(item) for item in risk_profiles],
        "stage_a_candidates_count": len(list(stage_a_candidates)),
    }


def failed_stage_summary() -> dict[str, Any]:
    return {
        "oos_pass": False,
        "constraint_dd_cap_pass": False,
        "constraint_dd_cap_pass_peak": False,
        "constraint_dd_cap_pass_initial": False,
        "objective_value": OBJECTIVE_FAIL_VALUE,
    }
