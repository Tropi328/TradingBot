from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from bot.data.candles import Candle


REASON_CODE_MAP: dict[str, str] = {
    "BIAS_FAIL": "H1_BIAS_NEUTRAL",
    "PD_FAIL": "H1_PD_FAIL",
    "ATR_M15_MISSING": "M15_ATR_WARMUP",
    "SWEEP_FAIL": "M15_NO_SWEEP",
    "MSS_FAIL": "M5_NO_MSS",
    "ATR_M5_MISSING": "M5_ATR_WARMUP",
    "DISPLACEMENT_FAIL": "M5_NO_DISPLACEMENT",
    "FVG_FAIL": "M5_NO_FVG",
    "SPREAD_FAIL": "M5_SPREAD_FAIL",
    "NEWS_BLOCK": "NEWS_BLOCKED",
    "INSUFFICIENT_DATA": "PIPELINE_INSUFFICIENT_DATA",
    "M15_WAIT_H1_SNAPSHOT": "M15_WAIT_H1",
    "M15_H1_GATE_FAIL": "M15_H1_GATE_FAIL",
    "M15_H1_PD_FAIL": "M15_WAIT_H1_PD",
}


@dataclass(slots=True)
class H1TraceState:
    updated: bool = False
    bias_state: str = "NEUTRAL"
    safe_mode: bool = False
    ema200_ready: bool = False
    ema200_value: float | None = None
    bos_state: str = "NONE"
    bos_age: int | None = None
    bars: int = 0
    required_bars: int = 0
    pd_state: str = "UNKNOWN"
    close: float | None = None
    eq: float | None = None
    dealing_low: float | None = None
    dealing_high: float | None = None


@dataclass(slots=True)
class M15TraceState:
    updated: bool = False
    setup_state: str = "WAIT"
    sweep_dir: str = "NONE"
    reject_ok: bool = False
    sweep_level: float | None = None
    invalidation_level: float | None = None
    setup_age_minutes: float | None = None


@dataclass(slots=True)
class M5TraceState:
    updated: bool = False
    entry_state: str = "WAIT"
    mss_ok: bool = False
    displacement_ok: bool = False
    fvg_ok: bool = False
    fvg_range: tuple[float, float] | None = None
    fvg_mid: float | None = None
    limit_price: float | None = None


@dataclass(slots=True)
class DecisionTrace:
    asset: str
    created_at: datetime
    strategy_name: str = "UNKNOWN"
    score_total: float | None = None
    score_layers: dict[str, float] = field(default_factory=dict)
    score_breakdown: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    gates: dict[str, bool] = field(default_factory=dict)
    gate_blocked: str | None = None
    reasons_blocking: list[str] = field(default_factory=list)
    would_enter_if: list[str] = field(default_factory=list)
    snapshot: dict[str, Any] = field(default_factory=dict)
    h1_last_closed_ts: datetime | None = None
    h1_new_close: bool = False
    m15_last_closed_ts: datetime | None = None
    m15_new_close: bool = False
    m5_last_closed_ts: datetime | None = None
    m5_new_close: bool = False
    h1: H1TraceState = field(default_factory=H1TraceState)
    m15: M15TraceState = field(default_factory=M15TraceState)
    m5: M5TraceState = field(default_factory=M5TraceState)
    final_decision: str = "NO_SIGNAL"
    reasons: list[str] = field(default_factory=list)


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def map_reason_codes(reason_codes: list[str]) -> list[str]:
    mapped = [REASON_CODE_MAP.get(code, code) for code in reason_codes]
    return _dedupe(mapped)


def last_closed_candle_ts(candles: list[Candle]) -> datetime | None:
    if len(candles) < 2:
        return None
    return candles[-2].timestamp


def is_new_closed_candle(
    candles: list[Candle],
    last_processed_closed_ts: datetime | None,
) -> tuple[bool, datetime | None]:
    closed_ts = last_closed_candle_ts(candles)
    if closed_ts is None:
        return False, None
    if last_processed_closed_ts is None:
        return True, closed_ts
    return closed_ts > last_processed_closed_ts, closed_ts


def closed_candles(candles: list[Candle]) -> list[Candle]:
    if len(candles) < 2:
        return []
    return candles[:-1]


def _to_tz(dt: datetime | None, timezone_name: str) -> datetime | None:
    if dt is None:
        return None
    try:
        return dt.astimezone(ZoneInfo(timezone_name))
    except ZoneInfoNotFoundError:
        return dt


def _fmt_ts(dt: datetime | None, timezone_name: str) -> str:
    localized = _to_tz(dt, timezone_name)
    if localized is None:
        return "-"
    return localized.isoformat(timespec="seconds")


def format_trace_text(trace: DecisionTrace, timezone_name: str) -> str:
    h1_update = "updated" if trace.h1_new_close else "SKIP(no new close)"
    m15_update = "updated" if trace.m15_new_close else "SKIP(no new close)"
    m5_update = "updated" if trace.m5_new_close else "SKIP(no new close)"
    reasons = ",".join(trace.reasons) if trace.reasons else "NONE"
    blocking = ",".join(trace.reasons_blocking) if trace.reasons_blocking else "NONE"
    would_enter_if = ",".join(trace.would_enter_if) if trace.would_enter_if else "NONE"
    score_total = f"{trace.score_total:.2f}" if trace.score_total is not None else "-"
    edge = trace.score_layers.get("edge", 0.0)
    trigger = trace.score_layers.get("trigger", 0.0)
    execution = trace.score_layers.get("execution", 0.0)
    strategy_name = trace.strategy_name or "UNKNOWN"
    fvg = "none"
    if trace.m5.fvg_range is not None:
        fvg = f"{trace.m5.fvg_range[0]:.3f}-{trace.m5.fvg_range[1]:.3f}"
    pd_diag = ""
    if "H1_PD_FAIL" in trace.reasons and trace.h1.close is not None:
        pd_diag = (
            f" pd_diag(close={trace.h1.close:.3f},eq={trace.h1.eq if trace.h1.eq is not None else '-'},"
            f"low={trace.h1.dealing_low if trace.h1.dealing_low is not None else '-'},"
            f"high={trace.h1.dealing_high if trace.h1.dealing_high is not None else '-'})"
        )
    penalties = ",".join(f"{key}:{value:.1f}" for key, value in trace.penalties.items()) if trace.penalties else "none"
    gates = ",".join(f"{key}:{1 if value else 0}" for key, value in trace.gates.items()) if trace.gates else "none"
    blocked_gate = trace.gate_blocked or "-"
    return (
        f"{trace.asset} state | strategy={strategy_name} score={score_total} | "
        f"layers(edge={edge:.1f},trigger={trigger:.1f},execution={execution:.1f}) | "
        f"H1:{h1_update} close={_fmt_ts(trace.h1_last_closed_ts, timezone_name)} "
        f"bias={trace.h1.bias_state} safe={int(trace.h1.safe_mode)} ema_ready={int(trace.h1.ema200_ready)} "
        f"bars={trace.h1.bars}/{trace.h1.required_bars} bos={trace.h1.bos_state} "
        f"bos_age={trace.h1.bos_age if trace.h1.bos_age is not None else '-'} | "
        f"M15:{m15_update} close={_fmt_ts(trace.m15_last_closed_ts, timezone_name)} "
        f"setup={trace.m15.setup_state} sweep={trace.m15.sweep_dir} "
        f"reject={int(trace.m15.reject_ok)} age_min={trace.m15.setup_age_minutes if trace.m15.setup_age_minutes is not None else '-'} | "
        f"M5:{m5_update} close={_fmt_ts(trace.m5_last_closed_ts, timezone_name)} "
        f"entry={trace.m5.entry_state} mss={int(trace.m5.mss_ok)} "
        f"disp={int(trace.m5.displacement_ok)} fvg={int(trace.m5.fvg_ok)} "
        f"fvg_range={fvg} | gates={gates} blocked_gate={blocked_gate} penalties={penalties} "
        f"| decision={trace.final_decision} | reasons={reasons} | blocking={blocking} | would_enter_if={would_enter_if}"
        f"{pd_diag}"
    )


def _json_normalize(value: Any, timezone_name: str) -> Any:
    if isinstance(value, datetime):
        return _fmt_ts(value, timezone_name)
    if isinstance(value, dict):
        return {k: _json_normalize(v, timezone_name) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_json_normalize(v, timezone_name) for v in value]
    if isinstance(value, list):
        return [_json_normalize(v, timezone_name) for v in value]
    return value


def trace_to_json(trace: DecisionTrace, timezone_name: str) -> str:
    payload = _json_normalize(asdict(trace), timezone_name)
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
