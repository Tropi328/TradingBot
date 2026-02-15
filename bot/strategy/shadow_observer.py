"""
Shadow Observer — simulate outcomes for candidates that were NOT traded.

For every signal candidate (including OBSERVE tier), we record the features
and simulate what WOULD have happened if the order had been placed and filled.
This creates a training dataset for ScoreV3 and lets us measure how many
OBSERVE signals would have been profitable.

The shadow simulation uses the same _calc_exit logic as the real engine:
- Entry at the candidate's limit price (FVG mid)
- TP at the candidate's target
- SL at the candidate's stop
- Same TP1/BE/trailing rules
- Same spread/slippage

Output: JSONL file with one record per candidate.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from bot.data.candles import Candle


@dataclass(slots=True)
class ShadowCandidate:
    """Features + shadow outcome for a single signal candidate."""

    # ── Identification ──
    timestamp: str
    symbol: str
    side: str
    action: str           # TRADE / SMALL / OBSERVE
    tier: str             # A_plus / A / B / OBSERVE / NONE
    score_v2: float       # V2 composite score
    score_v3: float | None = None

    # ── Features: HTF alignment ──
    h1_bias_direction: str = "NEUTRAL"
    h1_bos_state: str = "NONE"
    h1_pd_fail: bool = False
    h1_pd_eq: float | None = None
    h1_close: float | None = None
    m15_trend_aligned: bool = False

    # ── Features: FVG quality ──
    fvg_size: float = 0.0
    fvg_size_atr_ratio: float = 0.0
    fvg_distance_to_price: float = 0.0
    fvg_age_bars: int = 0
    fvg_mitigation_depth: float = 0.0

    # ── Features: Trigger quality ──
    trigger_confirmations: int = 0
    mss_ok: bool = False
    displacement_ok: bool = False
    displacement_ratio: float = 0.0
    sweep_magnitude: float = 0.0
    sweep_atr_ratio: float = 0.0

    # ── Features: Volatility regime ──
    atr_m5: float = 0.0
    atr_percentile: float = 0.5
    spread: float = 0.0
    spread_ratio: float = 0.0
    spread_atr_ratio: float = 0.0

    # ── Features: Session / Time ──
    hour_utc: int = 0
    day_of_week: int = 0
    session: str = "OTHER"        # LONDON / NY / OVERLAP / ASIA / OTHER
    news_blocked: bool = False
    near_rollover: bool = False

    # ── Features: Entry quality ──
    entry_price: float = 0.0
    stop_price: float = 0.0
    tp_price: float = 0.0
    risk_distance: float = 0.0
    rr_ratio: float = 0.0
    entry_distance_from_close: float = 0.0
    entry_distance_atr_ratio: float = 0.0

    # ── Features: Score components ──
    edge_score: float = 0.0
    trigger_score: float = 0.0
    execution_score: float = 0.0
    penalty_total: float = 0.0

    # ── Shadow outcome ──
    shadow_filled: bool = False
    shadow_exit_reason: str = ""    # TP / STOP / BE / EXPIRE / NOT_FILLED
    shadow_exit_price: float = 0.0
    shadow_pnl: float = 0.0
    shadow_r: float = 0.0
    shadow_hold_bars: int = 0
    shadow_max_favorable: float = 0.0
    shadow_max_adverse: float = 0.0

    # ── Gate reasons (why blocked) ──
    gate_reasons: list[str] = field(default_factory=list)
    would_enter_if: list[str] = field(default_factory=list)

    # ── Raw metadata for debugging ──
    raw_score_breakdown: dict[str, float] = field(default_factory=dict)
    raw_penalties: dict[str, float] = field(default_factory=dict)


def classify_session(hour_utc: int) -> str:
    """Classify trading session by UTC hour."""
    if 7 <= hour_utc < 12:
        return "LONDON"
    elif 12 <= hour_utc < 16:
        return "OVERLAP"
    elif 16 <= hour_utc < 21:
        return "NY"
    elif 0 <= hour_utc < 7:
        return "ASIA"
    return "OTHER"


def compute_atr_percentile(
    current_atr: float,
    atr_history: list[float | None],
    lookback: int = 500,
) -> float:
    """Compute percentile rank of current ATR value in recent history."""
    valid: list[float] = []
    for v in atr_history[-lookback:]:
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > 0:
            valid.append(fv)
    if not valid or current_atr <= 0:
        return 0.5
    below = sum(1 for v in valid if v <= current_atr)
    return below / len(valid)


def simulate_shadow_outcome(
    *,
    side: str,
    entry_price: float,
    stop_price: float,
    tp_price: float,
    candles: list[Candle],
    start_index: int,
    max_bars: int = 120,
    tp1_trigger_r: float = 0.7,
    tp1_fraction: float = 0.35,
    be_offset_r: float = 0.05,
    assumed_spread: float = 0.2,
) -> dict[str, Any]:
    """Simulate what would happen if a candidate were traded.

    Returns dict with: filled, exit_reason, exit_price, pnl, r, hold_bars,
    max_favorable, max_adverse.
    """
    result: dict[str, Any] = {
        "filled": False,
        "exit_reason": "NOT_FILLED",
        "exit_price": 0.0,
        "pnl": 0.0,
        "r": 0.0,
        "hold_bars": 0,
        "max_favorable": 0.0,
        "max_adverse": 0.0,
    }

    risk_distance = abs(entry_price - stop_price)
    if risk_distance <= 0:
        return result

    # Phase 1: Check if limit order gets filled (entry touched)
    fill_index: int | None = None
    ttl_bars = min(max_bars, 40)  # order TTL for fill check
    for i in range(start_index + 1, min(start_index + 1 + ttl_bars, len(candles))):
        c = candles[i]
        if entry_price >= c.low and entry_price <= c.high:
            fill_index = i
            break

    if fill_index is None:
        return result

    result["filled"] = True

    # Phase 2: Simulate position management (TP1/BE/trailing) + exit
    actual_entry = entry_price + (assumed_spread * 0.5) if side == "LONG" else entry_price - (assumed_spread * 0.5)
    current_stop = stop_price
    current_tp = tp_price
    tp1_taken = False
    be_moved = False
    remaining_size = 1.0
    realized_partial = 0.0
    max_fav = 0.0
    max_adv = 0.0

    tp1_level = actual_entry + (risk_distance * tp1_trigger_r) if side == "LONG" else actual_entry - (risk_distance * tp1_trigger_r)

    for i in range(fill_index + 1, min(fill_index + 1 + max_bars, len(candles))):
        c = candles[i]
        bars_held = i - fill_index

        # Track max favorable / adverse excursion
        if side == "LONG":
            fav = c.high - actual_entry
            adv = actual_entry - c.low
        else:
            fav = actual_entry - c.low
            adv = c.high - actual_entry
        max_fav = max(max_fav, fav)
        max_adv = max(max_adv, adv)

        # TP1 check
        if not tp1_taken:
            tp1_hit = (c.high >= tp1_level) if side == "LONG" else (c.low <= tp1_level)
            if tp1_hit:
                close_size = remaining_size * tp1_fraction
                if side == "LONG":
                    partial_pnl = (tp1_level - actual_entry) * close_size
                else:
                    partial_pnl = (actual_entry - tp1_level) * close_size
                realized_partial += partial_pnl
                remaining_size -= close_size
                tp1_taken = True
                # Move stop to BE
                be_price = actual_entry + (risk_distance * be_offset_r) if side == "LONG" else actual_entry - (risk_distance * be_offset_r)
                current_stop = be_price
                be_moved = True

        # Exit check
        if side == "LONG":
            stop_hit = c.low <= current_stop
            tp_hit = c.high >= current_tp
        else:
            stop_hit = c.high >= current_stop
            tp_hit = c.low <= current_tp

        if stop_hit and tp_hit:
            # Conservative: STOP wins
            stop_hit = True
            tp_hit = False

        if stop_hit:
            exit_price = current_stop
            if side == "LONG":
                final_pnl = (exit_price - actual_entry) * remaining_size
            else:
                final_pnl = (actual_entry - exit_price) * remaining_size
            total_pnl = realized_partial + final_pnl
            reason = "BE" if be_moved and abs(current_stop - actual_entry) < risk_distance * 0.15 else "STOP"
            result.update({
                "exit_reason": reason,
                "exit_price": exit_price,
                "pnl": total_pnl,
                "r": total_pnl / risk_distance if risk_distance > 0 else 0,
                "hold_bars": bars_held,
                "max_favorable": max_fav,
                "max_adverse": max_adv,
            })
            return result

        if tp_hit:
            exit_price = current_tp
            if side == "LONG":
                final_pnl = (exit_price - actual_entry) * remaining_size
            else:
                final_pnl = (actual_entry - exit_price) * remaining_size
            total_pnl = realized_partial + final_pnl
            result.update({
                "exit_reason": "TP",
                "exit_price": exit_price,
                "pnl": total_pnl,
                "r": total_pnl / risk_distance if risk_distance > 0 else 0,
                "hold_bars": bars_held,
                "max_favorable": max_fav,
                "max_adverse": max_adv,
            })
            return result

    # Expired (max_bars reached without exit)
    last_bar = candles[min(fill_index + max_bars, len(candles) - 1)]
    if side == "LONG":
        exit_price = last_bar.close - (assumed_spread * 0.5)
        final_pnl = (exit_price - actual_entry) * remaining_size
    else:
        exit_price = last_bar.close + (assumed_spread * 0.5)
        final_pnl = (actual_entry - exit_price) * remaining_size
    total_pnl = realized_partial + final_pnl
    result.update({
        "exit_reason": "EXPIRE",
        "exit_price": exit_price,
        "pnl": total_pnl,
        "r": total_pnl / risk_distance if risk_distance > 0 else 0,
        "hold_bars": max_bars,
        "max_favorable": max_fav,
        "max_adverse": max_adv,
    })
    return result


class ShadowObserver:
    """Collects shadow candidates and writes them to JSONL."""

    def __init__(self, output_path: Path | None = None) -> None:
        self._records: list[ShadowCandidate] = []
        self._output_path = output_path
        self._file = None

    def record(self, candidate: ShadowCandidate) -> None:
        self._records.append(candidate)
        if self._output_path and self._file is None:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._output_path.open("a", encoding="utf-8")
        if self._file:
            self._file.write(json.dumps(asdict(candidate), default=str) + "\n")

    def flush(self) -> None:
        if self._file:
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    @property
    def records(self) -> list[ShadowCandidate]:
        return self._records

    def summary(self) -> dict[str, Any]:
        """Produce a summary of shadow observations."""
        total = len(self._records)
        if total == 0:
            return {"total": 0}

        by_action: dict[str, list[ShadowCandidate]] = {}
        for r in self._records:
            by_action.setdefault(r.action, []).append(r)

        result: dict[str, Any] = {"total": total, "by_action": {}}
        for action, recs in by_action.items():
            filled = [r for r in recs if r.shadow_filled]
            profitable = [r for r in filled if r.shadow_r > 0]
            avg_r = sum(r.shadow_r for r in filled) / len(filled) if filled else 0
            result["by_action"][action] = {
                "count": len(recs),
                "filled": len(filled),
                "fill_rate": round(len(filled) / len(recs) * 100, 1) if recs else 0,
                "profitable": len(profitable),
                "win_rate": round(len(profitable) / len(filled) * 100, 1) if filled else 0,
                "avg_r": round(avg_r, 4),
                "avg_score_v2": round(sum(r.score_v2 for r in recs) / len(recs), 2),
            }

        # Key metric: how many OBSERVE signals would have been profitable?
        observe = by_action.get("OBSERVE", [])
        observe_filled = [r for r in observe if r.shadow_filled]
        observe_profitable = [r for r in observe_filled if r.shadow_r > 0.2]
        result["observe_missed_opportunities"] = {
            "total_observe": len(observe),
            "would_fill": len(observe_filled),
            "would_profit_r_gt_0.2": len(observe_profitable),
            "potential_pnl": round(sum(r.shadow_pnl for r in observe_profitable), 4),
            "avg_r_if_profitable": round(
                sum(r.shadow_r for r in observe_profitable) / len(observe_profitable), 4
            ) if observe_profitable else 0,
        }

        return result

    def save_summary(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.summary(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
