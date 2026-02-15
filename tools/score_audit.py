#!/usr/bin/env python3
"""
Score Audit Tool — diagnostic report for backtest decision funnel.

Reads a backtest report.json + trades.csv and produces:
  1. Full decision funnel stage-by-stage count
  2. Score histogram / percentile distribution
  3. Tier breakdown (A+/A/B/OBSERVE) with size and PnL
  4. TOP-N gate/block reasons
  5. Shadow observe statistics (if available)
  6. Year-by-year throughput table
  7. Baseline vs ScoreV3 comparison (if both reports exist)

Usage:
    python tools/score_audit.py <report_dir> [--compare <other_report_dir>]
    python tools/score_audit.py reports/backtest/XAUUSD_5m_..._W0_off_... --top 10
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trades(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _pct(num: float, denom: float) -> str:
    if denom <= 0:
        return "n/a"
    return f"{num / denom * 100:.1f}%"


def _quantiles(values: list[float], qs: list[float]) -> dict[str, float]:
    if not values:
        return {f"p{int(q*100)}": 0.0 for q in qs}
    s = sorted(values)
    result: dict[str, float] = {}
    for q in qs:
        idx = q * (len(s) - 1)
        lo = int(math.floor(idx))
        hi = min(lo + 1, len(s) - 1)
        frac = idx - lo
        result[f"p{int(q*100)}"] = s[lo] * (1 - frac) + s[hi] * frac
    return result


# ── Funnel Report ─────────────────────────────────────────────────────────

def build_funnel_report(report: dict[str, Any]) -> dict[str, Any]:
    """Extract decision funnel metrics from a backtest report.json."""
    src = report.get("extra", {}).get("source_report", report)

    signal_candidates = int(src.get("signal_candidates", 0))
    decision_counts = src.get("decision_counts", {})
    score_bins = src.get("count_score_bins", src.get("score_bins", {}))
    orders_submitted = int(src.get("orders_submitted", 0))
    trades_filled = int(src.get("trades_filled", 0))
    trades_closed = int(src.get("trades", 0))
    rejected = src.get("rejected_by_reason", {})
    top_blockers = src.get("top_blockers", {})
    gate_blocks = src.get("gate_block_counts", {})
    exec_fail = src.get("execution_fail_breakdown", {})
    margin_capped = int(src.get("margin_capped_count", 0))
    exit_dist = src.get("exit_reason_distribution", {})

    total_scored = sum(int(v) for v in score_bins.values()) if score_bins else 0

    return {
        "total_scored_evaluations": total_scored,
        "score_bins": dict(score_bins),
        "signal_candidates": signal_candidates,
        "decision_counts": dict(decision_counts),
        "orders_submitted": orders_submitted,
        "trades_filled": trades_filled,
        "trades_closed": trades_closed,
        "fill_rate_pct": round(trades_filled / orders_submitted * 100, 2) if orders_submitted else 0,
        "candidate_to_fill_pct": round(trades_filled / signal_candidates * 100, 2) if signal_candidates else 0,
        "exit_distribution": dict(exit_dist),
        "margin_capped_count": margin_capped,
        "rejected_by_reason": dict(rejected),
        "gate_block_counts": dict(gate_blocks),
        "execution_fail_breakdown": dict(exec_fail),
    }


# ── Score Distribution ────────────────────────────────────────────────────

def build_score_distribution(report: dict[str, Any]) -> dict[str, Any]:
    src = report.get("extra", {}).get("source_report", report)
    avg_score = src.get("avg_score")
    score_bins = src.get("count_score_bins", src.get("score_bins", {}))
    return {
        "avg_score": avg_score,
        "bins": dict(score_bins),
    }


# ── Top Blockers ──────────────────────────────────────────────────────────

def build_top_blockers(report: dict[str, Any], top_n: int = 10) -> list[dict[str, Any]]:
    src = report.get("extra", {}).get("source_report", report)
    blockers = src.get("top_blockers", {})
    sorted_blockers = sorted(blockers.items(), key=lambda x: -x[1])[:top_n]
    total = sum(blockers.values()) or 1
    return [
        {"reason": reason, "count": count, "pct": round(count / total * 100, 2)}
        for reason, count in sorted_blockers
    ]


# ── Year-by-Year Throughput ──────────────────────────────────────────────

def build_yearly_throughput(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_year: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0,
        "tp": 0, "stop": 0, "be": 0, "margin_capped": 0,
        "r_values": [],
    })
    for t in trades:
        ts = str(t.get("entry_ts", ""))
        year = ts[:4] if len(ts) >= 4 else "unknown"
        d = by_year[year]
        d["trades"] += 1
        pnl = _safe_float(t.get("pnl"))
        r = _safe_float(t.get("r"))
        d["pnl"] += pnl
        d["r_values"].append(r)
        if pnl > 0:
            d["wins"] += 1
        else:
            d["losses"] += 1
        reason = str(t.get("reason_close", "")).upper()
        if reason == "TP":
            d["tp"] += 1
        elif reason == "STOP":
            d["stop"] += 1
        elif reason == "BE":
            d["be"] += 1
        if str(t.get("margin_capped", "")).lower() in ("true", "1"):
            d["margin_capped"] += 1

    rows: list[dict[str, Any]] = []
    for year in sorted(by_year.keys()):
        d = by_year[year]
        total = d["trades"]
        wr = d["wins"] / total * 100 if total else 0
        avg_r = sum(d["r_values"]) / len(d["r_values"]) if d["r_values"] else 0
        rows.append({
            "year": year,
            "trades": total,
            "wins": d["wins"],
            "losses": d["losses"],
            "win_rate_pct": round(wr, 1),
            "pnl": round(d["pnl"], 2),
            "avg_r": round(avg_r, 4),
            "tp": d["tp"],
            "stop": d["stop"],
            "be": d["be"],
            "margin_capped": d["margin_capped"],
        })
    return rows


# ── Comparison Table ──────────────────────────────────────────────────────

def build_comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    def _m(report: dict[str, Any]) -> dict[str, Any]:
        return report.get("metrics", report.get("extra", {}).get("source_report", {}))

    bm = _m(baseline)
    cm = _m(candidate)

    keys = [
        "trades_count", "wins", "losses", "win_rate_pct",
        "total_pnl", "profit_factor", "max_drawdown", "max_drawdown_pct",
    ]
    comparison: dict[str, Any] = {}
    for k in keys:
        bv = _safe_float(bm.get(k, bm.get(k.replace("_count", ""), 0)))
        cv = _safe_float(cm.get(k, cm.get(k.replace("_count", ""), 0)))
        delta = cv - bv
        pct_change = (delta / bv * 100) if bv != 0 else float("inf") if delta != 0 else 0
        comparison[k] = {
            "baseline": round(bv, 4),
            "candidate": round(cv, 4),
            "delta": round(delta, 4),
            "pct_change": round(pct_change, 2),
        }
    return comparison


# ── Gate Reason Categories ────────────────────────────────────────────────

_CATEGORY_MAP = {
    "score_observe": [
        "SCORE_BELOW_MIN", "SCALP_SCORE_TOO_LOW",
    ],
    "structural": [
        "SCALP_NO_MSS", "SCALP_NO_DISPLACEMENT", "SCALP_NO_FVG",
        "SCALP_BIAS_MISMATCH", "H1_NO_BOS", "H1_BIAS_NEUTRAL", "H1_PD_FAIL",
        "NEWS_BLOCKED", "SCALP_CANDIDATE_EXPIRED", "SCALP_ATR_WARMUP",
    ],
    "reaction_wait": [
        "GATE_REACTION_WAIT_REACTION", "GATE_REACTION_WAIT_MITIGATION",
        "REACTION_TIMEOUT_SOFT_REACTION", "REACTION_TIMEOUT_SOFT_MITIGATION",
        "SOFT_REASON_REACTION_WAIT_SOFT_MITIGATION", "SOFT_REASON_REACTION_WAIT_SOFT_REACTION",
        "SOFT_REASON_WAIT_TIMEOUT_SOFT_MODE",
    ],
    "execution": [
        "EXEC_FAIL_MISSING_FEATURES", "EXEC_FAIL_SPREAD_TOO_HIGH",
        "EXEC_FAIL_CONFIRMATIONS_LOW", "EXEC_FAIL_MARKET_CLOSED",
        "EXEC_FAIL_INVALID_ATR", "EXEC_FAIL_NO_PRICE",
    ],
    "risk_budget": [
        "KILL_SWITCH_DAILY_LOSS", "DAILY_PROFIT_LOCKED",
        "PER_TRADE_RISK_TOO_HIGH", "OPEN_RISK_CAP_EXCEEDED",
        "RISK_MAX_TRADES_DAY", "RISK_DAILY_STOP",
    ],
    "margin_size": [
        "SIZE_TOO_SMALL", "SIZE_MARGIN_LIMIT", "SIZE_INVALID",
        "INSUFFICIENT_EQUITY", "INSUFFICIENT_MARGIN",
        "EDGE_TOO_SMALL",
    ],
    "spread_cost": [
        "SPREAD_EXCEEDS_MAX", "SPREAD_EXCEEDS_PERCENTILE",
        "SOFT_REASON_ASSUMED_OHLC_SPREAD",
    ],
    "correlation": [
        "CORRELATION_EXPOSURE", "SOFT_REASON_CORRELATION_EXPOSURE",
    ],
}


def categorize_blockers(blockers: dict[str, int]) -> dict[str, dict[str, int]]:
    categorized: dict[str, dict[str, int]] = defaultdict(dict)
    for reason, count in blockers.items():
        placed = False
        for cat, patterns in _CATEGORY_MAP.items():
            if reason in patterns or any(reason.startswith(p) for p in patterns):
                categorized[cat][reason] = count
                placed = True
                break
        if not placed:
            categorized["other"][reason] = count
    return dict(categorized)


# ── Main Report ───────────────────────────────────────────────────────────

def generate_audit(
    report_dir: Path,
    *,
    top_n: int = 10,
    compare_dir: Path | None = None,
) -> dict[str, Any]:
    report_json_path = report_dir / "report.json"
    summary_json_path = report_dir / "summary.json"
    trades_csv_path = report_dir / "trades.csv"

    report = _load_json(report_json_path) if report_json_path.exists() else {}
    summary = _load_json(summary_json_path) if summary_json_path.exists() else {}
    trades = _load_trades(trades_csv_path) if trades_csv_path.exists() else []

    src = report.get("extra", {}).get("source_report", {})

    audit: dict[str, Any] = {
        "meta": report.get("meta", summary.get("meta", {})),
        "funnel": build_funnel_report(report),
        "score_distribution": build_score_distribution(report),
        "top_blockers": build_top_blockers(report, top_n),
        "blocker_categories": categorize_blockers(src.get("top_blockers", {})),
        "yearly_throughput": build_yearly_throughput(trades),
    }

    if compare_dir:
        other_report = _load_json(compare_dir / "report.json") if (compare_dir / "report.json").exists() else {}
        audit["comparison"] = build_comparison(report, other_report)

    return audit


# ── CLI ───────────────────────────────────────────────────────────────────

def _print_section(title: str, data: Any, indent: int = 0) -> None:
    prefix = "  " * indent
    if isinstance(data, dict):
        print(f"{prefix}{title}:")
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                _print_section(str(k), v, indent + 1)
            else:
                print(f"{prefix}  {k}: {v}")
    elif isinstance(data, list):
        print(f"{prefix}{title}:")
        for item in data:
            if isinstance(item, dict):
                line = " | ".join(f"{k}={v}" for k, v in item.items())
                print(f"{prefix}  {line}")
            else:
                print(f"{prefix}  {item}")
    else:
        print(f"{prefix}{title}: {data}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Audit — backtest decision funnel diagnostic")
    parser.add_argument("report_dir", type=Path, help="Path to backtest report directory")
    parser.add_argument("--compare", type=Path, default=None, help="Compare with another report directory")
    parser.add_argument("--top", type=int, default=10, help="Top N blockers to show")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted text")
    args = parser.parse_args()

    audit = generate_audit(args.report_dir, top_n=args.top, compare_dir=args.compare)

    if args.json:
        print(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
        return

    print("=" * 72)
    print("  SCORE AUDIT REPORT")
    print("=" * 72)
    print()

    meta = audit.get("meta", {})
    if meta:
        print(f"  Symbol: {meta.get('symbol', '?')}  |  Variant: {meta.get('variant', '?')}")
        print(f"  Period: {meta.get('start', '?')} → {meta.get('end', '?')}")
        print()

    funnel = audit["funnel"]
    print("── DECISION FUNNEL ──")
    print(f"  Total scored evaluations : {funnel['total_scored_evaluations']:>10,}")
    for bin_name, count in funnel["score_bins"].items():
        pct = _pct(count, funnel["total_scored_evaluations"])
        print(f"    {bin_name:<20} : {count:>10,}  ({pct})")
    print(f"  Signal candidates        : {funnel['signal_candidates']:>10,}")
    for action, count in funnel["decision_counts"].items():
        print(f"    {action:<20} : {count:>10,}")
    print(f"  Orders submitted         : {funnel['orders_submitted']:>10,}")
    print(f"  Trades filled            : {funnel['trades_filled']:>10,}  (fill rate: {funnel['fill_rate_pct']}%)")
    print(f"  Trades closed            : {funnel['trades_closed']:>10,}")
    print(f"  Margin capped            : {funnel['margin_capped_count']:>10,}")
    print(f"  Candidate→Fill rate      : {funnel['candidate_to_fill_pct']}%")
    print()

    print("── EXIT DISTRIBUTION ──")
    for reason, count in funnel["exit_distribution"].items():
        print(f"    {reason:<10} : {count:>6}")
    print()

    print("── SCORE DISTRIBUTION ──")
    sd = audit["score_distribution"]
    print(f"  Avg score: {sd['avg_score']}")
    print()

    print("── TOP BLOCKERS ──")
    for item in audit["top_blockers"]:
        print(f"  {item['reason']:<45} : {item['count']:>10,}  ({item['pct']}%)")
    print()

    print("── BLOCKER CATEGORIES ──")
    cats = audit["blocker_categories"]
    for cat in sorted(cats.keys()):
        cat_total = sum(cats[cat].values())
        print(f"  {cat} (total={cat_total:,}):")
        for reason, count in sorted(cats[cat].items(), key=lambda x: -x[1]):
            print(f"    {reason:<45} : {count:>10,}")
    print()

    print("── YEARLY THROUGHPUT ──")
    header = f"  {'Year':<6} {'Trades':>7} {'Wins':>5} {'Loss':>5} {'WR%':>6} {'PnL':>10} {'AvgR':>8} {'TP':>4} {'SL':>4} {'BE':>4} {'MCap':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in audit["yearly_throughput"]:
        print(
            f"  {row['year']:<6} {row['trades']:>7} {row['wins']:>5} {row['losses']:>5} "
            f"{row['win_rate_pct']:>5.1f}% {row['pnl']:>10.2f} {row['avg_r']:>8.4f} "
            f"{row['tp']:>4} {row['stop']:>4} {row['be']:>4} {row['margin_capped']:>5}"
        )
    print()

    if "comparison" in audit:
        print("── BASELINE vs CANDIDATE ──")
        for k, vals in audit["comparison"].items():
            print(f"  {k:<25} : baseline={vals['baseline']:>10.4f}  candidate={vals['candidate']:>10.4f}  Δ={vals['delta']:>+10.4f}  ({vals['pct_change']:>+.1f}%)")
        print()

    # Rejected by reason
    if funnel["rejected_by_reason"]:
        print("── REJECTED BY REASON ──")
        for reason, count in sorted(funnel["rejected_by_reason"].items(), key=lambda x: -x[1]):
            print(f"  {reason:<45} : {count:>10,}")
        print()

    # Gate blocks
    if funnel["gate_block_counts"]:
        print("── GATE BLOCKS ──")
        for reason, count in sorted(funnel["gate_block_counts"].items(), key=lambda x: -x[1]):
            print(f"  {reason:<45} : {count:>10,}")
        print()


if __name__ == "__main__":
    main()
