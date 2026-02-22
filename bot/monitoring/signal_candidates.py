"""Signal Candidate Logger — telemetry for PAPER/LIVE.

Records every decision-window candidate (including rejected ones) with full
context so operators can answer *"why no entries?"* after any session.

Storage: SQLite table ``signal_candidates`` in the main state DB.
Aggregation: ``SignalCandidateAggregator`` produces periodic summaries.
Export: ``export_diagnostics()`` dumps JSON/CSV at session end.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SignalCandidate:
    """One decision-window candidate — always created, even if rejected."""

    timestamp: datetime
    symbol: str
    timeframe: str
    strategy_name: str
    setup_name: str
    side: str | None
    score: float | None
    # Feature snapshot
    spread: float | None
    atr: float | None
    trend_regime: str | None          # e.g. "TRENDING", "RANGE", "UNKNOWN"
    volatility_regime: str | None     # e.g. "HIGH", "NORMAL", "LOW"
    session_name: str | None          # e.g. "LONDON", "NY", "ASIAN", "OFF"
    bias_direction: str | None        # e.g. "LONG", "SHORT", "NEUTRAL"
    # SL/TP/edge
    sl_distance: float | None
    tp_distance: float | None
    expected_rr: float | None
    expected_move: float | None
    estimated_roundtrip_cost: float | None
    # Decision
    action: str                       # TRADE / SMALL / OBSERVE / SKIP
    accepted: bool
    rejection_reasons: list[str] = field(default_factory=list)
    # Rich context
    score_breakdown: dict[str, float] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS signal_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    setup_name TEXT NOT NULL,
    side TEXT,
    score REAL,
    spread REAL,
    atr REAL,
    trend_regime TEXT,
    volatility_regime TEXT,
    session_name TEXT,
    bias_direction TEXT,
    sl_distance REAL,
    tp_distance REAL,
    expected_rr REAL,
    expected_move REAL,
    estimated_roundtrip_cost REAL,
    action TEXT NOT NULL,
    accepted INTEGER NOT NULL DEFAULT 0,
    rejection_reasons TEXT NOT NULL DEFAULT '[]',
    score_breakdown TEXT NOT NULL DEFAULT '{}',
    features TEXT NOT NULL DEFAULT '{}',
    gates TEXT NOT NULL DEFAULT '{}',
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_sc_timestamp ON signal_candidates(timestamp);
CREATE INDEX IF NOT EXISTS idx_sc_symbol ON signal_candidates(symbol);
CREATE INDEX IF NOT EXISTS idx_sc_action ON signal_candidates(action);
"""

_INSERT_SQL = """
INSERT INTO signal_candidates (
    timestamp, symbol, timeframe, strategy_name, setup_name, side,
    score, spread, atr, trend_regime, volatility_regime, session_name,
    bias_direction, sl_distance, tp_distance, expected_rr, expected_move,
    estimated_roundtrip_cost, action, accepted, rejection_reasons,
    score_breakdown, features, gates, metadata
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def init_signal_candidates_table(conn: sqlite3.Connection) -> None:
    """Create table if missing — safe to call on every startup."""
    conn.executescript(_CREATE_TABLE_SQL)
    conn.commit()


# ---------------------------------------------------------------------------
# Logger (writer)
# ---------------------------------------------------------------------------
class SignalCandidateLogger:
    """Thread-safe writer of SignalCandidate rows to SQLite."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._lock = threading.Lock()

    def log(self, candidate: SignalCandidate) -> None:
        with self._lock:
            self.conn.execute(
                _INSERT_SQL,
                (
                    _to_iso(candidate.timestamp),
                    candidate.symbol,
                    candidate.timeframe,
                    candidate.strategy_name,
                    candidate.setup_name,
                    candidate.side,
                    candidate.score,
                    candidate.spread,
                    candidate.atr,
                    candidate.trend_regime,
                    candidate.volatility_regime,
                    candidate.session_name,
                    candidate.bias_direction,
                    candidate.sl_distance,
                    candidate.tp_distance,
                    candidate.expected_rr,
                    candidate.expected_move,
                    candidate.estimated_roundtrip_cost,
                    candidate.action,
                    int(candidate.accepted),
                    json.dumps(candidate.rejection_reasons),
                    json.dumps(candidate.score_breakdown),
                    json.dumps(candidate.features),
                    json.dumps(candidate.gates),
                    json.dumps(candidate.metadata),
                ),
            )
            self.conn.commit()

    def log_many(self, candidates: list[SignalCandidate]) -> None:
        if not candidates:
            return
        rows = [
            (
                _to_iso(c.timestamp), c.symbol, c.timeframe, c.strategy_name,
                c.setup_name, c.side, c.score, c.spread, c.atr, c.trend_regime,
                c.volatility_regime, c.session_name, c.bias_direction,
                c.sl_distance, c.tp_distance, c.expected_rr, c.expected_move,
                c.estimated_roundtrip_cost, c.action, int(c.accepted),
                json.dumps(c.rejection_reasons), json.dumps(c.score_breakdown),
                json.dumps(c.features), json.dumps(c.gates), json.dumps(c.metadata),
            )
            for c in candidates
        ]
        with self._lock:
            self.conn.executemany(_INSERT_SQL, rows)
            self.conn.commit()


# ---------------------------------------------------------------------------
# Aggregator — periodic summary
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CandidateAggregation:
    """Snapshot of aggregated stats over a window."""

    window_start: datetime
    window_end: datetime
    candidates_count: int
    accepted_trades_count: int
    rejection_reasons_top10: list[tuple[str, int]]
    score_p50: float | None
    score_p75: float | None
    score_p90: float | None
    breakdown_by_gate: dict[str, int]
    action_distribution: dict[str, int]
    symbols: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "candidates_count": self.candidates_count,
            "accepted_trades_count": self.accepted_trades_count,
            "rejection_reasons_top10": [
                {"reason": r, "count": c} for r, c in self.rejection_reasons_top10
            ],
            "score_p50": self.score_p50,
            "score_p75": self.score_p75,
            "score_p90": self.score_p90,
            "breakdown_by_gate": self.breakdown_by_gate,
            "action_distribution": self.action_distribution,
            "symbols": self.symbols,
        }


class SignalCandidateAggregator:
    """Produces real-time aggregation summaries from the DB."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        interval_seconds: int = 600,
    ) -> None:
        self.conn = conn
        self.interval_seconds = interval_seconds
        self._last_emit: float = 0.0

    def should_emit(self) -> bool:
        return (time.monotonic() - self._last_emit) >= self.interval_seconds

    def aggregate_window(
        self,
        window_start: datetime,
        window_end: datetime,
    ) -> CandidateAggregation:
        start_iso = _to_iso(window_start)
        end_iso = _to_iso(window_end)
        rows = self.conn.execute(
            "SELECT * FROM signal_candidates WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
            (start_iso, end_iso),
        ).fetchall()

        candidates_count = len(rows)
        accepted_count = 0
        scores: list[float] = []
        reasons_counter: Counter[str] = Counter()
        gate_counter: Counter[str] = Counter()
        action_counter: Counter[str] = Counter()
        symbol_counter: Counter[str] = Counter()

        for row in rows:
            action_counter[row["action"]] += 1
            symbol_counter[row["symbol"]] += 1
            if row["accepted"]:
                accepted_count += 1
            if row["score"] is not None:
                scores.append(float(row["score"]))
            try:
                reasons = json.loads(row["rejection_reasons"])
            except (json.JSONDecodeError, TypeError):
                reasons = []
            for reason in reasons:
                reasons_counter[reason] += 1
            try:
                gates = json.loads(row["gates"])
            except (json.JSONDecodeError, TypeError):
                gates = {}
            for gate_name, passed in gates.items():
                if not passed:
                    gate_counter[gate_name] += 1

        scores.sort()

        def _percentile(data: list[float], pct: float) -> float | None:
            if not data:
                return None
            k = (len(data) - 1) * pct / 100.0
            f = int(k)
            c = f + 1
            if c >= len(data):
                return data[f]
            return data[f] + (k - f) * (data[c] - data[f])

        return CandidateAggregation(
            window_start=window_start,
            window_end=window_end,
            candidates_count=candidates_count,
            accepted_trades_count=accepted_count,
            rejection_reasons_top10=reasons_counter.most_common(10),
            score_p50=_percentile(scores, 50),
            score_p75=_percentile(scores, 75),
            score_p90=_percentile(scores, 90),
            breakdown_by_gate=dict(gate_counter),
            action_distribution=dict(action_counter),
            symbols=dict(symbol_counter),
        )

    def maybe_log_summary(self, window_start: datetime, now: datetime) -> CandidateAggregation | None:
        if not self.should_emit():
            return None
        agg = self.aggregate_window(window_start, now)
        self._last_emit = time.monotonic()
        rejection_str = ", ".join(f"{r}:{c}" for r, c in agg.rejection_reasons_top10[:5])
        LOGGER.info(
            "SignalCandidate summary | candidates=%d accepted=%d "
            "score_p50=%.1f score_p75=%.1f score_p90=%.1f "
            "top_rejections=[%s] actions=%s",
            agg.candidates_count,
            agg.accepted_trades_count,
            agg.score_p50 or 0.0,
            agg.score_p75 or 0.0,
            agg.score_p90 or 0.0,
            rejection_str,
            json.dumps(agg.action_distribution),
        )
        return agg


# ---------------------------------------------------------------------------
# Diagnostics export
# ---------------------------------------------------------------------------
def export_diagnostics(
    conn: sqlite3.Connection,
    output_path: Path,
    *,
    session_start: datetime | None = None,
    fmt: str = "json",
) -> Path:
    """Dump signal_candidates + aggregation to a file.

    ``fmt`` can be ``"json"`` or ``"csv"``.
    """
    if session_start is None:
        row = conn.execute(
            "SELECT MIN(timestamp) AS min_ts FROM signal_candidates"
        ).fetchone()
        ts_raw = row["min_ts"] if row else None
        if ts_raw:
            session_start = datetime.fromisoformat(ts_raw)
        else:
            session_start = datetime.now(timezone.utc)
    session_end = datetime.now(timezone.utc)

    agg = SignalCandidateAggregator(conn).aggregate_window(session_start, session_end)

    rows = conn.execute(
        "SELECT * FROM signal_candidates WHERE timestamp >= ? ORDER BY timestamp",
        (_to_iso(session_start),),
    ).fetchall()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        import csv

        csv_path = output_path.with_suffix(".csv")
        if rows:
            fieldnames = rows[0].keys()
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(row))
        # Also dump summary
        summary_path = output_path.with_name(output_path.stem + "_summary.json")
        summary_path.write_text(
            json.dumps(agg.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        LOGGER.info("Diagnostics exported: %s + %s", csv_path, summary_path)
        return csv_path

    # Default: JSON
    json_path = output_path.with_suffix(".json")
    payload = {
        "summary": agg.to_dict(),
        "candidates": [dict(r) for r in rows],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    LOGGER.info("Diagnostics exported: %s", json_path)
    return json_path
