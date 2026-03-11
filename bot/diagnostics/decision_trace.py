"""Centralised decision-trace emitter.

Writes one JSON line per event to a JSONL file (flushed immediately).
Two event types:
  - **decision**: emitted on every evaluation bar / signal candidate.
  - **fill**: emitted when a position is opened or closed.

The module is designed to be safe to call in any mode (BACKTEST, PAPER, LIVE).
If the path is ``None`` or the feature is disabled, every call is a no-op.

Log rotation: when the file exceeds *max_bytes* it is renamed to ``.1``
and older backups are shifted (``.1`` -> ``.2``, etc.) up to *max_backups*.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimal schema keys (used for validation in tests)
# ---------------------------------------------------------------------------
DECISION_REQUIRED_KEYS = frozenset(
    {
        "type",
        "ts",
        "symbol",
        "tf",
        "candidates",
        "signal",
        "features_ok",
        "score",
        "threshold",
        "reject_reason",
    }
)

FILL_REQUIRED_KEYS = frozenset(
    {
        "type",
        "ts",
        "symbol",
        "side",
        "pnl",
        "equity_after",
        "reason_close",
    }
)

# ---------------------------------------------------------------------------
# Defaults for rotation
# ---------------------------------------------------------------------------
_DEFAULT_MAX_BYTES = 50 * 1024 * 1024  # 50 MB
_DEFAULT_MAX_BACKUPS = 5


class DecisionTraceWriter:
    """Append-only JSONL writer with per-line flush and automatic rotation."""

    def __init__(
        self,
        path: str | Path | None,
        *,
        enabled: bool = True,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        max_backups: int = _DEFAULT_MAX_BACKUPS,
    ) -> None:
        self._path: Path | None = Path(path) if path else None
        self._enabled = enabled and self._path is not None
        self._handle: IO[str] | None = None
        self._max_bytes = max(1024, max_bytes)
        self._max_backups = max(0, max_backups)
        self._bytes_written: int = 0

    # -- lifecycle -----------------------------------------------------------

    def open(self) -> None:
        if not self._enabled or self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")
        self._bytes_written = self._path.stat().st_size if self._path.exists() else 0
        LOGGER.info(
            "DecisionTrace → %s (append, rotation at %d MB)",
            self._path,
            self._max_bytes // (1024 * 1024),
        )

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> DecisionTraceWriter:
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -- rotation ------------------------------------------------------------

    def _rotate(self) -> None:
        """Rotate log files: current -> .1, .1 -> .2, ... up to max_backups."""
        if self._path is None:
            return
        self.close()
        name = self._path.name  # e.g. "decision_trace.jsonl"
        parent = self._path.parent
        # Shift existing backups: drop oldest, shift rest up by 1
        for i in range(self._max_backups, 0, -1):
            src = parent / f"{name}.{i}"
            if i == self._max_backups:
                if src.exists():
                    src.unlink()
            else:
                dst = parent / f"{name}.{i + 1}"
                if src.exists():
                    src.rename(dst)
        # Move current file to .1
        backup_1 = parent / f"{name}.1"
        if self._path.exists():
            self._path.rename(backup_1)
        # Re-open fresh file
        self._handle = self._path.open("a", encoding="utf-8")
        self._bytes_written = 0
        LOGGER.info("DecisionTrace rotated → %s", backup_1)

    # -- public API ----------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._enabled and self._handle is not None

    def emit(self, event: dict[str, Any]) -> None:
        """Write one JSON line and flush immediately. Rotates if size exceeded."""
        if not self.active:
            return
        if self._handle is None:
            return
        try:
            line = json.dumps(event, ensure_ascii=False, default=str)
            data = line + "\n"
            self._handle.write(data)
            self._handle.flush()
            self._bytes_written += len(data.encode("utf-8"))
            if self._bytes_written >= self._max_bytes and self._max_backups > 0:
                self._rotate()
        except Exception:
            LOGGER.exception("DecisionTrace write error")

    # -- convenience builders ------------------------------------------------

    def decision(
        self,
        *,
        ts: datetime | str,
        symbol: str,
        tf: str = "5m",
        candidates: int = 0,
        signal: str | None = None,
        features_ok: bool = True,
        missing: list[str] | None = None,
        score: float | None = None,
        threshold: float | None = None,
        score_breakdown: list[dict[str, Any]] | None = None,
        reject_reason: str | None = None,
        spread_points: float | None = None,
        session_ok: bool = True,
        size_raw: float | None = None,
        size_final: float | None = None,
        min_lot: float | None = None,
        lot_step: float | None = None,
        margin_capped: bool = False,
        cooldown_active: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        ts_str = ts if isinstance(ts, str) else _iso(ts)
        event: dict[str, Any] = {
            "type": "decision",
            "ts": ts_str,
            "symbol": symbol,
            "tf": tf,
            "candidates": candidates,
            "signal": signal,
            "features_ok": features_ok,
            "missing": missing or [],
            "score": _r(score),
            "threshold": _r(threshold),
            "score_breakdown": score_breakdown or [],
            "reject_reason": reject_reason,
            "spread_points": _r(spread_points),
            "session_ok": session_ok,
            "size_raw": _r(size_raw),
            "size_final": _r(size_final),
            "min_lot": _r(min_lot),
            "lot_step": _r(lot_step),
            "margin_capped": margin_capped,
            "cooldown_active": cooldown_active,
        }
        if extra:
            event.update(extra)
        self.emit(event)

    def fill(
        self,
        *,
        ts: datetime | str,
        symbol: str,
        side: str,
        pnl: float = 0.0,
        equity_after: float = 0.0,
        reason_close: str = "",
        holding_min: float = 0.0,
        spread_cost: float = 0.0,
        swap_cost: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        ts_str = ts if isinstance(ts, str) else _iso(ts)
        event: dict[str, Any] = {
            "type": "fill",
            "ts": ts_str,
            "symbol": symbol,
            "side": side,
            "pnl": _r(pnl),
            "equity_after": _r(equity_after),
            "reason_close": reason_close,
            "holding_min": _r(holding_min),
            "spread_cost": _r(spread_cost),
            "swap_cost": _r(swap_cost),
        }
        if extra:
            event.update(extra)
        self.emit(event)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _r(val: float | None, digits: int = 4) -> float | None:
    if val is None:
        return None
    return round(float(val), digits)


# ---------------------------------------------------------------------------
# Module-level convenience (lazy singleton)
# ---------------------------------------------------------------------------

_GLOBAL_WRITER: DecisionTraceWriter | None = None


def get_global_writer() -> DecisionTraceWriter | None:
    return _GLOBAL_WRITER


def init_global_writer(path: str | Path | None, *, enabled: bool = True) -> DecisionTraceWriter:
    global _GLOBAL_WRITER
    if _GLOBAL_WRITER is not None:
        _GLOBAL_WRITER.close()
    _GLOBAL_WRITER = DecisionTraceWriter(path, enabled=enabled)
    _GLOBAL_WRITER.open()
    return _GLOBAL_WRITER


def close_global_writer() -> None:
    global _GLOBAL_WRITER
    if _GLOBAL_WRITER is not None:
        _GLOBAL_WRITER.close()
        _GLOBAL_WRITER = None
