"""Shared type-conversion helpers for strategy plugins."""

from __future__ import annotations

from typing import Any

# ── Strategy-wide defaults ─────────────────────────────────────────
DEFAULT_MAX_SPREAD_RATIO: float = 0.15
"""Maximum spread/ATR ratio allowed by quality gates (used in
decision_core, score_v3 and strategy plugins as the fallback when
no per-route override is configured)."""


def as_float(value: Any, default: float) -> float:
    """Safely convert *value* to float, returning *default* on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_float_or_none(value: Any) -> float | None:
    """Safely convert *value* to float, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any, default: int) -> int:
    """Safely convert *value* to int, returning *default* on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool) -> bool:
    """Safely convert *value* to bool, returning *default* on failure."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def as_mode(value: Any, default: str, allowed: set[str]) -> str:
    """Convert *value* to uppercase string if it matches *allowed* set."""
    if value is None:
        return default
    mode = str(value).strip().upper()
    return mode if mode in allowed else default
