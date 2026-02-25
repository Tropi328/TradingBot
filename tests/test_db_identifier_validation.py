from __future__ import annotations

import sqlite3

import pytest

from bot.storage.db import _ensure_column, _table_columns, init_db


def _build_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def test_ensure_column_blocks_unknown_table() -> None:
    conn = _build_conn()
    with pytest.raises(ValueError):
        _ensure_column(conn, "unknown_table", "epic", "TEXT")


def test_ensure_column_blocks_unknown_column() -> None:
    conn = _build_conn()
    with pytest.raises(ValueError):
        _ensure_column(conn, "orders", "not_allowed", "TEXT")


def test_table_columns_rejects_invalid_identifier() -> None:
    conn = _build_conn()
    with pytest.raises(ValueError):
        _table_columns(conn, "orders; DROP TABLE orders;")
