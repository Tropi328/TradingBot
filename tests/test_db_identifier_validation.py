from __future__ import annotations

import sqlite3

import pytest

from bot.storage.db import _ensure_column, _table_columns, init_db


@pytest.fixture
def conn() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    init_db(connection)
    yield connection
    connection.close()


def test_ensure_column_blocks_unknown_table(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError):
        _ensure_column(conn, "unknown_table", "epic", "TEXT")


def test_ensure_column_blocks_unknown_column(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError):
        _ensure_column(conn, "orders", "not_allowed", "TEXT")


def test_table_columns_rejects_invalid_identifier(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError):
        _table_columns(conn, "orders; DROP TABLE orders;")
