from __future__ import annotations

import sqlite3
from pathlib import Path

_ALLOWED_MIGRATION_TABLES = frozenset(
    {
        "journal_trades",
        "orders",
        "positions",
        "daily_stats",
        "spreads",
        "risk_state",
        "capital_ramp_state",
        "capital_ramp_events",
    }
)
_ALLOWED_MIGRATION_COLUMNS = frozenset(
    {
        "epic",
        "request_id",
        "remote_status",
        "filled_size",
    }
)


def _validate_sql_identifier(identifier: str) -> str:
    normalized = str(identifier).strip()
    if not normalized:
        raise ValueError("SQL identifier cannot be empty")
    if not normalized.replace("_", "").isalnum():
        raise ValueError(f"Invalid SQL identifier: {identifier!r}")
    return normalized


def _quote_identifier(identifier: str) -> str:
    validated = _validate_sql_identifier(identifier)
    return f'"{validated}"'


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    quoted_table = _quote_identifier(table_name)
    rows = conn.execute(f"PRAGMA table_info({quoted_table})").fetchall()
    return {str(row[1]) for row in rows}


def _ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_sql: str,
) -> None:
    if table_name not in _ALLOWED_MIGRATION_TABLES:
        raise ValueError(f"Unsupported migration table: {table_name!r}")
    if column_name not in _ALLOWED_MIGRATION_COLUMNS:
        raise ValueError(f"Unsupported migration column: {column_name!r}")
    if column_name in _table_columns(conn, table_name):
        return
    quoted_table = _quote_identifier(table_name)
    quoted_column = _quote_identifier(column_name)
    conn.execute(f"ALTER TABLE {quoted_table} ADD COLUMN {quoted_column} {column_sql}")


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _daily_stats_pk_columns(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("PRAGMA table_info(daily_stats)").fetchall()
    pk_rows = sorted((row for row in rows if int(row[5]) > 0), key=lambda row: int(row[5]))
    return [str(row[1]) for row in pk_rows]


def _rebuild_daily_stats_table(conn: sqlite3.Connection) -> None:
    has_epic = "epic" in _table_columns(conn, "daily_stats")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS daily_stats_new (
            trading_day TEXT NOT NULL,
            epic TEXT NOT NULL DEFAULT 'GLOBAL',
            pnl REAL NOT NULL DEFAULT 0,
            trades_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'ON',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (trading_day, epic)
        );
        """
    )
    if has_epic:
        conn.execute(
            """
            INSERT OR REPLACE INTO daily_stats_new (
                trading_day, epic, pnl, trades_count, status, updated_at
            )
            SELECT
                trading_day,
                COALESCE(NULLIF(epic, ''), 'GLOBAL'),
                pnl,
                trades_count,
                status,
                updated_at
            FROM daily_stats
            """
        )
    else:
        conn.execute(
            """
            INSERT OR REPLACE INTO daily_stats_new (
                trading_day, epic, pnl, trades_count, status, updated_at
            )
            SELECT
                trading_day,
                'GLOBAL',
                pnl,
                trades_count,
                status,
                updated_at
            FROM daily_stats
            """
        )
    conn.execute("DROP TABLE daily_stats")
    conn.execute("ALTER TABLE daily_stats_new RENAME TO daily_stats")


def _drop_legacy_daily_stats_indexes(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA index_list(daily_stats)").fetchall()
    for row in rows:
        index_name = str(row[1])
        is_unique = int(row[2]) == 1
        if not is_unique or index_name.startswith("sqlite_autoindex"):
            continue
        try:
            quoted_index = _quote_identifier(index_name)
        except ValueError:
            continue
        idx_cols = conn.execute(f"PRAGMA index_info({quoted_index})").fetchall()
        col_names = [str(idx_col[2]) for idx_col in idx_cols]
        if col_names == ["trading_day"]:
            conn.execute(f"DROP INDEX IF EXISTS {quoted_index}")


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS journal_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            epic TEXT NOT NULL DEFAULT 'UNKNOWN',
            side TEXT,
            bias TEXT NOT NULL,
            pd_state TEXT NOT NULL,
            sweep INTEGER NOT NULL,
            mss INTEGER NOT NULL,
            displacement INTEGER NOT NULL,
            fvg INTEGER NOT NULL,
            spread_ok INTEGER NOT NULL,
            news_blocked INTEGER NOT NULL,
            rr REAL,
            reason_codes TEXT NOT NULL,
            payload TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            deal_reference TEXT,
            request_id TEXT,
            epic TEXT NOT NULL,
            side TEXT NOT NULL,
            size REAL NOT NULL,
            entry_price REAL NOT NULL,
            stop_price REAL NOT NULL,
            take_profit REAL NOT NULL,
            status TEXT NOT NULL,
            remote_status TEXT,
            filled_size REAL NOT NULL DEFAULT 0,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            reason_codes TEXT NOT NULL,
            metadata TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS positions (
            deal_id TEXT PRIMARY KEY,
            epic TEXT NOT NULL,
            side TEXT NOT NULL,
            size REAL NOT NULL,
            entry_price REAL NOT NULL,
            stop_price REAL NOT NULL,
            take_profit REAL NOT NULL,
            status TEXT NOT NULL,
            opened_at TEXT NOT NULL,
            closed_at TEXT,
            partial_closed_size REAL NOT NULL DEFAULT 0,
            pnl REAL NOT NULL DEFAULT 0,
            metadata TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS daily_stats (
            trading_day TEXT NOT NULL,
            epic TEXT NOT NULL DEFAULT 'GLOBAL',
            pnl REAL NOT NULL DEFAULT 0,
            trades_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'ON',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (trading_day, epic)
        );

        CREATE TABLE IF NOT EXISTS spreads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            epic TEXT NOT NULL DEFAULT 'UNKNOWN',
            spread REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS risk_state (
            scope TEXT PRIMARY KEY,
            loss_streak INTEGER NOT NULL DEFAULT 0,
            cooldown_until TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS capital_ramp_state (
            scope TEXT PRIMARY KEY,
            timezone_name TEXT NOT NULL,
            start_ts_utc TEXT NOT NULL,
            start_year_local INTEGER NOT NULL,
            start_equity REAL NOT NULL,
            monthly_topup REAL NOT NULL,
            target_equity REAL NOT NULL,
            pnl_baseline_at_start REAL NOT NULL DEFAULT 0,
            topups_total REAL NOT NULL DEFAULT 0,
            topups_count INTEGER NOT NULL DEFAULT 0,
            next_topup_date_local TEXT,
            last_topup_date_local TEXT,
            stopped_reason TEXT,
            stopped_at_utc TEXT,
            updated_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS capital_ramp_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_ts_utc TEXT NOT NULL,
            local_date TEXT NOT NULL,
            amount REAL NOT NULL DEFAULT 0,
            model_equity REAL NOT NULL DEFAULT 0,
            payload TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
        CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
        CREATE INDEX IF NOT EXISTS idx_spreads_timestamp ON spreads(timestamp);
        CREATE INDEX IF NOT EXISTS idx_capital_ramp_events_scope_ts ON capital_ramp_events(scope, event_ts_utc);
        """
    )
    # Runtime migration support for existing databases.
    _ensure_column(conn, "journal_trades", "epic", "TEXT NOT NULL DEFAULT 'UNKNOWN'")
    _ensure_column(conn, "orders", "request_id", "TEXT")
    _ensure_column(conn, "orders", "remote_status", "TEXT")
    _ensure_column(conn, "orders", "filled_size", "REAL NOT NULL DEFAULT 0")
    _ensure_column(conn, "daily_stats", "epic", "TEXT NOT NULL DEFAULT 'GLOBAL'")
    _ensure_column(conn, "spreads", "epic", "TEXT NOT NULL DEFAULT 'UNKNOWN'")
    if _table_exists(conn, "daily_stats") and _daily_stats_pk_columns(conn) != ["trading_day", "epic"]:
        _rebuild_daily_stats_table(conn)
    _drop_legacy_daily_stats_indexes(conn)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_stats_day_epic ON daily_stats(trading_day, epic)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_request_id ON orders(request_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_spreads_epic_timestamp ON spreads(epic, timestamp)")
    conn.commit()
