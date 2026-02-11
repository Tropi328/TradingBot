from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

from bot.storage.models import (
    DailyStats,
    OrderRecord,
    PositionRecord,
    RiskState,
    StrategyDecisionRecord,
)


def _to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _from_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


class Journal:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.lock = threading.Lock()

    def log_decision(self, record: StrategyDecisionRecord) -> None:
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO journal_trades (
                    created_at, epic, side, bias, pd_state, sweep, mss, displacement, fvg,
                    spread_ok, news_blocked, rr, reason_codes, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _to_iso(record.created_at),
                    record.epic,
                    record.side,
                    record.bias,
                    record.pd_state,
                    int(record.sweep),
                    int(record.mss),
                    int(record.displacement),
                    int(record.fvg),
                    int(record.spread_ok),
                    int(record.news_blocked),
                    record.rr,
                    json.dumps(record.reason_codes),
                    json.dumps(record.payload),
                ),
            )
            self.conn.commit()

    def upsert_order(self, order: OrderRecord) -> None:
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO orders (
                    order_id, deal_reference, request_id, epic, side, size, entry_price, stop_price, take_profit,
                    status, remote_status, filled_size, expires_at, created_at, updated_at, reason_codes, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) DO UPDATE SET
                    deal_reference=excluded.deal_reference,
                    request_id=excluded.request_id,
                    status=excluded.status,
                    remote_status=excluded.remote_status,
                    filled_size=excluded.filled_size,
                    size=excluded.size,
                    entry_price=excluded.entry_price,
                    stop_price=excluded.stop_price,
                    take_profit=excluded.take_profit,
                    expires_at=excluded.expires_at,
                    updated_at=excluded.updated_at,
                    reason_codes=excluded.reason_codes,
                    metadata=excluded.metadata
                """,
                (
                    order.order_id,
                    order.deal_reference,
                    order.request_id,
                    order.epic,
                    order.side,
                    order.size,
                    order.entry_price,
                    order.stop_price,
                    order.take_profit,
                    order.status,
                    order.remote_status,
                    order.filled_size,
                    _to_iso(order.expires_at),
                    _to_iso(order.created_at),
                    _to_iso(order.updated_at),
                    json.dumps(order.reason_codes),
                    json.dumps(order.metadata),
                ),
            )
            self.conn.commit()

    def update_order_status(
        self,
        order_id: str,
        status: str,
        updated_at: datetime,
        *,
        remote_status: str | None = None,
        filled_size: float | None = None,
    ) -> None:
        with self.lock:
            if remote_status is None and filled_size is None:
                self.conn.execute(
                    "UPDATE orders SET status = ?, updated_at = ? WHERE order_id = ?",
                    (status, _to_iso(updated_at), order_id),
                )
            else:
                self.conn.execute(
                    """
                    UPDATE orders
                    SET status = ?, remote_status = COALESCE(?, remote_status),
                        filled_size = COALESCE(?, filled_size), updated_at = ?
                    WHERE order_id = ?
                    """,
                    (status, remote_status, filled_size, _to_iso(updated_at), order_id),
                )
            self.conn.commit()

    def get_pending_orders(self, epic: str | None = None) -> list[OrderRecord]:
        if epic:
            rows = self.conn.execute(
                "SELECT * FROM orders WHERE status = 'PENDING' AND epic = ? ORDER BY created_at ASC",
                (epic,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM orders WHERE status = 'PENDING' ORDER BY created_at ASC"
            ).fetchall()
        return [self._row_to_order(row) for row in rows]

    def get_order_by_request_id(self, request_id: str) -> OrderRecord | None:
        row = self.conn.execute(
            "SELECT * FROM orders WHERE request_id = ? ORDER BY created_at DESC LIMIT 1",
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_order(row)

    def upsert_position(self, position: PositionRecord) -> None:
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO positions (
                    deal_id, epic, side, size, entry_price, stop_price, take_profit, status,
                    opened_at, closed_at, partial_closed_size, pnl, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(deal_id) DO UPDATE SET
                    status=excluded.status,
                    size=excluded.size,
                    stop_price=excluded.stop_price,
                    take_profit=excluded.take_profit,
                    closed_at=excluded.closed_at,
                    partial_closed_size=excluded.partial_closed_size,
                    pnl=excluded.pnl,
                    metadata=excluded.metadata
                """,
                (
                    position.deal_id,
                    position.epic,
                    position.side,
                    position.size,
                    position.entry_price,
                    position.stop_price,
                    position.take_profit,
                    position.status,
                    _to_iso(position.opened_at),
                    _to_iso(position.closed_at),
                    position.partial_closed_size,
                    position.pnl,
                    json.dumps(position.metadata),
                ),
            )
            self.conn.commit()

    def get_open_positions(self, epic: str | None = None) -> list[PositionRecord]:
        if epic:
            rows = self.conn.execute(
                "SELECT * FROM positions WHERE status = 'OPEN' AND epic = ? ORDER BY opened_at ASC",
                (epic,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM positions WHERE status = 'OPEN' ORDER BY opened_at ASC"
            ).fetchall()
        return [self._row_to_position(row) for row in rows]

    def get_daily_stats(self, trading_day: str, epic: str = "GLOBAL") -> DailyStats:
        row = self.conn.execute(
            "SELECT * FROM daily_stats WHERE trading_day = ? AND epic = ?",
            (trading_day, epic),
        ).fetchone()
        if row is None:
            stats = DailyStats(
                trading_day=trading_day,
                epic=epic,
                updated_at=datetime.now(timezone.utc),
            )
            self.upsert_daily_stats(stats)
            return stats
        return DailyStats(
            trading_day=row["trading_day"],
            epic=row["epic"] if "epic" in row.keys() else epic,
            pnl=float(row["pnl"]),
            trades_count=int(row["trades_count"]),
            status=row["status"],
            updated_at=_from_iso(row["updated_at"]),
        )

    def upsert_daily_stats(self, stats: DailyStats) -> None:
        updated_at = stats.updated_at or datetime.now(timezone.utc)
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO daily_stats (trading_day, epic, pnl, trades_count, status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(trading_day, epic) DO UPDATE SET
                    pnl=excluded.pnl,
                    trades_count=excluded.trades_count,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (
                    stats.trading_day,
                    stats.epic,
                    stats.pnl,
                    stats.trades_count,
                    stats.status,
                    _to_iso(updated_at),
                ),
            )
            self.conn.commit()

    def increment_daily_trades(self, trading_day: str, increment: int = 1, epic: str = "GLOBAL") -> DailyStats:
        stats = self.get_daily_stats(trading_day, epic=epic)
        stats.trades_count += increment
        stats.updated_at = datetime.now(timezone.utc)
        self.upsert_daily_stats(stats)
        return stats

    def add_daily_pnl(self, trading_day: str, pnl_delta: float, epic: str = "GLOBAL") -> DailyStats:
        stats = self.get_daily_stats(trading_day, epic=epic)
        stats.pnl += pnl_delta
        stats.updated_at = datetime.now(timezone.utc)
        self.upsert_daily_stats(stats)
        return stats

    def set_daily_status(self, trading_day: str, status: str, epic: str = "GLOBAL") -> DailyStats:
        stats = self.get_daily_stats(trading_day, epic=epic)
        stats.status = status
        stats.updated_at = datetime.now(timezone.utc)
        self.upsert_daily_stats(stats)
        return stats

    def get_risk_state(self, scope: str) -> RiskState:
        row = self.conn.execute(
            "SELECT * FROM risk_state WHERE scope = ?",
            (scope,),
        ).fetchone()
        if row is None:
            state = RiskState(scope=scope, updated_at=datetime.now(timezone.utc))
            self.upsert_risk_state(state)
            return state
        return RiskState(
            scope=row["scope"],
            loss_streak=int(row["loss_streak"]),
            cooldown_until=_from_iso(row["cooldown_until"]),
            updated_at=_from_iso(row["updated_at"]),
        )

    def upsert_risk_state(self, state: RiskState) -> None:
        now = state.updated_at or datetime.now(timezone.utc)
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO risk_state (scope, loss_streak, cooldown_until, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(scope) DO UPDATE SET
                    loss_streak=excluded.loss_streak,
                    cooldown_until=excluded.cooldown_until,
                    updated_at=excluded.updated_at
                """,
                (
                    state.scope,
                    state.loss_streak,
                    _to_iso(state.cooldown_until),
                    _to_iso(now),
                ),
            )
            self.conn.commit()

    def save_spread(self, timestamp: datetime, spread: float, epic: str) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO spreads (timestamp, epic, spread) VALUES (?, ?, ?)",
                (_to_iso(timestamp), epic, spread),
            )
            self.conn.commit()

    def load_recent_spreads(self, limit: int, epic: str) -> list[float]:
        rows = self.conn.execute(
            "SELECT spread FROM spreads WHERE epic = ? ORDER BY id DESC LIMIT ?",
            (epic, limit),
        ).fetchall()
        return [float(row["spread"]) for row in reversed(rows)]

    @staticmethod
    def _row_to_order(row: sqlite3.Row) -> OrderRecord:
        row_keys = row.keys()
        return OrderRecord(
            order_id=row["order_id"],
            deal_reference=row["deal_reference"],
            request_id=row["request_id"] if "request_id" in row_keys else None,
            epic=row["epic"],
            side=row["side"],
            size=float(row["size"]),
            entry_price=float(row["entry_price"]),
            stop_price=float(row["stop_price"]),
            take_profit=float(row["take_profit"]),
            status=row["status"],
            remote_status=row["remote_status"] if "remote_status" in row_keys else None,
            filled_size=float(row["filled_size"]) if "filled_size" in row_keys else 0.0,
            expires_at=_from_iso(row["expires_at"]) or datetime.now(timezone.utc),
            created_at=_from_iso(row["created_at"]) or datetime.now(timezone.utc),
            updated_at=_from_iso(row["updated_at"]) or datetime.now(timezone.utc),
            reason_codes=json.loads(row["reason_codes"] or "[]"),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _row_to_position(row: sqlite3.Row) -> PositionRecord:
        return PositionRecord(
            deal_id=row["deal_id"],
            epic=row["epic"],
            side=row["side"],
            size=float(row["size"]),
            entry_price=float(row["entry_price"]),
            stop_price=float(row["stop_price"]),
            take_profit=float(row["take_profit"]),
            status=row["status"],
            opened_at=_from_iso(row["opened_at"]) or datetime.now(timezone.utc),
            closed_at=_from_iso(row["closed_at"]),
            partial_closed_size=float(row["partial_closed_size"]),
            pnl=float(row["pnl"]),
            metadata=json.loads(row["metadata"] or "{}"),
        )
