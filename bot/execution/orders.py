from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from bot.data.capital_client import CapitalAPIError, CapitalClient
from bot.storage.journal import Journal
from bot.storage.models import OrderRecord, PositionRecord
from bot.strategy.state_machine import StrategySignal

LOGGER = logging.getLogger(__name__)


def _map_remote_order_status(raw: str | None) -> str:
    if raw is None:
        return "PENDING"
    status = raw.strip().upper()
    if status in {"OPEN", "WORKING", "ACTIVE", "PENDING"}:
        return "PENDING"
    if status in {"DELETED", "CANCELLED", "CANCELED"}:
        return "CANCELLED"
    if status in {"ACCEPTED", "FILLED", "OPENED"}:
        return "FILLED"
    if status in {"REJECTED"}:
        return "REJECTED"
    if status in {"EXPIRED"}:
        return "EXPIRED"
    return "PENDING"


def _strip_mode_prefix(identifier: str) -> str:
    if "-" not in identifier:
        return identifier
    prefix, rest = identifier.split("-", 1)
    if prefix in {"DRY", "PAPER"} and rest:
        return rest
    return identifier


class OrderExecutor:
    def __init__(
        self,
        *,
        client: CapitalClient | None,
        journal: Journal,
        dry_run: bool,
        default_epic: str,
        default_currency: str,
    ):
        self.client = client
        self.journal = journal
        self.dry_run = dry_run
        self.default_epic = default_epic
        self.default_currency = default_currency
        self.mode_prefix = "DRY" if dry_run else "PAPER"

    def place_limit_order(
        self,
        signal: StrategySignal,
        size: float,
        *,
        epic: str | None = None,
        currency: str | None = None,
        idempotency_key: str | None = None,
    ) -> OrderRecord:
        now = datetime.now(timezone.utc)
        order_epic = epic or self.default_epic
        order_currency = currency or self.default_currency
        request_id = idempotency_key or f"{self.mode_prefix}-REQ-{uuid.uuid4().hex[:20]}"

        existing = self.journal.get_order_by_request_id(request_id)
        if existing is not None and existing.status in {"PENDING", "FILLED"}:
            LOGGER.info("Idempotent order reuse for request_id=%s", request_id)
            return existing

        if self.dry_run or self.client is None:
            order_id = f"{self.mode_prefix}-{uuid.uuid4().hex[:10]}"
            record = OrderRecord(
                order_id=order_id,
                deal_reference=request_id,
                request_id=request_id,
                epic=order_epic,
                side=signal.side,
                size=size,
                entry_price=signal.entry_price,
                stop_price=signal.stop_price,
                take_profit=signal.take_profit,
                status="PENDING",
                remote_status="DRY_PENDING",
                filled_size=0.0,
                expires_at=signal.expires_at,
                created_at=now,
                updated_at=now,
                reason_codes=signal.reason_codes,
                metadata=signal.metadata,
            )
            self.journal.upsert_order(record)
            LOGGER.info("DRY-RUN: placed limit order %s (%s)", record.order_id, order_epic)
            return record

        response = self.client.place_working_order(
            epic=order_epic,
            side=signal.side,
            size=size,
            level=signal.entry_price,
            stop_level=signal.stop_price,
            profit_level=signal.take_profit,
            currency=order_currency,
            expires_at=signal.expires_at,
            deal_reference=request_id,
        )
        remote_id = str(response.get("dealId") or response.get("dealReference") or uuid.uuid4().hex)
        order_id = f"{self.mode_prefix}-{remote_id}"
        deal_reference = str(response.get("dealReference") or request_id)
        remote_status = str(response.get("status") or response.get("dealStatus") or "PENDING")
        record = OrderRecord(
            order_id=order_id,
            deal_reference=deal_reference,
            request_id=request_id,
            epic=order_epic,
            side=signal.side,
            size=size,
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
            take_profit=signal.take_profit,
            status=_map_remote_order_status(remote_status),
            remote_status=remote_status,
            filled_size=0.0,
            expires_at=signal.expires_at,
            created_at=now,
            updated_at=now,
            reason_codes=signal.reason_codes,
            metadata=signal.metadata,
        )
        self.journal.upsert_order(record)
        LOGGER.info("Placed limit order %s (%s)", order_id, order_epic)
        return record

    def cancel_order(self, order_id: str) -> None:
        now = datetime.now(timezone.utc)
        if not self.dry_run and self.client is not None:
            try:
                self.client.cancel_working_order(_strip_mode_prefix(order_id))
            except CapitalAPIError as exc:
                LOGGER.warning("Cancel order failed (%s): %s", order_id, exc)
        self.journal.update_order_status(
            order_id,
            "CANCELLED",
            now,
            remote_status="CANCELLED",
        )
        LOGGER.info("Cancelled pending order %s", order_id)

    def get_pending_orders(self, epic: str | None = None) -> list[OrderRecord]:
        prefix = f"{self.mode_prefix}-"
        return [
            order
            for order in self.journal.get_pending_orders(epic=epic)
            if order.order_id.startswith(prefix)
        ]

    def cancel_expired_orders(self, now: datetime, epic: str | None = None) -> list[str]:
        cancelled: list[str] = []
        for order in self.get_pending_orders(epic=epic):
            if now >= order.expires_at:
                self.cancel_order(order.order_id)
                cancelled.append(order.order_id)
        return cancelled

    def sync_remote_pending_orders(self) -> list[tuple[str, str]]:
        """
        Returns list of (order_id, mapped_status) updates.
        """
        updates: list[tuple[str, str]] = []
        if self.dry_run or self.client is None:
            return updates
        try:
            remote_orders = self.client.get_working_orders()
        except CapitalAPIError as exc:
            LOGGER.warning("Could not sync pending orders: %s", exc)
            return updates

        remote_by_id: dict[str, dict] = {}
        for item in remote_orders:
            data = item.get("workingOrderData", item)
            deal_id = data.get("dealId")
            if deal_id:
                remote_by_id[str(deal_id)] = data

        now = datetime.now(timezone.utc)
        for order in self.get_pending_orders():
            remote = remote_by_id.get(_strip_mode_prefix(order.order_id))
            if remote is not None:
                raw_status = str(remote.get("status") or remote.get("orderStatus") or "PENDING")
                mapped_status = _map_remote_order_status(raw_status)
                self.journal.update_order_status(
                    order.order_id,
                    mapped_status,
                    now,
                    remote_status=raw_status,
                    filled_size=0.0,
                )
                updates.append((order.order_id, mapped_status))
                continue

            # Order disappeared from /workingorders; check confirmation when available.
            mapped_status = "EXTERNAL_CHANGE"
            raw_status = "MISSING_FROM_WORKING_ORDERS"
            if order.deal_reference:
                try:
                    confirm = self.client.get_confirmation(order.deal_reference)
                except CapitalAPIError:
                    confirm = {}
                if confirm:
                    raw_status = str(
                        confirm.get("dealStatus")
                        or confirm.get("status")
                        or confirm.get("reason")
                        or raw_status
                    )
                    mapped_status = _map_remote_order_status(raw_status)
            self.journal.update_order_status(
                order.order_id,
                mapped_status,
                now,
                remote_status=raw_status,
            )
            updates.append((order.order_id, mapped_status))
        return updates

    def process_pending_fills(
        self,
        *,
        quotes_by_epic: dict[str, tuple[float, float, float]],
        now: datetime,
    ) -> list[PositionRecord]:
        opened: list[PositionRecord] = []
        for order in self.get_pending_orders():
            quote = quotes_by_epic.get(order.epic)
            if quote is None:
                continue
            bid, ask, _ = quote
            if order.side == "LONG":
                filled = ask <= order.entry_price
                fill_price = ask
            else:
                filled = bid >= order.entry_price
                fill_price = bid
            if not filled:
                continue
            if not self.dry_run and self.client is not None:
                # In paper mode fills should be confirmed by API sync (positions endpoint).
                continue

            self.journal.update_order_status(
                order.order_id,
                "FILLED",
                now,
                remote_status="DRY_FILLED",
                filled_size=order.size,
            )
            position = PositionRecord(
                deal_id=order.order_id,
                epic=order.epic,
                side=order.side,
                size=order.size,
                entry_price=fill_price,
                stop_price=order.stop_price,
                take_profit=order.take_profit,
                status="OPEN",
                opened_at=now,
                metadata={
                    "from_order_id": order.order_id,
                    "fill_price": fill_price,
                    "strategy_meta": order.metadata,
                    "request_id": order.request_id,
                },
            )
            self.journal.upsert_position(position)
            opened.append(position)
            LOGGER.info("DRY-RUN fill simulated for %s at %.5f", order.order_id, fill_price)
        return opened
