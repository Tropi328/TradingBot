from __future__ import annotations

import logging
from datetime import datetime, timezone

from bot.data.capital_client import CapitalAPIError, CapitalClient
from bot.execution.sizing import r_multiple
from bot.storage.journal import Journal
from bot.storage.models import ClosedPositionEvent, PositionRecord

LOGGER = logging.getLogger(__name__)


def _strip_mode_prefix(identifier: str) -> str:
    if "-" not in identifier:
        return identifier
    prefix, rest = identifier.split("-", 1)
    if prefix in {"DRY", "PAPER"} and rest:
        return rest
    return identifier


class PositionManager:
    def __init__(self, *, client: CapitalClient | None, journal: Journal, dry_run: bool):
        self.client = client
        self.journal = journal
        self.dry_run = dry_run
        self.mode_prefix = "DRY" if dry_run else "PAPER"

    def get_open_positions(self, epic: str | None = None) -> list[PositionRecord]:
        prefix = f"{self.mode_prefix}-"
        return [
            position
            for position in self.journal.get_open_positions(epic=epic)
            if position.deal_id.startswith(prefix)
        ]

    def sync_positions_from_api(self) -> list[ClosedPositionEvent]:
        closed_events: list[ClosedPositionEvent] = []
        if self.dry_run or self.client is None:
            return closed_events
        try:
            positions = self.client.get_positions()
        except CapitalAPIError as exc:
            LOGGER.warning("Could not sync positions: %s", exc)
            return closed_events

        remote_ids: set[str] = set()
        for item in positions:
            pos = item.get("position", item)
            market = item.get("market", {})
            remote_deal_id = str(pos.get("dealId", ""))
            if not remote_deal_id:
                continue
            deal_id = f"{self.mode_prefix}-{remote_deal_id}"
            remote_ids.add(deal_id)
            side = "LONG" if pos.get("direction") == "BUY" else "SHORT"
            opened_at_raw = pos.get("createdDateUTC") or datetime.now(timezone.utc).isoformat()
            opened_at = datetime.fromisoformat(str(opened_at_raw).replace("Z", "+00:00"))
            record = PositionRecord(
                deal_id=deal_id,
                epic=str(market.get("epic") or pos.get("epic") or ""),
                side=side,
                size=float(pos.get("size", 0.0)),
                entry_price=float(pos.get("level", 0.0)),
                stop_price=float(pos.get("stopLevel") or pos.get("stop", 0.0)),
                take_profit=float(pos.get("limitLevel") or pos.get("profitLevel") or 0.0),
                status="OPEN",
                opened_at=opened_at,
                partial_closed_size=float(pos.get("partialClosedSize") or 0.0),
                pnl=float(pos.get("upl") or 0.0),
                metadata={"api_snapshot": pos},
            )
            self.journal.upsert_position(record)

        now = datetime.now(timezone.utc)
        for local in self.get_open_positions():
            if local.deal_id in remote_ids:
                continue
            local.status = "CLOSED"
            local.closed_at = now
            self.journal.upsert_position(local)
            closed_events.append(
                ClosedPositionEvent(
                    deal_id=local.deal_id,
                    epic=local.epic,
                    pnl=local.pnl,
                    closed_at=now,
                )
            )
        return closed_events

    def manage_open_positions(
        self,
        *,
        now: datetime,
        quotes_by_epic: dict[str, tuple[float, float, float]],
    ) -> list[ClosedPositionEvent]:
        closed_events: list[ClosedPositionEvent] = []
        for position in self.get_open_positions():
            quote = quotes_by_epic.get(position.epic)
            if quote is None:
                continue
            bid, ask, _ = quote
            current_price = bid if position.side == "LONG" else ask
            current_r = r_multiple(
                side=position.side,
                entry_price=position.entry_price,
                stop_price=position.stop_price,
                current_price=current_price,
            )

            half_size = round(position.size * 0.5, 8)
            already_scaled = position.partial_closed_size >= half_size and half_size > 0

            if current_r >= 1.0 and not already_scaled:
                self._move_sl_to_be_and_partial(position, half_size)

            closed, pnl = self._simulate_close_if_hit(position, current_price, now)
            if closed:
                closed_events.append(
                    ClosedPositionEvent(
                        deal_id=position.deal_id,
                        epic=position.epic,
                        pnl=pnl,
                        closed_at=now,
                    )
                )

        return closed_events

    def _move_sl_to_be_and_partial(self, position: PositionRecord, half_size: float) -> None:
        LOGGER.info("Position %s reached +1R: moving SL->BE and partial close", position.deal_id)
        position.stop_price = position.entry_price
        position.partial_closed_size = max(position.partial_closed_size, half_size)
        metadata = dict(position.metadata)
        metadata["be_moved"] = True
        position.metadata = metadata

        if not self.dry_run and self.client is not None:
            try:
                remote_deal_id = _strip_mode_prefix(position.deal_id)
                self.client.update_position(remote_deal_id, stop_level=position.entry_price)
                if half_size > 0:
                    self.client.partial_close_position(remote_deal_id, half_size)
            except CapitalAPIError as exc:
                LOGGER.warning("Could not execute +1R management for %s: %s", position.deal_id, exc)

        self.journal.upsert_position(position)

    def _simulate_close_if_hit(
        self,
        position: PositionRecord,
        current_price: float,
        now: datetime,
    ) -> tuple[bool, float]:
        if not self.dry_run:
            return False, 0.0

        if position.side == "LONG":
            stop_hit = current_price <= position.stop_price
            tp_hit = current_price >= position.take_profit
        else:
            stop_hit = current_price >= position.stop_price
            tp_hit = current_price <= position.take_profit
        if not stop_hit and not tp_hit:
            return False, 0.0

        exit_price = position.stop_price if stop_hit else position.take_profit
        if position.side == "LONG":
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size
        position.pnl = pnl
        position.status = "CLOSED"
        position.closed_at = now
        self.journal.upsert_position(position)
        return True, pnl
