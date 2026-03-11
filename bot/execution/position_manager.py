from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from bot.data.capital_client import CapitalAPIError, CapitalClient
from bot.execution.sizing import r_multiple
from bot.execution.utils import strip_mode_prefix as _strip_mode_prefix
from bot.storage.journal import Journal
from bot.storage.models import ClosedPositionEvent, PositionRecord
from bot.strategy.utils import as_float

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-TP runtime config (frozen for speed)
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class TPLevel:
    name: str
    trigger_r: float
    close_fraction: float
    move_sl_to_be: bool


@dataclass(slots=True, frozen=True)
class MultiTPProfile:
    """Runtime multi-TP profile — built once from config."""

    enabled: bool = False
    levels: tuple[TPLevel, ...] = ()
    be_offset_r: float = 0.05
    be_delay_bars: int = 4
    trailing_enabled: bool = True
    trailing_swing_window: int = 12
    trailing_buffer_r: float = 0.12


def build_multi_tp_profile(config: Any) -> MultiTPProfile:
    """Convert MultiTPConfig pydantic model → frozen dataclass."""
    if config is None or not getattr(config, "enabled", False):
        return MultiTPProfile(enabled=False)
    levels: list[TPLevel] = []
    for lv in getattr(config, "levels", []):
        levels.append(
            TPLevel(
                name=lv.name,
                trigger_r=lv.trigger_r,
                close_fraction=lv.close_fraction,
                move_sl_to_be=lv.move_sl_to_be,
            )
        )
    # Sort by trigger_r ascending
    levels.sort(key=lambda x: x.trigger_r)
    return MultiTPProfile(
        enabled=True,
        levels=tuple(levels),
        be_offset_r=getattr(config, "be_offset_r", 0.05),
        be_delay_bars=getattr(config, "be_delay_bars", 4),
        trailing_enabled=getattr(config, "trailing_enabled", True),
        trailing_swing_window=getattr(config, "trailing_swing_window", 12),
        trailing_buffer_r=getattr(config, "trailing_buffer_r", 0.12),
    )


# _strip_mode_prefix imported from bot.execution.utils


class PositionManager:
    def __init__(
        self, *, client: CapitalClient | None, journal: Journal, dry_run: bool, multi_tp: MultiTPProfile | None = None
    ):
        self.client = client
        self.journal = journal
        self.dry_run = dry_run
        self.mode_prefix = "DRY" if dry_run else "PAPER"
        self.multi_tp = multi_tp or MultiTPProfile()

    def get_open_positions(self, epic: str | None = None) -> list[PositionRecord]:
        prefix = f"{self.mode_prefix}-"
        return [
            position for position in self.journal.get_open_positions(epic=epic) if position.deal_id.startswith(prefix)
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
            opened_at_raw = pos.get("createdDateUTC") or datetime.now(UTC).isoformat()
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

        now = datetime.now(UTC)
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
                    side=local.side,
                    exit_type=str(local.metadata.get("exit_type", "MANUAL")),
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

            # ── Multi-TP path ──────────────────────────
            if self.multi_tp.enabled:
                self._apply_multi_tp(position, current_r, current_price, now)
            else:
                # ── Legacy single-TP path ──────────────
                half_size = round(position.size * 0.5, 8)
                already_scaled = position.partial_closed_size >= half_size and half_size > 0
                if current_r >= 1.0 and not already_scaled:
                    self._move_sl_to_be_and_partial(position, half_size)

            closed, pnl = self._simulate_close_if_hit(position, current_price, now)
            if closed:
                _exit_type = str(position.metadata.get("exit_type", ""))
                closed_events.append(
                    ClosedPositionEvent(
                        deal_id=position.deal_id,
                        epic=position.epic,
                        pnl=pnl,
                        closed_at=now,
                        side=position.side,
                        exit_type=_exit_type,
                    )
                )

        return closed_events

    # ── Multi-TP management ──────────────────────────────
    def _apply_multi_tp(
        self,
        position: PositionRecord,
        current_r: float,
        current_price: float,
        now: datetime,
    ) -> None:
        """Apply multi-level take-profit, BE movement, and trailing stop."""
        meta = dict(position.metadata)
        tp_levels_hit: list[str] = meta.get("tp_levels_hit", [])
        initial_size = float(meta.get("initial_size", position.size + position.partial_closed_size))
        tp1_hit = bool(meta.get("tp1_hit", False))
        be_moved = bool(meta.get("be_moved", False))
        tp1_hit_at_bar = meta.get("tp1_hit_at_bar")
        bars_since_open = int(meta.get("bars_managed", 0)) + 1
        meta["bars_managed"] = bars_since_open

        profile = self.multi_tp

        for level in profile.levels:
            if level.name in tp_levels_hit:
                continue
            if current_r >= level.trigger_r:
                # Close fraction of remaining size
                close_size = round(position.size * level.close_fraction, 8)
                if close_size > 0 and position.size > close_size:
                    position.partial_closed_size += close_size
                    position.size = round(position.size - close_size, 8)
                    tp_levels_hit.append(level.name)
                    LOGGER.info(
                        "Multi-TP %s hit for %s at %.2fR — closed %.4f (remaining %.4f)",
                        level.name,
                        position.deal_id,
                        current_r,
                        close_size,
                        position.size,
                    )
                    if level.move_sl_to_be and not be_moved:
                        tp1_hit = True
                        tp1_hit_at_bar = bars_since_open
                elif level.close_fraction >= 1.0:
                    # Full close at this level
                    tp_levels_hit.append(level.name)
                    LOGGER.info(
                        "Multi-TP %s hit for %s — full close signal at %.2fR",
                        level.name,
                        position.deal_id,
                        current_r,
                    )

        # BE movement (after TP1, with delay)
        if tp1_hit and not be_moved and tp1_hit_at_bar is not None:
            elapsed_bars = bars_since_open - int(tp1_hit_at_bar)
            if elapsed_bars >= profile.be_delay_bars:
                risk_dist = abs(position.entry_price - float(meta.get("initial_stop", position.stop_price)))
                cost_offset = self._get_cost_offset(meta)
                if position.side == "LONG":
                    be_price = position.entry_price + max(cost_offset, risk_dist * profile.be_offset_r)
                    if be_price > position.stop_price:
                        position.stop_price = be_price
                        be_moved = True
                else:
                    be_price = position.entry_price - max(cost_offset, risk_dist * profile.be_offset_r)
                    if be_price < position.stop_price:
                        position.stop_price = be_price
                        be_moved = True
                if be_moved:
                    meta["be_moved"] = True
                    meta["be_price"] = position.stop_price
                    LOGGER.info("Multi-TP BE moved for %s to %.5f", position.deal_id, position.stop_price)

        # Trailing stop (after BE moved)
        if profile.trailing_enabled and be_moved and position.size > 0:
            risk_dist = abs(position.entry_price - float(meta.get("initial_stop", position.stop_price)))
            buffer_value = risk_dist * profile.trailing_buffer_r
            if position.side == "LONG":
                trail_stop = current_price - buffer_value
                if trail_stop > position.stop_price:
                    position.stop_price = trail_stop
                    meta["trailing_stop"] = trail_stop
            else:
                trail_stop = current_price + buffer_value
                if trail_stop < position.stop_price:
                    position.stop_price = trail_stop
                    meta["trailing_stop"] = trail_stop

        meta["tp_levels_hit"] = tp_levels_hit
        meta["tp1_hit"] = tp1_hit
        meta["tp1_hit_at_bar"] = tp1_hit_at_bar
        meta["initial_size"] = initial_size
        position.metadata = meta
        self.journal.upsert_position(position)

    @staticmethod
    def _get_cost_offset(meta: dict) -> float:
        """Read cost offset from position metadata."""
        spread = as_float(meta.get("spread_at_entry"), 0.0)
        buf = as_float(meta.get("be_buffer_ticks"), 0.0)
        return spread + buf

    def _move_sl_to_be_and_partial(self, position: PositionRecord, half_size: float) -> None:
        LOGGER.info("Position %s reached +1R: moving SL->BE and partial close", position.deal_id)
        # Cost-aware BE: SL = entry + spread + slippage + buffer (covers costs)
        be_price = position.entry_price
        meta = dict(position.metadata)
        spread_at_entry = as_float(meta.get("spread_at_entry"), 0.0)
        be_buffer = as_float(meta.get("be_buffer_ticks"), 0.0)
        cost_offset = spread_at_entry + be_buffer
        if position.side == "LONG":
            be_price = position.entry_price + cost_offset
        else:
            be_price = position.entry_price - cost_offset
        position.stop_price = be_price
        position.partial_closed_size = max(position.partial_closed_size, half_size)
        meta["be_moved"] = True
        meta["be_price"] = be_price
        meta["be_cost_offset"] = cost_offset
        position.metadata = meta

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

        # Subtract estimated roundtrip cost for DRY mode realism
        rtc = as_float(position.metadata.get("estimated_roundtrip_cost"), 0.0)
        net_pnl = pnl - rtc

        position.pnl = net_pnl
        position.status = "CLOSED"
        position.closed_at = now
        meta = dict(position.metadata)
        meta["gross_pnl"] = pnl
        meta["net_pnl"] = net_pnl
        meta["roundtrip_cost"] = rtc
        meta["exit_type"] = "STOP" if stop_hit else "TP"
        position.metadata = meta
        self.journal.upsert_position(position)
        return True, net_pnl
