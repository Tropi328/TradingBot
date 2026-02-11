from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bot.config import AppConfig, AssetConfig
from bot.data.candles import Candle
from bot.execution.sizing import position_size_from_risk
from bot.strategy.risk import RiskEngine
from bot.strategy.state_machine import StrategyEngine


@dataclass(slots=True)
class BacktestTrade:
    epic: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    r_multiple: float
    reason: str


@dataclass(slots=True)
class BacktestReport:
    epic: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    expectancy: float
    avg_r: float
    max_drawdown: float
    time_in_market_bars: int
    equity_end: float
    trade_log: list[BacktestTrade]

    def to_dict(self) -> dict[str, Any]:
        return {
            "epic": self.epic,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "expectancy": self.expectancy,
            "avg_r": self.avg_r,
            "max_drawdown": self.max_drawdown,
            "time_in_market_bars": self.time_in_market_bars,
            "equity_end": self.equity_end,
        }


@dataclass(slots=True)
class WalkForwardReport:
    epic: str
    splits: list[BacktestReport]
    aggregate: BacktestReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "epic": self.epic,
            "splits": [split.to_dict() for split in self.splits],
            "aggregate": self.aggregate.to_dict(),
        }


@dataclass(slots=True)
class _PendingOrder:
    side: str
    entry: float
    stop: float
    tp: float
    size: float
    expiry_index: int
    created_at: datetime


@dataclass(slots=True)
class _OpenPosition:
    side: str
    entry: float
    stop: float
    tp: float
    size: float
    opened_at: datetime
    be_moved: bool = False
    realized_partial: float = 0.0


def _parse_dt(value: str) -> datetime:
    normalized = value.replace(" ", "T")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_candles_csv(path: str | Path) -> list[Candle]:
    csv_path = Path(path)
    candles: list[Candle] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames:
            reader.fieldnames = [name.lstrip("\ufeff").strip() for name in reader.fieldnames]
        required = {"timestamp", "open", "high", "low", "close"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must include: timestamp,open,high,low,close")
        for row in reader:
            candles.append(
                Candle(
                    timestamp=_parse_dt(str(row["timestamp"])),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume") or 0.0),
                )
            )
    return sorted(candles, key=lambda c: c.timestamp)


def _bucket_time(dt: datetime, minutes: int) -> datetime:
    unix = int(dt.timestamp())
    size = minutes * 60
    return datetime.fromtimestamp(unix - (unix % size), tz=timezone.utc)


def aggregate_candles(candles: list[Candle], timeframe_minutes: int) -> list[Candle]:
    if not candles:
        return []
    result: list[Candle] = []
    bucket_start = _bucket_time(candles[0].timestamp, timeframe_minutes)
    open_price = candles[0].open
    high = candles[0].high
    low = candles[0].low
    close = candles[0].close
    volume = candles[0].volume or 0.0

    for candle in candles[1:]:
        current_bucket = _bucket_time(candle.timestamp, timeframe_minutes)
        if current_bucket != bucket_start:
            result.append(
                Candle(
                    timestamp=bucket_start,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                )
            )
            bucket_start = current_bucket
            open_price = candle.open
            high = candle.high
            low = candle.low
            close = candle.close
            volume = candle.volume or 0.0
            continue
        high = max(high, candle.high)
        low = min(low, candle.low)
        close = candle.close
        volume += candle.volume or 0.0

    result.append(
        Candle(
            timestamp=bucket_start,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
    )
    return result


def _calc_exit(position: _OpenPosition, candle: Candle) -> tuple[bool, float, str]:
    if position.side == "LONG":
        stop_hit = candle.low <= position.stop
        tp_hit = candle.high >= position.tp
    else:
        stop_hit = candle.high >= position.stop
        tp_hit = candle.low <= position.tp
    if not stop_hit and not tp_hit:
        return False, 0.0, ""
    # Conservative fill order when both happen in one bar.
    if stop_hit:
        return True, position.stop, "STOP"
    return True, position.tp, "TP"


def run_backtest(
    *,
    config: AppConfig,
    asset: AssetConfig,
    candles_m5: list[Candle],
    assumed_spread: float = 0.2,
) -> BacktestReport:
    strategy_engine = StrategyEngine(config)
    risk_engine = RiskEngine(config.risk)
    candles_m15 = aggregate_candles(candles_m5, 15)
    candles_h1 = aggregate_candles(candles_m5, 60)

    by_time_m15 = {c.timestamp: i for i, c in enumerate(candles_m15)}
    by_time_h1 = {c.timestamp: i for i, c in enumerate(candles_h1)}

    equity = config.risk.equity
    peak_equity = equity
    max_drawdown = 0.0
    trades: list[BacktestTrade] = []
    pending: _PendingOrder | None = None
    open_pos: _OpenPosition | None = None
    time_in_market_bars = 0

    daily_trades: dict[str, int] = {}
    daily_pnl: dict[str, float] = {}

    start_idx = max(config.indicators.ema_period_h1 + 10, 250)
    for i in range(start_idx, len(candles_m5)):
        candle = candles_m5[i]
        day_key = candle.timestamp.date().isoformat()
        daily_trades.setdefault(day_key, 0)
        daily_pnl.setdefault(day_key, 0.0)

        if pending is not None and i > pending.expiry_index:
            pending = None

        if pending is not None:
            touched = pending.entry >= candle.low and pending.entry <= candle.high
            if touched:
                open_pos = _OpenPosition(
                    side=pending.side,
                    entry=pending.entry,
                    stop=pending.stop,
                    tp=pending.tp,
                    size=pending.size,
                    opened_at=candle.timestamp,
                )
                pending = None

        if open_pos is not None:
            time_in_market_bars += 1
            risk_dist = abs(open_pos.entry - open_pos.stop)
            if risk_dist > 0 and not open_pos.be_moved:
                one_r = open_pos.entry + risk_dist if open_pos.side == "LONG" else open_pos.entry - risk_dist
                reached_1r = candle.high >= one_r if open_pos.side == "LONG" else candle.low <= one_r
                if reached_1r:
                    half = open_pos.size * 0.5
                    open_pos.be_moved = True
                    open_pos.stop = open_pos.entry
                    open_pos.size = open_pos.size - half
                    open_pos.realized_partial += half * risk_dist

            should_close, exit_price, reason = _calc_exit(open_pos, candle)
            if should_close:
                if open_pos.side == "LONG":
                    remaining_pnl = (exit_price - open_pos.entry) * open_pos.size
                else:
                    remaining_pnl = (open_pos.entry - exit_price) * open_pos.size
                total_pnl = open_pos.realized_partial + remaining_pnl
                equity += total_pnl
                daily_pnl[day_key] += total_pnl
                r_denom = risk_engine.per_trade_risk_amount(equity=config.risk.equity)
                r_mult = (total_pnl / r_denom) if r_denom > 0 else 0.0
                trades.append(
                    BacktestTrade(
                        epic=asset.epic,
                        side=open_pos.side,
                        entry_time=open_pos.opened_at,
                        exit_time=candle.timestamp,
                        entry_price=open_pos.entry,
                        exit_price=exit_price,
                        size=open_pos.size,
                        pnl=total_pnl,
                        r_multiple=r_mult,
                        reason=reason,
                    )
                )
                open_pos = None
                peak_equity = max(peak_equity, equity)
                drawdown = peak_equity - equity
                max_drawdown = max(max_drawdown, drawdown)

        if open_pos is not None or pending is not None:
            continue

        if daily_trades[day_key] >= config.risk.max_trades_per_day:
            continue
        if risk_engine.should_turn_off_for_day(daily_pnl[day_key], equity=config.risk.equity):
            continue

        t = candle.timestamp
        m15_idx = max([idx for ts, idx in by_time_m15.items() if ts <= t], default=-1)
        h1_idx = max([idx for ts, idx in by_time_h1.items() if ts <= t], default=-1)
        if m15_idx <= 20 or h1_idx <= 50:
            continue

        slice_m5 = candles_m5[: i + 1]
        slice_m15 = candles_m15[: m15_idx + 1]
        slice_h1 = candles_h1[: h1_idx + 1]
        decision = strategy_engine.evaluate(
            epic=asset.epic,
            minimal_tick_buffer=asset.minimal_tick_buffer,
            candles_h1=slice_h1,
            candles_m15=slice_m15,
            candles_m5=slice_m5,
            current_spread=assumed_spread,
            spread_history=[assumed_spread] * config.spread_filter.window,
            news_blocked=False,
        )
        if decision.signal is None:
            continue

        size = position_size_from_risk(
            equity=config.risk.equity,
            risk_per_trade=config.risk.risk_per_trade,
            entry_price=decision.signal.entry_price,
            stop_price=decision.signal.stop_price,
            min_size=asset.min_size,
            size_step=asset.size_step,
        )
        if size <= 0:
            continue

        pending = _PendingOrder(
            side=decision.signal.side,
            entry=decision.signal.entry_price,
            stop=decision.signal.stop_price,
            tp=decision.signal.take_profit,
            size=size,
            expiry_index=i + config.execution.limit_ttl_bars,
            created_at=t,
        )
        daily_trades[day_key] += 1

    wins = sum(1 for trade in trades if trade.pnl > 0)
    losses = sum(1 for trade in trades if trade.pnl <= 0)
    total_pnl = sum(trade.pnl for trade in trades)
    trade_count = len(trades)
    expectancy = (total_pnl / trade_count) if trade_count else 0.0
    avg_r = (sum(trade.r_multiple for trade in trades) / trade_count) if trade_count else 0.0
    win_rate = (wins / trade_count) if trade_count else 0.0

    return BacktestReport(
        epic=asset.epic,
        trades=trade_count,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        expectancy=expectancy,
        avg_r=avg_r,
        max_drawdown=max_drawdown,
        time_in_market_bars=time_in_market_bars,
        equity_end=config.risk.equity + total_pnl,
        trade_log=trades,
    )


def run_backtest_from_csv(
    *,
    config: AppConfig,
    asset: AssetConfig,
    csv_path: str | Path,
    assumed_spread: float = 0.2,
) -> BacktestReport:
    candles = load_candles_csv(csv_path)
    return run_backtest(
        config=config,
        asset=asset,
        candles_m5=candles,
        assumed_spread=assumed_spread,
    )


def run_walk_forward_from_csv(
    *,
    config: AppConfig,
    asset: AssetConfig,
    csv_path: str | Path,
    wf_splits: int = 4,
    assumed_spread: float = 0.2,
) -> WalkForwardReport:
    candles = load_candles_csv(csv_path)
    if wf_splits < 2:
        wf_splits = 2
    chunk = len(candles) // wf_splits
    if chunk < 260:
        raise ValueError("Not enough candles for walk-forward splits")

    reports: list[BacktestReport] = []
    for split in range(wf_splits):
        start = split * chunk
        end = (split + 1) * chunk if split < (wf_splits - 1) else len(candles)
        part = candles[start:end]
        if len(part) < 260:
            continue
        reports.append(
            run_backtest(
                config=config,
                asset=asset,
                candles_m5=part,
                assumed_spread=assumed_spread,
            )
        )
    if not reports:
        raise ValueError("No valid walk-forward splits produced")

    total_trades = sum(report.trades for report in reports)
    total_wins = sum(report.wins for report in reports)
    total_losses = sum(report.losses for report in reports)
    total_pnl = sum(report.total_pnl for report in reports)
    total_time_in_market = sum(report.time_in_market_bars for report in reports)
    avg_r = (
        sum(report.avg_r * report.trades for report in reports) / total_trades
        if total_trades
        else 0.0
    )
    win_rate = (total_wins / total_trades) if total_trades else 0.0
    expectancy = (total_pnl / total_trades) if total_trades else 0.0
    aggregate = BacktestReport(
        epic=asset.epic,
        trades=total_trades,
        wins=total_wins,
        losses=total_losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        expectancy=expectancy,
        avg_r=avg_r,
        max_drawdown=max((report.max_drawdown for report in reports), default=0.0),
        time_in_market_bars=total_time_in_market,
        equity_end=config.risk.equity + total_pnl,
        trade_log=[],
    )
    return WalkForwardReport(epic=asset.epic, splits=reports, aggregate=aggregate)
