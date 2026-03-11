"""Debug/diagnostic collection and writing — extracted from engine.py."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from bot.backtest.models import _ExecutionFailSample, _NoPriceSample, _ReactionTimeoutSample
from bot.diagnostics.decision_trace import DecisionTraceWriter
from bot.strategy.contracts import StrategyEvaluation


def _collect_execution_fail_sample(
    *,
    samples: list[_ExecutionFailSample],
    max_samples: int,
    ts: datetime,
    symbol: str,
    strategy: str,
    reason: str,
    evaluation: StrategyEvaluation,
) -> None:
    if len(samples) >= max_samples:
        return
    spread_ratio = evaluation.metadata.get("spread_ratio")
    atr_value = evaluation.metadata.get("atr_m5", evaluation.snapshot.get("atr_m5"))
    try:
        spread_ratio_float = float(spread_ratio) if spread_ratio is not None else None
    except (TypeError, ValueError):
        spread_ratio_float = None
    try:
        atr_float = float(atr_value) if atr_value is not None else None
    except (TypeError, ValueError):
        atr_float = None
    samples.append(
        _ExecutionFailSample(
            ts_utc=ts.isoformat(),
            symbol=symbol,
            strategy=strategy,
            reason=reason,
            spread_ratio=spread_ratio_float,
            atr_m5=atr_float,
            missing_features=[
                str(item) for item in evaluation.metadata.get("missing_features", []) if str(item).strip()
            ]
            if isinstance(evaluation.metadata.get("missing_features"), list)
            else [],
        )
    )


def _build_decision_trace_record(
    *,
    ts: datetime,
    symbol: str,
    strategy: str,
    evaluation: StrategyEvaluation,
    order_request: Any | None,
    spread_points: float,
    fate: str,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a single decision-trace JSONL record for a candidate that reached
    the order-evaluation stage (i.e. had order_request != None)."""
    meta = evaluation.metadata if isinstance(evaluation.metadata, dict) else {}
    layers = evaluation.score_layers or {}
    penalties = evaluation.penalties or {}
    entry = float(order_request.entry_price) if order_request is not None else None
    stop = float(order_request.stop_price) if order_request is not None else None
    tp = float(order_request.take_profit) if order_request is not None else None
    side = str(order_request.side) if order_request is not None else str(meta.get("side", ""))
    return {
        "ts_utc": ts.isoformat(),
        "symbol": symbol,
        "strategy": strategy,
        "side": side,
        "score_total": round(float(evaluation.score_total), 4) if evaluation.score_total is not None else None,
        "action": str(evaluation.action.value) if evaluation.action else "OBSERVE",
        "entry": entry,
        "stop": stop,
        "tp": tp,
        "spread_points": round(spread_points, 2),
        "expected_rr": meta.get("expected_rr"),
        "layers": {k: round(float(v), 2) for k, v in layers.items()},
        "penalties": {k: round(float(v), 2) for k, v in penalties.items()},
        "reasons_blocking": list(evaluation.reasons_blocking) if evaluation.reasons_blocking else [],
        "fate": fate,
        "detail": detail or {},
    }


def _write_decision_trace(path: Path | None, records: list[dict[str, Any]]) -> None:
    if path is None or not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")


def _emit_decision(
    dtw: DecisionTraceWriter,
    *,
    ts: datetime,
    symbol: str,
    tf: str = "5m",
    candidates: int = 0,
    signal: str | None = None,
    features_ok: bool = True,
    missing: list[str] | None = None,
    score: float | None = None,
    threshold: float | None = None,
    evaluation: Any | None = None,
    reject_reason: str | None = None,
    spread_points: float | None = None,
    session_ok: bool = True,
    size_raw: float | None = None,
    size_final: float | None = None,
    min_lot: float = 0.01,
    lot_step: float = 0.01,
    margin_capped: bool = False,
    cooldown_active: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit one decision event via the live-flushing writer."""
    if not dtw.active:
        return
    breakdown: list[dict[str, Any]] = []
    if evaluation is not None:
        layers = getattr(evaluation, "score_layers", None) or {}
        penalties = getattr(evaluation, "penalties", None) or {}
        for k, v in layers.items():
            breakdown.append({"k": str(k), "v": round(float(v), 2)})
        for k, v in penalties.items():
            breakdown.append({"k": f"penalty_{k}", "v": round(float(v), 2)})
    dtw.decision(
        ts=ts,
        symbol=symbol,
        tf=tf,
        candidates=candidates,
        signal=signal,
        features_ok=features_ok,
        missing=missing,
        score=score,
        threshold=threshold,
        score_breakdown=breakdown or None,
        reject_reason=reject_reason,
        spread_points=spread_points,
        session_ok=session_ok,
        size_raw=size_raw,
        size_final=size_final,
        min_lot=min_lot,
        lot_step=lot_step,
        margin_capped=margin_capped,
        cooldown_active=cooldown_active,
        extra=extra,
    )


def _emit_fill(
    dtw: DecisionTraceWriter,
    *,
    ts: datetime,
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
    if not dtw.active:
        return
    dtw.fill(
        ts=ts,
        symbol=symbol,
        side=side,
        pnl=pnl,
        equity_after=equity_after,
        reason_close=reason_close,
        holding_min=holding_min,
        spread_cost=spread_cost,
        swap_cost=swap_cost,
        extra=extra,
    )


def _write_execution_fail_debug(path: Path | None, samples: list[_ExecutionFailSample]) -> None:
    if path is None or not samples:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = {
                "ts_utc": sample.ts_utc,
                "symbol": sample.symbol,
                "strategy": sample.strategy,
                "reason": sample.reason,
                "spread_ratio": sample.spread_ratio,
                "atr_m5": sample.atr_m5,
                "missing_features": sample.missing_features,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _collect_no_price_sample(
    *,
    samples: list[_NoPriceSample],
    max_samples: int,
    ts: datetime,
    symbol: str,
    strategy: str,
    evaluation: StrategyEvaluation,
    data_context: dict[str, Any] | None,
) -> None:
    if len(samples) >= max_samples:
        return
    ctx = data_context or {}
    snapshot = evaluation.snapshot if isinstance(evaluation.snapshot, dict) else {}
    metadata = evaluation.metadata if isinstance(evaluation.metadata, dict) else {}
    missing_fields: list[str] = []
    for field_name, value in (
        ("snapshot.spread", snapshot.get("spread", metadata.get("spread"))),
        ("snapshot.close", snapshot.get("close", metadata.get("close"))),
        ("metadata.atr_m5", metadata.get("atr_m5", snapshot.get("atr_m5"))),
    ):
        if value is None or (isinstance(value, float) and value != value):
            missing_fields.append(field_name)

    record = {
        "close": snapshot.get("close", metadata.get("close")),
        "spread": snapshot.get("spread", metadata.get("spread")),
        "atr_m5": metadata.get("atr_m5", snapshot.get("atr_m5")),
        "spread_ratio": metadata.get("spread_ratio"),
        "bid": metadata.get("bid"),
        "ask": metadata.get("ask"),
        "price_mode_requested": ctx.get("price_mode_requested"),
    }
    source_files_raw = ctx.get("source_files")
    source_files = source_files_raw if isinstance(source_files_raw, list) else []
    source_datasets_raw = ctx.get("source_datasets")
    source_datasets = source_datasets_raw if isinstance(source_datasets_raw, list) else []
    timeframe = str(ctx.get("timeframe") or "5m")
    price_mode = str(ctx.get("price_mode_requested") or metadata.get("price_mode") or "unknown")
    samples.append(
        _NoPriceSample(
            ts_utc=ts.isoformat(),
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            price_mode=price_mode,
            missing_fields=missing_fields,
            source_files=[str(item) for item in source_files],
            source_datasets=[str(item) for item in source_datasets],
            record=record,
        )
    )


def _write_no_price_debug(path: Path | None, samples: list[_NoPriceSample]) -> None:
    if path is None or not samples:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = {
                "ts_utc": sample.ts_utc,
                "symbol": sample.symbol,
                "timeframe": sample.timeframe,
                "strategy": sample.strategy,
                "price_mode": sample.price_mode,
                "missing_fields": sample.missing_fields,
                "source_files": sample.source_files,
                "source_datasets": sample.source_datasets,
                "record": sample.record,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _write_reaction_timeout_debug(path: Path | None, samples: list[_ReactionTimeoutSample]) -> None:
    if path is None or not samples:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = {
                "ts_utc": sample.ts_utc,
                "symbol": sample.symbol,
                "strategy": sample.strategy,
                "state": sample.state,
                "waited_bars": sample.waited_bars,
                "reason": sample.reason,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
