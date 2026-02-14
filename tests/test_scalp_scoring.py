from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bot.strategy.scalp_ict_pa as scalp_module
from bot.config import AppConfig
from bot.data.candles import Candle
from bot.strategy.contracts import BiasState, DecisionAction, SetupCandidate, StrategyDataBundle
from bot.strategy.ict import FVGSignal, MSSSignal
from bot.strategy.scalp_ict_pa import ScalpIctPriceActionStrategy


def _candles(start: datetime, minutes: int, count: int, base: float, step: float) -> list[Candle]:
    items: list[Candle] = []
    value = base
    for i in range(count):
        ts = start + timedelta(minutes=i * minutes)
        open_price = value
        close = value + step + (0.02 if i % 3 == 0 else -0.01)
        high = max(open_price, close) + 0.03
        low = min(open_price, close) - 0.03
        items.append(Candle(timestamp=ts, open=open_price, high=high, low=low, close=close))
        value = close
    return items


def _bundle(
    *,
    symbol: str,
    now: datetime,
    candles_m15: list[Candle],
    candles_m5: list[Candle],
    strategy_params: dict | None = None,
    m5_new_close: bool = True,
) -> StrategyDataBundle:
    return StrategyDataBundle(
        symbol=symbol,
        now=now,
        candles_h1=[],
        candles_m15=candles_m15,
        candles_m5=candles_m5,
        spread=0.2,
        spread_history=[0.2] * 140,
        news_blocked=False,
        entry_state="WAIT",
        m15_new_close=True,
        m5_new_close=m5_new_close,
        extra={
            "minimal_tick_buffer": 0.05,
            "strategy_params": strategy_params or {},
        },
    )


def _candidate(*, symbol: str, now: datetime, candles_m5: list[Candle], side: str, setup_id: str) -> SetupCandidate:
    sweep_time = candles_m5[-40].timestamp.isoformat()
    return SetupCandidate(
        candidate_id=f"SCALP-{setup_id}",
        symbol=symbol,
        strategy_name="SCALP_ICT_PA",
        side=side,
        created_at=now,
        expires_at=now + timedelta(minutes=60),
        source_timeframe="M5",
        setup_type="SWEEP_REJECT",
        metadata={
            "setup_id": setup_id,
            "sweep_level": candles_m5[-40].low,
            "sweep_magnitude": 0.3,
            "sweep_time": sweep_time,
            "atr_m5": 0.2,
        },
    )


def test_scalp_scoring_is_deterministic() -> None:
    config = AppConfig()
    strategy = ScalpIctPriceActionStrategy(config)
    now = datetime(2026, 2, 11, 14, 0, tzinfo=timezone.utc)
    candles_m15 = _candles(now - timedelta(hours=40), 15, 220, 2000.0, 0.12)
    candles_m5 = _candles(now - timedelta(hours=20), 5, 700, 2010.0, 0.03)
    bundle = _bundle(symbol="GOLD", now=now, candles_m15=candles_m15, candles_m5=candles_m5)
    strategy.preprocess("GOLD", bundle)
    strategy.compute_bias("GOLD", bundle)
    candidate = _candidate(symbol="GOLD", now=now, candles_m5=candles_m5, side="LONG", setup_id="DETERMINISTIC")

    first = strategy.evaluate_candidate("GOLD", candidate, bundle)
    second = strategy.evaluate_candidate("GOLD", candidate, bundle)

    assert first.action == second.action
    assert first.score_total == second.score_total
    assert first.score_breakdown == second.score_breakdown
    assert first.reasons_blocking == second.reasons_blocking


def test_btc_h1_no_bos_is_soft_penalty_not_blocker(monkeypatch) -> None:
    config = AppConfig()
    strategy = ScalpIctPriceActionStrategy(config)
    now = datetime(2026, 2, 11, 15, 0, tzinfo=timezone.utc)
    candles_m15 = _candles(now - timedelta(hours=24), 15, 240, 50000.0, 5.0)
    candles_m5 = _candles(now - timedelta(hours=12), 5, 420, 50100.0, 1.5)
    mss_index = len(candles_m5) - 8
    candles_m5[mss_index] = Candle(
        timestamp=candles_m5[mss_index].timestamp,
        open=50080.0,
        high=50110.0,
        low=50070.0,
        close=50105.0,
    )
    bundle = _bundle(
        symbol="BTCUSD",
        now=now,
        candles_m15=candles_m15,
        candles_m5=candles_m5,
        strategy_params={
            "h1_bos_mode": "SCORE",
            "h1_no_bos_penalty": 10,
            "pd_filter_mode": "IGNORE",
            "allow_neutral_bias": False,
            "displacement_multiplier": 0.2,
            "fvg_min_atr": 0.0,
        },
    )
    strategy.preprocess("BTCUSD", bundle)
    strategy.compute_bias("BTCUSD", bundle)
    candidate = _candidate(symbol="BTCUSD", now=now, candles_m5=candles_m5, side="LONG", setup_id="SOFT-H1")

    def _fake_mss(*args, **kwargs) -> MSSSignal:
        return MSSSignal(
            side="LONG",
            broken_level=50090.0,
            source_swing_index=mss_index - 2,
            candle_index=mss_index,
            candle_time=candles_m5[mss_index].timestamp,
        )

    def _fake_fvg(*args, **kwargs) -> FVGSignal:
        return FVGSignal(
            side="LONG",
            c1_index=mss_index - 2,
            c2_index=mss_index - 1,
            c3_index=mss_index,
            lower=50095.0,
            upper=50100.0,
            midpoint=50097.5,
            timestamp=candles_m5[mss_index].timestamp,
        )

    monkeypatch.setattr(scalp_module, "detect_mss", _fake_mss)
    monkeypatch.setattr(scalp_module, "detect_latest_fvg", _fake_fvg)

    evaluation = strategy.evaluate_candidate("BTCUSD", candidate, bundle)

    assert evaluation.action in {DecisionAction.TRADE, DecisionAction.SMALL}
    assert "H1_NO_BOS" not in evaluation.reasons_blocking
    assert "penalty_h1_no_bos" in evaluation.score_breakdown


def test_btc_neutral_bias_allows_small_with_risk_override(monkeypatch) -> None:
    config = AppConfig()
    strategy = ScalpIctPriceActionStrategy(config)
    now = datetime(2026, 2, 11, 15, 30, tzinfo=timezone.utc)
    candles_m15 = _candles(now - timedelta(hours=20), 15, 220, 50000.0, 1.2)
    candles_m5 = _candles(now - timedelta(hours=10), 5, 360, 50100.0, 1.1)
    mss_index = len(candles_m5) - 6
    candles_m5[mss_index] = Candle(
        timestamp=candles_m5[mss_index].timestamp,
        open=50080.0,
        high=50130.0,
        low=50070.0,
        close=50120.0,
    )
    bundle = _bundle(
        symbol="BTCUSD",
        now=now,
        candles_m15=candles_m15,
        candles_m5=candles_m5,
        strategy_params={
            "allow_neutral_bias": True,
            "neutral_bias_trade_mode": "SMALL",
            "neutral_bias_risk_multiplier": 0.35,
            "h1_bos_mode": "SCORE",
            "pd_filter_mode": "IGNORE",
            "displacement_multiplier": 0.2,
            "fvg_min_atr": 0.0,
        },
    )
    state = strategy._state("BTCUSD")
    state.bias = BiasState(
        symbol="BTCUSD",
        strategy_name="SCALP_ICT_PA",
        direction="NEUTRAL",
        timeframe="M15",
        updated_at=now,
        metadata={
            "h1_bos_state": "NONE",
            "h1_pd_available": False,
        },
    )
    candidate = _candidate(symbol="BTCUSD", now=now, candles_m5=candles_m5, side="LONG", setup_id="NEUTRAL-SMALL")
    candidate.metadata["sweep_magnitude"] = 5000.0

    def _fake_mss(*args, **kwargs) -> MSSSignal:
        return MSSSignal(
            side="LONG",
            broken_level=50090.0,
            source_swing_index=mss_index - 2,
            candle_index=mss_index,
            candle_time=candles_m5[mss_index].timestamp,
        )

    def _fake_fvg(*args, **kwargs) -> FVGSignal:
        return FVGSignal(
            side="LONG",
            c1_index=mss_index - 2,
            c2_index=mss_index - 1,
            c3_index=mss_index,
            lower=50090.0,
            upper=50100.0,
            midpoint=50095.0,
            timestamp=candles_m5[mss_index].timestamp,
        )

    monkeypatch.setattr(scalp_module, "detect_mss", _fake_mss)
    monkeypatch.setattr(scalp_module, "detect_latest_fvg", _fake_fvg)

    evaluation = strategy.evaluate_candidate("BTCUSD", candidate, bundle)

    assert evaluation.action == DecisionAction.SMALL
    assert evaluation.metadata.get("risk_multiplier_override") == 0.35
    assert "H1_BIAS_NEUTRAL" not in evaluation.reasons_blocking


def test_missed_opportunity_dedupes_same_key(monkeypatch) -> None:
    config = AppConfig()
    strategy = ScalpIctPriceActionStrategy(config)
    now = datetime(2026, 2, 11, 16, 0, tzinfo=timezone.utc)
    candles_m15 = _candles(now - timedelta(hours=20), 15, 220, 30000.0, 2.0)
    candles_m5 = _candles(now - timedelta(hours=8), 5, 280, 30100.0, 0.2)
    bundle = _bundle(symbol="BTCUSD", now=now, candles_m15=candles_m15, candles_m5=candles_m5)
    state = strategy._state("BTCUSD")
    state.bias = BiasState(
        symbol="BTCUSD",
        strategy_name="SCALP_ICT_PA",
        direction="LONG",
        timeframe="M15",
        updated_at=now,
        metadata={},
    )
    candidate = _candidate(symbol="BTCUSD", now=now, candles_m5=candles_m5, side="LONG", setup_id="DEDUPE")

    monkeypatch.setattr(scalp_module, "detect_mss", lambda *args, **kwargs: None)
    monkeypatch.setattr(scalp_module, "detect_latest_fvg", lambda *args, **kwargs: None)

    strategy.evaluate_candidate("BTCUSD", candidate, bundle)
    strategy.evaluate_candidate("BTCUSD", candidate, bundle)

    assert len(state.probes) == 1


def test_fallback_setup_forces_small_and_caps_risk(monkeypatch) -> None:
    config = AppConfig()
    strategy = ScalpIctPriceActionStrategy(config)
    now = datetime(2026, 2, 11, 16, 20, tzinfo=timezone.utc)
    candles_m15 = _candles(now - timedelta(hours=24), 15, 240, 2300.0, 0.1)
    candles_m5 = _candles(now - timedelta(hours=12), 5, 420, 2310.0, 0.04)
    anchor_idx = len(candles_m5) - 12
    mss_index = anchor_idx + 2
    candles_m5[mss_index] = Candle(
        timestamp=candles_m5[mss_index].timestamp,
        open=2311.0,
        high=2312.5,
        low=2310.8,
        close=2312.4,
    )
    bundle = _bundle(
        symbol="XAUUSD",
        now=now,
        candles_m15=candles_m15,
        candles_m5=candles_m5,
        strategy_params={
            "allow_neutral_bias": False,
            "h1_bos_mode": "IGNORE",
            "pd_filter_mode": "IGNORE",
            "displacement_multiplier": 0.2,
            "fvg_min_atr": 0.0,
            "fallback_score_penalty": 2.0,
            "fallback_force_small": True,
            "fallback_small_risk_multiplier": 0.3,
            "fallback_min_displacement_ratio": 1.1,
            "fallback_max_trigger_bars": 10,
        },
    )
    state = strategy._state("XAUUSD")
    state.bias = BiasState(
        symbol="XAUUSD",
        strategy_name="SCALP_ICT_PA",
        direction="LONG",
        timeframe="M15",
        updated_at=now,
        metadata={},
    )
    candidate = SetupCandidate(
        candidate_id="SCALP-FALLBACK-FORCE-SMALL",
        symbol="XAUUSD",
        strategy_name="SCALP_ICT_PA",
        side="LONG",
        created_at=now,
        expires_at=now + timedelta(minutes=45),
        source_timeframe="M5",
        setup_type="DISPLACEMENT_RETEST",
        metadata={
            "setup_id": "XAUUSD:LONG:FB:test",
            "setup_origin": "DISPLACEMENT_RETEST",
            "fallback_anchor_index": anchor_idx,
            "sweep_level": candles_m5[anchor_idx].low,
            "sweep_magnitude": 4.0,
            "sweep_time": candles_m5[anchor_idx - 1].timestamp.isoformat(),
            "atr_m5": 0.2,
        },
    )

    def _fake_mss(*args, **kwargs) -> MSSSignal:
        return MSSSignal(
            side="LONG",
            broken_level=2312.0,
            source_swing_index=mss_index - 2,
            candle_index=mss_index,
            candle_time=candles_m5[mss_index].timestamp,
        )

    def _fake_fvg(*args, **kwargs) -> FVGSignal:
        return FVGSignal(
            side="LONG",
            c1_index=mss_index - 2,
            c2_index=mss_index - 1,
            c3_index=mss_index,
            lower=2311.2,
            upper=2311.8,
            midpoint=2311.5,
            timestamp=candles_m5[mss_index].timestamp,
        )

    monkeypatch.setattr(scalp_module, "detect_mss", _fake_mss)
    monkeypatch.setattr(scalp_module, "detect_latest_fvg", _fake_fvg)

    evaluation = strategy.evaluate_candidate("XAUUSD", candidate, bundle)

    assert evaluation.action == DecisionAction.SMALL
    assert evaluation.metadata.get("risk_multiplier_override") == 0.3
    assert evaluation.metadata.get("fallback_forced_small") is True


def test_detect_candidates_creates_displacement_fallback_when_no_sweep(monkeypatch) -> None:
    config = AppConfig()
    strategy = ScalpIctPriceActionStrategy(config)
    now = datetime(2026, 2, 11, 16, 10, tzinfo=timezone.utc)
    candles_m15 = _candles(now - timedelta(hours=30), 15, 220, 2300.0, 0.05)
    candles_m5 = _candles(now - timedelta(hours=8), 5, 220, 2300.0, 0.01)
    anchor_idx = len(candles_m5) - 8
    candles_m5[anchor_idx] = Candle(
        timestamp=candles_m5[anchor_idx].timestamp,
        open=2300.0,
        high=2301.1,
        low=2299.9,
        close=2301.0,
    )

    bundle = _bundle(
        symbol="XAUUSD",
        now=now,
        candles_m15=candles_m15,
        candles_m5=candles_m5,
        strategy_params={
            "displacement_retest_fallback": True,
            "fallback_min_displacement_atr": 0.6,
            "fallback_lookback_bars": 30,
            "candidate_min_interval_minutes": 0,
        },
    )
    state = strategy._state("XAUUSD")
    state.bias = BiasState(
        symbol="XAUUSD",
        strategy_name="SCALP_ICT_PA",
        direction="LONG",
        timeframe="M15",
        updated_at=now,
        metadata={},
    )
    monkeypatch.setattr(scalp_module, "detect_sweep_reject", lambda *args, **kwargs: None)

    out = strategy.detect_candidates("XAUUSD", bundle)

    assert out
    fallback = next((item for item in out if item.setup_type == "DISPLACEMENT_RETEST"), None)
    assert fallback is not None
    assert fallback.metadata.get("setup_origin") == "DISPLACEMENT_RETEST"
    assert fallback.metadata.get("fallback_anchor_index") is not None
    assert fallback.metadata.get("sweep_time") is not None
    assert fallback.metadata.get("sweep_level") is not None


def test_miss_rate_is_hits_divided_by_probes() -> None:
    config = AppConfig()
    strategy = ScalpIctPriceActionStrategy(config)
    now = datetime(2026, 2, 11, 16, 30, tzinfo=timezone.utc)
    candles_m15 = _candles(now - timedelta(hours=20), 15, 220, 1000.0, 0.5)
    candles_m5 = [
        Candle(timestamp=now - timedelta(minutes=10), open=100.0, high=100.5, low=99.5, close=100.1),
        Candle(timestamp=now - timedelta(minutes=5), open=100.1, high=102.5, low=99.8, close=101.0),
        Candle(timestamp=now, open=101.0, high=101.5, low=100.5, close=101.2),
    ]
    bundle = _bundle(symbol="BTCUSD", now=now, candles_m15=candles_m15, candles_m5=candles_m5)
    state = strategy._state("BTCUSD")

    strategy._register_miss_probe(
        symbol="BTCUSD",
        setup_id="MISS-HIT",
        direction="LONG",
        start_price=100.0,
        start_ts=now - timedelta(minutes=5),
        atr_value=1.0,
        data=bundle,
    )
    strategy._register_miss_probe(
        symbol="BTCUSD",
        setup_id="MISS-EXPIRE",
        direction="LONG",
        start_price=120.0,
        start_ts=now - timedelta(minutes=11),
        atr_value=1.0,
        data=bundle,
    )
    state.probes[1].expires_at = now - timedelta(seconds=1)

    strategy.preprocess("BTCUSD", bundle)
    hits, probes = strategy.missed_opportunity_stats("BTCUSD")

    assert hits == 1
    assert probes == 2
    assert strategy.missed_opportunity_rate("BTCUSD") == 0.5


def test_missed_opportunity_dedupes_semantic_context_across_setup_ids() -> None:
    config = AppConfig()
    strategy = ScalpIctPriceActionStrategy(config)
    now = datetime(2026, 2, 11, 16, 40, tzinfo=timezone.utc)
    candles_m15 = _candles(now - timedelta(hours=20), 15, 220, 1000.0, 0.5)
    candles_m5 = [
        Candle(timestamp=now - timedelta(minutes=10), open=100.0, high=100.4, low=99.8, close=100.2),
        Candle(timestamp=now - timedelta(minutes=5), open=100.2, high=100.6, low=100.0, close=100.3),
        Candle(timestamp=now, open=100.3, high=100.7, low=100.1, close=100.4),
    ]
    bundle = _bundle(symbol="XAUUSD", now=now, candles_m15=candles_m15, candles_m5=candles_m5)
    state = strategy._state("XAUUSD")

    strategy._register_miss_probe(
        symbol="XAUUSD",
        setup_id="SETUP-A",
        direction="LONG",
        start_price=100.0,
        start_ts=now - timedelta(minutes=5),
        atr_value=1.0,
        data=bundle,
    )
    strategy._register_miss_probe(
        symbol="XAUUSD",
        setup_id="SETUP-B",
        direction="LONG",
        start_price=100.0,
        start_ts=now - timedelta(minutes=5),
        atr_value=1.0,
        data=bundle,
    )

    assert len(state.probes) == 1


def test_fvg_detection_uses_sweep_window_not_only_mss_window(monkeypatch) -> None:
    config = AppConfig()
    strategy = ScalpIctPriceActionStrategy(config)
    now = datetime(2026, 2, 11, 18, 0, tzinfo=timezone.utc)
    candles_m15 = _candles(now - timedelta(hours=20), 15, 220, 2500.0, -0.2)
    candles_m5 = _candles(now - timedelta(hours=10), 5, 300, 2500.0, -0.05)

    # Bearish FVG appears earlier after sweep, before later MSS candle.
    fvg_c1 = 120
    candles_m5[fvg_c1] = Candle(
        timestamp=candles_m5[fvg_c1].timestamp,
        open=2498.0,
        high=2499.0,
        low=2496.0,
        close=2496.5,
    )
    candles_m5[fvg_c1 + 1] = Candle(
        timestamp=candles_m5[fvg_c1 + 1].timestamp,
        open=2496.5,
        high=2497.0,
        low=2493.0,
        close=2493.5,
    )
    candles_m5[fvg_c1 + 2] = Candle(
        timestamp=candles_m5[fvg_c1 + 2].timestamp,
        open=2493.2,
        high=2495.0,  # < c1.low(2496.0) -> valid SHORT FVG
        low=2490.5,
        close=2491.0,
    )

    mss_index = 150
    candles_m5[mss_index] = Candle(
        timestamp=candles_m5[mss_index].timestamp,
        open=2492.0,
        high=2492.5,
        low=2486.0,
        close=2486.5,
    )

    bundle = _bundle(
        symbol="GOLD",
        now=now,
        candles_m15=candles_m15,
        candles_m5=candles_m5,
        strategy_params={
            "displacement_multiplier": 0.2,
            "fvg_min_atr": 0.0,
            "allow_neutral_bias": False,
            "h1_bos_mode": "IGNORE",
            "pd_filter_mode": "IGNORE",
        },
    )

    state = strategy._state("GOLD")
    state.bias = BiasState(
        symbol="GOLD",
        strategy_name="SCALP_ICT_PA",
        direction="SHORT",
        timeframe="M15",
        updated_at=now,
        metadata={},
    )
    candidate = _candidate(symbol="GOLD", now=now, candles_m5=candles_m5, side="SHORT", setup_id="SWEEP-FVG-WINDOW")
    candidate.metadata["sweep_time"] = candles_m5[100].timestamp.isoformat()
    candidate.metadata["sweep_level"] = candles_m5[100].high

    def _fake_mss(*args, **kwargs) -> MSSSignal:
        return MSSSignal(
            side="SHORT",
            broken_level=2488.0,
            source_swing_index=mss_index - 3,
            candle_index=mss_index,
            candle_time=candles_m5[mss_index].timestamp,
        )

    monkeypatch.setattr(scalp_module, "detect_mss", _fake_mss)

    evaluation = strategy.evaluate_candidate("GOLD", candidate, bundle)

    assert "SCALP_NO_FVG" not in evaluation.reasons_blocking
    assert evaluation.metadata.get("fvg_detected") is True
    assert evaluation.metadata.get("fvg_mid") is not None
