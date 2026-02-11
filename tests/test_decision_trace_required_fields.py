from __future__ import annotations

import json
from datetime import datetime, timezone

from bot.strategy.trace import DecisionTrace, trace_to_json


def test_decision_trace_contains_required_multi_strategy_fields() -> None:
    trace = DecisionTrace(
        asset="XAUUSD",
        created_at=datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc),
        strategy_name="SCALP_ICT_PA",
        score_total=74.5,
        score_breakdown={"bias": 20.0, "mss": 15.0},
        reasons_blocking=["SCALP_NO_FVG"],
        would_enter_if=["FVG"],
        snapshot={"spread": 0.2, "atr_m5": 1.4},
        reasons=["SCALP_NO_FVG"],
        final_decision="NO_SIGNAL",
    )
    payload = json.loads(trace_to_json(trace, "Europe/Warsaw"))

    assert payload["asset"] == "XAUUSD"
    assert payload["strategy_name"] == "SCALP_ICT_PA"
    assert payload["score_total"] == 74.5
    assert payload["score_breakdown"]["bias"] == 20.0
    assert payload["reasons_blocking"] == ["SCALP_NO_FVG"]
    assert payload["would_enter_if"] == ["FVG"]
    assert payload["snapshot"]["spread"] == 0.2

