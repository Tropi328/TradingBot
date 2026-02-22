from __future__ import annotations

from bot.config import ResearchSearchSpaceConfig
from bot.research.optimizer import build_stage_a_gate_candidates, build_stage_b_candidates


def test_stage_a_deep_search_space_has_expected_count() -> None:
    cfg = ResearchSearchSpaceConfig()
    stage_a = build_stage_a_gate_candidates(
        search_space_gate=cfg.gate.model_dump(),
        runtime_budget="deep",
    )
    assert len(stage_a) == 82


def test_stage_a_order_is_deterministic() -> None:
    cfg = ResearchSearchSpaceConfig()
    first = build_stage_a_gate_candidates(search_space_gate=cfg.gate.model_dump(), runtime_budget="deep")
    second = build_stage_a_gate_candidates(search_space_gate=cfg.gate.model_dump(), runtime_budget="deep")
    assert [item["candidate_id"] for item in first] == [item["candidate_id"] for item in second]


def test_stage_b_count_matches_top_gate_x_risk_profiles() -> None:
    cfg = ResearchSearchSpaceConfig()
    stage_a = build_stage_a_gate_candidates(search_space_gate=cfg.gate.model_dump(), runtime_budget="deep")
    top_gate = stage_a[:10]
    stage_b = build_stage_b_candidates(
        top_gate_candidates=top_gate,
        risk_profiles=[profile.model_dump() for profile in cfg.risk_profiles],
        runtime_budget="deep",
    )
    assert len(stage_b) == 120
