from __future__ import annotations

from bot.research.optimizer import (
    get_checkpoint_record,
    load_checkpoint,
    save_checkpoint,
    upsert_checkpoint_record,
)


def test_checkpoint_roundtrip_for_stage_records(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)

    stage_a_record = {"status": "done", "candidate_id": "A_off_1", "summary": {"objective_value": 1.0}}
    upsert_checkpoint_record(
        checkpoint,
        stage="A",
        candidate_id="A_off_1",
        record=stage_a_record,
    )
    save_checkpoint(checkpoint_path, checkpoint)

    restored = load_checkpoint(checkpoint_path)
    loaded_record = get_checkpoint_record(restored, stage="A", candidate_id="A_off_1")
    assert loaded_record is not None
    assert loaded_record["status"] == "done"
    assert loaded_record["summary"]["objective_value"] == 1.0


def test_per_capital_checkpoints_are_isolated(tmp_path) -> None:
    cap_usd = tmp_path / "capital_10000_USD" / "checkpoint.json"
    cap_pln = tmp_path / "capital_100_PLN" / "checkpoint.json"

    checkpoint_usd = load_checkpoint(cap_usd)
    checkpoint_pln = load_checkpoint(cap_pln)

    upsert_checkpoint_record(
        checkpoint_usd,
        stage="B",
        candidate_id="B_trend_1",
        record={"status": "done", "summary": {"objective_value": 11.0}},
    )
    upsert_checkpoint_record(
        checkpoint_pln,
        stage="B",
        candidate_id="B_trend_1",
        record={"status": "done", "summary": {"objective_value": 22.0}},
    )
    save_checkpoint(cap_usd, checkpoint_usd)
    save_checkpoint(cap_pln, checkpoint_pln)

    restored_usd = load_checkpoint(cap_usd)
    restored_pln = load_checkpoint(cap_pln)
    usd_record = get_checkpoint_record(restored_usd, stage="B", candidate_id="B_trend_1")
    pln_record = get_checkpoint_record(restored_pln, stage="B", candidate_id="B_trend_1")

    assert usd_record is not None
    assert pln_record is not None
    assert usd_record["summary"]["objective_value"] == 11.0
    assert pln_record["summary"]["objective_value"] == 22.0
