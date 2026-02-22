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
