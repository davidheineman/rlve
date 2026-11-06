import json
import shutil

import pytest

from .snapshot import preserve, validate_snapshot


@pytest.fixture
def active_checkpoint(tmp_path):
    run = tmp_path / "Qwen3-1.7B-math"
    iteration = run / "ckpt/iter_0000049"
    iteration.mkdir(parents=True)
    (iteration / "weights.distcp").write_bytes(b"trained weights")
    (run / "ckpt/latest_checkpointed_iteration.txt").write_text("49")
    (run / "ckpt/rlve_checkpoint_state.json").write_text(json.dumps({
        "rollout_id": 49, "evaluated_rollout_id": 49, "completed": False,
    }))
    (run / "progress.jsonl").write_text(json.dumps({
        "phase": "eval", "completed_updates": 50, "train_rollouts": 400, "accuracy": .75,
    }) + "\n")
    return run


def test_snapshot_survives_pruning_and_keeps_partial_training_budget(active_checkpoint, tmp_path):
    run, saved = active_checkpoint, tmp_path / "saved"
    value = preserve(run, saved, 50)
    source_weights = run / "ckpt/iter_0000049/weights.distcp"
    saved_weights = saved / "iter_0000049/weights.distcp"
    assert source_weights.stat().st_ino == saved_weights.stat().st_ino
    assert value["completed_updates"] == 50 and value["train_rollouts"] == 400
    assert not value["training_complete"]
    assert not json.loads((saved / "rlve_checkpoint_state.json").read_text())["completed"]
    shutil.rmtree(run / "ckpt/iter_0000049")
    (run / "ckpt/latest_checkpointed_iteration.txt").write_text("99")
    (run / "ckpt/rlve_checkpoint_state.json").write_text("{}")
    assert saved_weights.read_bytes() == b"trained weights"
    assert validate_snapshot(saved)["rollout_id"] == 49


def test_snapshot_rejects_checkpoint_without_matching_evaluation(active_checkpoint, tmp_path):
    path = active_checkpoint / "ckpt/rlve_checkpoint_state.json"
    state = json.loads(path.read_text())
    state["evaluated_rollout_id"] = 48
    path.write_text(json.dumps(state))
    destination = tmp_path / "saved"
    with pytest.raises(ValueError, match="committed and evaluated"):
        preserve(active_checkpoint, destination, 50)
    assert not destination.exists()


def test_snapshot_rejects_changed_weights(active_checkpoint, tmp_path):
    destination = tmp_path / "saved"
    preserve(active_checkpoint, destination, 50)
    (destination / "iter_0000049/weights.distcp").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="checkpoint changed"):
        validate_snapshot(destination)
