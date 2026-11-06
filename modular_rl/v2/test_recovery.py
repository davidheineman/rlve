import fcntl
import json

import pytest

from .recovery import archive_attempt


@pytest.mark.parametrize("committed,evaluated", [(-1, -1), (49, 49), (49, -1)])
def test_recovery_retains_committed_samples_and_archives_uncommitted_metrics(tmp_path, committed, evaluated):
    run = tmp_path / "run"
    checkpoint = run / "ckpt"
    checkpoint.mkdir(parents=True)
    (checkpoint / "rlve_checkpoint_state.json").write_text(json.dumps({
        "completed": False, "rollout_id": committed, "evaluated_rollout_id": evaluated,
        "data_source": "global_dataset"}))
    if committed >= 0:
        (checkpoint / f"iter_{committed:07d}").mkdir()
        (checkpoint / "latest_checkpointed_iteration.txt").write_text(str(committed))
        (checkpoint / "rollout").mkdir()
        for prefix in ("global_dataset_state_dict_", "runtime_state_"):
            (checkpoint / "rollout" / f"{prefix}{committed}.pt").write_bytes(b"saved state")
    (run / "wandb_run_id.txt").write_text("original-id")
    progress = "".join(json.dumps({"phase": "train", "completed_updates": n}) + "\n" for n in range(1, 61))
    progress += "".join(json.dumps({"phase": "eval", "completed_updates": n}) + "\n" for n in (0, 50))
    (run / "progress.jsonl").write_text(progress + '{"partial":')
    (run / "signal.jsonl").write_text("".join(json.dumps({"david/step": n}) + "\n" for n in range(60)))
    teacher_scores="".join(json.dumps({"rollout_id":n,"samples":[]})+'\n' for n in range(60))
    (run/'teacher-scores.jsonl').write_text(teacher_scores+'{"rollout_id":')
    for phase in ("train", "eval"):
        directory = run / "rollouts" / phase
        directory.mkdir(parents=True)
        for n in (0, 49, 59):
            (directory / f"{n:04d}.jsonl").write_text("sample\n")
    original_state = (checkpoint / "rlve_checkpoint_state.json").read_bytes()
    archive = run / "resume-audit/attempt"
    result = archive_attempt(run, archive)
    retained = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()]
    assert len([r for r in retained if r["phase"] == "train"]) == committed + 1
    assert [r["completed_updates"] for r in retained if r["phase"] == "eval"] == (
        [] if committed < 0 else ([0, 50] if evaluated >= 0 else [0]))
    assert (archive / "progress.jsonl").read_text() == progress + '{"partial":'
    assert [r['rollout_id'] for r in map(json.loads,(run/'teacher-scores.jsonl').read_text().splitlines())]==list(range(committed+1))
    assert (archive/'teacher-scores.jsonl').read_text()==teacher_scores+'{"rollout_id":'
    assert result["discarded_uncommitted_updates"] == 60 - committed - 1
    assert (run / "wandb_run_id.txt").exists() == (committed >= 0)
    assert (checkpoint / "rlve_checkpoint_state.json").read_bytes() == original_state
    for phase in ("train", "eval"):
        expected = [] if committed < 0 else (["0000"] if phase == "eval" and evaluated < 0 else ["0000", "0049"])
        assert sorted(p.stem for p in (run / "rollouts" / phase).glob("*.jsonl")) == expected
        assert (archive / "rollouts" / phase / "0059.jsonl").exists()


def test_recovery_refuses_a_run_with_a_live_worker_lock(tmp_path):
    lock_path = tmp_path / "run.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            archive_attempt(tmp_path, tmp_path / "archive")
    assert not (tmp_path / "archive").exists()
