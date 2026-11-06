"""Commit model and rollout checkpoints together for restartable RLVE jobs."""

import json
import os
import re
import shutil
from pathlib import Path
from uuid import uuid4

STATE_FILE = "rlve_checkpoint_state.json"
TRACKER_FILE = "latest_checkpointed_iteration.txt"
ROLLOUT_PREFIXES = (
    "rlve_manager_state_dict_",
    "global_dataset_state_dict_",
    "runtime_state_",
)


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_checkpoint_state(directory):
    return json.loads((Path(directory) / STATE_FILE).read_text())


def write_checkpoint_state(directory, state):
    atomic_write(Path(directory) / STATE_FILE, json.dumps(state, sort_keys=True) + "\n")


def validate_checkpoint(directory, rollout_id):
    directory = Path(directory)
    source = read_checkpoint_state(directory).get("data_source", "rlve")
    if source not in ("rlve", "global_dataset"):
        raise ValueError(f"Unknown checkpoint data source: {source}")
    required = [directory / f"iter_{rollout_id:07d}"]
    required.extend(
        directory / "rollout" / f"{prefix}{rollout_id}.pt"
        for prefix in ROLLOUT_PREFIXES
        if source == "rlve" or prefix != "rlve_manager_state_dict_"
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Incomplete committed checkpoint: {missing}")


def remove_checkpoint(directory, rollout_id):
    directory = Path(directory)
    model_path = directory / f"iter_{rollout_id:07d}"
    if model_path.exists():
        shutil.rmtree(model_path)
    for prefix in ROLLOUT_PREFIXES:
        (directory / "rollout" / f"{prefix}{rollout_id}.pt").unlink(missing_ok=True)


def checkpoint_iterations(directory):
    return sorted(
        int(match.group(1))
        for path in Path(directory).iterdir()
        if (match := re.fullmatch(r"iter_(\d+)", path.name))
    )


def begin_final_only_attempt(run_directory):
    """Give an interrupted final-only run fresh model, optimizer, and W&B identity."""
    run_directory = Path(run_directory)
    checkpoint = run_directory / "ckpt"
    attempt_file = run_directory / "attempt.txt"
    identity = run_directory / "wandb_run_id.txt"
    state_file = checkpoint / STATE_FILE
    previous_attempt = int(attempt_file.read_text()) if attempt_file.exists() else 1
    if previous_attempt < 1:
        raise ValueError("Attempt number must be positive")
    if state_file.exists() and not json.loads(state_file.read_text())["completed"]:
        new_attempt = previous_attempt + 1
        atomic_write(attempt_file, f"{new_attempt}\n")
        shutil.rmtree(checkpoint)
        identity.unlink(missing_ok=True)
        return new_attempt, True
    if not attempt_file.exists():
        atomic_write(attempt_file, "1\n")
    return previous_attempt, False


def prepare_checkpoint(directory, data_source="rlve"):
    directory = Path(directory)
    if data_source not in ("rlve", "global_dataset"):
        raise ValueError(f"Unknown checkpoint data source: {data_source}")
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / STATE_FILE).exists():
        if (directory / TRACKER_FILE).exists() or checkpoint_iterations(directory):
            raise RuntimeError(
                "Refusing to initialize transactional state over an unmanaged checkpoint"
            )
        write_checkpoint_state(
            directory,
            {"rollout_id": -1, "evaluated_rollout_id": -1,
             "completed": False, "data_source": data_source},
        )
    state = read_checkpoint_state(directory)
    if state.get("data_source", "rlve") != data_source:
        raise RuntimeError("Checkpoint data source differs from the requested source")
    rollout_id = state["rollout_id"]
    if rollout_id >= 0:
        validate_checkpoint(directory, rollout_id)
    for iteration in checkpoint_iterations(directory):
        if iteration > rollout_id:
            remove_checkpoint(directory, iteration)
    if rollout_id < 0:
        (directory / TRACKER_FILE).unlink(missing_ok=True)
    else:
        atomic_write(directory / TRACKER_FILE, str(rollout_id))
    return state


def commit_checkpoint(directory, rollout_id):
    keep = int(os.environ.get("RLVE_CHECKPOINT_KEEP", "2"))
    if keep < 1:
        raise ValueError("RLVE_CHECKPOINT_KEEP must be at least 1")
    validate_checkpoint(directory, rollout_id)
    if int((Path(directory) / TRACKER_FILE).read_text()) != rollout_id:
        raise RuntimeError("Model checkpoint tracker does not match the rollout state")
    state = read_checkpoint_state(directory)
    if rollout_id <= state["rollout_id"]:
        raise RuntimeError("Checkpoint commits must advance training")
    state.update(rollout_id=rollout_id, completed=False)
    write_checkpoint_state(directory, state)
    for iteration in checkpoint_iterations(directory)[:-keep]:
        remove_checkpoint(directory, iteration)


def mark_evaluated(directory, rollout_id):
    state = read_checkpoint_state(directory)
    if state["rollout_id"] == rollout_id:
        state["evaluated_rollout_id"] = rollout_id
        write_checkpoint_state(directory, state)


def mark_early_stopped(directory, rollout_id, val_accuracy, reason="accuracy"):
    state = read_checkpoint_state(directory)
    if state["rollout_id"] != rollout_id or state["evaluated_rollout_id"] != rollout_id:
        raise RuntimeError("Cannot early stop before checkpoint and evaluation are committed")
    state["early_stopped"] = True
    state["early_stop_val_accuracy"] = val_accuracy
    state["early_stop_reason"] = reason
    write_checkpoint_state(directory, state)


def mark_plateau_progress(directory, rollout_id, best_accuracy, stale_evals):
    state = read_checkpoint_state(directory)
    if state["rollout_id"] != rollout_id or state["evaluated_rollout_id"] != rollout_id:
        raise RuntimeError("Cannot record plateau progress before checkpoint and evaluation")
    state["plateau_best_accuracy"] = best_accuracy
    state["plateau_stale_evals"] = stale_evals
    write_checkpoint_state(directory, state)


def mark_completed(directory):
    state = read_checkpoint_state(directory)
    state["completed"] = True
    write_checkpoint_state(directory, state)
