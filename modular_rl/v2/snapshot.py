"""Preserve a committed validation checkpoint while its source run continues."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from modular_rl.prepare import write_json

from .inspect_rollouts import read_events

MANIFEST = "evaluation_snapshot.json"


def file_identity(path):
    stat = path.stat()
    result = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size <= 1024 * 1024:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def validate_snapshot(checkpoint):
    checkpoint = Path(checkpoint)
    value = json.loads((checkpoint / MANIFEST).read_text())
    state = json.loads((checkpoint / "rlve_checkpoint_state.json").read_text())
    tracker = (checkpoint / "latest_checkpointed_iteration.txt").read_text().strip()
    if (not value["immutable_evaluation_snapshot"] or tracker != str(value["rollout_id"])
            or state["rollout_id"] != value["rollout_id"]
            or state["evaluated_rollout_id"] != value["rollout_id"]
            or value["completed_updates"] != value["rollout_id"] + 1
            or value["train_rollouts"] <= 0
            or not any(name.startswith(f"iter_{value['rollout_id']:07d}/") for name in value["files"])):
        raise ValueError("Snapshot does not identify an evaluated training checkpoint")
    for name, expected in value["files"].items():
        path = checkpoint / name
        if path.is_symlink() or file_identity(path) != expected:
            raise ValueError(f"Preserved checkpoint changed: {path}")
    return value


def preserve(run, destination, updates):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Choose a fresh snapshot directory")
    checkpoint = run / "ckpt"
    state_text = (checkpoint / "rlve_checkpoint_state.json").read_text()
    state = json.loads(state_text)
    tracker_text = (checkpoint / "latest_checkpointed_iteration.txt").read_text()
    rollout_id = updates - 1
    evaluations = [row for row in read_events(run / "progress.jsonl")
                   if row["phase"] == "eval" and row["completed_updates"] == updates]
    if (updates <= 0 or len(evaluations) != 1 or state["rollout_id"] != rollout_id
            or state["evaluated_rollout_id"] != rollout_id or tracker_text.strip() != str(rollout_id)):
        raise ValueError("Snapshot must match the current committed and evaluated checkpoint")
    iteration = f"iter_{rollout_id:07d}"
    source = checkpoint / iteration
    if not source.is_dir() or any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Expected regular committed checkpoint files")
    destination.mkdir(parents=True)
    try:
        # Iteration files are write-once. Hard links survive the trainer's pruning.
        # Copy mutable tracker/state metadata independently; never chmod shared files.
        shutil.copytree(source, destination / iteration, copy_function=os.link)
        (destination / "latest_checkpointed_iteration.txt").write_text(tracker_text)
        (destination / "rlve_checkpoint_state.json").write_text(state_text)
        if ((checkpoint / "latest_checkpointed_iteration.txt").read_text() != tracker_text
                or (checkpoint / "rlve_checkpoint_state.json").read_text() != state_text):
            raise ValueError("Source checkpoint advanced during preservation; retry after validation")
        value = {"immutable_evaluation_snapshot": True, "source_run": str(run),
                 "model": run.name.rsplit("-", 1)[0], "rollout_id": rollout_id,
                 "completed_updates": updates, "train_rollouts": evaluations[0]["train_rollouts"],
                 "training_complete": state["completed"], "monitored_accuracy": evaluations[0]["accuracy"],
                 "files": {str(path.relative_to(destination)): file_identity(path)
                           for path in destination.rglob("*") if path.is_file()}}
        write_json(destination / MANIFEST, value)
        validate_snapshot(destination)
    except Exception:
        shutil.rmtree(destination)
        raise
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--updates", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(preserve(args.run, args.destination, args.updates)))
