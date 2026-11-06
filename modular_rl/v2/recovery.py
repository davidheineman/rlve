"""Archive an interrupted attempt and align exported metrics to its checkpoint."""

import fcntl
import json
import shutil
from pathlib import Path

from slime.utils.checkpoint_state import (
    atomic_write,
    read_checkpoint_state,
    validate_checkpoint,
)


def archive_attempt(run, archive):
    run, archive = Path(run), Path(archive)
    with (run / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_checkpoint_state(run / "ckpt")
        if state["completed"]:
            raise ValueError("Do not restart an already completed run")
        committed = state["rollout_id"]
        evaluated = state.get("evaluated_rollout_id", -1)
        if committed >= 0:
            validate_checkpoint(run / "ckpt", committed)
            if (run / "ckpt/latest_checkpointed_iteration.txt").read_text().strip() != str(committed):
                raise ValueError("Checkpoint tracker and committed state disagree")
        indexed = {"progress.jsonl": ("completed_updates", committed + 1),
                   "generation.jsonl": ("rollout_id", committed),
                   "signal.jsonl": ("david/step", committed),
                   "update-probes.jsonl": ("rollout_id", committed),
                   "optimizer-decisions.jsonl": ("rollout_id", committed),
                   "teacher-scores.jsonl": ("rollout_id", committed)}
        retained, original_updates = {}, 0
        for name, (key, maximum) in indexed.items():
            path = run / name
            if not path.exists():
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines(keepends=True)
                    if line.endswith("\n")]
            if name == "progress.jsonl":
                original_updates = max((r["completed_updates"] for r in rows), default=0)
            retained[name] = [
                r for r in rows
                if committed >= 0 and r[key] <= maximum
                and (r.get("phase") != "eval" or r[key] == 0
                     or r[key] <= evaluated + int(name == "progress.jsonl"))
            ]
        # Finish validation before changing files. Preserve the full original byte
        # stream, including a partial final JSONL record from an abrupt termination.
        archive.mkdir(parents=True, exist_ok=False)
        for name in ("train.log", "initial-model-identity.json", "wandb_run_id.txt",
                     "ckpt/rlve_checkpoint_state.json", "ckpt/latest_checkpointed_iteration.txt"):
            path = run / name
            if path.exists():
                shutil.copy2(path, archive / path.name)
        for name, rows in retained.items():
            path = run / name
            shutil.copy2(path, archive / name)
            atomic_write(path, "".join(json.dumps(r) + "\n" for r in rows))
        for phase in ("train", "eval"):
            for path in (run / "rollouts" / phase).glob("*.jsonl"):
                index = int(path.stem)
                if index > committed or (phase == "eval" and index > max(evaluated, 0)):
                    target = archive / "rollouts" / phase / path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(target))
        if committed < 0:
            (run / "wandb_run_id.txt").unlink(missing_ok=True)
        result = {"resume_from_updates": committed + 1,
                  "discarded_uncommitted_updates": max(0, original_updates - committed - 1),
                  "fresh_wandb_identity": committed < 0}
        atomic_write(archive / "recovery.json", json.dumps(result, indent=2) + "\n")
        return result
