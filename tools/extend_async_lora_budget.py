"""Extend a held async LoRA job's durable budget without changing its frozen launch inputs.

First run `scontrol requeuehold JOB_ID`; after this command succeeds, release
the same job with `scontrol release JOB_ID`. Resume uses State.budget, while
plan.json remains the immutable record of the original launch.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def extended_state(before, budget):
    if type(budget) is not int or budget <= before["budget"] or budget % 8:
        raise ValueError("New budget must increase the current budget and be a multiple of eight")
    if budget <= before["student_rollouts"]:
        raise ValueError("New budget must exceed completed student rollouts")
    if before.get("stopped") not in (None, "student_budget"):
        raise ValueError(f"Investigate the existing stop reason before resuming: {before['stopped']}")
    after = copy.deepcopy(before)
    after["budget"] = budget
    after["stopped"] = None
    return after


def extend(experiment, budget):
    job = json.loads((experiment / "job.json").read_text())
    with (experiment / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        slurm = subprocess.check_output(["scontrol", "show", "job", str(job["job_id"])], text=True)
        held = re.search(r"\bReason=(?:JobHeldUser|job_requeued_in_held_state)\b", slurm)
        if not re.search(r"\bJobState=PENDING\b", slurm) or not held or not re.search(r"\bPriority=0\b", slurm):
            raise RuntimeError("Job must be stopped and held with scontrol requeuehold before changing its budget")
        state_path = experiment / "state.json"
        original_bytes = state_path.read_bytes()
        before = json.loads(original_bytes)
        after = extended_state(before, budget)
        plan = json.loads((experiment / "plan.json").read_text())
        inputs = json.loads((experiment / "input_sha256.json").read_text())
        plan_hash = hashlib.sha256((experiment / "plan.json").read_bytes()).hexdigest()
        if plan_hash != inputs["plan.json"]:
            raise RuntimeError("Frozen launch plan has changed")
        continuation = before["snapshots"][-1]
        if not Path(continuation["hf"]).is_dir() or not (Path(continuation["resume"]) / ".metadata").is_file():
            raise RuntimeError("Latest committed student adapter or optimizer checkpoint is missing")
        # Validate using this experiment's actual frozen scheduler, including
        # its checkpoint schema, rather than a potentially newer workspace copy.
        sys.path.insert(0, str(experiment / "source/rlve"))
        from async_branching.io import write_json
        from async_branching.schedule import Simulation, State

        Simulation(
            State.from_dict(after),
            teacher_cap=plan["teacher_branch_rollouts"],
            merge_cap=plan["student_rollouts_per_teacher_version"],
            recovery=plan["teacher_recovery_fraction"],
            quantum=plan["student_quantum_rollouts"],
        )
        timestamp = datetime.now(timezone.utc)
        audit = experiment / "budget-extensions" / timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
        audit.mkdir(parents=True)
        (audit / "state.before.json").write_bytes(original_bytes)
        record = {
            "at": timestamp.isoformat(),
            "job_id": str(job["job_id"]),
            "previous_budget": before["budget"],
            "new_budget": budget,
            "student_rollouts": before["student_rollouts"],
            "student_checkpoint": continuation,
            "original_plan_sha256": plan_hash,
            "state_before_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "changed_state_fields": [key for key in after if after[key] != before[key]],
        }
        write_json(audit / "state.after.json", after)
        write_json(state_path, after)
        completed = experiment / "completed.json"
        if completed.exists():
            shutil.copy2(completed, audit / "completed.before.json")
            completed.unlink()
        event = {
            "event": "budget_extended",
            "clock": after["clock"],
            "student_rollouts": after["student_rollouts"],
            "previous_budget": before["budget"],
            "new_budget": budget,
            "at": record["at"],
        }
        with (experiment / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        write_json(audit / "extension.json", record)
        write_json(experiment / "budget_extension.json", record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("student_rollouts", type=int)
    args = parser.parse_args()
    print(json.dumps(extend(args.experiment.resolve(), args.student_rollouts), indent=2))


if __name__ == "__main__":
    main()
