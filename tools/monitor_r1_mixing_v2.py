"""Track every task of the R1 mixing v2 array through final evaluation."""

import argparse
import json
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TRAIN = re.compile(r"\bdavid (\d+): \{")
EVAL = re.compile(r"\beval (-?\d+): \{")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}


def states(array_job):
    result = subprocess.run(
        ["sacct", "-j", array_job, "-n", "-P", "-X", "-o", "JobID,State,ExitCode,ElapsedRaw"],
        capture_output=True, text=True, check=True,
    )
    output = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        match = re.fullmatch(re.escape(array_job) + r"_(\d+)", fields[0])
        if match and len(fields) >= 4:
            output[int(match.group(1))] = {
                "state": fields[1].split()[0].split("+")[0],
                "exit_code": fields[2], "elapsed_seconds": int(fields[3] or 0),
            }
    return output


def read_new_lines(path, offsets, partials, index):
    if not path.exists():
        return []
    size = path.stat().st_size
    previous = offsets.get(index, 0)
    if size < previous:
        previous = 0
    with path.open("rb") as stream:
        stream.seek(previous)
        raw = stream.read()
    offsets[index] = size
    pieces = (partials.get(index, b"") + raw).split(b"\n")
    partials[index] = pieces.pop()
    return [piece.decode("utf-8", errors="replace") for piece in pieces]


def monitor(root, array_job, interval, hours, once=False):
    offsets = {}
    partials = {}
    runs = [{"index": i, "last_training_update": None, "id_eval_updates": [],
             "ood_eval_updates": [], "errors": []} for i in range(73)]
    deadline = time.monotonic() + hours * 3600
    previous = None
    while True:
        slurm = states(array_job)
        for run in runs:
            index = run["index"]
            path = root / f"{index:02d}-mix-{index:02d}.log"
            for line in read_new_lines(path, offsets, partials, index):
                training = TRAIN.search(line)
                if training:
                    run["last_training_update"] = max(
                        run["last_training_update"] or 0, int(training.group(1))
                    )
                evaluation = EVAL.search(line)
                if evaluation:
                    update = int(evaluation.group(1))
                    if "'eval/val_envs'" in line and update not in run["id_eval_updates"]:
                        run["id_eval_updates"].append(update)
                    if "'eval/test_envs'" in line and update not in run["ood_eval_updates"]:
                        run["ood_eval_updates"].append(update)
                if any(marker in line for marker in (
                    "Traceback (most recent call last)", "FAILED mix-", "RuntimeError:", "AssertionError:"
                )):
                    run["errors"].append(line[-300:])
                    run["errors"] = run["errors"][-5:]
            run.update(slurm.get(index, {"state": "PENDING", "exit_code": None, "elapsed_seconds": 0}))
            outer = root.parent / f"rlve-mixing-v3-{array_job}_{index}.out"
            if outer.exists() and "GPU preflight passed" in outer.read_text(errors="replace"):
                run["gpu_preflight"] = True
            else:
                run["gpu_preflight"] = False

        counts = dict(Counter(run["state"] for run in runs))
        expected_id = {0, *range(49, 500, 50)}
        expected_ood = {479, 489, 499}
        for run in runs:
            run["verified_complete"] = (
                run["state"] == "COMPLETED"
                and run["last_training_update"] is not None
                and run["last_training_update"] >= 499
                and expected_id <= set(run["id_eval_updates"])
                and expected_ood <= set(run["ood_eval_updates"])
            )
        summary = {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "array_job": array_job,
            "wandb_project": "david-heineman/rl-data-mixing-r1-distil-v2",
            "counts": counts,
            "gpu_preflight_passed": sum(run["gpu_preflight"] for run in runs),
            "baseline_id_eval_logged": sum(0 in run["id_eval_updates"] for run in runs),
            "final_ood_complete": sum(expected_ood <= set(run["ood_eval_updates"]) for run in runs),
            "verified_complete": sum(run["verified_complete"] for run in runs),
            "runs": runs,
        }
        output = root / "monitor.json"
        temporary = root / "monitor.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
        progress = (tuple(sorted(counts.items())), summary["gpu_preflight_passed"],
                    summary["baseline_id_eval_logged"], summary["final_ood_complete"])
        if progress != previous:
            print(summary["updated_utc"], progress, flush=True)
            previous = progress
        if once:
            return 0
        if all(run["state"] in TERMINAL for run in runs):
            failures = [run["index"] for run in runs if not run["verified_complete"]]
            print(f"All tasks terminal; failures={failures}", flush=True)
            return 1 if failures else 0
        if time.monotonic() >= deadline:
            print("Monitoring deadline expired before all tasks finished", flush=True)
            return 2
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("array_job")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    raise SystemExit(monitor(args.root, args.array_job, args.interval, args.hours, args.once))


if __name__ == "__main__":
    main()
