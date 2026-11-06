"""Compare ID convergence per allocated GPU-hour, including reused teacher cost.

Run inside the slime container after the student jobs have produced W&B evals.
Slurm allocations are read with --duplicates so preempted attempts count too.
"""

import argparse
import csv
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ID_KEY = "david/eval/val_envs/pass@1"
OOD_KEY = "david/eval/test_envs/pass@1"


def allocations(job):
    env = dict(os.environ, TZ="UTC")
    result = subprocess.run(
        ["sacct", "-n", "-P", "--duplicates", "--allocations", "-j", job,
         "--format=JobID,Start,End,State,AllocTRES"],
        check=True, capture_output=True, text=True, env=env,
    )
    result_rows = []
    for line in result.stdout.splitlines():
        job_id, start, end, state, tres = line.split("|", 4)
        if job_id != job or start in {"Unknown", "None", ""}:
            continue
        gpu = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", tres)
        if gpu is None or int(gpu.group(1)) != 4:
            raise ValueError(f"Expected four GPUs for {job}: {tres}")
        begin = datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp()
        finish = None if end in {"Unknown", "None", ""} else datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp()
        result_rows.append({"start": begin, "end": finish, "state": state})
    return result_rows


def spent_gpu_hours(allocs, timestamp):
    return 4 * sum(
        max(0.0, min(timestamp, row["end"] if row["end"] is not None else timestamp) - row["start"])
        for row in allocs
    ) / 3600


def eval_history(run, allocs, teacher_gpu_hours):
    points = []
    rollouts = 0
    for row in run.scan_history(page_size=1000):
        if row.get("david/rollouts") is not None:
            rollouts = int(row["david/rollouts"])
        accuracy = row.get(ID_KEY)
        timestamp = row.get("_timestamp")
        if accuracy is None or timestamp is None:
            continue
        timestamp = float(timestamp)
        student_gpu_hours = spent_gpu_hours(allocs, timestamp)
        points.append({
            "id_pass_at_1": float(accuracy),
            "ood_pass_at_1": None if row.get(OOD_KEY) is None else float(row[OOD_KEY]),
            "id_stderr": row.get("eval/val_envs-stderr"),
            "rollouts": rollouts,
            "eval_step": row.get("eval/step"),
            "student_gpu_hours": student_gpu_hours,
            "total_gpu_hours": teacher_gpu_hours + student_gpu_hours,
            "timestamp_utc": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
        })
    return points


def first_to_target(points, target):
    return next((point for point in points if point["id_pass_at_1"] >= target), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--entity", default="david-heineman")
    parser.add_argument("--project", default="rl-data-hero-similarity-v2")
    parser.add_argument("--targets", type=float, nargs="+", default=[0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--collect-allocations", action="store_true", help="Run on the Slurm host before W&B analysis in the container")
    args = parser.parse_args()
    if any(not 0 < target <= 1 for target in args.targets):
        parser.error("Targets must be within (0, 1]")
    root = args.experiment_dir
    costs = json.loads((root / "teacher_costs.json").read_text())["per_variant_teacher_gpu_hours"]
    jobs = list(csv.DictReader((root / "jobs.tsv").open(), delimiter="\t"))
    allocation_file = root / "student_allocations.json"
    if args.collect_allocations:
        data = {row["job"]: allocations(row["job"]) for row in jobs}
        allocation_file.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"Wrote Slurm allocations for {len(data)} jobs to {allocation_file}")
        return
    if not allocation_file.is_file():
        raise FileNotFoundError(f"Collect Slurm allocations first: {allocation_file}")
    alloc_by_job = json.loads(allocation_file.read_text())
    import wandb

    api = wandb.Api(timeout=60)
    output = {"metric": ID_KEY, "ood_metric": OOD_KEY, "targets": args.targets,
              "cost_definition": "Slurm allocated GPU time through each evaluation; MOPD adds original per-variant teacher GPU time, including preempted allocations",
              "variants": {}}
    for row in jobs:
        variant, kind, job = row["variant"], row["kind"], row["job"]
        section = output["variants"].setdefault(variant, {"teacher_gpu_hours": costs[variant], "runs": {}})
        run_label = row.get("run_label") or f"{variant}-{kind}"
        run_id_path = root.parent / row["group"] / "resumable" / run_label / "wandb_run_id.txt"
        if not run_id_path.is_file():
            section["runs"][kind] = {"state": "not_started", "job": job}
            continue
        run_id = run_id_path.read_text().strip()
        run = api.run(f"{args.entity}/{args.project}/{run_id}")
        allocs = alloc_by_job[job]
        student_cost = 0.0 if kind == "rl" else costs[variant]
        points = eval_history(run, allocs, student_cost)
        section["runs"][kind] = {"state": run.state, "job": job, "url": run.url,
                                  "allocations": allocs, "evaluations": points}
    for variant, section in output["variants"].items():
        mopd = section["runs"].get("mopd", {}).get("evaluations", [])
        rl = section["runs"].get("rl", {}).get("evaluations", [])
        gains = {}
        for target in args.targets:
            m = first_to_target(mopd, target)
            r = first_to_target(rl, target)
            gains[str(target)] = None if m is None or r is None else {
                "mopd_total_gpu_hours": m["total_gpu_hours"],
                "hero_rl_gpu_hours": r["total_gpu_hours"],
                "compute_gain": r["total_gpu_hours"] / m["total_gpu_hours"],
                "mopd_rollouts": m["rollouts"],
                "hero_rl_rollouts": r["rollouts"],
            }
        section["gain_at_id_pass_at_1"] = gains
        print(variant, {target: value["compute_gain"] if value else None for target, value in gains.items()})
    destination = args.output or root / "compute_gain.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
