"""Measure teacher slopes and MOPD compute gain for the horizon experiment.

Run --collect-allocations on the Slurm host after jobs finish, then run the
default analysis inside the slime container where wandb is installed.
"""

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

from analyze_hero_similarity_v2 import allocations, eval_history, first_to_target


TARGETS = (0.25, 0.5, 0.75, 0.9)
TEACHER_METRIC = "david/eval/val_envs/pass@1"


def teacher_allocations(array_job):
    result = subprocess.run(
        ["sacct", "-n", "-P", "--duplicates", "--allocations", "-j", array_job,
         "--format=JobID,ElapsedRaw,State,AllocTRES"],
        check=True, capture_output=True, text=True,
    )
    tasks = {str(index): [] for index in range(4)}
    for line in result.stdout.splitlines():
        job, elapsed, state, tres = line.split("|", 3)
        match = re.fullmatch(rf"{re.escape(array_job)}_(\d+)", job)
        if match is None:
            continue
        index = match.group(1)
        if index not in tasks or "gres/gpu=1" not in tres:
            raise RuntimeError(f"Invalid teacher allocation: {line}")
        tasks[index].append({"job": job, "state": state, "gpu_hours": int(elapsed) / 3600})
    return tasks


def effective_jobs(root, by_variant):
    """Use replacement Slurm jobs while retaining the original teacher costs."""
    repairs_path = root / "repairs.tsv"
    repairs = list(csv.DictReader(repairs_path.open(), delimiter="\t")) if repairs_path.exists() else []
    teacher_repairs = {variant: [] for variant in by_variant}
    for row in repairs:
        variant, kind = row["variant"], row["kind"]
        if kind == "teacher":
            teacher_repairs[variant].append(row["job"])
        elif kind in ("mopd", "rl"):
            by_variant[variant][kind] = row
    return teacher_repairs


def teacher_curve(run):
    points = []
    rollouts = 0
    for row in run.scan_history(page_size=1000):
        if row.get("david/rollouts") is not None:
            rollouts = int(row["david/rollouts"])
        value = row.get(TEACHER_METRIC)
        if value is not None:
            points.append({"rollouts": rollouts, "val_pass_at_1": float(value)})
    return points


def initial_slope(points):
    start = next((point for point in points if point["rollouts"] == 0), None)
    window = [point for point in points if 0 < point["rollouts"] <= 4000]
    if start is None or not window:
        return None
    end = window[-1]
    return (end["val_pass_at_1"] - start["val_pass_at_1"]) / (end["rollouts"] / 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--collect-allocations", action="store_true")
    parser.add_argument("--entity", default="david-heineman")
    args = parser.parse_args()
    root = args.experiment_dir
    source = json.loads((Path(__file__).resolve().parent.parent / "data/HORIZON-DIFFICULTY/manifest.json").read_text())
    jobs = list(csv.DictReader((root / "jobs.tsv").open(), delimiter="\t"))
    by_variant = {variant: {row["kind"]: row for row in jobs if row["variant"] == variant}
                  for variant in source["groups"]}
    teacher_repairs = effective_jobs(root, by_variant)
    allocations_path = root / "allocations.json"
    if args.collect_allocations:
        data = {}
        for variant, rows in by_variant.items():
            teacher_jobs = [rows["teacher"]["job"], *teacher_repairs[variant]]
            tasks = {str(index): [] for index in range(4)}
            for job in teacher_jobs:
                for index, attempts in teacher_allocations(job).items():
                    tasks[index].extend(attempts)
            for index, attempts in tasks.items():
                if not attempts or not any(row["state"] == "COMPLETED" for row in attempts):
                    raise RuntimeError(f"Teacher {variant} task {index} has not completed")
            data[variant] = {
                "teachers": tasks,
                "mopd": allocations(rows["mopd"]["job"]),
                "rl": allocations(rows["rl"]["job"]),
            }
        allocations_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"Collected teacher and student allocations in {allocations_path}")
        return

    import wandb
    api = wandb.Api(timeout=60)
    allocs = json.loads(allocations_path.read_text())
    result = {"metric": "david/eval/val_envs/pass@1", "test_metric": "david/eval/test_envs/pass@1",
              "teacher_slope_definition": "change in mixed-difficulty validation pass@1 per 1000 teacher rollouts from baseline through the last evaluation at or before 4000 rollouts",
              "cost_definition": "Slurm allocated GPU-hours, including requeued teacher and student allocations",
              "targets": TARGETS, "variants": {}}
    for variant, rows in by_variant.items():
        names = source["groups"][variant]["environments"]
        section = {"environments": names, "source_validation_rollout_speedup": source["groups"][variant]["source_validation_rollout_speedup"],
                   "teachers": {}, "students": {}}
        for index, environment in enumerate(names):
            run_dir = root.parent / rows["teacher"]["group"] / "resumable" / f"{index:02d}-{environment}"
            run_id = (run_dir / "wandb_run_id.txt").read_text().strip()
            run = api.run(f"{args.entity}/rl-data-horizon-difficulty-teachers/{run_id}")
            points = teacher_curve(run)
            cost = sum(row["gpu_hours"] for row in allocs[variant]["teachers"][str(index)])
            section["teachers"][environment] = {"url": run.url, "state": run.state,
                                                    "gpu_hours": cost, "initial_slope_per_1000_rollouts": initial_slope(points),
                                                    "evaluations": points}
        section["teacher_gpu_hours"] = sum(row["gpu_hours"] for row in section["teachers"].values())
        slopes = [row["initial_slope_per_1000_rollouts"] for row in section["teachers"].values()]
        section["mean_teacher_initial_slope_per_1000_rollouts"] = None if any(value is None for value in slopes) else sum(slopes) / len(slopes)
        for kind in ("mopd", "rl"):
            row = rows[kind]
            run_dir = root.parent / row["group"] / "resumable" / f"{variant}-{kind}"
            run_id = (run_dir / "wandb_run_id.txt").read_text().strip()
            run = api.run(f"{args.entity}/rl-data-horizon-difficulty-students/{run_id}")
            teacher_cost = section["teacher_gpu_hours"] if kind == "mopd" else 0
            section["students"][kind] = {"url": run.url, "state": run.state,
                "evaluations": eval_history(run, allocs[variant][kind], teacher_cost)}
        mopd = section["students"]["mopd"]["evaluations"]
        rl = section["students"]["rl"]["evaluations"]
        section["gain_at_id_pass_at_1"] = {}
        for target in TARGETS:
            m, r = first_to_target(mopd, target), first_to_target(rl, target)
            section["gain_at_id_pass_at_1"][str(target)] = None if m is None or r is None else {
                "compute_gain": r["total_gpu_hours"] / m["total_gpu_hours"],
                "mopd_total_gpu_hours": m["total_gpu_hours"],
                "hero_rl_gpu_hours": r["total_gpu_hours"],
                "mopd_rollouts": m["rollouts"], "hero_rl_rollouts": r["rollouts"],
            }
        result["variants"][variant] = section
        print(variant, "teacher slope", section["mean_teacher_initial_slope_per_1000_rollouts"],
              "compute gains", {target: value["compute_gain"] if value else None
                                for target, value in section["gain_at_id_pass_at_1"].items()})
    destination = root / "compute_gain.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
