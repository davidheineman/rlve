"""Freeze a runtime tree; submit an elastic CPU coordinator or local simulation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from async_branching.io import read_json, write_json
from tools.prepare_async_branching import OUTPUT, validate, with_replay


def freeze_runtime(directory):
    runtime = directory / "runtime"
    runtime.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git")
    for name in ("async_branching", "Gym", "slime", "slime_plugins", "tools", "configs", "scripts/models"):
        shutil.copytree(ROOT / name, runtime / name, ignore=ignore)
    for name in ("train.py", "scripts/slurm/common.sh", "scripts/slurm/port_slot.sh",
                 "scripts/slurm/async-branching-worker.sh",
                 "scripts/slurm/async-branching-elastic-worker.sbatch",
                 "scripts/slurm/async-branching-elastic.sbatch",
                 "scripts/slurm/async-branching-monitor.sbatch",
                 "scripts/slurm/async-branching.sbatch"):
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    shutil.copytree(ROOT / "data/DATA-SCALING-V1", runtime / "data/DATA-SCALING-V1")
    files = sorted(p for p in runtime.rglob("*") if p.is_file())
    lines = [hashlib.sha256(p.read_bytes()).hexdigest() + "  " + str(p.relative_to(runtime)) + "\n" for p in files]
    (directory / "runtime_sha256.txt").write_text("".join(lines))
    return runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=(16, 32, 64), required=True)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--backend", choices=("slurm", "local"), default="slurm")
    parser.add_argument("--max-gpus", type=int, default=16, help="Elastic cap including two student GPUs")
    parser.add_argument("--replay-percent", type=int, choices=(0, 10, 30, 50), default=0,
                        help="Replace this percent of student trajectories with cached OPD replay")
    parser.add_argument("--wait-for-storage", action="store_true",
                        help="Queue the CPU coordinator now; wait for launch headroom before submitting GPUs")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    elastic = args.backend == "slurm"
    plan = read_json(OUTPUT / f"n{args.size}{'-elastic' if elastic else ''}.json")
    with_replay(plan, args.replay_percent)
    if elastic:
        if not 4 <= args.max_gpus <= 16:
            parser.error("--max-gpus must be between 4 and 16")
        plan["physical_gpus"] = args.max_gpus
    validate(plan)
    for path in (Path(plan["base_hf"]) / "config.json", Path(plan["base_ref"]) / "latest_checkpointed_iteration.txt"):
        assert path.is_file(), path
    scratch = Path(plan["base_hf"]).parent.parent
    required_gib = {16: 400, 32: 750, 64: 1400}[args.size]
    if args.wait_for_storage:
        if not elastic:
            parser.error("--wait-for-storage requires the elastic Slurm backend")
        plan["minimum_launch_free_gib"] = required_gib
    else:
        assert shutil.disk_usage(scratch).free >= required_gib * 1024**3, f"Need {required_gib} GiB free for live branch models and checkpoints"
    assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY is not set"
    if args.check_only:
        layout = f"CPU coordinator, 1-GPU teachers, 2-GPU student, total cap {args.max_gpus}" if elastic else "one four-GPU job"
        storage = f"; GPU launch waits for {required_gib} GiB free" if args.wait_for_storage else ""
        print(f"N{args.size} preflight passed: {layout}; no arrays or other-job dependencies{storage}; not submitted")
        return
    replay_label = f"-replay{args.replay_percent}" if args.replay_percent else ""
    directory = args.directory or scratch / "runs" / f"data-scaling-async-n{args.size}{replay_label}-{datetime.now():%Y%m%d-%H%M%S}"
    directory = directory.resolve()
    directory.mkdir(exist_ok=False)
    runtime = freeze_runtime(directory)
    write_json(directory / "plan.json", plan)
    if elastic:
        write_json(directory / "resources.json", {"max_gpus": args.max_gpus})
    script = "async-branching-elastic.sbatch" if elastic else "async-branching.sbatch"
    command = ["sbatch", "--parsable", f"--job-name=scaling-async-n{args.size}{replay_label}",
               f"--export=ALL,RLVE_ASYNC_EXPERIMENT={directory},RLVE_REPO={runtime},RLVE_SCRATCH={scratch}",
               str(runtime / "scripts/slurm" / script)]
    job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError(f"Unexpected sbatch job ID: {job}")
    write_json(directory / "submission.json", {"job": job, "size": args.size, "physical_gpus": plan["physical_gpus"],
                                               "backend": args.backend,
                                               "utc": datetime.now(timezone.utc).isoformat(), "command": command})
    print(f"Submitted {job}; experiment={directory}")


if __name__ == "__main__":
    main()
