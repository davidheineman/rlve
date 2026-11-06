"""Submit prepared jobs as checkpoint headroom becomes available during cleanup."""

import argparse
import fcntl
import math
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from submit_clustering_mopd import verify_sweep

from async_branching.io import read_json, write_json


def launch(sweep):
    with (sweep / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify_sweep(sweep)
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("Missing W&B credentials")
        reservations = {
            r["label"]: math.ceil(
                0.6
                * (
                    r["adapter_slots"]
                    + math.ceil(manifest["simulated_clock_budget"] / 400)
                    + 2
                )
                + 16 / len(manifest["experiments"])
            )
            for r in manifest["experiments"]
        }
        jobs = {}
        for r in manifest["experiments"]:
            path = Path(r["experiment"]) / "job.json"
            if path.exists():
                jobs[r["label"]] = read_json(path)["job_id"]
        manifest.update(
            submission_strategy="staggered with conservative cumulative peak checkpoint reservations",
            checkpoint_reservations_gib=reservations,
            status="submitting",
        )
        write_json(sweep / "sweep.json", manifest)
        for r in manifest["experiments"]:
            if r["label"] in jobs:
                continue
            required = (
                sum(reservations[label] for label in jobs) + reservations[r["label"]]
            )
            while shutil.disk_usage(sweep).free < required * 2**30:
                print(
                    "WAITING_FOR_STORAGE",
                    r["label"],
                    "required_gib",
                    required,
                    "available_gib",
                    round(shutil.disk_usage(sweep).free / 2**30, 1),
                    flush=True,
                )
                time.sleep(30)
            directory = Path(r["experiment"])
            job = (
                subprocess.check_output(
                    read_json(directory / "submit_argv.json"), text=True
                )
                .strip()
                .split(";")[0]
            )
            write_json(
                directory / "job.json",
                {
                    "job_id": job,
                    "experiment": str(directory),
                    "student_project": manifest["student_project"],
                    "teacher_project": manifest["teacher_project"],
                },
            )
            jobs[r["label"]] = job
            manifest["job_ids"] = list(jobs.values())
            write_json(sweep / "sweep.json", manifest)
            print("SUBMITTED", r["label"], job, "gpus", 4, flush=True)
        manifest["status"] = "submitted"
        write_json(sweep / "sweep.json", manifest)
        if not manifest.get("watchdog_job_id"):
            command = [
                "python3",
                str(sweep / "source/rlve/tools/submit_clustering_mopd.py"),
                "--monitor",
                "--recover",
                "--experiment",
                str(sweep),
            ]
            watch = (
                subprocess.check_output(
                    [
                        "sbatch",
                        "--parsable",
                        "--account=miso",
                        "--partition=sc-freecpu",
                        "--job-name=clustering-mopd-watch",
                        "--cpus-per-task=1",
                        "--mem=1G",
                        "--time=7-00:00:00",
                        "--requeue",
                        f"--output={sweep}/monitor-%j.out",
                        "--wrap",
                        "exec " + shlex.join(command),
                    ],
                    text=True,
                )
                .strip()
                .split(";")[0]
            )
            manifest["watchdog_job_id"] = watch
            write_json(sweep / "sweep.json", manifest)
            print("WATCHDOG_SUBMITTED", watch, flush=True)
        print("ALL_JOBS_SUBMITTED", len(jobs), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path)
    launch(parser.parse_args().sweep)
