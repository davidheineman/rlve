"""Watch only the six jobs explicitly owned by a parallel clustering manifest."""

import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path

from monitor_mopd_clustering import (
    ACTIVE,
    inspect,
    read,
    recover,
    slurm,
    tail,
    validate_completion,
    write,
)


def apply_node_exclusions(directory, job_id, state):
    """Slurm permits changing node exclusions only while the job is pending."""
    requested = set(read(directory / "node-exclusions.json", []))
    requested.update(read(directory / "full-tmp-nodes.json", []))
    if state["state"] != "PENDING" or not requested:
        return
    description = subprocess.check_output(
        ["scontrol", "show", "job", job_id, "-o"], text=True, timeout=20
    )
    fields = dict(item.split("=", 1) for item in description.split() if "=" in item)
    if fields.get("JobState") != "PENDING":
        return
    existing = fields.get("ExcNodeList", "(null)")
    current = set() if existing == "(null)" else set(existing.split(","))
    if requested <= current:
        return
    result = subprocess.run(
        [
            "scontrol",
            "update",
            f"JobId={job_id}",
            "ExcNodeList=" + ",".join(sorted(current | requested)),
        ],
        text=True,
        capture_output=True,
        timeout=20,
    )
    # A job may start between the read and update. Retain its allocation and
    # retry on its next pending interval instead of disrupting active work.
    if result.returncode and "no longer pending" not in result.stderr.lower():
        result.check_returncode()


def check(experiment, manifest, *, repair=False):
    health = inspect(experiment)
    shards = []
    failures = []
    for backend in manifest["backends"]:
        directory = Path(backend["backend"])
        state = slurm(backend["job_id"])
        if repair:
            apply_node_exclusions(directory, backend["job_id"], state)
        stage = read(directory / "pipeline-status.json", {}).get("stage", "queued")
        row = {
            "index": backend["index"],
            "job_id": backend["job_id"],
            "indices": backend["indices"],
            "slurm": state,
            "stage": stage,
        }
        if state["state"] not in ACTIVE | {"UNKNOWN", "COMPLETED"}:
            journal = read(directory / "watchdog/journal.json", {})
            local = dict(health, job_id=backend["job_id"], slurm=state)
            # Early frozen workers attempted to edit exclusions while running.
            # Their recorded /tmp guard still identifies the failure, allowing
            # the normal requeuehold -> update -> release recovery to handle it.
            log = tail(directory / f"slurm-{backend['job_id']}.out")
            if (
                state["node"] in read(directory / "full-tmp-nodes.json", [])
                and "no longer pending" in log.lower()
                and "ExcNodeList=" in log
            ):
                local["errors"] = [
                    *health["errors"],
                    {"traceback": "No space left on device: startup /tmp guard"},
                ]
            if repair and recover(directory, local, journal):
                row["requeued"] = True
            else:
                failures.append(backend["job_id"])
        shards.append(row)
    health.update(
        job_ids=[b["job_id"] for b in manifest["backends"]],
        shards=shards,
        allocated_gpus=4 * sum(s["slurm"]["state"] == "RUNNING" for s in shards),
        requested_gpus=manifest["total_gpus"],
        failed_job_ids=failures,
    )
    stages = {s["stage"] for s in shards}
    if health["active_teachers"]:
        health["stage"] = "training" if "training" in stages else "backend_startup"
    elif "finalizing" in stages:
        health["stage"] = "finalizing"
    elif stages & {"merged_confirmation", "waiting_for_confirmation"}:
        health["stage"] = "merged_confirmation"
    elif "balanced_search" in stages:
        health["stage"] = "balanced_search"
    else:
        health["stage"] = "transfer_evaluation"
    if validate_completion(experiment):
        health.update(status="complete", stage="complete")
    elif failures or health["errors"]:
        health["status"] = "needs_attention"
    else:
        health["status"] = "healthy"
    return health


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=90)
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    directory = args.experiment / "watchdog"
    directory.mkdir(exist_ok=True)
    manifest = read(args.manifest)
    with (directory / "parallel-monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            health = check(args.experiment, manifest, repair=args.recover)
            write(directory / "health.json", health)
            with (directory / "health.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            k: v
                            for k, v in health.items()
                            if k not in ("active", "completed", "errors")
                        }
                    )
                    + "\n"
                )
            print(
                json.dumps(
                    {
                        k: health[k]
                        for k in (
                            "status",
                            "stage",
                            "allocated_gpus",
                            "completed_teachers",
                            "min_active_rollouts",
                            "max_active_rollouts",
                            "evaluation_cells",
                            "failed_job_ids",
                        )
                    }
                ),
                flush=True,
            )
            if health["status"] == "complete" or args.once:
                return
            time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
