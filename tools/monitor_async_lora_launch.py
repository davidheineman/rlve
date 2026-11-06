"""Record queued startup, first optimizer steps and optional-generator admission."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def inspect(experiment):
    job = read(experiment / "job.json")["job_id"]
    queue = subprocess.check_output(["squeue", "-h", "-j", job, "-o", "%T|%R"], text=True, timeout=20).strip()
    if queue:
        state, node = queue.split("|", 1)
    else:
        state = (
            subprocess.check_output(["sacct", "-n", "-X", "-j", job, "--format=State", "-P"], text=True, timeout=20)
            .strip()
            .splitlines()[0]
            .strip("|")
        )
        node = None
    core = read(experiment / "flex/core.json")
    extras = [read(path) for path in (experiment / "flex").glob("worker-*.json")]
    ready = [
        worker
        for worker in extras
        if worker.get("generation") == core.get("generation")
        and worker.get("stage") == "ready"
        and time.time() - worker.get("time", 0) < 30
    ]
    workers = {path.parent.name: read(path) for path in (experiment / "workers").glob("*/heartbeat.json")}
    errors = {path.parent.name: read(path) for path in (experiment / "workers").glob("*/error.json")}
    plan = read(experiment / "plan.json")
    slots = len(list((experiment / "workers").glob("*/ready.json")))
    preflight = (experiment / "preflight/passed.json").is_file()
    steps = {name: value.get("used", 0) // 8 for name, value in workers.items()}
    identity = read(experiment / "workers/student/identity.json")
    result = {
        "time": time.time(),
        "job_id": job,
        "slurm_state": state,
        "node": node,
        "core_gpus": 4,
        "core_trainers": 2,
        "core_generators": 2,
        "max_flex_generators": 8,
        "preflight_passed": preflight,
        "ready_slots": slots,
        "expected_slots": plan["adapter_slots"],
        "optimizer_steps": steps,
        "ready_flex_generators": ready,
        "flex_capacity": read(experiment / "flex/status.json"),
        "core_gpu_mapping": read(experiment / "gpu_mapping.json"),
        "student_identity": identity,
        "worker_errors": errors,
        "tests_passed": read(experiment / "validation.json").get("total_passed"),
    }
    result["launch_verified"] = bool(
        state == "RUNNING"
        and preflight
        and slots == plan["adapter_slots"]
        and max(steps.values(), default=0) >= 3
        and ready
        and not errors
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    deadline = time.monotonic() + 3.9 * 3600
    while True:
        result = inspect(args.experiment)
        temporary = args.experiment / "launch_status.json.monitor-tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(temporary, args.experiment / "launch_status.json")
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "time",
                        "job_id",
                        "slurm_state",
                        "preflight_passed",
                        "ready_slots",
                        "optimizer_steps",
                        "launch_verified",
                    )
                }
            ),
            flush=True,
        )
        if args.once or result["launch_verified"]:
            return
        if result["worker_errors"] or result["slurm_state"].startswith(
            ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")
        ):
            raise SystemExit("Launch failed; inspect launch_status.json and worker/gateway logs")
        if time.monotonic() > deadline:
            raise SystemExit("Launch monitor ended before full verification; inspect launch_status.json")
        time.sleep(15)


if __name__ == "__main__":
    main()
