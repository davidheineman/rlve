"""Evaluate the unique k=1 composition, then update the existing figure."""

import argparse
import fcntl
import hashlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from async_branching.io import read_json, write_json
from composed_clustering.plot import render
from composed_clustering.runner import Evaluator


def execute(directory):
    plan = read_json(directory / "plan.json")
    if (directory / "completed.json").exists():
        return
    origin = Path(plan["origin"])
    if (
        hashlib.sha256((origin / "completed.json").read_bytes()).hexdigest()
        != plan["origin_completed_sha256"]
    ):
        raise ValueError("Original completed evaluation changed")
    original = read_json(origin / "completed.json")
    if len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")) != 1:
        raise RuntimeError("The k=1 endpoint expects exactly one allocated GPU")
    for name in [
        "workers",
        "tasks",
        "scores",
        "cells",
        "task-locks",
        "completed-tasks",
    ]:
        (directory / name).mkdir(exist_ok=True)
    work = directory / "workers/worker-0"
    work.mkdir(exist_ok=True)
    for name in ["ready.json", "error.json"]:
        (work / name).unlink(missing_ok=True)
    (directory / "stop").unlink(missing_ok=True)

    def stop(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    with (work / "worker.log").open("a") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "composed_clustering.worker",
                "--experiment",
                str(directory),
                "--index",
                "0",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 1200
            while not (work / "ready.json").exists():
                if process.poll() is not None:
                    raise RuntimeError("k=1 inference worker failed during startup")
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "k=1 inference initialization exceeded 20 minutes"
                    )
                time.sleep(2)
            write_json(
                directory / "status.json",
                {
                    "stage": "all_task_composition_evaluation",
                    "k": 1,
                    "rank": 2048,
                    "time": time.time(),
                },
            )
            evaluator = Evaluator(directory, plan, [process])
            result = evaluator.partition(
                [list(range(len(plan["environments"])))], "confirmation"
            )
            result.update(
                k=1,
                extension=str(directory),
                composition=plan["composition"],
                finished_at=time.time(),
            )
            write_json(directory / "result.json", result)
            write_json(origin / "held-k01.json", result)
            render(
                origin,
                original["candidates"],
                original["base"],
                singleton=original["singleton"],
                all_tasks=result,
            )
            write_json(
                directory / "completed.json",
                {
                    "complete": True,
                    "result": result,
                    "figure_output": read_json(origin / "plan.json")["output"],
                    "finished_at": time.time(),
                },
            )
            write_json(
                directory / "status.json", {"stage": "complete", "time": time.time()}
            )
        finally:
            (directory / "stop").touch()
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    with (args.experiment / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args.experiment)
