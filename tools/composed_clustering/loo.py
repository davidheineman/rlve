"""Run one resumable inference worker against a shared leave-one-out queue."""

import signal
import subprocess
import sys
import time
from pathlib import Path

from async_branching.io import read_json, write_json


def execute(directory):
    directory = Path(directory)
    plan = read_json(directory / "plan.json")
    expected = plan["loo_task_ids"]

    def finished():
        return all((directory / "scores" / f"{key}.json").exists() for key in expected)

    if finished():
        write_json(directory / "completed.json", {"complete": True, "time": time.time()})
        return
    work = directory / "workers/worker-0"
    work.mkdir(parents=True, exist_ok=True)
    for name in ("ready.json", "error.json"):
        (work / name).unlink(missing_ok=True)
    (directory / "stop").unlink(missing_ok=True)
    process = None

    def stop(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    try:
        with (work / "worker.log").open("a") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "composed_clustering.worker", "--experiment",
                 str(directory), "--index", "0"], stdout=log, stderr=subprocess.STDOUT,
            )
            while not finished():
                if process.poll() is not None:
                    raise RuntimeError("LOO worker exited before its shared queue completed")
                time.sleep(5)
            write_json(directory / "completed.json", {"complete": True, "time": time.time()})
    finally:
        (directory / "stop").touch()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    execute(Path(sys.argv[1]))
