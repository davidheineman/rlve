"""An independent one-GPU task or persistent two-GPU student service."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .io import read_json


def execute(path):
    task = read_json(path)
    directory = Path(task["directory"])
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "result.json").exists():
            return
        for name in ("ckpt", "hf-final", "hf.incomplete"):
            if (directory / name).exists():
                shutil.rmtree(directory / name)
        (directory / "progress.json").unlink(missing_ok=True)
        with (directory / "worker.log").open("a") as log:
            if os.environ.get("RLVE_ASYNC_RUNTIME_PREFLIGHT") == "1":
                # Exercise the real parser for teachers and both student
                # continuation modes before opening a large submission window.
                import sys
                subprocess.run([sys.executable, "tools/check_async_branching.py", "--runtime",
                                "--plan", str(Path(task["experiment"]) / "plan.json")],
                               check=True, stdout=log, stderr=subprocess.STDOUT)
            process = subprocess.Popen(
                ["bash", "scripts/slurm/async-branching-worker.sh", str(path)],
                env={**os.environ, "RLVE_ASYNC_TASK": str(path), "RLVE_FINAL_WEIGHTS_ONLY": "0"},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

            def terminate(signum, frame):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    pass
                raise SystemExit(128 + signum)

            old_term = signal.signal(signal.SIGTERM, terminate)
            old_int = signal.signal(signal.SIGINT, terminate)
            try:
                status = process.wait()
            finally:
                signal.signal(signal.SIGTERM, old_term)
                signal.signal(signal.SIGINT, old_int)
        if status or not (directory / "result.json").exists():
            raise RuntimeError(f"Task {task['id']} failed with exit {status}; inspect {directory / 'worker.log'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task", type=Path)
    mode.add_argument("--service", type=Path)
    args = parser.parse_args()
    if args.task:
        execute(args.task)
        return
    args.service.mkdir(parents=True, exist_ok=True)
    with (args.service / "service.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            lease = args.service.parent / "coordinator-lease.json"
            if lease.exists() and time.time() - read_json(lease)["timestamp"] > 900:
                # An abandoned CPU coordinator must not strand two GPUs in
                # an idle service forever. In-flight segments finish first.
                return
            request = args.service / "request.json"
            if request.exists():
                path = Path(read_json(request)["task"])
                if not (path.parent / "result.json").exists():
                    execute(path)
            time.sleep(.5)


if __name__ == "__main__":
    main()
