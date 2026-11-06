"""Use the shared-base Miles backend with an ordinary synchronous phase barrier."""

import fcntl
import os
import subprocess
import sys
import time

from async_lora.backend import LoRABackend

from transfer_regimes.controller import execute


class Backend(LoRABackend):
    def task(self, raw):
        return raw

    def close(self):
        (self.directory / "workers.stop").touch()
        student = self.processes["student"]
        if student.poll() is None:
            try:
                student.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        super().close()


def run_clients(experiment, plan, *, base_url, model_path, gateway):
    with (experiment / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (experiment / "completed.json").exists():
            return
        (experiment / "workers.stop").unlink(missing_ok=True)
        processes, logs = {}, []
        roles = [("student", 0)]
        if plan["algorithm"] == "sync-mopd":
            roles.extend(("teacher", i) for i in range(4))
        for role, index in roles:
            name = "student" if role == "student" else f"teacher-{index:02d}"
            directory = experiment / "workers" / name
            directory.mkdir(parents=True, exist_ok=True)
            for filename in [
                "ready.json",
                "error.json",
                "inbox.json",
                "heartbeat.json",
            ]:
                (directory / filename).unlink(missing_ok=True)
            log = (directory / "worker.log").open("a")
            logs.append(log)
            processes[name] = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "transfer_regimes.worker",
                    "--experiment",
                    str(experiment),
                    "--role",
                    role,
                    "--index",
                    str(index),
                    "--model-path",
                    str(model_path),
                    "--base-url",
                    base_url,
                ],
                env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        backend = Backend(experiment, plan, processes, gateway)
        try:
            deadline = time.monotonic() + 1800
            while not all(
                (experiment / "workers" / name / "ready.json").exists()
                for name in processes
            ):
                backend.check()
                if time.monotonic() > deadline:
                    raise TimeoutError("Synchronous adapter workers did not initialize")
                time.sleep(1)
            execute(plan, experiment, backend)
        finally:
            backend.close()
            for log in logs:
                log.close()
