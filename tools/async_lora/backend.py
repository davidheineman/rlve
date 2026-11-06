"""Bridge the unchanged causal Runner to persistent, shared-base LoRA workers."""

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lora_teachers.runner import stop_process

from async_branching.io import read_json, write_json


class LoRABackend:
    def __init__(self, experiment, plan, processes, gateway):
        self.directory, self.plan = Path(experiment), plan
        self.processes, self.gateway = processes, gateway
        self.executor = ThreadPoolExecutor(max_workers=plan["size"])
        self.stopping = threading.Event()

    def check(self):
        if self.stopping.is_set():
            raise RuntimeError("LoRA backend is stopping")
        if self.gateway.poll() is not None:
            raise RuntimeError("Miles gateway exited; inspect gateway.log")
        for name, process in self.processes.items():
            if process.poll() is not None:
                raise RuntimeError(f"Persistent worker {name} exited; inspect workers/{name}/worker.log")

    def reserve_student(self, active):
        # The gateway batches requests across independent slots; both roles
        # always share the same two trainer ranks and two sampling engines.
        pass

    def task(self, raw):
        student = raw["kind"] == "student"
        index = self.plan["environments"].index(raw["environment"])
        return {
            **raw,
            "directory": str(self.directory / "tasks" / raw["id"]),
            "train_seed": (
                2_000_000_000 + raw["start_rollouts"] * 1024 + index
                if student
                else 1_000_000_000 + index * 1_000_000 + raw["generation"] * 10000
            ),
        }

    def execute(self, raw):
        task = self.task(raw)
        directory = Path(task["directory"])
        result = directory / "result.json"
        if result.exists():
            value = read_json(result)
            if not Path(value["hf"]).is_dir() or (value.get("resume") and not Path(value["resume"]).is_dir()):
                raise RuntimeError(f"Committed checkpoint missing: {result}")
            if value.get("replay") and not Path(value["replay"]).is_file():
                raise RuntimeError(f"Committed replay buffer missing: {result}")
            return value
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "task.json", task)
        name = "student" if task["kind"] == "student" else f"teacher-{task['environment_index']:02d}"
        worker = self.directory / "workers" / name
        write_json(worker / "inbox.json", task)
        began = time.time()
        last_notice = 0
        while not result.exists():
            self.check()
            heartbeat = worker / "heartbeat.json"
            record = read_json(heartbeat) if heartbeat.exists() else {}
            advancing = record.get("task_id") == task["id"]
            last = max(began, record.get("time", began)) if advancing else began
            limit = 1800 if advancing else 2700
            if time.time() - last > limit:
                raise TimeoutError(f"No advancing optimizer step for {task['id']} within {limit // 60} minutes")
            if time.time() - last_notice >= 60:
                print(f"ASYNC_LORA_WAIT {task['id']} used={record.get('used', 0) if advancing else 0}", flush=True)
                last_notice = time.time()
            time.sleep(0.25)
        return read_json(result)

    def teacher(self, task):
        return self.executor.submit(self.execute, task)

    def student(self, task):
        return self.execute(task)

    def information(self, task, future):
        progress = self.directory / "tasks" / task["id"] / "progress.json"
        while not future.done():
            self.check()
            record = read_json(progress) if progress.exists() else {}
            bound = record.get("next_validation_rollouts")
            if bound is not None and bound > task["minimum_rollouts"]:
                return {"lower_bound_rollouts": bound}
            time.sleep(0.2)
        return future.result()

    def collect(self, state):
        write_json(
            self.directory / "status.json",
            {
                "student_rollouts": state.student_rollouts,
                "simulated_clock": state.clock,
                "teacher_versions": len(state.versions),
                "active_teachers": sum(d.active is not None for d in state.devices),
                "available_targets": sum(not v.recovered and v.used < 1000 for v in state.versions),
                "last_update": time.time(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
        )

    def close(self):
        self.stopping.set()
        (self.directory / "workers.stop").touch()
        for process in self.processes.values():
            stop_process(process)
        self.executor.shutdown(wait=True, cancel_futures=True)


def start_workers(experiment, plan, model_path, base_url):
    processes, logs = {}, []
    for role, index in [("student", 0), *[("teacher", i) for i in range(plan["size"])]]:
        name = "student" if role == "student" else f"teacher-{index:02d}"
        directory = experiment / "workers" / name
        directory.mkdir(parents=True, exist_ok=True)
        for filename in ("ready.json", "error.json", "inbox.json", "heartbeat.json"):
            (directory / filename).unlink(missing_ok=True)
        log = (directory / "worker.log").open("a")
        logs.append(log)
        processes[name] = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "async_lora.worker",
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
    return processes, logs
