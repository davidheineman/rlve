"""Physical GPU time sharing; logical publication belongs exclusively to the scheduler."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import signal
import subprocess
import threading
import time

from .io import read_json, write_json


class GPUPool:
    def __init__(self, devices):
        if len(devices) < 4 or len(devices) != len(set(devices)):
            raise ValueError("Need at least four distinct physical GPU IDs")
        self.devices = list(devices)
        self.available = set(range(len(devices)))
        self.reserved = False
        self.stopping = False
        self.condition = threading.Condition()

    def reserve_student(self, active):
        with self.condition:
            self.reserved = active
            self.condition.notify_all()

    @contextmanager
    def claim(self, student=False):
        with self.condition:
            while True:
                if self.stopping:
                    raise RuntimeError("GPU pool is stopping")
                candidates = ({0, 1} & self.available if student else
                              self.available - ({0, 1} if self.reserved else set()))
                count = 2 if student else 1
                if len(candidates) >= count:
                    indices = sorted(candidates)[:count]
                    self.available.difference_update(indices)
                    break
                self.condition.wait()
        try:
            yield [self.devices[i] for i in indices]
        finally:
            with self.condition:
                self.available.update(indices)
                self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()


class SlimeBackend:
    def __init__(self, plan, directory, devices):
        self.plan, self.directory = plan, Path(directory).resolve()
        self.pool = GPUPool(devices)
        self.executor = ThreadPoolExecutor(max_workers=len(devices))
        self.processes, self.process_lock = {}, threading.Lock()
        self.closing = False

    def reserve_student(self, active):
        self.pool.reserve_student(active)

    def information(self, task, future):
        path = self.directory / "tasks" / task["id"] / "progress.json"
        while not future.done():
            if path.exists():
                progress = read_json(path)
                bound = progress.get("next_validation_rollouts")
                if bound is not None and bound > task["minimum_rollouts"]:
                    return {"lower_bound_rollouts": bound}
            time.sleep(.2)
        return future.result()

    def task(self, task):
        task = dict(task)
        d = self.directory / "tasks" / task["id"]
        student = task["kind"] == "student"
        index = self.plan["environments"].index(task["environment"])
        task.update(directory=str(d), checkpoint=str(d / "ckpt"),
                    base_hf=self.plan["base_hf"], base_ref=self.plan["base_ref"],
                    config=self.plan["student_config"] if student else self.plan["teacher_config"],
                    gate_eval=self.plan["teacher_eval"][task["environment"]],
                    hero_eval=self.plan["id_eval"], test_eval=self.plan["test_eval"],
                    teacher_project=self.plan["teacher_project"], student_project=self.plan["student_project"],
                    experiment=str(self.directory), group=self.directory.name,
                    train_seed=(2_000_000_000 + task["start_rollouts"] * 1024 + index if student
                                else 1_000_000_000 + index * 1_000_000 + task["generation"] * 10000))
        if student and self.plan.get("opd_replay"):
            task["opd_replay"] = dict(self.plan["opd_replay"])
        return task

    def execute(self, raw_task):
        task = self.task(raw_task)
        d = Path(task["directory"])
        if (d / "result.json").exists():
            result = read_json(d / "result.json")
            if result["rollouts"] and not Path(result["hf"]).is_dir():
                raise RuntimeError(f"Committed task weights are missing: {d}")
            return result
        d.mkdir(parents=True, exist_ok=True)
        # Uncommitted outputs belong only to this task and can be replayed.
        for name in ("ckpt", "hf-final", "hf.incomplete"):
            if (d / name).exists():
                shutil.rmtree(d / name)
        (d / "progress.json").unlink(missing_ok=True)
        write_json(d / "task.json", task)
        student = task["kind"] == "student"
        with self.pool.claim(student=student) as devices:
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(devices),
                           "RLVE_ASYNC_TASK": str(d / "task.json"),
                           "RLVE_FINAL_WEIGHTS_ONLY": "0"}
            with (d / "worker.log").open("a") as log:
                process = subprocess.Popen(["bash", "scripts/slurm/async-branching-worker.sh", str(d / "task.json")],
                                           env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                with self.process_lock:
                    self.processes[task["id"]] = process
                    if self.closing:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                try:
                    status = process.wait()
                finally:
                    with self.process_lock:
                        self.processes.pop(task["id"], None)
            if status:
                raise RuntimeError(f"Task {task['id']} exited {status}; see {d / 'worker.log'}")
        return read_json(d / "result.json")

    def teacher(self, task):
        d = self.directory / "tasks" / task["id"]
        if not (d / "result.json").exists():
            # Clear an interrupted attempt's watermark before the future is
            # visible to the scheduler; a replay may stop at an earlier eval.
            (d / "progress.json").unlink(missing_ok=True)
        return self.executor.submit(self.execute, task)

    def student(self, task):
        return self.execute(task)

    def collect(self, state):
        """Keep branch-selectable models and one student continuation; no traces."""
        # Preserve every snapshot that can be selected by a future teacher.
        # Each device's rollout counter is monotone, so older prefixes than
        # its current floor can never become branch parents again.
        keep = {state.snapshots[-1].hf, self.plan["base_hf"]}
        floor = min(d.rollouts for d in state.devices)
        for p in state.snapshots:
            if p.rollouts >= floor:
                keep.add(p.hf)
        for d in state.devices:
            candidates = [p for p in state.snapshots if p.rollouts <= d.rollouts]
            if candidates:
                keep.add(max(candidates, key=lambda p: p.rollouts).hf)
            if d.active:
                keep.add(d.active["parent"]["hf"])
                keep.add(str(self.directory / "tasks" / d.active["id"] / "hf-final"))
                if "result" in d.active:
                    keep.add(d.active["result"]["hf"])
        keep.update(v.hf for v in state.versions if not v.recovered and v.used < 1000)
        # Retain the last specialist per environment for inspection.
        for environment in state.environments:
            versions = [v for v in state.versions if v.environment == environment]
            if versions:
                keep.add(max(versions, key=lambda v: v.generation).hf)
        resume = state.snapshots[-1].resume
        replay = state.snapshots[-1].replay
        for result_path in (self.directory / "tasks").glob("*/result.json"):
            result = read_json(result_path)
            task_dir = result_path.parent
            hf = task_dir / "hf-final"
            if hf.is_dir() and str(hf) not in keep:
                shutil.rmtree(hf)
            checkpoint = task_dir / "ckpt"
            if checkpoint.is_dir() and str(checkpoint) != resume:
                shutil.rmtree(checkpoint)
            replay_path = task_dir / "opd-replay.json.gz"
            if str(replay_path) != replay:
                replay_path.unlink(missing_ok=True)

    def close(self):
        with self.process_lock:
            self.closing = True
            processes = list(self.processes.values())
        self.pool.stop()
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.executor.shutdown(wait=True, cancel_futures=True)

    def finalize(self, state):
        return self.finalize_storage(self.directory, state)

    @staticmethod
    def finalize_storage(directory, state):
        """After workers stop, retain the final student and last teacher per device."""
        directory = Path(directory)
        final_teachers = {}
        for d in state.devices:
            versions = [v for v in state.versions if v.environment == d.environment]
            if versions:
                hf = max(versions, key=lambda v: v.generation).hf
                final_teachers[d.environment] = hf
            if d.active and "result" in d.active and d.active["end_clock"] <= state.clock:
                hf = d.active["result"]["hf"]
                final_teachers[d.environment] = hf
        keep = {state.snapshots[-1].hf, *final_teachers.values()}
        for task_dir in (directory / "tasks").glob("*"):
            for name in ("hf-final", "ckpt", "hf.incomplete"):
                path = task_dir / name
                if path.is_dir() and str(path) not in keep:
                    shutil.rmtree(path)
            replay_path = task_dir / "opd-replay.json.gz"
            if str(replay_path) != state.snapshots[-1].replay:
                replay_path.unlink(missing_ok=True)
        if (directory / "parents").exists():
            shutil.rmtree(directory / "parents")
        return final_teachers
