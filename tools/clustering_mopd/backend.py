"""Map many logical clusters onto a bounded pool of reusable adapter slots."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from async_lora.backend import LoRABackend

from async_branching.io import read_json, write_json


def prune_history(experiment, state, merge_cap):
    """Persist a safe frontier before deleting only this run's obsolete bundles."""
    if not state.snapshots:
        return
    frontier = min((d.rollouts for d in state.devices), default=0)
    snapshots = state.snapshots
    floors = [p for p in snapshots if p.rollouts <= frontier]
    protected = {snapshots[0].hf, snapshots[-1].hf}
    if floors:
        protected.add(max(floors, key=lambda p: (p.rollouts, p.clock)).hf)
    protected.update(d.active["parent"]["hf"] for d in state.devices if d.active)
    keep = [p for p in snapshots if p.hf in protected or p.rollouts >= frontier]
    retire = [Path(p.hf).parent for p in snapshots if p not in keep]
    for version in state.versions:
        if version.recovered or version.used >= merge_cap:
            retire.append(Path(version.hf).parent)
    destinations = set()
    for path in set(retire):
        resolved = path.resolve()
        if (
            not resolved.is_relative_to(experiment.resolve() / "tasks")
            or resolved.name != "final"
        ):
            raise ValueError(
                f"Refusing to prune outside this run's task bundles: {path}"
            )
        destinations.add(resolved)
    state.snapshots = keep
    # All possible future lane bounds are >= frontier. Retaining every newer
    # snapshot is essential: keeping only each lane's current parent is unsafe.
    write_json(experiment / "state.json", state.to_dict())
    for resolved in destinations:
        shutil.rmtree(resolved, ignore_errors=True)


class ClusterBackend(LoRABackend):
    def __init__(self, experiment, plan, processes, gateway):
        super().__init__(
            experiment,
            {**plan, "size": plan["teacher_worker_count"]},
            processes,
            gateway,
        )
        self.plan = plan

    def close(self):
        # Let the completed student flush W&B before terminating any unfinished
        # teacher work censored by the simulated deadline.
        (self.directory / "workers.stop").touch()
        student = self.processes.get("student")
        if student is not None and student.poll() is None:
            try:
                student.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        super().close()

    def task(self, raw):
        cluster = next(
            c for c in self.plan["clusters"] if c["id"] == raw["environment"]
        )
        student = raw["kind"] == "student"
        return {
            **raw,
            "directory": str(self.directory / "tasks" / raw["id"]),
            "environments": cluster["environments"],
            "cluster_index": cluster["index"],
            "train_seed": self.plan.get("training_seed_offset", 0)
            + (
                2_000_000_000 + raw["start_rollouts"] * 1024 + cluster["index"]
                if student
                else 1_000_000_000
                + cluster["index"] * 1_000_000
                + raw["generation"] * 10000
            ),
        }

    def collect(self, state):
        super().collect(state)
        value = read_json(self.directory / "status.json")
        work = sum(state.teacher_rollout_counts().values()) + state.student_rollouts
        value.update(
            simulated_clock_budget=state.clock_budget,
            simulated_devices=self.plan["simulated_devices"],
            teacher_devices=len(state.devices),
            cluster_count=len(state.environments),
            queued_clusters=len(state.cluster_queue),
            training_device_utilization=work
            / (self.plan["simulated_devices"] * state.clock)
            if state.clock
            else 0,
            physical_elapsed_seconds=time.time()
            - read_json(self.directory / "physical-start.json")["time"],
        )
        write_json(self.directory / "status.json", value)
        prune_history(
            self.directory, state, self.plan["student_rollouts_per_teacher_version"]
        )


def start_workers(experiment, plan, model_path, base_url):
    processes, logs = {}, []
    for role, index in [
        ("student", 0),
        *[("teacher", i) for i in range(plan["teacher_worker_count"])],
    ]:
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
                "clustering_mopd.worker",
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
