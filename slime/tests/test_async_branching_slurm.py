"""Exercise the real elastic dispatcher against a CPU-only fake Slurm."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from async_branching.io import read_json, write_json
from async_branching.runner import Runner
from async_branching.schedule import Simulation, initial_state
from async_branching.slurm import SlurmBackend
from tools.prepare_async_branching import plan


class FakeSlurm:
    def __init__(self, directory, reverse=False, finish=True):
        self.directory, self.reverse, self.finish = Path(directory), reverse, finish
        self.jobs, self.commands, self.completions = {}, [], []
        self.serial, self.peak_gpus = 0, 0

    def submit(self, command):
        options = dict(part[2:].split("=", 1) for part in command if part.startswith("--") and "=" in part)
        exports = dict(part.split("=", 1) for part in options["export"].split(",") if "=" in part)
        role = exports["RLVE_ASYNC_ROLE"]
        count = int(options["gres"].split(":")[1])
        assert count == (2 if role == "student" else 1)
        assert "array" not in options and "dependency" not in options
        self.serial += 1
        key = str(self.serial)
        self.jobs[key] = {"comment": options["comment"], "role": role, "count": count,
                          "task": exports.get("RLVE_ASYNC_TASK")}
        self.commands.append(command)
        self.peak_gpus = max(self.peak_gpus, sum(j["count"] for j in self.jobs.values()))
        return key

    def complete(self, task_path):
        task = read_json(task_path)
        directory = Path(task["directory"])
        hf = directory / "hf-final"
        hf.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / "ckpt"
        checkpoint.mkdir(exist_ok=True)
        result = {"rollouts": task["cap"], "accuracy": .2 if task["kind"] == "student" else .5,
                  "hf": str(hf), "resume": str(checkpoint)}
        write_json(directory / "result.json", result)
        self.completions.append(task["id"])

    def live(self):
        if self.finish:
            teachers = [key for key, job in self.jobs.items() if job["role"] == "teacher"]
            if teachers:
                key = teachers[-1] if self.reverse else teachers[0]
                self.complete(self.jobs.pop(key)["task"])
            request = self.directory / "student-service/request.json"
            if any(j["role"] == "student" for j in self.jobs.values()) and request.exists():
                task = Path(read_json(request)["task"])
                if not (task.parent / "result.json").exists():
                    self.complete(task)
        return {key: job["comment"] for key, job in self.jobs.items()}

    def cancel(self, jobs):
        for job in jobs:
            self.jobs.pop(job, None)


class ElasticAsyncTest(unittest.TestCase):
    def make(self, directory, size=16, cap=16, reverse=False, finish=True):
        config = plan(size, Path(directory) / "scratch", "slurm", cap)
        state = initial_state(config["environments"], config["base_hf"], config["base_ref"], 16000)
        client = FakeSlurm(directory, reverse, finish)
        backend = SlurmBackend(config, directory, state, client=client, poll_seconds=0)
        return config, state, client, backend

    def test_full_simulation_equivalent_across_limits_and_completion_orders(self):
        for size in (16, 32):
            canonical = None
            for cap, reverse in ((4, False), (16, True)):
                with self.subTest(size=size, cap=cap), tempfile.TemporaryDirectory() as directory:
                    _, state, client, backend = self.make(directory, size, cap, reverse)
                    runner = Runner(Simulation(state), backend, directory)
                    runner.run()
                    self.assertEqual(state.student_rollouts, 16000)
                    self.assertLessEqual(client.peak_gpus, cap)
                    self.assertFalse(client.jobs)  # Workers stopped before cleanup.
                    events = []
                    for event in runner.events:
                        if event["event"] == "branch":
                            t = event["task"]
                            events.append(("branch", t["id"], t["start_clock"], t["parent"]["rollouts"]))
                        elif event["event"] == "merge_end":
                            t = event["task"]
                            events.append(("merge", t["version_key"], t["start_clock"], t["cap"]))
                    if canonical is None:
                        canonical = events
                    else:
                        self.assertEqual(events, canonical)
                    student_jobs = [c for c in client.commands if "--gres=gpu:2" in c]
                    self.assertEqual(len(student_jobs), 1)  # No queue wait per merge.

    def test_dynamic_cap_drains_then_resumes_without_changing_pending_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            _, state, client, backend = self.make(directory, size=32, finish=False)
            for task in Simulation(state).branches():
                backend.teacher(task)
            backend._pump()
            self.assertEqual(len(client.jobs), 14)
            write_json(Path(directory) / "resources.json", {"max_gpus": 8})
            backend._pump()
            self.assertEqual(len(client.commands), 14)  # No new submissions while over limit.
            client.finish = True
            for _ in range(10):
                backend._pump()
            self.assertEqual(len(client.jobs), 6)
            write_json(Path(directory) / "resources.json", {"max_gpus": 16})
            backend._pump()
            self.assertEqual(len(client.jobs), 14)

    def test_coordinator_restart_reattaches_live_jobs_without_duplicate_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, client, first = self.make(directory, finish=False)
            tasks = Simulation(state).branches()
            for task in tasks:
                first.teacher(task)
            first._pump()
            self.assertEqual(len(client.commands), 14)
            first.close()  # Interruption preserves live workers.
            second = SlurmBackend(config, directory, state, client=client, poll_seconds=0)
            for task in tasks:
                second.teacher(task)
            second._pump()
            self.assertEqual(len(client.commands), 14)
            self.assertEqual(len(client.jobs), 14)

    def test_submit_response_journal_gap_reattaches_by_comment(self):
        with tempfile.TemporaryDirectory() as directory:
            _, state, client, backend = self.make(directory, finish=False)
            task = Simulation(state).branches()[0]
            # Emulate sbatch succeeding immediately before a coordinator crash.
            raw = backend.task(task)
            path = Path(raw["directory"]) / "task.json"
            write_json(path, raw)
            client.submit(backend.command(task["id"], path))
            backend.teacher(task)
            backend._pump()
            self.assertEqual(len(client.commands), 1)
            self.assertEqual(backend.jobs[task["id"]]["job"], "1")

    def test_invalid_live_cap_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, client, backend = self.make(directory, finish=False)
            write_json(Path(directory) / "resources.json", {"max_gpus": 17})
            with self.assertRaises(ValueError):
                backend._pump()
            self.assertEqual(client.commands, [])

    def test_failed_teacher_surfaces_without_unbounded_retry(self):
        import time
        with tempfile.TemporaryDirectory() as directory:
            _, state, client, backend = self.make(directory, finish=False)
            task = Simulation(state).branches()[0]
            future = backend.teacher(task)
            backend._pump()
            client.jobs.clear()  # A failed job writes no terminal commit.
            backend._pump()
            with patch("async_branching.slurm.time.monotonic", return_value=time.monotonic() + 31):
                backend._pump()
            with self.assertRaisesRegex(RuntimeError, "ended without a result"):
                future.result()
            self.assertEqual(len(client.commands), 1)

    def test_shared_node_worker_exposes_only_allocated_gpu_ids(self):
        import hashlib
        import os
        import shutil
        import subprocess
        root = Path(__file__).parents[2]
        for role, devices in (("teacher", "2"), ("student", "2,5")):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                runtime = directory / "runtime"
                slurm = runtime / "scripts/slurm"
                slurm.mkdir(parents=True)
                shutil.copy2(root / "scripts/slurm/common.sh", slurm / "common.sh")
                marker = runtime / "marker"
                marker.write_text("frozen")
                (directory / "runtime_sha256.txt").write_text(hashlib.sha256(marker.read_bytes()).hexdigest() + "  marker\n")
                binary = directory / "bin"
                binary.mkdir()
                (binary / "nvidia-smi").write_text(
                    '#!/usr/bin/python3\nimport sys\nassert sys.argv[1] == "-i"\n'
                    'for index in sys.argv[2].split(","): print("GPU-"+index)\n')
                (binary / "apptainer").write_text(
                    '#!/usr/bin/python3\nimport json,os,sys\n'
                    'json.dump({"devices":os.environ["APPTAINERENV_CUDA_VISIBLE_DEVICES"],'
                    '"args":sys.argv[1:]},open(os.environ["ASYNC_GPU_PROBE"],"w"))\n')
                for file in binary.iterdir():
                    file.chmod(0o755)
                scratch = directory / "scratch"
                scratch.mkdir()
                probe = directory / "probe.json"
                environment = {**os.environ, "PATH": str(binary) + ":" + os.environ["PATH"],
                               "RLVE_ASYNC_EXPERIMENT": str(directory), "RLVE_REPO": str(runtime),
                               "RLVE_SCRATCH": str(scratch), "RLVE_ASYNC_ROLE": role,
                               "RLVE_ASYNC_TASK": str(directory / "task.json"), "CUDA_VISIBLE_DEVICES": devices,
                               "SLURM_JOB_ID": "12345", "WANDB_API_KEY": "test-only-placeholder",
                               "ASYNC_GPU_PROBE": str(probe)}
                subprocess.run(["bash", str(root / "scripts/slurm/async-branching-elastic-worker.sbatch")],
                               env=environment, capture_output=True, text=True, check=True)
                result = read_json(probe)
                self.assertEqual(result["devices"], ",".join("GPU-" + v for v in devices.split(",")))
                self.assertIn("--service" if role == "student" else "--task", result["args"])


if __name__ == "__main__":
    unittest.main()
