"""Elastic physical execution. All rollout-clock decisions stay in Simulation.

The CPU coordinator owns a bounded submission window, not an allocation. Pending
jobs count against that window; Slurm may run any subset of them. Two slots are
reserved for a persistent student service, avoiding a new queue wait per merge.
"""

from concurrent.futures import Future
import getpass
import hashlib
from pathlib import Path
import subprocess
import time

from .backend import SlimeBackend
from .io import read_json, write_json


class SlurmClient:
    def live(self):
        output = subprocess.check_output(
            ["squeue", "--noheader", "--user", getpass.getuser(), "--format=%A|%k"], text=True)
        return {job: comment for line in output.splitlines() if "|" in line
                for job, comment in [line.strip().split("|", 1)]}

    def submit(self, command):
        job = subprocess.check_output(command, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"Unexpected sbatch response: {job}")
        return job

    def cancel(self, jobs):
        if jobs:
            # Completed job IDs may already have left Slurm's active table.
            subprocess.run(["scancel", *sorted(jobs)], check=False, capture_output=True, text=True)


def gpu_cap(directory, default):
    path = Path(directory) / "resources.json"
    value = read_json(path).get("max_gpus", default) if path.exists() else default
    if type(value) is not int or not 4 <= value <= 16:
        raise ValueError("resources.json max_gpus must be an integer from 4 to 16")
    return value


class SlurmBackend(SlimeBackend):
    """Reuse task construction/retention, without a local GPU pool or executor."""

    def __init__(self, plan, directory, state, client=None, poll_seconds=3):
        self.plan, self.directory, self.state = plan, Path(directory).resolve(), state
        self.client, self.poll_seconds = client or SlurmClient(), poll_seconds
        self.pending, self.futures = [], {}
        self.raw_tasks = {}
        self.missing_since = {}
        self.student_request = None
        self.journal = self.directory / "slurm_jobs.json"
        self.jobs = read_json(self.journal) if self.journal.exists() else {}
        self.restored_jobs = set(self.jobs)
        self.token = hashlib.sha256(str(self.directory).encode()).hexdigest()[:16]
        self.runtime = self.directory / "runtime"
        self.last_poll = float("-inf")
        self.live = {}
        if not (self.directory / "resources.json").exists():
            write_json(self.directory / "resources.json", {"max_gpus": plan["physical_gpus"]})
        write_json(self.directory / "coordinator-lease.json", {"timestamp": time.time()})

    def reserve_student(self, active):
        # Always reserve two submission slots. Running teachers need never be
        # killed to admit the student, even when many one-GPU cards are free.
        pass

    def comment(self, key):
        return f"rlve-async:{self.token}:{key}"

    def _save(self):
        write_json(self.journal, self.jobs)

    def command(self, key, task_path=None):
        student = key == "student-service"
        count = 2 if student else 1
        replay = self.plan.get("opd_replay", {}).get("fraction", 0)
        replay_label = f"-replay{round(100 * replay)}" if replay else ""
        scratch = Path(self.plan["base_hf"]).parent.parent
        export = (f"ALL,RLVE_ASYNC_EXPERIMENT={self.directory},RLVE_REPO={self.runtime},"
                  f"RLVE_SCRATCH={scratch},RLVE_ASYNC_ROLE={'student' if student else 'teacher'}")
        if task_path:
            export += f",RLVE_ASYNC_TASK={task_path}"
        return ["sbatch", "--parsable", "--account=miso", "--nodes=1", "--ntasks=1",
                f"--job-name=async-n{self.plan['size']}{replay_label}-{'student' if student else key}",
                f"--comment={self.comment(key)}", f"--gres=gpu:{count}",
                f"--cpus-per-task={count * 6}", f"--mem={count * 64}G",
                "--partition=miso,sphinx,jag-standard,miso-lo,sphinx-lo,jag-lo,sc-loprio",
                "--constraint=ada|48G|80G|141G", "--exclude=viscam9,viscam10,iliad6,pasteur8",
                f"--time={'1-00:00:00' if student else '06:00:00'}", "--requeue", "--signal=B:USR1@300", "--open-mode=append",
                f"--output={self.directory}/slurm-%j.out", f"--export={export}",
                str(self.runtime / "scripts/slurm/async-branching-elastic-worker.sbatch")]

    def _submit(self, key, task_path=None):
        # Recover submission even if a coordinator died between sbatch and
        # journaling its response. Per-task worker locks provide a second guard.
        matches = [j for j, c in self.live.items() if c == self.comment(key)]
        if len(matches) > 1:
            raise RuntimeError(f"Duplicate live jobs for {key}: {matches}")
        job = matches[0] if matches else self.client.submit(self.command(key, task_path))
        self.jobs[key] = {"job": job, "kind": "student" if key == "student-service" else "teacher"}
        self.live[job] = self.comment(key)
        self._save()

    def teacher(self, raw):
        key = raw["id"]
        future = Future()
        self.futures[key], self.raw_tasks[key] = future, raw
        result_path = self.directory / "tasks" / key / "result.json"
        if result_path.exists():
            future.set_result(self._result(result_path))
        else:
            self.pending.append(key)
        return future

    @staticmethod
    def _result(path):
        result = read_json(path)
        if result["rollouts"] and not Path(result["hf"]).is_dir():
            raise RuntimeError(f"Committed task weights are missing: {path}")
        if result.get("replay") and not Path(result["replay"]).is_file():
            raise RuntimeError(f"Committed student replay snapshot is missing: {path}")
        return result

    def _pump(self):
        if time.monotonic() - self.last_poll < self.poll_seconds:
            return
        self.live = self.client.live()  # A failed Slurm query must fail closed.
        self.last_poll = time.monotonic()
        write_json(self.directory / "coordinator-lease.json", {"timestamp": time.time()})
        cap = gpu_cap(self.directory, self.plan["physical_gpus"])
        # Attach live jobs before counting slots, including submit/journal gaps.
        for job, comment in self.live.items():
            prefix = f"rlve-async:{self.token}:"
            if comment.startswith(prefix):
                key = comment[len(prefix):]
                previous = self.jobs.get(key)
                if previous and previous["job"] != job and previous["job"] in self.live:
                    raise RuntimeError(f"Duplicate live jobs for {key}")
                self.jobs[key] = {"job": job, "kind": "student" if key == "student-service" else "teacher"}
        self._save()
        # A restarted coordinator may replay tasks whose previous jobs ended.
        # Live jobs reattach without touching their progress. During this run,
        # failures surface instead of being silently retried indefinitely.
        for key in list(self.restored_jobs):
            if self.jobs[key]["job"] not in self.live:
                self.jobs.pop(key)
                progress = self.directory / "tasks" / key / "progress.json"
                progress.unlink(missing_ok=True)
            self.restored_jobs.remove(key)
        self._save()
        for key, future in self.futures.items():
            if future.done():
                continue
            path = self.directory / "tasks" / key / "result.json"
            if path.exists():
                future.set_result(self._result(path))
                continue
            record = self.jobs.get(key)
            if record and record["job"] not in self.live:
                # Allow the terminal atomic commit to become visible on NFS.
                since = self.missing_since.setdefault(key, time.monotonic())
                if time.monotonic() - since >= 30:
                    future.set_exception(RuntimeError(
                        f"Teacher job {record['job']} ended without a result: {key}; inspect slurm log"))
            else:
                self.missing_since.pop(key, None)
        active_teachers = sum(r["kind"] == "teacher" and r["job"] in self.live for r in self.jobs.values())
        remaining = []
        for key in self.pending:
            if self.futures[key].done() or key in self.jobs:
                continue
            if active_teachers >= cap - 2:
                remaining.append(key)
                continue
            task = self.task(self.raw_tasks[key])
            path = Path(task["directory"]) / "task.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            (path.parent / "progress.json").unlink(missing_ok=True)
            write_json(path, task)
            self._submit(key, path)
            active_teachers += 1
        self.pending = remaining
        if self.student_request:
            record = self.jobs.get("student-service")
            if not record:
                self._submit("student-service")
            elif record["job"] not in self.live:
                # Never replay a failed student silently: optimizer continuation
                # and the causal checkpoint must be reviewed on failure.
                since = self.missing_since.setdefault("student-service", time.monotonic())
                if time.monotonic() - since >= 30:
                    raise RuntimeError(f"Student service job {record['job']} ended; inspect slurm log")
            else:
                self.missing_since.pop("student-service", None)

    def information(self, task, future):
        path = self.directory / "tasks" / task["id"] / "progress.json"
        while True:
            self._pump()
            if future.done():
                return future.result()
            if path.exists():
                bound = read_json(path).get("next_validation_rollouts")
                if bound is not None and bound > task["minimum_rollouts"]:
                    return {"lower_bound_rollouts": bound}
            time.sleep(min(self.poll_seconds, .2))

    def student(self, raw):
        task = self.task(raw)
        path = Path(task["directory"])
        path.mkdir(parents=True, exist_ok=True)
        result_path = path / "result.json"
        if result_path.exists():
            return self._result(result_path)
        write_json(path / "task.json", task)
        self.student_request = task["id"]
        write_json(self.directory / "student-service/request.json", {"task": str(path / "task.json")})
        while not result_path.exists():
            self._pump()
            time.sleep(min(self.poll_seconds, .2))
        self.student_request = None
        return self._result(result_path)

    def close(self):
        # On coordinator interruption leave workers alive. A requeued CPU job
        # reattaches to their job IDs and durable immutable requests.
        if self.state.stopped:
            self.stop_workers(self.directory, self.client)

    @staticmethod
    def stop_workers(directory, client=None):
        client = client or SlurmClient()
        token = hashlib.sha256(str(Path(directory).resolve()).encode()).hexdigest()[:16]
        prefix = f"rlve-async:{token}:"
        live = {j for j, c in client.live().items() if c.startswith(prefix)}
        client.cancel(live)
        deadline = time.monotonic() + 55
        while live:
            live.intersection_update(client.live())
            if live and time.monotonic() >= deadline:
                raise RuntimeError("Workers are still stopping; refusing checkpoint cleanup")
            if live:
                time.sleep(.5)
