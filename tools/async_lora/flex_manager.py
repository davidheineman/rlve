"""Bounded Slurm generation capacity owned by one running core allocation."""

import hashlib
import os
import shlex
import subprocess
import time

from async_branching.io import read_json, write_json
from async_lora.flex_protocol import read_optional, unregister


class FlexManager:
    def __init__(self, args):
        self.args = args
        self.experiment = args.experiment.resolve()
        self.directory = self.experiment / "flex"
        self.directory.mkdir(exist_ok=True)
        self.limit = read_json(self.experiment / "plan.json").get("elastic_generation", {}).get("max_extra_gpus", 0)
        self.prefix = "rlve-flex-" + hashlib.sha256(str(self.experiment).encode()).hexdigest()[:16] + ":"
        self.owned = {}
        self.attempts = {}
        self.last_poll = 0

    def jobs(self):
        output = subprocess.check_output(
            ["squeue", "--noheader", "--user", os.environ["USER"], "--format=%i|%k|%T"], text=True, timeout=20
        )
        return [line.strip().split("|") for line in output.splitlines() if self.prefix in line]

    def submit(self, core, slot):
        command = [
            "python3",
            str(self.experiment / "source/rlve/tools/submit_async_lora.py"),
            "--flex-worker",
            "--experiment",
            str(self.experiment),
            "--slot",
            str(slot),
            "--generation",
            core["generation"],
            "--scratch",
            str(self.args.scratch),
            "--image",
            str(self.args.image),
            "--model-path",
            str(self.args.model_path),
        ]
        comment = f"{self.prefix}{core['generation']}:{slot}"
        argv = [
            "sbatch",
            "--parsable",
            "--account=miso",
            "--partition=miso,sphinx,jag-standard,miso-lo,sphinx-lo,jag-lo,sc-loprio",
            "--nodes=1",
            "--ntasks=1",
            "--gres=gpu:1",
            "--cpus-per-task=4",
            "--mem=24G",
            "--time=06:00:00",
            "--constraint=48G|80G|141G",
            "--exclude=pasteur8,jagupard30",
            "--requeue",
            "--signal=B:USR1@120",
            f"--comment={comment}",
            f"--job-name=async-lora-flex-{slot}",
            f"--chdir={self.experiment}",
            f"--output={self.directory}/slurm-%j.out",
            "--wrap",
            "exec " + shlex.join(command),
        ]
        job = subprocess.check_output(argv, text=True, timeout=30).strip().split(";")[0]
        self.owned[job] = {"slot": slot, "generation": core["generation"], "comment": comment}
        self.attempts[slot] = self.attempts.get(slot, 0) + 1
        write_json(self.directory / "jobs.json", self.owned)

    def remove_worker(self, core, slot, job):
        record = read_optional(self.directory / f"worker-{slot}.json")
        if record and record.get("job_id") == job and record.get("url"):
            unregister(core["router_url"], record["url"])

    def tick(self):
        if not self.limit or time.monotonic() - self.last_poll < 5:
            return
        self.last_poll = time.monotonic()
        core = read_optional(self.directory / "core.json")
        if not core or core["core_job_id"] != os.environ["SLURM_JOB_ID"]:
            return
        write_json(self.directory / "lease.json", {"generation": core["generation"], "time": time.time()})
        try:
            records = self.jobs()
            active = {}
            for job, comment, state in records:
                generation, slot = comment.removeprefix(self.prefix).split(":")
                if generation != core["generation"]:
                    subprocess.run(["scancel", job], check=True, timeout=20)
                    continue
                active[job] = {"slot": int(slot), "generation": generation, "state": state}
                self.owned.setdefault(job, active[job])
            for job, record in list(self.owned.items()):
                if job not in active or active[job]["state"] != "RUNNING":
                    self.remove_worker(core, record["slot"], job)
                else:
                    heartbeat = read_optional(self.directory / f"worker-{record['slot']}.json")
                    if (
                        heartbeat
                        and heartbeat.get("job_id") == job
                        and heartbeat.get("stage") == "ready"
                        and time.time() - heartbeat["time"] > 30
                    ):
                        self.remove_worker(core, record["slot"], job)
                        subprocess.run(["scancel", job], check=True, timeout=20)
            used = {record["slot"] for record in active.values()}
            vacancies = max(0, self.limit - len(records))
            if read_optional(self.directory / "probe.json"):
                for slot in range(self.limit):
                    if vacancies and slot not in used and self.attempts.get(slot, 0) < 3:
                        self.submit(core, slot)
                        vacancies -= 1
            write_json(
                self.directory / "status.json",
                {
                    "time": time.time(),
                    "generation": core["generation"],
                    "max_extra_gpus": self.limit,
                    "active_jobs": active,
                    "submitted_jobs": self.owned,
                    "attempts": self.attempts,
                },
            )
        except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
            write_json(self.directory / "manager_error.json", {"time": time.time(), "error": str(error)})

    def close(self):
        if not self.limit:
            return
        # Expire the lease first, including if Slurm is temporarily unreachable.
        write_json(self.directory / "lease.json", {"generation": "stopped", "time": 0})
        try:
            jobs = {record[0] for record in self.jobs()} | set(self.owned)
            if jobs:
                subprocess.run(["scancel", *sorted(jobs)], check=False, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass
