"""Bounded Slurm requeues based on completed training steps, not log traffic."""

import hashlib
from pathlib import Path
import re
import subprocess
import time

from .io import read_json, write_json


STEP = re.compile(rb"(?:^|[\r\n])david (\d+):[^\r\n]*['\"]david/rollouts['\"]:\s*(\d+)")
MAX_READ = 4 * 1024 * 1024


def training_steps(path, observation):
    """Read only new log bytes, keeping a partial final line for the next poll."""
    if not path.exists():
        return []
    stat = path.stat()
    offset = observation.get("offset", 0)
    if observation.get("inode") != stat.st_ino or stat.st_size < offset:
        offset = 0
        observation["rollouts"] = -1
    offset = max(offset, stat.st_size - MAX_READ)
    with path.open("rb") as handle:
        handle.seek(offset)
        content = handle.read(MAX_READ)
    end = content.rfind(b"\n") + 1
    observation.update(offset=offset + end, inode=stat.st_ino)
    return [(int(step), int(rollouts)) for step, rollouts in STEP.findall(content[:end])]


class ProgressWatchdog:
    def __init__(self, directory, stall_seconds=1800, startup_seconds=2700, max_requeues=2):
        if stall_seconds <= 0 or startup_seconds <= 0 or max_requeues < 0:
            raise ValueError("Watchdog timeouts must be positive and retry limit nonnegative")
        self.directory = Path(directory).resolve()
        self.path = self.directory / "watchdog.json"
        self.state = read_json(self.path) if self.path.exists() else {"workers": {}, "recoveries": []}
        self.stall_seconds, self.startup_seconds = stall_seconds, startup_seconds
        self.max_requeues = max_requeues
        token = hashlib.sha256(str(self.directory).encode()).hexdigest()[:16]
        self.prefix = f"rlve-async:{token}:"

    def task(self, key, worker):
        if worker["kind"] == "teacher":
            path = self.directory / "tasks" / key
        else:
            request = self.directory / "student-service/request.json"
            if not request.exists():
                return None
            path = Path(read_json(request)["task"]).resolve().parent
        if path.parent != self.directory / "tasks":
            raise ValueError(f"Worker task is outside this experiment: {path}")
        return None if (path / "result.json").exists() else path

    def check(self, workers, refresh_queue, now=None):
        now = time.time() if now is None else now
        observations, alerts = {}, []
        for key, worker in workers.items():
            # Pending, suspended and idle services must not consume timeout.
            if worker["state"] != "RUNNING":
                continue
            task = self.task(key, worker)
            if task is None:
                continue
            identity = {"job": worker["job"], "start_time": worker.get("start_time"), "task": task.name}
            previous = self.state["workers"].get(key, {})
            log = task / "worker.log"
            if any(previous.get(k) != v for k, v in identity.items()) or previous.get("phase") == "not_running":
                # On attachment/requeue ignore historical bytes from old attempts.
                # Grant the startup allowance from observation rather than guessing
                # timestamps from interleaved or inherited worker logs.
                stat = log.stat() if log.exists() else None
                observation = {**identity, "last_progress_at": now, "step": None, "rollouts": -1,
                               "phase": "startup", "offset": stat.st_size if stat else 0,
                               "inode": stat.st_ino if stat else None}
            else:
                observation = dict(previous)
            for step, rollouts in training_steps(log, observation):
                if rollouts > observation["rollouts"]:
                    observation.update(last_progress_at=now, step=step, rollouts=rollouts, phase="training")
            age = max(0, now - observation["last_progress_at"])
            limit = self.startup_seconds if observation["step"] is None else self.stall_seconds
            observation.update(no_progress_seconds=age, timeout_seconds=limit)
            observations[key] = observation
            if age < limit:
                continue
            retries = sum(item["task"] == task.name for item in self.state["recoveries"])
            if retries >= self.max_requeues:
                observation["phase"] = "alert_retry_limit_reached"
                alerts.append({"task": task.name, "reason": "retry_limit_reached"})
                continue
            # Recheck ownership, attempt identity and the terminal commit just
            # before acting; never requeue an unrelated/replaced or finished job.
            fresh = refresh_queue().get(worker["job"], {})
            if (fresh.get("state") != "RUNNING" or fresh.get("comment") != self.prefix + key
                    or fresh.get("start_time") != worker.get("start_time")
                    or (task / "result.json").exists()):
                alerts.append({"task": task.name, "reason": "requeue_precondition_changed"})
                continue
            event = {"timestamp": now, **identity, "step": observation["step"],
                     "rollouts": observation["rollouts"], "no_progress_seconds": age,
                     "action": "requeue_requested"}
            self.state["recoveries"].append(event)
            # Persist intent before the scheduler mutation. A monitor crash cannot
            # forget an attempt and bypass the per-task recovery limit.
            observation.update(last_progress_at=now, phase="requeue_requested")
            self.state["workers"].update(observations)
            write_json(self.path, self.state)
            try:
                subprocess.run(["scontrol", "requeue", worker["job"]], check=True,
                               capture_output=True, text=True, timeout=30)
                event["action"] = "requeued"
                print(f"Progress watchdog requeued {task.name} job={worker['job']} after {age:.0f}s "
                      f"without step progress (last step={observation['step']})", flush=True)
            except (OSError, subprocess.SubprocessError) as error:
                event.update(action="requeue_failed", error=str(error))
                observation["phase"] = "alert_requeue_failed"
                alerts.append({"task": task.name, "reason": "requeue_failed"})
        # A worker returning from suspension/queue gets a fresh startup allowance.
        for key, worker in workers.items():
            if worker["state"] != "RUNNING" and key in self.state["workers"]:
                observations[key] = {**self.state["workers"][key], "phase": "not_running"}
        self.state.update(timestamp=now, workers=observations, alerts=alerts,
                          stall_seconds=self.stall_seconds, startup_seconds=self.startup_seconds,
                          max_requeues_per_task=self.max_requeues)
        write_json(self.path, self.state)
        return {"alerts": alerts, "requeues": len(self.state["recoveries"]), "workers": observations,
                "stall_seconds": self.stall_seconds, "startup_seconds": self.startup_seconds,
                "max_requeues_per_task": self.max_requeues}
