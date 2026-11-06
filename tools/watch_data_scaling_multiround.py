"""Bounded Slurm recovery for final-only multi-round experiments.

Training jobs keep their IDs across retries; pending students have their
dependencies refreshed because Slurm remembers a dependency's earlier failure.
No intermediate training state is introduced.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from uuid import uuid4

from finalize_data_scaling import validate_terminal_checkpoint


ROOT = Path(__file__).resolve().parents[1]
LARGE_GPU_PARTITIONS = "miso,sphinx,miso-lo,sphinx-lo,sc-loprio"
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
          "REQUEUED", "RESIZING", "REQUEUE_HOLD"}
FAILURES = {"FAILED", "NODE_FAIL", "BOOT_FAIL", "PREEMPTED", "OUT_OF_MEMORY"}


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.PIPE, timeout=45).strip()


def job_info(job):
    text = command(["scontrol", "show", "job", str(job), "-o"])
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:]*)=(\S+)", text))
    if fields.get("JobId") != str(job) or not fields.get("UserId", "").endswith(f"({os.getuid()})"):
        raise RuntimeError(f"Cannot establish ownership of job {job}")
    return fields


def log_tail(run):
    path = run / "train.log"
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 4 * 1024**2))
        text = handle.read().decode("utf-8", errors="replace")
    # Old failed attempts share a log. Only classify the latest attempt.
    return text.rsplit("Final-model run ", 1)[-1]


def retry_reason(state, text):
    if state == "OUT_OF_MEMORY":
        return "gpu_or_host_memory"
    if state in {"NODE_FAIL", "BOOT_FAIL", "PREEMPTED"}:
        return "node_or_preemption"
    if state != "FAILED":
        return None
    if re.search(r"CUDA out of memory|OutOfMemoryError|CUresult error: 2 \(out of memory\)", text):
        return "gpu_or_host_memory"
    # Never retry arbitrary configuration or application exceptions.
    if re.search(r"ActorDiedError|ActorUnavailableError|RaySystemError|Raylet is terminated", text):
        return "worker_crash"
    if re.search(r"NCCL.*(?:connection|remote process exited)|ConnectionResetError|Connection refused", text):
        return "connection_failure"
    return None


def exported(run):
    marker = run / "weights_only_complete.json"
    if not marker.exists():
        return False
    data = json.loads(marker.read_text())
    index = run / "hf-final/model.safetensors.index.json"
    if data.get("format") != "hf-safetensors" or data.get("weight_bytes", 0) <= 0 or not index.is_file():
        raise RuntimeError(f"Invalid final export marker in {run}")
    shards = set(json.loads(index.read_text())["weight_map"].values())
    if not shards or any(not (index.parent / name).is_file() or (index.parent / name).stat().st_size == 0
                         for name in shards):
        raise RuntimeError(f"Missing final export weights in {run}")
    return True


class Watchdog:
    def __init__(self, experiment, max_retries=2):
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        self.experiment = Path(experiment).resolve()
        self.path = self.experiment / "recovery_watchdog.json"
        self.plan = json.loads((self.experiment / "run_plan.json").read_text())
        if self.plan.get("method") != "multi-round" or self.plan.get("round") != 2:
            raise ValueError("Expected a round-2 multi-round experiment")
        self.specs = {run["label"]: run for run in self.plan["runs"]}
        self.max_retries = max_retries
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "version": 1, "events": [], "alerts": [], "status": "watching"}

    def save(self):
        self.state.update(updated_at=time.time(), max_retries=self.max_retries)
        atomic_json(self.path, self.state)

    def rows(self):
        with (self.experiment / "jobs.tsv").open() as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def write_rows(self, rows):
        path = self.experiment / "jobs.tsv"
        backup = self.experiment / "jobs.before-watchdog-finalization.tsv"
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        temporary = path.with_name(f".jobs.{uuid4().hex}.tmp")
        with temporary.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def run_dir(self, row):
        path = self.experiment / row["kind"] / row["label"]
        if path.resolve().parent != self.experiment / row["kind"]:
            raise ValueError("Recovery is restricted to this experiment's new runs")
        return path

    def verify_inputs(self):
        command(["sha256sum", "--check", "--status", str(self.experiment / "input_sha256.txt")])

    def refresh_students(self, rows):
        waiting = [row["job"] for row in rows if row["kind"] == "teacher"
                   and row.get("phase") != "shared" and not exported(self.run_dir(row))]
        dependency = "afterok:" + ":".join(waiting) if waiting else ""
        for row in rows:
            if row["kind"] != "mopd" or exported(self.run_dir(row)):
                continue
            info = job_info(row["job"])
            if info["JobState"] == "PENDING":
                command(["scontrol", "update", f"JobId={row['job']}", f"Dependency={dependency}"])

    def resume_retry(self, event, rows):
        """Persist each step so monitor restart cannot duplicate a requeue."""
        job = event["job"]
        info = job_info(job)
        if event["stage"] == "requested":
            restarts = int(info.get("Restarts", 0))
            if restarts == event["restarts_before"] and info["JobState"] in FAILURES:
                self.verify_inputs()
                command(["scontrol", "requeuehold", job])
                info = job_info(job)
            if int(info.get("Restarts", 0)) <= event["restarts_before"]:
                raise RuntimeError(f"Cannot confirm held requeue of {job}")
            if info["JobState"] not in {"PENDING", "REQUEUE_HOLD"}:
                raise RuntimeError(f"Requeue of {job} is not held: {info['JobState']}")
            event["stage"] = "held"
            self.save()
        if event["stage"] == "held":
            if event["reason"] in {"gpu_or_host_memory", "worker_crash"}:
                command(["scontrol", "update", f"JobId={job}", "Features=80G|141G",
                         f"Partition={LARGE_GPU_PARTITIONS}"])
                event["gpu_constraint"] = "80G|141G"
            event["stage"] = "configured"
            self.save()
        if event["stage"] == "configured":
            # Explicitly replace Slurm's remembered failed dependency before release.
            self.refresh_students(rows)
            event["stage"] = "dependencies_repaired"
            self.save()
        if event["stage"] == "dependencies_repaired":
            info = job_info(job)
            if info["JobState"] == "PENDING" and int(info.get("Priority", 0)) == 0:
                command(["scontrol", "release", job])
            elif info["JobState"] not in ACTIVE:
                raise RuntimeError(f"Unexpected state releasing {job}: {info['JobState']}")
            event["stage"] = "requeued"
            self.save()
            print(f"Recovered {event['label']} job={job}: {event['reason']}", flush=True)

    def finalize(self, row):
        """Recover a terminal checkpoint without rerunning the trainer."""
        run = self.run_dir(row)
        validate_terminal_checkpoint(run, self.specs[row["label"]]["updates"])
        # Export can need tens of GiB of RAM: run it in its own CPU allocation.
        # A deterministic comment lets us discover a submission after monitor death.
        token = hashlib.sha256(str(run).encode()).hexdigest()[:16]
        comment = "rlve-multiround-finalize:" + token
        found = command(["squeue", "--me", "-h", "-o", "%i|%k"])
        jobs = [line.split("|", 1)[0] for line in found.splitlines() if line.endswith("|" + comment)]
        if len(jobs) > 1:
            raise RuntimeError(f"Multiple finalizers for {row['label']}")
        if jobs:
            job = jobs[0]
        else:
            exports = ",".join(["ALL", f"RLVE_REPO={ROOT}",
                                f"RLVE_SCRATCH={self.experiment.parent.parent}",
                                f"RLVE_SCALING_EXPERIMENT={self.experiment}",
                                f"RLVE_SCALING_FINALIZE_LABEL={row['label']}"])
            job = command(["sbatch", "--parsable", "--hold", "--comment=" + comment, "--export=" + exports,
                           str(ROOT / "scripts/slurm/finalize-multiround-resilient.sbatch")]).split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"Invalid finalization job ID: {job}")
        previous = row["job"]
        row.update(job=job, gpus="0", phase="finalize", dependency="none")
        self.state["events"].append({"label": row["label"], "job": job,
                                     "previous_job": previous, "stage": "finalizer_submitted", "at": time.time()})
        self.save()

    def finish_finalizer_submission(self, event, rows):
        row = next(row for row in rows if row["label"] == event["label"])
        if row["job"] not in {event["previous_job"], event["job"]}:
            raise RuntimeError("Manifest changed during finalizer submission")
        row.update(job=event["job"], gpus="0", phase="finalize", dependency="none")
        self.write_rows(rows)
        self.refresh_students(rows)
        info = job_info(event["job"])
        if info["JobState"] == "PENDING" and int(info.get("Priority", 0)) == 0:
            command(["scontrol", "release", event["job"]])
        elif info["JobState"] not in ACTIVE and info["JobState"] != "COMPLETED":
            raise RuntimeError("Finalizer changed state before submission completed")
        event["stage"] = "finalizer_ready"
        self.save()

    def tick(self):
        rows = self.rows()
        monitored = [row for row in rows if row.get("phase") != "shared"
                     and row["kind"] in {"teacher", "mopd"}]
        self.state["alerts"] = []
        # Finish an interrupted recovery before taking any new action.
        for event in self.state["events"]:
            if event["stage"] == "finalizer_submitted":
                self.finish_finalizer_submission(event, rows)
            if event["stage"] in {"requested", "held", "configured", "dependencies_repaired"}:
                row = next(row for row in rows if row["label"] == event["label"])
                if row["job"] != event["job"]:
                    raise RuntimeError("Manifest changed during recovery")
                self.resume_retry(event, rows)
        complete = 0
        for row in monitored:
            run = self.run_dir(row)
            if exported(run):
                complete += 1
                continue
            info = job_info(row["job"])
            state = info["JobState"]
            if state in ACTIVE:
                continue
            # A user's cancellation is authoritative; never resurrect it.
            if state.startswith("CANCELLED"):
                self.state["alerts"].append({"label": row["label"], "reason": "cancelled"})
                continue
            if row.get("phase") != "finalize":
                try:
                    validate_terminal_checkpoint(run, self.specs[row["label"]]["updates"])
                except (OSError, ValueError, RuntimeError):
                    pass
                else:
                    self.verify_inputs()
                    self.finalize(row)
                    self.finish_finalizer_submission(self.state["events"][-1], rows)
                    continue
            reason = retry_reason(state, log_tail(run))
            if row.get("phase") == "finalize":
                # Finalization retries rerun export only, never training.
                reason = "export_failure" if state in FAILURES else None
            used = sum(event["label"] == row["label"] and "restarts_before" in event
                       for event in self.state["events"])
            # Include Slurm's automatic restarts in the budget after attachment.
            used = max(used, int(info.get("Restarts", 0)))
            if reason is None or used >= self.max_retries:
                self.state["alerts"].append({"label": row["label"], "job": row["job"],
                                             "reason": "retry_limit" if reason else "unrecognized_failure",
                                             "state": state})
                continue
            self.verify_inputs()
            event = {"label": row["label"], "job": row["job"], "reason": reason,
                     "restarts_before": int(info.get("Restarts", 0)), "stage": "requested", "at": time.time()}
            self.state["events"].append(event)
            self.save()
            self.resume_retry(event, rows)
        # Also repairs a stale failed dependency if export finished while this
        # monitor was down. Readiness comes from exports, never a failed job ID.
        for row in monitored:
            if row["kind"] == "mopd" and not exported(self.run_dir(row)):
                info = job_info(row["job"])
                if info["JobState"] == "PENDING" and (
                    "failed" in info.get("Dependency", "")
                    or info.get("Reason") == "DependencyNeverSatisfied"
                ):
                    self.refresh_students(rows)
                    break
        self.state.update(status="complete" if complete == len(monitored) else "watching",
                          exports_complete=complete, runs=len(monitored))
        self.save()
        return self.state["status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=60)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true", help="Attach a single CPU recovery job")
    mode.add_argument("--once", action="store_true", help="Perform one recovery pass")
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("poll-seconds must be at least 1")
    os.chdir(ROOT)
    watchdog = Watchdog(args.experiment, args.max_retries)
    lock_name = "recovery_watchdog_submit.lock" if args.submit else "recovery_watchdog.lock"
    with (watchdog.experiment / lock_name).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.submit:
            registration = watchdog.experiment / "recovery_watchdog_job.json"
            if registration.exists():
                prior = json.loads(registration.read_text())
                if job_info(prior["job"])["JobState"] in ACTIVE:
                    print(f"Recovery monitor already queued: {prior['job']}")
                    return
            token = hashlib.sha256(str(watchdog.experiment).encode()).hexdigest()[:16]
            comment = "rlve-multiround-watchdog:" + token
            queued = command(["squeue", "--me", "-h", "-o", "%i|%k"])
            matches = [line.split("|", 1)[0] for line in queued.splitlines() if line.endswith("|" + comment)]
            if matches:
                if len(matches) != 1:
                    raise RuntimeError("Multiple recovery monitors are already queued")
                atomic_json(registration, {"job": matches[0], "recovered_registration_at": time.time()})
                print(f"Recovery monitor already queued: {matches[0]}")
                return
            # Freeze the monitor separately; existing training input hashes stay intact.
            paths = [Path(__file__).resolve(), ROOT / "tools/finalize_data_scaling.py",
                     ROOT / "tools/export_data_scaling_checkpoint.py",
                     ROOT / "scripts/slurm/watch-multiround-resilient.sbatch",
                     ROOT / "scripts/slurm/finalize-multiround-resilient.sbatch"]
            manifest = watchdog.experiment / "recovery_runtime_sha256.txt"
            manifest.write_text("".join(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + str(path) + "\n"
                                        for path in paths))
            exports = ",".join(["ALL", f"RLVE_REPO={ROOT}",
                                f"RLVE_SCALING_EXPERIMENT={watchdog.experiment}",
                                f"RLVE_RECOVERY_MAX_RETRIES={args.max_retries}",
                                f"RLVE_RECOVERY_POLL_SECONDS={args.poll_seconds}"])
            job = command(["sbatch", "--parsable", "--comment=" + comment, "--export=" + exports,
                           str(ROOT / "scripts/slurm/watch-multiround-resilient.sbatch")]).split(";")[0]
            if not job.isdigit():
                raise RuntimeError(f"Invalid monitor job ID: {job}")
            atomic_json(registration, {"job": job, "submitted_at": time.time()})
            print(f"Attached CPU recovery monitor {job} to {watchdog.experiment}")
            return
        while True:
            try:
                status = watchdog.tick()
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                # Keep polling on controller/network outages; never launch blindly.
                watchdog.state["last_error"] = str(error)
                watchdog.save()
                print(f"Recovery monitor error: {error}", flush=True)
                if args.once:
                    raise
                status = "watching"
            if args.once or status == "complete":
                break
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
