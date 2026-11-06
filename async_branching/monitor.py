"""Persistent health reporting and bounded coordinator recovery for elastic runs."""

import argparse
import fcntl
import getpass
import os
from pathlib import Path
import subprocess
import time

from .io import read_json, write_json
from .watchdog import ProgressWatchdog


def queue():
    output = subprocess.check_output(
        ["squeue", "--noheader", "--user", getpass.getuser(), "--format=%A|%T|%R|%S|%k"], text=True)
    return {parts[0]: {"state": parts[1], "reason": parts[2],
                       "start_time": parts[3], "comment": parts[4]}
            for line in output.splitlines() if len(parts := line.strip().split("|", 4)) == 5}


def inspect(directory, jobs, current_job):
    directory = Path(directory)
    state = read_json(directory / "state.json") if (directory / "state.json").exists() else {}
    journal = read_json(directory / "slurm_jobs.json") if (directory / "slurm_jobs.json").exists() else {}
    workers = {key: {**value, **jobs.get(value["job"], {"state": "ENDED"})}
               for key, value in journal.items()}
    lease = directory / "coordinator-lease.json"
    return {"timestamp": time.time(), "coordinator_job": current_job,
            "coordinator": jobs.get(current_job, {"state": "ENDED"}),
            "coordinator_heartbeat_age_seconds": (time.time() - read_json(lease)["timestamp"] if lease.exists() else None),
            "student_rollouts": state.get("student_rollouts", 0), "clock": state.get("clock", 0),
            "stopped": state.get("stopped"), "completed": (directory / "completed.json").exists(),
            "workers": workers,
            "running_worker_gpus": sum((2 if item["kind"] == "student" else 1)
                                       for item in workers.values() if item["state"] == "RUNNING")}


def restart(directory):
    submission = read_json(directory / "submission.json")
    job = subprocess.check_output(submission["command"], text=True).strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError(f"Unexpected coordinator sbatch response: {job}")
    return job


def teacher_validation(directory, job, live, task="teacher-00-g0001"):
    """Require the real three-mode parser probe and a completed training step."""
    directory = Path(directory)
    path = directory / "tasks" / task
    terminal = (path / "result.json").exists()
    if job not in live and not terminal:
        return "missing"
    log = path / "worker.log"
    content = log.read_text() if log.exists() else ""
    import json
    parsed = set()
    for line in content.splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict) and "kind" in value and "start_update" in value:
                parsed.add((value["kind"], value["start_update"]))
        except ValueError:
            pass
    modes = {("teacher", 0), ("student", 0), ("student", 50)}
    if modes <= parsed and ("david 0:" in content or terminal):
        return "passed"
    return "waiting"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--stall-minutes", type=float, default=30)
    parser.add_argument("--startup-minutes", type=float, default=45)
    parser.add_argument("--worker-requeues", type=int, default=2)
    parser.add_argument("--validate-teacher-job", help="Resume the coordinator only after this GPU startup probe passes")
    args = parser.parse_args()
    directory = args.directory.resolve()
    monitor = directory / "monitor.json"
    saved = read_json(monitor) if monitor.exists() else {}
    current = saved.get("coordinator_job", read_json(directory / "submission.json")["job"])
    restarts = saved.get("restarts", [])
    missing = 0
    with (directory / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        watchdog = ProgressWatchdog(directory, args.stall_minutes * 60,
                                    args.startup_minutes * 60, args.worker_requeues)
        if args.validate_teacher_job:
            validation = directory / "gpu-startup-validation.json"
            record = read_json(validation)
            if record.get("status") == "passed" and (
                    not current or saved.get("timestamp", 0) <= record.get("validated_at", 0)):
                current = record["coordinator_job"]
            if record.get("status") != "passed":
                absent = 0
                while True:
                    try:
                        live = queue()
                        status = teacher_validation(directory, args.validate_teacher_job, live,
                                                    task=record.get("task", "teacher-00-g0001"))
                        if status == "passed" and record.get("require_terminal"):
                            result = directory / "tasks" / record["task"] / "result.json"
                            if args.validate_teacher_job in live or not result.exists():
                                status = "waiting"
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        print(f"Startup validation check unavailable: {error}", flush=True)
                        time.sleep(args.interval)
                        continue
                    absent = absent + 1 if status == "missing" else 0
                    if (directory / "monitor-stop.json").exists():
                        return
                    if status == "passed":
                        current = restart(directory)
                        record.update(status="passed", validated_at=time.time(), coordinator_job=current)
                        write_json(validation, record)
                        write_json(monitor, {"timestamp": time.time(), "coordinator_job": current,
                                            "monitor_job": os.environ.get("SLURM_JOB_ID"),
                                            "monitor_status": "validation_passed_coordinator_submitted", "restarts": restarts})
                        print(f"GPU teacher validation passed; coordinator={current}", flush=True)
                        break
                    progress = {"timestamp": time.time(), "monitor_job": os.environ.get("SLURM_JOB_ID"),
                                "coordinator_job": None, "student_rollouts": 0, "restarts": restarts,
                                "validation_job": args.validate_teacher_job,
                                "validation_job_state": live.get(args.validate_teacher_job, {"state": "ENDED"}),
                                "monitor_status": "alert_teacher_validation_failed" if absent >= 2 else "validating_teacher"}
                    write_json(monitor, progress)
                    print(f"{progress['monitor_status']}: job={args.validate_teacher_job}, {progress['validation_job_state']}", flush=True)
                    if absent >= 2:
                        record.update(status="failed", checked_at=time.time())
                        write_json(validation, record)
                        return
                    time.sleep(args.interval)
        while True:
            try:
                status = inspect(directory, queue(), current)
                status.update(monitor_job=os.environ.get("SLURM_JOB_ID"), restarts=restarts,
                              recovery_limit=args.max_restarts)
                if (directory / "monitor-stop.json").exists():
                    status["monitor_status"] = "stopped_by_user"
                    write_json(monitor, status)
                    return
                if status["completed"]:
                    status["monitor_status"] = "completed"
                    write_json(monitor, status)
                    return
                if status["coordinator"]["state"] == "RUNNING" and not status["stopped"]:
                    status["watchdog"] = watchdog.check(status["workers"], queue)
                ended = status["coordinator"]["state"] == "ENDED"
                missing = missing + 1 if ended else 0
                status["monitor_status"] = "coordinator_missing" if ended else "watching"
                age = status["coordinator_heartbeat_age_seconds"]
                if status["coordinator"]["state"] == "RUNNING" and age is not None and age > 1800:
                    status["monitor_status"] = "alert_stale_coordinator_heartbeat"
                storage = directory / "storage-wait.json"
                if (status["monitor_status"] == "watching" and storage.exists()
                        and not (directory / "state.json").exists()):
                    status["storage"] = read_json(storage)
                    if status["storage"]["status"] == "waiting":
                        status["monitor_status"] = "waiting_for_storage"
                if missing >= 2:
                    accounting = subprocess.check_output(
                        ["sacct", "-X", "--noheader", "--parsable2", "-j", current, "--format=State"], text=True)
                    if "CANCELLED" in accounting:
                        status["monitor_status"] = "coordinator_cancelled_no_restart"
                        write_json(monitor, status)
                        return
                    if len(restarts) >= args.max_restarts:
                        status["monitor_status"] = "alert_recovery_limit_reached"
                    else:
                        previous = current
                        current = restart(directory)
                        restarts.append({"timestamp": time.time(), "previous_job": previous, "job": current})
                        status.update(coordinator_job=current, monitor_status="coordinator_restarted", restarts=restarts)
                        missing = 0
                if status["monitor_status"] == "watching" and status.get("watchdog", {}).get("alerts"):
                    status["monitor_status"] = "alert_worker_watchdog"
                elif status["monitor_status"] == "watching" and any(
                        item.get("phase") == "requeue_requested"
                        for item in status.get("watchdog", {}).get("workers", {}).values()):
                    status["monitor_status"] = "worker_requeued"
                write_json(monitor, status)
                print(f"{status['monitor_status']}: coordinator={current}, student={status['student_rollouts']}, "
                      f"running_gpu={status['running_worker_gpus']}, restarts={len(restarts)}", flush=True)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                # Do not submit recovery jobs based on an unavailable scheduler
                # or partially unavailable storage. Recheck on the next poll.
                print(f"Monitor check failed: {type(error).__name__}: {error}", flush=True)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
