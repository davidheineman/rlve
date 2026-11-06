"""Monitor the recorded jobs; recover recognized infrastructure loss."""

import fcntl
import json
import subprocess
import time
from pathlib import Path

from async_branching.io import read_json, write_json

ACTIVE = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED"}
INFRA = {"NODE_FAIL", "BOOT_FAIL", "TIMEOUT", "PREEMPTED"}


def state(job):
    queue = subprocess.run(
        ["squeue", "-h", "-j", job, "-o", "%T|%N|%R"],
        text=True,
        capture_output=True,
        timeout=20,
    )
    # Finished jobs eventually disappear from slurmctld. squeue then exits 1
    # instead of returning an empty listing; accounting still owns their state.
    if queue.returncode and "invalid job id" not in queue.stderr.lower():
        queue.check_returncode()
    row = queue.stdout.strip()
    if row:
        status, node, reason = row.splitlines()[0].split("|", 2)
        return {"state": status, "node": node, "reason": reason}
    rows = subprocess.check_output(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            job,
            "--format=JobID,State,NodeList,ExitCode",
        ],
        text=True,
        timeout=20,
    )
    for row in reversed(rows.splitlines()):
        fields = row.split("|")
        if fields[0] == job:
            return {
                "state": fields[1].split()[0].rstrip("+"),
                "node": fields[2],
                "reason": fields[3],
            }
    return {"state": "UNKNOWN", "node": "", "reason": "not yet in accounting"}


def tail(path, limit=1024 * 1024):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode(errors="replace")


def recover_job(sweep, directory, job, *, exclude_node=None):
    argv = read_json(directory / "submit_argv.json")
    if exclude_node:
        for index, value in enumerate(argv):
            if value.startswith("--exclude="):
                excluded = set(value.split("=", 1)[1].split(","))
                argv[index] = "--exclude=" + ",".join(sorted(excluded | {exclude_node}))
                break
        else:
            index = argv.index("--wrap") if "--wrap" in argv else len(argv)
            argv.insert(index, "--exclude=" + exclude_node)
        write_json(directory / "submit_argv.json", argv)
    held = subprocess.run(
        ["scontrol", "requeuehold", job], text=True, capture_output=True, timeout=30
    )
    if held.returncode:
        if "invalid job id" not in held.stderr.lower():
            held.check_returncode()
        # Once purged, Slurm cannot requeue the original ID. Reuse the same
        # experiment, durable checkpoints, W&B identities, and frozen arguments.
        replacement = (
            subprocess.check_output(argv, text=True, timeout=30).strip().split(";")[0]
        )
        record = read_json(directory / "job.json")
        record["previous_job_ids"] = [*record.get("previous_job_ids", []), job]
        record["job_id"] = replacement
        write_json(directory / "job.json", record)
        manifest = read_json(sweep / "sweep.json")
        manifest["job_ids"] = [
            replacement if str(value) == job else value for value in manifest["job_ids"]
        ]
        write_json(sweep / "sweep.json", manifest)
        return replacement
    if exclude_node:
        description = subprocess.check_output(
            ["scontrol", "show", "job", job, "-o"], text=True, timeout=20
        )
        fields = dict(part.split("=", 1) for part in description.split() if "=" in part)
        prior = fields.get("ExcNodeList", "(null)")
        excluded = set() if prior == "(null)" else set(prior.split(","))
        subprocess.run(
            [
                "scontrol",
                "update",
                f"JobId={job}",
                "ExcNodeList=" + ",".join(sorted(excluded | {exclude_node})),
            ],
            check=True,
            timeout=20,
        )
    subprocess.run(["scontrol", "release", job], check=True, timeout=20)
    return job


def check(sweep, *, repair=False):
    manifest = read_json(sweep / "sweep.json")
    if not manifest.get("job_ids"):
        return {"status": "prepared_not_launched", "experiments": []}
    records = []
    journal_path = sweep / "recovery-journal.json"
    journal = read_json(journal_path) if journal_path.exists() else {}
    for item in manifest["experiments"]:
        directory = Path(item["experiment"])
        job = read_json(directory / "job.json")["job_id"]
        live = state(job)
        progress = (
            read_json(directory / "status.json")
            if (directory / "status.json").exists()
            else {}
        )
        complete = (directory / "completed.json").exists()
        record = {
            **item,
            "job_id": job,
            "slurm": live,
            "progress": progress,
            "complete": complete,
        }
        if complete:
            result = read_json(directory / "completed.json")
            if result["simulated_clock"] != manifest["simulated_clock_budget"]:
                raise ValueError(
                    "Completed experiment did not reach the common comparison deadline"
                )
            if (
                not Path(result["student"]["hf"]).is_dir()
                or not Path(result["student"]["resume"]).is_dir()
            ):
                raise ValueError("Final student continuation is missing")
        elif live["state"] not in ACTIVE | {"UNKNOWN"}:
            text = tail(directory / f"slurm-{job}.out") + tail(
                directory / "gateway.log"
            )
            tmp_full = "Need 12 GiB of free node-local /tmp" in text or any(
                "No space" in line and "/tmp/" in line for line in text.splitlines()
            )
            transient = any(
                word in text
                for word in (
                    "ActorDiedError",
                    "ncclRemoteError",
                    "GPU has fallen off",
                    "Watchdog caught collective operation timeout",
                )
            )
            token = f"{job}:{progress.get('student_rollouts')}:{progress.get('teacher_versions')}"
            if (
                repair
                and live["state"] != "CANCELLED"
                and (live["state"] in INFRA or tmp_full or transient)
                and journal.get(token, 0) < 3
            ):
                journal[token] = journal.get(token, 0) + 1
                write_json(journal_path, journal)
                record["job_id"] = recover_job(
                    sweep,
                    directory,
                    job,
                    exclude_node=live["node"] if tmp_full else None,
                )
                record["requeued"] = True
            else:
                record["needs_attention"] = True
        records.append(record)
    return {
        "at": time.time(),
        "status": "complete"
        if all(r["complete"] for r in records)
        else "needs_attention"
        if any(r.get("needs_attention") for r in records)
        else "running",
        "experiments": records,
    }


def monitor(sweep, *, repair=False, once=False):
    with (sweep / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        next_plot = 0
        while True:
            try:
                report = check(sweep, repair=repair)
            except (OSError, subprocess.SubprocessError) as exc:
                # A transient scheduler/RPC failure must not disable all future
                # recovery. Leave the failed operation visible and retry later.
                report = {
                    "at": time.time(),
                    "status": "monitor_error",
                    "error": str(exc),
                }
            write_json(sweep / "status.json", report)
            print(json.dumps(report), flush=True)
            manifest = read_json(sweep / "sweep.json")
            interval = manifest.get("refresh_plot_seconds", 0)
            if interval and time.time() >= next_plot and report["status"] != "complete":
                try:
                    subprocess.run(manifest["analysis_argv"], check=True)
                except (OSError, subprocess.SubprocessError) as exc:
                    print(f"PLOT_REFRESH_FAILED: {exc}", flush=True)
                next_plot = time.time() + interval
            if report["status"] == "complete":
                # CPU-watch Python lacks matplotlib; the pinned image provides
                # plotting without allocating GPUs.
                subprocess.run(manifest["analysis_argv"], check=True)
                manifest["status"] = "complete"
                write_json(sweep / "sweep.json", manifest)
                return
            if once or report["status"] == "prepared_not_launched":
                return
            time.sleep(60)
