"""Watch one logical clustering experiment through requeues and final plots.

Only the job recorded in experiment/job.json is managed. Known infrastructure
failures can be retried from the existing checkpoints. Python/model failures
and persistent stalls are surfaced to the supervising agent for diagnosis.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import traceback
from pathlib import Path

ACTIVE = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "REQUEUED", "SUSPENDED"}
INFRA_STATES = {"NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "TIMEOUT"}
INFRA_ERRORS = (
    "CUDA error: driver shutting down",
    "GPU has fallen off",
    "ncclRemoteError",
    "ActorDiedError",
    "RayActorError",
    "Node unexpectedly terminated",
    "The node was marked dead",
    "Connection reset by peer",
)
PLOTS = [
    "individual_transfer",
    "signed_affinity",
    "cluster_transfer",
    "confirmation_performance",
    "within_between_transfer",
    "candidate_scores",
    "teacher_curves",
]


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def tail(path, length=14000):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(max(0, Path(path).stat().st_size - length))
            return stream.read().decode(errors="replace")
    except FileNotFoundError:
        return ""


def command(argv):
    return subprocess.check_output(argv, text=True, timeout=20).strip()


def slurm(job_id):
    output = command(["squeue", "-h", "-j", job_id, "-o", "%T|%N|%R"])
    if output:
        state, node, reason = output.splitlines()[0].split("|", 2)
        return {"state": state, "node": node, "reason": reason}
    output = command(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            job_id,
            "--format=JobID,State,NodeList,ExitCode",
        ]
    )
    rows = [line.split("|") for line in output.splitlines()]
    row = next((r for r in reversed(rows) if r[0] == job_id), None)
    if row is None:
        return {"state": "UNKNOWN", "node": "", "reason": "not yet in accounting"}
    return {"state": row[1].split()[0].rstrip("+"), "node": row[2], "reason": row[3]}


def inspect(experiment):
    now = time.time()
    plan, job = read(experiment / "plan.json"), read(experiment / "job.json")
    completed, active, errors, checkpoints = [], [], [], []
    for teacher in plan["teachers"]:
        name = teacher["source_label"]
        directory = experiment / "teacher" / name
        final = read(directory / "completed.json")
        if final:
            completed.append({"teacher": name, **final})
            continue
        heartbeat = read(experiment / "workers" / name / "heartbeat.json", {})
        checkpoint = read(directory / "latest.json")
        if checkpoint:
            checkpoints.append(checkpoint)
        active.append(
            {
                "teacher": name,
                "rollouts": heartbeat.get("used", 0),
                "heartbeat_age_s": now - heartbeat.get("time", now),
            }
        )
        error = read(experiment / "workers" / name / "error.json")
        if error:
            errors.append({"teacher": name, **error})
    cells = {
        split: sum(
            1 for _ in (experiment / "clustering/evaluations" / split).glob("*/*.json")
        )
        for split in ("selection", "confirmation")
    }
    state = slurm(str(job["job_id"]))
    rollouts = [t["rollouts"] for t in active]
    return {
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "time": now,
        "job_id": str(job["job_id"]),
        "slurm": state,
        "stage": read(experiment / "pipeline-status.json", {}).get("stage", "startup"),
        "completed_teachers": len(completed),
        "active_teachers": len(active),
        "min_active_rollouts": min(rollouts, default=0),
        "max_active_rollouts": max(rollouts, default=0),
        "sum_training_rollouts": sum(rollouts)
        + sum(t["training_rollouts"] for t in completed),
        "max_heartbeat_age_s": max((t["heartbeat_age_s"] for t in active), default=0),
        "checkpoints": len(checkpoints),
        "evaluation_cells": cells,
        "scratch_free_gib": shutil.disk_usage(experiment).free / 2**30,
        "active": active,
        "completed": completed,
        "errors": errors,
    }


def validate_completion(experiment):
    plan = read(experiment / "plan.json")
    final = read(experiment / "completed.json")
    if not final or final.get("stage") != "complete":
        return False
    expected = {t["environment"] for t in plan["teachers"]}
    records = [
        read(experiment / "teacher" / t["source_label"] / "completed.json")
        for t in plan["teachers"]
    ]
    if (
        any(r is None for r in records)
        or {r["environment"] for r in records} != expected
    ):
        raise ValueError("Completed pipeline is missing teacher results")
    for r in records:
        if r["training_rollouts"] > plan["teacher_rollouts"]:
            raise ValueError("Teacher exceeded the rollout cap")
        if (
            r["training_rollouts"] < plan["teacher_rollouts"]
            and r["accuracy"] < plan["teacher_validation_threshold"]
        ):
            raise ValueError("Teacher stopped before either requested limit")
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (Path(r["hf"]) / name).is_file():
                raise ValueError("Completed teacher adapter is missing")
    output = Path(plan["plot_output"])
    manifest = read(output / "mopd-clusters.json")
    if not manifest:
        raise ValueError("Missing cluster manifest")
    environments = [e for c in manifest["clusters"] for e in c["environments"]]
    if len(environments) != len(expected) or set(environments) != expected:
        raise ValueError("Clusters do not partition the original environments")
    balanced = plan.get("clustering", {}).get("mode") == "balanced"
    plots = (
        [
            "balanced/k_selection",
            "balanced/balanced_transfer",
            "confirmation_performance",
            "teacher_curves",
        ]
        if balanced
        else PLOTS
    )
    hashes = {}
    for stem in plots:
        for suffix in ("png", "pdf"):
            path = output / f"{stem}.{suffix}"
            data = path.read_bytes()
            magic = b"\x89PNG\r\n\x1a\n" if suffix == "png" else b"%PDF-"
            if len(data) < 100 or not data.startswith(magic):
                raise ValueError(f"Incomplete plot: {path}")
            hashes[str(path.relative_to(output))] = hashlib.sha256(data).hexdigest()
    artifacts = (
        (
            "results.json",
            "confirmation.json",
            "teacher_summary.csv",
            "README.md",
            "balanced/results.json",
            "balanced/transfer.json",
            "balanced/assignments.csv",
            "balanced/README.md",
        )
        if balanced
        else (
            "results.json",
            "uncertainty.json",
            "transfer.json",
            "candidate-results.json",
            "individual_transfer.csv",
            "teacher_summary.csv",
            "README.md",
        )
    )
    for name in artifacts:
        if not (output / name).is_file():
            raise ValueError(f"Missing result artifact: {name}")
    if balanced:
        confirmation = read(output / "confirmation.json")
        for row in confirmation["assignments"]["controls"].values():
            if len(row) != len(expected):
                raise ValueError("Incomplete merged-adapter confirmation")
        for key in ("selected-loo", "random-loo", "outside-matched"):
            if any(
                i in members
                for i, members in enumerate(
                    confirmation["assignments"]["controls"][key]
                )
            ):
                raise ValueError("LOO confirmation included the target's own adapter")
    write(
        experiment / "watchdog/artifact-verification.json",
        {
            "complete": True,
            "teachers": len(records),
            "clusters": len(manifest["clusters"]),
            "plot_sha256": hashes,
        },
    )
    return True


def classify_failure(experiment, health):
    state = health["slurm"]["state"]
    if state == "CANCELLED":
        # A cancellation may be a deliberate user action, even when an older
        # infrastructure traceback remains in the append-only Slurm log.
        return None
    text = tail(experiment / f"slurm-{health['job_id']}.out")
    text += "\n".join(e.get("traceback", "") for e in health["errors"])
    if state in INFRA_STATES:
        return "slurm_infrastructure"
    if "No space left on device" in text and "/tmp" in text:
        return "node_tmp_full"
    if any(pattern in text for pattern in INFRA_ERRORS):
        return "transient_backend"
    return None


def recover(experiment, health, journal):
    reason = classify_failure(experiment, health)
    if not reason:
        return False
    # Count attempts against committed progress, not replayable in-flight steps.
    durable = sorted(
        str(read(p).get("hf")) for p in (experiment / "teacher").glob("*/latest.json")
    )
    progress = (
        health["completed_teachers"],
        durable,
        health["evaluation_cells"]["selection"],
        health["evaluation_cells"]["confirmation"],
    )
    token = json.dumps([reason, progress])
    attempts = journal.setdefault("attempts", {})
    if attempts.get(token, 0) >= 3:
        return False
    job = health["job_id"]
    record = {
        "job_id": job,
        "reason": reason,
        "node": health["slurm"]["node"],
        "progress": progress,
        "time": time.time(),
        "action": "requeue_from_durable_checkpoints",
    }
    journal["recovery_in_progress"] = record
    attempts[token] = attempts.get(token, 0) + 1
    write(experiment / "watchdog/journal.json", journal)
    command(["scontrol", "requeuehold", job])
    if reason == "node_tmp_full":
        excluded = set(
            read(
                experiment / "full-tmp-nodes.json",
                ["pasteur8", "jagupard30", "viscam10"],
            )
        )
        node = health["slurm"]["node"]
        if re.fullmatch(r"[A-Za-z0-9_-]+", node or ""):
            excluded.add(node)
        command(
            [
                "scontrol",
                "update",
                f"JobId={job}",
                "ExcNodeList=" + ",".join(sorted(excluded)),
            ]
        )
        write(experiment / "full-tmp-nodes.json", sorted(excluded))
    command(["scontrol", "release", job])
    journal["last_recovery"] = record
    journal.pop("recovery_in_progress", None)
    write(experiment / "watchdog/journal.json", journal)
    return True


def finish_interrupted_recovery(experiment, health, journal):
    record = journal.get("recovery_in_progress")
    if not record or record["job_id"] != health["job_id"]:
        return
    if (
        health["slurm"]["state"] == "PENDING"
        and "held" in health["slurm"]["reason"].lower()
    ):
        command(["scontrol", "release", health["job_id"]])
    journal["last_recovery"] = record
    journal.pop("recovery_in_progress")
    write(experiment / "watchdog/journal.json", journal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=45)
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    directory = experiment / "watchdog"
    directory.mkdir(exist_ok=True)
    with (directory / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(directory / "process.json", {"pid": os.getpid(), "started": time.time()})
        journal = read(directory / "journal.json", {})
        previous, advanced_at, errors = None, time.time(), 0
        while True:
            try:
                health = inspect(experiment)
                if args.recover:
                    finish_interrupted_recovery(experiment, health, journal)
                progress = (
                    health["job_id"],
                    health["completed_teachers"],
                    health["sum_training_rollouts"],
                    *health["evaluation_cells"].values(),
                )
                if progress != previous:
                    previous, advanced_at = progress, time.time()
                health["no_progress_seconds"] = time.time() - advanced_at
                state = health["slurm"]["state"]
                if validate_completion(experiment):
                    health["status"] = "complete"
                elif state not in ACTIVE and state != "UNKNOWN":
                    if args.recover and recover(experiment, health, journal):
                        health["status"] = "requeued"
                    else:
                        health["status"] = "needs_attention"
                        health["reason"] = (
                            f"Job ended in {state} before pipeline completion"
                        )
                elif state == "RUNNING" and health["no_progress_seconds"] > 5400:
                    health["status"] = "needs_attention"
                    health["reason"] = (
                        "No training, checkpoint, or evaluation progress for 90 minutes"
                    )
                elif health["errors"] and state == "RUNNING":
                    health["status"] = "needs_attention"
                    health["reason"] = "A teacher worker recorded an exception"
                else:
                    health["status"] = "healthy"
                write(directory / "health.json", health)
                with (directory / "health.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                k: v
                                for k, v in health.items()
                                if k not in {"active", "completed", "errors"}
                            }
                        )
                        + "\n"
                    )
                print(
                    json.dumps(
                        {
                            k: health[k]
                            for k in [
                                "status",
                                "job_id",
                                "stage",
                                "completed_teachers",
                                "min_active_rollouts",
                                "max_active_rollouts",
                                "checkpoints",
                                "evaluation_cells",
                            ]
                        }
                    ),
                    flush=True,
                )
                errors = 0
                if health["status"] in {"complete", "needs_attention"} or args.once:
                    return
            except Exception:
                errors += 1
                write(
                    directory / "monitor-error.json",
                    {
                        "time": time.time(),
                        "consecutive_errors": errors,
                        "traceback": traceback.format_exc(),
                    },
                )
                print(f"MONITOR_ERROR consecutive={errors}", flush=True)
                if args.once or errors >= 3:
                    raise
            time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
