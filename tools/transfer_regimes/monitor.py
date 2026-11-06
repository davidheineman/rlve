"""Recover only recorded Slurm array tasks without exceeding the array throttle."""

import fcntl
import json
import subprocess
import time
from pathlib import Path

from clustering_mopd.monitor import state, tail

from async_branching.io import read_json, write_json

PARTITIONS = "jag-standard,miso,sphinx,miso-lo,sphinx-lo,jag-lo,sc-loprio"


def durable_rollouts(directory):
    total = 0
    for task in (directory / "tasks").glob("*"):
        result, latest = task / "result.json", task / "latest.json"
        if result.exists():
            total += read_json(result).get("rollouts", 0)
        elif latest.exists():
            name = Path(read_json(latest)["resume"]).parent.name
            if name.startswith("u") and name[1:].isdigit():
                total += 8 * int(name[1:])
    return total


def update_placement(job, *, exclude_node=None, features=None, exclude_nodes=()):
    command = ["scontrol", "update", f"JobId={job}", f"Partition={PARTITIONS}"]
    if features:
        command.append(f"Features={features}")
    if exclude_node or exclude_nodes:
        description = subprocess.check_output(
            ["scontrol", "show", "job", job, "-o"], text=True, timeout=20
        )
        fields = dict(x.split("=", 1) for x in description.split() if "=" in x)
        prior = fields.get("ExcNodeList", "(null)")
        # Commas inside Slurm brackets are part of the host-list syntax.
        nodes = (
            set(
                subprocess.check_output(
                    ["scontrol", "show", "hostnames", prior], text=True, timeout=20
                ).splitlines()
            )
            if prior not in {"(null)", "", "None"}
            else set()
        )
        previous_nodes = set(nodes)
        nodes.update(exclude_nodes)
        if exclude_node:
            nodes.add(exclude_node)
        if (
            nodes == previous_nodes
            and not features
            and set(fields.get("Partition", "").split(","))
            == set(PARTITIONS.split(","))
        ):
            return
        command.append("ExcNodeList=" + ",".join(sorted(nodes)))
    subprocess.run(command, check=True, timeout=20)


def check(sweep):
    manifest = read_json(sweep / "sweep.json")
    records = []
    journal_path = sweep / "recovery-journal.json"
    journal = read_json(journal_path) if journal_path.exists() else {}
    pending_path = sweep / "recovery-pending.json"
    pending = read_json(pending_path) if pending_path.exists() else {}
    quarantine_path = sweep / "node-quarantine.json"
    quarantine = read_json(quarantine_path) if quarantine_path.exists() else {}
    for item in manifest["experiments"]:
        directory = Path(item["experiment"])
        job = item.get("job_id", f"{manifest['array_job_id']}_{item['index']}")
        live = state(job)
        complete = (directory / "completed.json").exists()
        record = {**item, "job_id": job, "slurm": live, "complete": complete}
        record["durable_training_rollouts"] = durable_rollouts(directory)
        if live["state"] == "PENDING" and job in pending:
            operation = pending[job]
            update_placement(
                job,
                exclude_node=operation.get("exclude_node"),
                features=operation.get("features"),
                exclude_nodes=quarantine,
            )
            if "held" in live.get("reason", "").lower():
                subprocess.run(["scontrol", "release", job], check=True, timeout=20)
            token = operation["token"]
            journal[token] = journal.get(token, 0) + 1
            write_json(journal_path, journal)
            del pending[job]
            write_json(pending_path, pending)
            record["requeued"] = True
            records.append(record)
            continue
        if complete:
            result = read_json(directory / "completed.json")
            if (
                not (directory / "final-evaluation.json").exists()
                or not Path(result["student"]["hf"]).is_dir()
                or not Path(result["student"]["resume"]).is_dir()
            ):
                raise ValueError("Completed endpoint or final continuation is missing")
        elif live["state"] in {
            "FAILED",
            "NODE_FAIL",
            "BOOT_FAIL",
            "TIMEOUT",
            "PREEMPTED",
            "OUT_OF_MEMORY",
        }:
            text = tail(sweep / f"slurm-{job}.out").rsplit(
                "TRANSFER_REGIMES_ATTEMPT", 1
            )[-1]
            text += tail(directory / "gateway.log")
            text += "\n".join(
                tail(p) for p in (directory / "workers").glob("*/error.json")
            )
            disk = "Need 12 GiB of free node-local /tmp" in text or any(
                "/tmp/" in line and "No space" in line for line in text.splitlines()
            )
            gpu_memory = (
                "CUDA out of memory" in text or "torch.OutOfMemoryError" in text
            )
            allocation = (
                "GPU allocation inventory" in text
                or "Expected 4 allocated GPUs with at least" in text
                or "CUDA GPU visibility disagrees with the allocation" in text
            )
            transient = (
                disk
                or gpu_memory
                or allocation
                or any(
                    word in text
                    for word in [
                        "ActorDiedError",
                        "ncclRemoteError",
                        "GPU has fallen off",
                        "Watchdog caught collective operation timeout",
                        "Miles gateway exited during startup",
                    ]
                )
            )
            token = f"{job}:{record['durable_training_rollouts']}"
            if (
                transient
                or live["state"] in {"NODE_FAIL", "BOOT_FAIL", "TIMEOUT", "PREEMPTED"}
            ) and journal.get(token, 0) < 3:
                if (disk or allocation) and live["node"]:
                    quarantine[live["node"]] = {
                        "reason": "temporary storage"
                        if disk
                        else "GPU allocation inventory",
                        "at": time.time(),
                    }
                    write_json(quarantine_path, quarantine)
                pending[job] = {
                    "token": token,
                    "exclude_node": live["node"] if disk or allocation else None,
                    "features": "48G|80G|141G" if gpu_memory else None,
                }
                write_json(pending_path, pending)
                held = subprocess.run(
                    ["scontrol", "requeuehold", job], text=True, capture_output=True
                )
                if held.returncode:
                    # Never create an out-of-array replacement that would
                    # escape the six-run resource throttle.
                    record["needs_attention"] = held.stderr
                    del pending[job]
                    write_json(pending_path, pending)
                else:
                    update_placement(
                        job,
                        exclude_node=pending[job]["exclude_node"],
                        features=pending[job]["features"],
                        exclude_nodes=quarantine,
                    )
                    subprocess.run(["scontrol", "release", job], check=True)
                    journal[token] = journal.get(token, 0) + 1
                    write_json(journal_path, journal)
                    del pending[job]
                    write_json(pending_path, pending)
                    record["requeued"] = True
            else:
                record["needs_attention"] = (
                    "Failure not automatically recoverable or retry limit reached"
                )
        elif live["state"] == "CANCELLED":
            record["needs_attention"] = "User cancellation respected"
        elif not complete and live["state"] == "PENDING":
            if quarantine:
                update_placement(job, exclude_nodes=quarantine)
            # Slurm can retain only the chosen partition after a requeue.
            description = subprocess.run(
                ["scontrol", "show", "job", job, "-o"],
                text=True,
                capture_output=True,
                timeout=20,
            )
            if description.returncode == 0:
                fields = dict(
                    x.split("=", 1) for x in description.stdout.split() if "=" in x
                )
                if set(fields.get("Partition", "").split(",")) != set(
                    PARTITIONS.split(",")
                ):
                    subprocess.run(
                        [
                            "scontrol",
                            "update",
                            f"JobId={job}",
                            f"Partition={PARTITIONS}",
                        ],
                        check=True,
                    )
        records.append(record)
    # Replacement arrays receive a separate share of the same six-run cap.
    # Reclaim that share only after an array has fully released its GPUs.
    capacities = manifest.get("array_capacities", {})
    if len(capacities) > 1:
        active = []
        for array in capacities:
            members = [r for r in records if r["job_id"].split("_")[0] == array]
            retired = members and all(
                r["complete"] and r["slurm"]["state"] == "COMPLETED" for r in members
            )
            if not retired:
                active.append(array)
        if (
            len(active) == 1
            and capacities[active[0]] != manifest["maximum_concurrent_runs"]
        ):
            array = active[0]
            changed = subprocess.run(
                [
                    "scontrol",
                    "update",
                    f"JobId={array}",
                    f"ArrayTaskThrottle={manifest['maximum_concurrent_runs']}",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if changed.returncode and any(
                "Job has already finished" not in line
                for line in changed.stderr.splitlines()
            ):
                changed.check_returncode()
            manifest["array_capacities"] = {array: manifest["maximum_concurrent_runs"]}
            write_json(sweep / "sweep.json", manifest)
    return {
        "at": time.time(),
        "status": "complete"
        if all(r["complete"] for r in records)
        else "needs_attention"
        if any(r.get("needs_attention") for r in records)
        else "running",
        "experiments": records,
    }


def monitor(sweep):
    with (sweep / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                report = check(sweep)
            except (OSError, subprocess.SubprocessError) as exc:
                report = {
                    "at": time.time(),
                    "status": "monitor_error",
                    "error": str(exc),
                }
            write_json(sweep / "status.json", report)
            print(json.dumps(report), flush=True)
            if report["status"] == "complete":
                manifest = read_json(sweep / "sweep.json")
                script = sweep / "source/rlve/tools/transfer_regimes/analysis.py"
                command = [
                    "apptainer",
                    "exec",
                    "--no-home",
                    "--bind",
                    manifest["scratch"],
                    "--bind",
                    str(Path(manifest["analysis_output"]).parent),
                    manifest["image"],
                    "python3",
                    str(script),
                    str(sweep),
                ]
                Path(manifest["analysis_output"]).parent.mkdir(
                    parents=True, exist_ok=True
                )
                subprocess.run(command, check=True)
                return
            time.sleep(60)
