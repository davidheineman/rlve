"""Matched own-task-excluded mixing probes in small low-priority GPU jobs."""

import argparse
import copy
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/juice5b/scr5b/dhei")
ORIGIN = SCRATCH / "runs/composed-clustering-n64-20261102-150018"
TEMPLATE = SCRATCH / "runs/composed-clustering-k01-20261103-115421"
PLOT = ROOT / "plot/mopd-clustering-v2"
PARTITIONS = "miso-lo,sphinx-lo,jag-lo,sc-loprio"
# Match the inference rank to the queue instead of reserving rank 2048 everywhere.
TIERS = {
    "large": (8, 2048, "80G|141G"),
    "medium": (12, 1024, "48G|80G|141G"),
    "small": (40, 512, "(24G&ampere)|48G|80G|141G"),
}
EXCLUDE = "pasteur8,jagupard28,jagupard30,viscam5,viscam10,tiger6,tiger7,tiger8,oriong12,arm1,sphinx2"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def task_specification(completed, n):
    tasks, records = {}, []

    def partition(groups):
        if sorted(i for group in groups for i in group) != list(range(n)):
            raise ValueError("Frozen assignments must cover every target once")
        ids = [None] * n
        for group in groups:
            for target in group:
                raw = {"members": sorted(i for i in group if i != target),
                       "targets": [target], "split": "confirmation"}
                if not raw["members"]:
                    raise ValueError("Singleton LOO has no transfer model to evaluate")
                key = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:24]
                tasks[key] = {"id": key, **raw}
                ids[target] = key
        return ids

    shared = partition([list(range(n))])
    records.append({"k": 1, "best": shared, "worst": shared, "random": [shared]})
    for row in completed["candidates"]:
        records.append({"k": row["k"],
                        "best": partition(row["best"]["groups"]),
                        "worst": partition(row["worst"]["groups"]),
                        "random": [partition(r["groups"]) for r in row["random"]]})
    return tasks, records


def tier_for(task):
    size = len(task["members"])
    return "large" if size > 31 else "medium" if size > 15 else "small"


def command(directory, *, tier=None, shard=None):
    script = directory / "code/submit_mixing_transfer_loo.py"
    argv = ["python3", str(script), "--worker", "--experiment", str(directory)]
    return argv + (["--shard", shard] if shard else ["--tier", tier])


def sbatch(directory, tier, *, index=None):
    count, _, constraint = TIERS[tier]
    argv = ["sbatch", "--parsable", "--account=miso", f"--partition={PARTITIONS}",
            "--nodes=1", "--ntasks=1", "--gres=gpu:1", "--cpus-per-task=6",
            "--mem=32G", "--time=06:00:00", "--time-min=00:30:00",
            f"--constraint={constraint}", f"--exclude={EXCLUDE}", "--requeue",
            "--signal=B:USR1@180", f"--job-name=mixing-loo-{tier}",
            f"--chdir={directory}", "--open-mode=append"]
    if index is None:
        argv += [f"--array=0-{count - 1}%{count}", f"--output={directory}/slurm-%A_%a.out"]
        worker = command(directory, tier=tier)
    else:
        label = f"{tier}-{index:02d}"
        argv += [f"--output={directory}/shards/{label}/slurm-%j.out"]
        worker = command(directory, shard=label)
    return argv + ["--wrap", "exec " + shlex.join(worker)]


def prepare(args):
    original = read(ORIGIN / "plan.json")
    completed = read(ORIGIN / "completed.json")
    n = len(original["environments"])
    tasks, records = task_specification(completed, n)
    for adapter in original["adapter_provenance"]:
        path = Path(adapter["path"])
        if (digest(path / "adapter_model.safetensors") != adapter["weights_sha256"]
                or digest(path / "adapter_config.json") != adapter["config_sha256"]):
            raise ValueError(f"Original specialist changed: {path}")
    for split, specification in original["evaluation"].items():
        if digest(specification["path"]) != specification["sha256"]:
            raise ValueError(f"Original {split} prompts changed")
    directory = args.experiment or SCRATCH / "runs" / f"mixing-transfer-loo-{datetime.now():%Y%m%d-%H%M%S}"
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copytree(TEMPLATE / "code", directory / "code",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "tools/composed_clustering/loo.py", directory / "code/composed_clustering/loo.py")
    shutil.copy2(Path(__file__), directory / "code/submit_mixing_transfer_loo.py")
    write(directory / "origin_completed.json", completed)
    write(directory / "specification.json", {"records": records, "tasks": tasks})
    for name in ("scores", "cells", "task-locks", "completed-tasks"):
        (directory / name).mkdir()
    for key, task in tasks.items():
        write(directory / "queues" / tier_for(task) / f"{key}.json", task)
    plan = {**original, "method": "matched-leave-own-adapter-out-confirmation",
            "origin": str(ORIGIN), "origin_completed_sha256": digest(ORIGIN / "completed.json"),
            "tasks": len(tasks), "sampled_responses": len(tasks) * 32 * 2,
            "requested_gpus": sum(t[0] for t in TIERS.values()), "partitions": PARTITIONS,
            "plot_output": str(PLOT), "prepared_at": time.time()}
    plan["plot_python"] = str(ROOT.parent / "nano-omni/.venv-midtraining/bin/python")
    write(directory / "plan.json", plan)
    common_hashes = {str(p.relative_to(directory)): digest(p) for p in (directory / "code").rglob("*.py")}
    for tier, (count, rank, _) in TIERS.items():
        ids = sorted(key for key, task in tasks.items() if tier_for(task) == tier)
        for index in range(count):
            shard = directory / "shards" / f"{tier}-{index:02d}"
            shard.mkdir(parents=True)
            for name in ("code", "scores", "cells", "task-locks", "completed-tasks"):
                (shard / name).symlink_to(directory / name, target_is_directory=True)
            (shard / "tasks").symlink_to(directory / "queues" / tier, target_is_directory=True)
            (shard / "home").mkdir()
            (shard / "workers").mkdir()
            shard_plan = copy.deepcopy(original)
            shard_plan.update(gpus=1, max_lora_rank=rank, required_local_free_gib=8,
                              entry_module="composed_clustering.loo", loo_task_ids=ids,
                              output=str(PLOT / ".cache/mixing-loo"),
                              loo_experiment=str(directory), loo_tier=tier)
            write(shard / "plan.json", shard_plan)
            write(shard / "input_sha256.json", {**common_hashes, "plan.json": digest(shard / "plan.json")})
            write(shard / "submit_argv.json", sbatch(directory, tier, index=index))
    write(directory / "execution.json", {"status": "prepared", "shards": [], "arrays": []})
    write(PLOT / ".cache/mixing_loo_launch.json", {"experiment": str(directory), "requested_gpus": plan["requested_gpus"]})
    collect(directory)
    print(json.dumps({"experiment": str(directory), "tasks": len(tasks), "requested_gpus": plan["requested_gpus"]}), flush=True)
    return directory


def submit(directory):
    manifest = read(directory / "execution.json")
    if manifest["arrays"]:
        raise ValueError("This experiment already has submitted arrays")
    for tier, (count, _, _) in TIERS.items():
        argv = sbatch(directory, tier)
        job = subprocess.check_output(argv, text=True).strip().split(";")[0]
        manifest["arrays"].append(job)
        for index in range(count):
            label = f"{tier}-{index:02d}"
            record = {"label": label, "tier": tier, "index": index, "job_id": f"{job}_{index}"}
            manifest["shards"].append(record)
            write(directory / "shards" / label / "job.json", record)
        write(directory / "execution.json", manifest)
        print("SUBMITTED", tier, job, "requested_gpus", count, flush=True)
    monitor_command = ["python3", str(directory / "code/submit_mixing_transfer_loo.py"),
                       "--monitor", "--experiment", str(directory)]
    monitor_job = subprocess.check_output([
        "sbatch", "--parsable", "--account=miso", "--partition=sc-freecpu", "--cpus-per-task=1",
        "--mem=2G", "--time=7-00:00:00", "--requeue", "--job-name=mixing-loo-watch",
        f"--output={directory}/monitor-%j.out", "--wrap", "exec " + shlex.join(monitor_command),
    ], text=True).strip().split(";")[0]
    manifest.update(status="submitted", monitor_job_id=monitor_job)
    write(directory / "execution.json", manifest)
    print("MONITOR", monitor_job, flush=True)


def collect(directory):
    plan = read(directory / "plan.json")
    specification = read(directory / "specification.json")
    expected = {}
    for row in read(plan["evaluation"]["confirmation"]["path"]):
        meta = row["metadata"]
        meta = json.loads(meta) if isinstance(meta, str) else meta
        expected.setdefault(meta["environment"], []).append(hashlib.sha256(row["user_prompt"].encode()).hexdigest())
    scores = {}
    for path in (directory / "scores").glob("*.json"):
        measured = read(path)
        task = specification["tasks"][path.stem]
        target = task["targets"][0]
        cell = measured["cells"][0]
        name = plan["environments"][target]
        rows = cell["observations"]
        if (measured["members"] != task["members"] or target in task["members"]
                or cell["environment"] != name or measured["split"] != "confirmation"
                or [r["prompt_id"] for r in rows] != expected[name]
                or any(r["total"] != 2 for r in rows)):
            raise ValueError(f"Mismatched LOO evaluation: {path}")
        accuracy = sum(r["correct"] for r in rows) / sum(r["total"] for r in rows)
        if abs(accuracy - cell["accuracy"]) > 1e-12:
            raise ValueError("LOO prompt outcomes and summary disagree")
        scores[path.stem] = accuracy
    records = []
    for row in specification["records"]:
        entry = {"k": row["k"]}
        for key in ("best", "worst"):
            entry[key] = [scores[k] for k in row[key]] if all(k in scores for k in row[key]) else None
        draws = row["random"]
        entry["random"] = ([sum(scores[draw[i]] for draw in draws) / len(draws)
                            for i in range(len(plan["environments"]))]
                           if all(k in scores for draw in draws for k in draw) else None)
        if any(entry[key] is not None for key in ("best", "worst", "random")):
            records.append(entry)
    result = {"environments": plan["environments"], "status": "complete" if len(scores) == len(specification["tasks"]) else "running",
              "records": records, "progress": {"completed": len(scores), "total": len(specification["tasks"])},
              "origin": plan["origin"], "assignments_refit": False, "time": time.time()}
    write(directory / "results.json", result)
    return result


def refresh_plot(directory):
    plan = read(directory / "plan.json")
    plotting = Path(plan["plot_output"]) / ".plotting/make_plots.py"
    subprocess.run([plan["plot_python"], str(plotting),
                    "--update-mixing-loo", str(directory / "results.json")], check=True)


def monitor(directory):
    sys.path.insert(0, str(directory / "code"))
    from submit_composed_clustering import recovery_placement

    last_progress = -1
    while True:
        result = collect(directory)
        if result["progress"]["completed"] != last_progress:
            refresh_plot(directory)
            last_progress = result["progress"]["completed"]
        manifest = read(directory / "execution.json")
        jobs = [r["job_id"] for r in manifest["shards"]]
        queue = subprocess.run(["squeue", "-r", "-h", "-j", ",".join(jobs), "-o", "%i|%T|%N|%R"], text=True, capture_output=True, timeout=30)
        live = {parts[0]: {"state": parts[1], "node": parts[2], "reason": parts[3]}
                for line in queue.stdout.splitlines() if len(parts := line.split("|", 3)) == 4}
        allocated = sum(r["state"] in ("RUNNING", "COMPLETING") for r in live.values())
        write(directory / "status.json", {**result, "allocated_gpus": allocated,
                                           "requested_gpus": read(directory / "plan.json")["requested_gpus"], "jobs": live})
        print(json.dumps({"progress": result["progress"], "allocated_gpus": allocated}), flush=True)
        if result["status"] == "complete":
            write(directory / "completed.json", result)
            manifest["status"] = "complete"
            write(directory / "execution.json", manifest)
            return
        accounting = subprocess.check_output(["sacct", "-X", "-nP", "-j", ",".join(set(r["job_id"].split("_")[0] for r in manifest["shards"])),
                                              "--format=JobID,State,NodeList"], text=True, timeout=30)
        states = {parts[0]: parts[1:] for line in accounting.splitlines() if len(parts := line.split("|")) >= 3}
        for record in manifest["shards"]:
            shard = directory / "shards" / record["label"]
            if record["job_id"] in live or (shard / "completed.json").exists():
                continue
            state, node, *_ = states.get(record["job_id"], ["UNKNOWN", ""])
            state = state.split()[0].rstrip("+")
            if state in ("UNKNOWN", "CANCELLED") or record.get("retries", 0) >= 4:
                continue
            if state in ("FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "TIMEOUT", "PREEMPTED"):
                # Retry the same queue; completed task files are never discarded.
                recovery_placement(shard, record["job_id"], node)
                argv = read(shard / "submit_argv.json")
                if state == "OUT_OF_MEMORY":
                    argv = ["--constraint=80G|141G" if a.startswith("--constraint=") else a for a in argv]
                record["previous_job_ids"] = [*record.get("previous_job_ids", []), record["job_id"]]
                record["job_id"] = subprocess.check_output(argv, text=True).strip().split(";")[0]
                record["retries"] = record.get("retries", 0) + 1
                write(shard / "job.json", record)
                write(directory / "execution.json", manifest)
        time.sleep(45)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "submit", "worker", "monitor", "collect"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--tier", choices=tuple(TIERS))
    parser.add_argument("--shard")
    args = parser.parse_args()
    if args.prepare:
        prepare(args)
    elif args.submit:
        submit(args.experiment)
    elif args.worker:
        directory = args.experiment
        label = args.shard or f"{args.tier}-{int(os.environ['SLURM_ARRAY_TASK_ID']):02d}"
        sys.path.insert(0, str(directory / "code"))
        from submit_composed_clustering import worker
        worker(directory / "shards" / label)
    elif args.monitor:
        monitor(args.experiment)
    else:
        print(json.dumps(collect(args.experiment)))


if __name__ == "__main__":
    main()
