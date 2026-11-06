"""Freeze and run inference-only exact-composition grouping on four GPUs."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

PARTITIONS = "jag-standard,miso,sphinx,miso-lo,sphinx-lo,jag-lo,sc-loprio"
SCRATCH = Path("/juice5b/scr5b/dhei")
ROOT = Path(__file__).resolve().parents[1]
IMAGE = SCRATCH / "images/miles-dev-202610231907.sif"
REFERENCE = SCRATCH / "runs/mopd-clustering-n64-20261031-165345"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(args):
    source = args.source.resolve()
    original = read(source / "plan.json")
    results = read(source / "teachers-completed.json")
    if [r["environment"] for r in results] != original["environments"]:
        raise ValueError("Teacher order and task panel disagree")
    experiment = (
        args.experiment
        or SCRATCH / "runs" / f"composed-clustering-n64-{datetime.now():%Y%m%d-%H%M%S}"
    )
    experiment = experiment.resolve()
    experiment.mkdir(parents=True, exist_ok=False)
    (experiment / "code").mkdir()
    (experiment / "home").mkdir()
    shutil.copytree(
        ROOT / "tools/composed_clustering",
        experiment / "code/composed_clustering",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(Path(__file__), experiment / "code/submit_composed_clustering.py")
    inputs = experiment / "inputs"
    inputs.mkdir()
    evaluation = {}
    seen = set()
    for split, count, samples in [("selection", 8, 2), ("confirmation", 32, 2)]:
        path = source / "source/rlve" / original["clustering"][split]
        if digest(path) != original["clustering"][split + "_sha256"]:
            raise ValueError("Original evaluation data changed")
        rows = read(path)
        prompts = {row["user_prompt"] for row in rows}
        if len(prompts) != len(rows) or prompts & seen:
            raise ValueError("Selection and confirmation prompts overlap")
        seen.update(prompts)
        if len(rows) != 64 * count:
            raise ValueError("Wrong prompt panel size")
        destination = inputs / f"{split}.json"
        shutil.copy2(path, destination)
        evaluation[split] = {
            "path": str(destination),
            "sha256": digest(destination),
            "samples": samples,
            "prompts_per_environment": count,
        }
    positive = ROOT / "plot/mopd-clustering/n64/balanced/results.json"
    low = ROOT / "plot/mopd-clustering/n64/low-transfer/results.json"
    transfer = positive.with_name("transfer.json")
    for path, name in [
        (positive, "positive.json"),
        (low, "low.json"),
        (transfer, "transfer.json"),
    ]:
        shutil.copy2(path, inputs / name)
    if (
        read(positive)["environments"] != original["environments"]
        or read(low)["environments"] != original["environments"]
    ):
        raise ValueError("Seed assignments use different environments")
    adapters = [str(Path(r["hf"]).resolve()) for r in results]
    provenance = []
    for adapter in adapters:
        config = read(Path(adapter) / "adapter_config.json")
        if config["r"] != 32:
            raise ValueError("Expected rank-32 trained adapters")
        provenance.append(
            {
                "path": adapter,
                "weights_sha256": digest(Path(adapter) / "adapter_model.safetensors"),
                "config_sha256": digest(Path(adapter) / "adapter_config.json"),
            }
        )
    plan = {
        "method": "direct-composed-adapter-id-search-v1",
        "source": str(source),
        "environments": original["environments"],
        "adapters": adapters,
        "adapter_provenance": provenance,
        "model_path": str(SCRATCH / "models/DeepSeek-R1-Distill-Qwen-1.5B"),
        "gpus": 4,
        "settings": original["teacher_config"],
        "evaluation": evaluation,
        "positive_results": str(inputs / "positive.json"),
        "low_results": str(inputs / "low.json"),
        "transfer": str(inputs / "transfer.json"),
        "ks": [2, 4, 8, 16, 32],
        "seed": 1234,
        "random_seeds": [1234, 4321, 2026],
        "rounds": args.rounds,
        "proposals_per_round": 4,
        "concurrency": 32,
        "composition": "uniform exact mean of member LoRA weight updates; no compression",
        "output": str(args.output.resolve()),
        "probe": str(args.probe.resolve()),
        "image": str(IMAGE),
        "frozen_at": time.time(),
    }
    write(experiment / "plan.json", plan)
    hashes = {
        str(p.relative_to(experiment)): digest(p)
        for p in [
            experiment / "plan.json",
            *list((experiment / "code").rglob("*.py")),
            *list(inputs.glob("*.json")),
        ]
    }
    write(experiment / "input_sha256.json", hashes)
    command = [
        "python3",
        str(experiment / "code/submit_composed_clustering.py"),
        "--worker",
        "--experiment",
        str(experiment),
    ]
    argv = [
        "sbatch",
        "--parsable",
        "--account=miso",
        f"--partition={PARTITIONS}",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:4",
        "--cpus-per-task=24",
        "--mem=96G",
        "--time=06:00:00",
        "--time-min=01:00:00",
        "--constraint=48G|80G|141G",
        "--exclude=pasteur8,jagupard28,jagupard30,viscam5,viscam10,tiger6,tiger8",
        "--requeue",
        "--signal=B:USR1@180",
        "--job-name=composed-clustering-n64",
        f"--chdir={experiment}",
        f"--output={experiment}/slurm-%j.out",
        "--open-mode=append",
        "--wrap",
        "exec " + shlex.join(command),
    ]
    write(experiment / "submit_argv.json", argv)
    return experiment


def submit(directory):
    plan = read(directory / "plan.json")
    if not (Path(plan["probe"]) / "output/passed.json").exists():
        raise RuntimeError(
            "Rank-1024 GPU composition preflight must pass before submission"
        )
    if (directory / "job.json").exists():
        raise RuntimeError("This run already has a submitted job")
    job = (
        subprocess.check_output(read(directory / "submit_argv.json"), text=True)
        .strip()
        .split(";")[0]
    )
    write(directory / "job.json", {"job_id": job, "submitted_at": time.time()})
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    write(
        output / "launch.json",
        {
            "job_id": job,
            "experiment": str(directory),
            "physical_gpus": plan["gpus"],
            "composition": plan["composition"],
        },
    )
    watch = [
        "python3",
        str(directory / "code/submit_composed_clustering.py"),
        "--watch",
        "--experiment",
        str(directory),
    ]
    watcher = (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--account=miso",
                "--partition=sc-freecpu",
                "--cpus-per-task=1",
                "--mem=1G",
                "--time=7-00:00:00",
                "--job-name=composed-clustering-watch",
                f"--output={directory}/monitor-%j.out",
                "--wrap",
                "exec " + shlex.join(watch),
            ],
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    write(directory / "watcher.json", {"job_id": watcher})
    print(
        json.dumps(
            {
                "experiment": str(directory),
                "job_id": job,
                "watcher_id": watcher,
                "output": str(output),
            }
        ),
        flush=True,
    )


def worker(directory):
    for relative, expected in read(directory / "input_sha256.json").items():
        if digest(directory / relative) != expected:
            raise ValueError("Frozen input changed: " + relative)
    plan = read(directory / "plan.json")
    required = plan.get("required_local_free_gib", 24)
    if shutil.disk_usage("/tmp").free < required * 2**30:
        raise RuntimeError(
            f"Need {required} GiB free node-local /tmp for composition workers"
        )
    reference = Path(plan["source"])
    pythonpath = ":".join(
        map(
            str,
            [
                directory / "code",
                reference / "source/rlve/tools",
                reference / "source/rlve",
                reference / "runtime",
            ],
        )
    )
    command = [
        "apptainer",
        "exec",
        "--nv",
        "--pid",
        "--home",
        str(directory / "home"),
        "--pwd",
        str(directory),
        "--bind",
        str(SCRATCH),
        "--bind",
        str(Path(plan["output"]).parent),
        "--env",
        f"PYTHONPATH={pythonpath}",
        "--env",
        "OMP_NUM_THREADS=2",
        "--env",
        "OPENBLAS_NUM_THREADS=1",
        "--env",
        "TOKENIZERS_PARALLELISM=false",
        "--env",
        f"MPLCONFIGDIR={directory}/plot-cache",
        "--env",
        f"HF_HOME={SCRATCH}/cache/hf",
        plan["image"],
        "python3",
        "-m",
        plan.get("entry_module", "composed_clustering.runner"),
        str(directory),
    ]
    process = subprocess.Popen(command, start_new_session=True)
    requeue = False

    def stop(signum, _):
        nonlocal requeue
        requeue = signum == signal.SIGUSR1
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    while process.poll() is None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    if requeue:
        subprocess.run(["scontrol", "requeue", os.environ["SLURM_JOB_ID"]], check=True)
    elif process.returncode:
        raise SystemExit(process.returncode)


def recovery_placement(directory, job, node):
    """Avoid repeating known capacity failures on the same placement."""
    logs = [directory / f"slurm-{job}.out"]
    logs += list((directory / "workers").glob("*/error.json"))
    logs += list((directory / "workers").glob("*/server.log"))
    tails = []
    for path in logs:
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 65536))
                tails.append(stream.read().decode(errors="replace").lower())
    diagnostic = "\n".join(tails)
    memory = any(
        s in diagnostic
        for s in [
            "cuda out of memory",
            "torch.outofmemoryerror",
            "not enough memory. please try",
        ]
    )
    disk = (
        "need 24 gib free node-local" in diagnostic
        or "need 6 gib free node-local" in diagnostic
        or "need 8 gib free node-local" in diagnostic
    )
    argv = read(directory / "submit_argv.json")
    features = next(a.split("=", 1)[1] for a in argv if a.startswith("--constraint="))
    excluded = next(a.split("=", 1)[1] for a in argv if a.startswith("--exclude="))
    if memory:
        features = read(directory / "plan.json").get(
            "memory_fallback_constraint", "48G|80G|141G"
        )
    if disk and node and node != "None":
        excluded += "," + node
    argv = [
        "--constraint=" + features
        if a.startswith("--constraint=")
        else "--exclude=" + excluded
        if a.startswith("--exclude=")
        else a
        for a in argv
    ]
    write(directory / "submit_argv.json", argv)
    return features, excluded


def watch(directory):
    import fcntl

    with (directory / "watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (directory / "completed.json").exists():
            try:
                job = read(directory / "job.json")["job_id"]
                queue = subprocess.run(
                    ["squeue", "-h", "-j", job, "-o", "%T|%N"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if queue.stdout.strip():
                    state, node = queue.stdout.strip().splitlines()[0].split("|", 1)
                else:
                    records = subprocess.check_output(
                        [
                            "sacct",
                            "-X",
                            "-nP",
                            "-j",
                            job,
                            "--format=JobID,State,NodeList",
                        ],
                        text=True,
                        timeout=20,
                    )
                    row = next(
                        (
                            line.split("|")
                            for line in reversed(records.splitlines())
                            if line.split("|")[0] == job
                        ),
                        [job, "UNKNOWN", ""],
                    )
                    state, node = row[1].split()[0], row[2]
                count = len(list((directory / "scores").glob("*.json")))
                cells = len(list((directory / "cells").glob("*/*.json")))
                write(
                    directory / "watch-status.json",
                    {
                        "time": time.time(),
                        "job_id": job,
                        "state": state,
                        "node": node,
                        "scored_groups": count,
                        "scored_environment_cells": cells,
                    },
                )
                if state not in [
                    "RUNNING",
                    "PENDING",
                    "COMPLETING",
                    "CONFIGURING",
                    "REQUEUED",
                    "UNKNOWN",
                ]:
                    journal = (
                        read(directory / "recovery.json")
                        if (directory / "recovery.json").exists()
                        else {}
                    )
                    token = f"{count}:{cells}"
                    if state == "CANCELLED" or journal.get(token, 0) >= 3:
                        write(
                            directory / "needs-attention.json",
                            {"state": state, "job_id": job, "time": time.time()},
                        )
                        return
                    features, excluded = recovery_placement(directory, job, node)
                    held = subprocess.run(
                        ["scontrol", "requeuehold", job],
                        capture_output=True,
                        text=True,
                        timeout=20,
                    )
                    if held.returncode:
                        if "invalid job id" not in held.stderr.lower():
                            held.check_returncode()
                        replacement = (
                            subprocess.check_output(
                                read(directory / "submit_argv.json"), text=True
                            )
                            .strip()
                            .split(";")[0]
                        )
                        write(
                            directory / "job.json",
                            {
                                "job_id": replacement,
                                "previous_job": job,
                                "time": time.time(),
                            },
                        )
                    else:
                        subprocess.run(
                            [
                                "scontrol",
                                "update",
                                f"JobId={job}",
                                f"Partition={PARTITIONS}",
                                f"Features={features}",
                                f"ExcNodeList={excluded}",
                            ],
                            check=True,
                        )
                        subprocess.run(["scontrol", "release", job], check=True)
                    journal[token] = journal.get(token, 0) + 1
                    write(directory / "recovery.json", journal)
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                write(
                    directory / "watch-error.json",
                    {"time": time.time(), "error": str(error)},
                )
            time.sleep(60)
        plan = read(directory / "plan.json")
        if plan.get("live_plot_script"):
            script = Path(plan["live_plot_script"])
            subprocess.run(
                [
                    "apptainer",
                    "exec",
                    "--no-home",
                    "--bind",
                    str(SCRATCH),
                    "--bind",
                    str(script.parents[2]),
                    "--env",
                    f"MPLCONFIGDIR={directory}/plot-cache",
                    plan["image"],
                    "python3",
                    str(script),
                    plan["origin"],
                ],
                check=True,
            )
        write(
            directory / "watch-status.json", {"state": "COMPLETE", "time": time.time()}
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for flag in ["prepare", "submit", "worker", "watch"]:
        modes.add_argument("--" + flag, action="store_true")
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--source", type=Path, default=REFERENCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "plot/mopd-clustering/n64/composed-search-v1",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=Path("/tmp/composed-clustering-probe-path").read_text().strip()
        if Path("/tmp/composed-clustering-probe-path").exists()
        else "/missing",
    )
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()
    if args.worker:
        worker(args.experiment.resolve())
    elif args.watch:
        watch(args.experiment.resolve())
    else:
        directory = (
            args.experiment.resolve()
            if args.experiment and (args.experiment / "plan.json").exists()
            else prepare(args)
        )
        if args.submit:
            submit(directory)
        else:
            print(directory)


if __name__ == "__main__":
    main()
