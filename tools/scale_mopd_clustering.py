"""Resume one logical clustering run on up to six independent four-GPU jobs."""

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lora_clustering.parallel import assign_teachers, checkpoint_inventory
from lora_teachers.plan import IMAGE_TAG, ROOT, sha256
from lora_teachers.runner import verify_inputs
from submit_data_scaling_lora import container_argv, gpu_environment

from async_branching.io import read_json, write_json

OVERLAY = [
    "tools/scale_mopd_clustering.py",
    "tools/monitor_parallel_mopd_clustering.py",
    "tools/lora_clustering/parallel.py",
    "tools/lora_clustering/pipeline.py",
    "tools/lora_clustering/teacher.py",
    "tools/lora_clustering/evaluation.py",
    "tools/lora_clustering/balanced_pipeline.py",
    "tools/submit_data_scaling_lora.py",
]


def prepare(experiment, shards):
    if not 1 <= shards <= 6:
        raise ValueError("This execution is limited to 24 GPUs")
    verify_inputs(experiment)
    plan = read_json(experiment / "plan.json")
    checkpoint_inventory(experiment, plan)
    original_job = read_json(experiment / "job.json")
    if original_job.get("mode") == "parallel":
        raise ValueError("This experiment already has a parallel execution")
    revision = (
        experiment / "runtime-revisions" / f"scale24-{datetime.now():%Y%m%dT%H%M%S}"
    )
    revision.mkdir(parents=True)
    shutil.copytree(
        experiment / "source",
        revision / "source",
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    for relative in OVERLAY:
        shutil.copy2(ROOT / relative, revision / "source/rlve" / relative)
    write_json(revision / "plan.json", plan)
    hashes = {
        str(path.relative_to(revision)): sha256(path)
        for path in (revision / "source").rglob("*")
        if path.is_file()
    }
    hashes["plan.json"] = sha256(revision / "plan.json")
    write_json(revision / "input_sha256.json", hashes)
    assignments = assign_teachers(experiment, plan, shards)
    backends = []
    for index, indices in enumerate(assignments):
        # Identical basenames retain the logical run's checkpoint tenant key;
        # runtime files, preflight artifacts and gateways remain independent.
        backend = revision / "backends" / f"shard-{index}" / experiment.name
        backend.mkdir(parents=True)
        (backend / "source").symlink_to(revision / "source", target_is_directory=True)
        (backend / "runtime").symlink_to(
            experiment / "runtime", target_is_directory=True
        )
        (backend / "teacher").symlink_to(
            experiment / "teacher", target_is_directory=True
        )
        physical_plan = dict(plan, adapter_slots=max(2, len(indices) + 1))
        write_json(backend / "plan.json", physical_plan)
        record = {
            "index": index,
            "experiment": str(experiment),
            "backend": str(backend),
            "revision": str(revision),
            "indices": indices,
            "gpus": 4,
        }
        write_json(backend / "backend.json", record)
        write_json(
            backend / "input_sha256.json",
            {name: sha256(backend / name) for name in ("plan.json", "backend.json")},
        )
        write_json(
            backend / "pipeline-status.json", {"stage": "queued", "time": time.time()}
        )
        backends.append(record)
    manifest = {
        "experiment": str(experiment),
        "revision": str(revision),
        "total_gpus": shards * 4,
        "original_job": original_job,
        "backends": backends,
        "state": "prepared",
    }
    write_json(revision / "execution.json", manifest)
    return revision, manifest


def submit(args):
    experiment = args.experiment.resolve()
    revision, manifest = prepare(experiment, args.shards)
    old = manifest["original_job"]
    jobs = []
    try:
        for backend in manifest["backends"]:
            directory = Path(backend["backend"])
            command = [
                "python3",
                str(revision / "source/rlve/tools/scale_mopd_clustering.py"),
                "--worker",
                "--experiment",
                str(directory),
                "--scratch",
                str(args.scratch),
                "--image",
                str(args.image),
                "--model-path",
                str(args.model_path),
            ]
            argv = [
                "sbatch",
                "--parsable",
                "--account=miso",
                "--partition=miso-lo,sphinx-lo,jag-lo,sc-loprio",
                f"--job-name=mopd-n64-shard-{backend['index']}",
                "--nodes=1",
                "--ntasks=1",
                "--gres=gpu:4",
                "--cpus-per-task=24",
                "--mem=96G",
                "--time=3-00:00:00",
                "--constraint=ada|48G|80G|141G",
                "--exclude=pasteur8,jagupard30,viscam10,tiger8",
                "--requeue",
                "--signal=B:USR1@180",
                f"--dependency=afterany:{old['job_id']}",
                f"--chdir={directory}",
                f"--output={directory}/slurm-%j.out",
                "--open-mode=append",
                "--wrap",
                "exec " + shlex.join(command),
            ]
            job_id = subprocess.check_output(argv, text=True).strip().split(";")[0]
            jobs.append(job_id)
            backend["job_id"] = job_id
            write_json(
                directory / "job.json",
                {"job_id": job_id, "indices": backend["indices"]},
            )
            write_json(directory / "submit_argv.json", argv)
        # Every new job is dependency-blocked until the original writer exits.
        checkpoint_inventory(experiment, read_json(experiment / "plan.json"))
    except BaseException:
        if jobs:
            subprocess.run(["scancel", *jobs], check=True)
        raise
    manifest["state"] = "handing_off"
    write_json(revision / "execution.json", manifest)
    if old.get("watchdog_job_id"):
        subprocess.run(["scancel", str(old["watchdog_job_id"])], check=True)
    subprocess.run(["scancel", str(old["job_id"])], check=True)
    record = dict(
        old,
        job_id=jobs[0],
        job_ids=jobs,
        mode="parallel",
        previous_job_id=old["job_id"],
        runtime_revision=str(revision),
        parallel_manifest=str(revision / "execution.json"),
        total_gpus=len(jobs) * 4,
    )
    record.pop("watchdog_job_id", None)
    write_json(experiment / "job.json", record)
    manifest["state"] = "running"
    write_json(revision / "execution.json", manifest)
    monitor = [
        "python3",
        str(revision / "source/rlve/tools/monitor_parallel_mopd_clustering.py"),
        "--experiment",
        str(experiment),
        "--manifest",
        str(revision / "execution.json"),
        "--interval",
        "90",
        "--recover",
    ]
    watchdog = (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--account=miso",
                "--partition=sc-freecpu",
                "--job-name=mopd-n64-parallel-watch",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=1",
                "--mem=1G",
                "--time=4-00:00:00",
                "--requeue",
                f"--output={experiment}/watchdog/parallel-slurm-%j.out",
                "--open-mode=append",
                "--wrap",
                "exec " + shlex.join(monitor),
            ],
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    record["watchdog_job_id"] = watchdog
    write_json(experiment / "job.json", record)
    plan = read_json(experiment / "plan.json")
    write_json(Path(plan["plot_output"]) / "launch.json", record)
    # After Slurm drains the old PID namespace, only the assigned new owners
    # can touch each teacher. Capture the durable state actually handed over.
    for _ in range(36):
        state = subprocess.check_output(
            ["squeue", "-h", "-j", str(old["job_id"]), "-o", "%T"], text=True
        ).strip()
        if not state:
            write_json(
                revision / "handoff-checkpoints.json",
                checkpoint_inventory(experiment, plan),
            )
            break
        time.sleep(5)
    print(json.dumps(record, indent=2), flush=True)


def worker(args):
    backend = args.experiment.resolve()
    config = read_json(backend / "backend.json")
    verify_inputs(Path(config["revision"]))
    verify_inputs(backend)
    if shutil.disk_usage("/tmp").free < 24 * 2**30:
        node = os.uname().nodename.split(".")[0]
        excluded = sorted(
            set(
                read_json(backend / "full-tmp-nodes.json")
                if (backend / "full-tmp-nodes.json").exists()
                else ["pasteur8", "jagupard30", "viscam10", "tiger8"]
            )
            | {node}
        )
        write_json(backend / "full-tmp-nodes.json", excluded)
        # The CPU watchdog requeues on hold before editing node exclusions;
        # Slurm rejects those edits on a running job.
        raise RuntimeError(
            f"No space left on device: startup /tmp guard on {node}; "
            "need 24 GiB free, watchdog will move this backend"
        )
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    physical_plan = read_json(backend / "plan.json")
    os.environ.update(
        gpu_environment(
            rows,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            minimum_gpu_mib=physical_plan.get("minimum_gpu_mib", 45000),
        )
    )
    command = container_argv(
        backend, scratch=args.scratch, image=args.image, model_path=args.model_path
    )
    original = str(backend / "source/rlve/tools/submit_data_scaling_lora.py")
    command[command.index(original)] = str(
        backend / "source/rlve/tools/scale_mopd_clustering.py"
    )
    output = read_json(Path(config["experiment"]) / "plan.json")["plot_output"]
    command[command.index("--nv") : command.index("--nv")] = ["--bind", output]
    script = 'source "$1"; rlve_reserve_port_slot "$2"; export LORA_PORT_SLOT="$RLVE_RESERVED_PORT_SLOT"; shift 2; exec "$@"'
    lock_dir = args.scratch / "locks/rlve-ports" / os.uname().nodename.split(".")[0]
    process = subprocess.Popen(
        [
            "bash",
            "-euc",
            script,
            "parallel-mopd-lock",
            str(backend / "source/rlve/scripts/slurm/port_slot.sh"),
            str(lock_dir),
            *command,
        ],
        start_new_session=True,
    )
    requeue = False

    def stop(signum, _frame):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("submit", "worker", "inside"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--scratch", type=Path, default=Path("/juice5b/scr5b/dhei"))
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(f"/juice5b/scr5b/dhei/images/miles-{IMAGE_TAG}.sif"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/juice5b/scr5b/dhei/models/DeepSeek-R1-Distill-Qwen-1.5B"),
    )
    args = parser.parse_args()
    if args.submit:
        submit(args)
    elif args.worker:
        worker(args)
    else:
        from lora_clustering.parallel import run_shard
        from lora_teachers.runner import run

        run(args.experiment, args.model_path, clients=run_shard)


if __name__ == "__main__":
    main()
