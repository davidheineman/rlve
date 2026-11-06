"""Submit one Miles job training the original N16 RLVE teachers as LoRAs.

Example:
    python3 tools/submit_data_scaling_lora.py --check-only
    python3 tools/submit_data_scaling_lora.py --submit

Uses a separate pinned Miles image. It never edits an existing experiment.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from lora_teachers.plan import IMAGE_TAG, PROJECT, ROOT, build_plan, sha256


def snapshot(experiment, miles_repo, plan):
    experiment.mkdir(parents=True, exist_ok=False)
    destination = experiment / "source"
    destination.mkdir()
    shutil.copytree(
        miles_repo,
        destination / "miles",
        symlinks=True,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            "docs",
            "tests",
            "examples",
            "docker",
            ".github",
            ".agents",
            ".claude",
        ),
    )
    rlve = destination / "rlve"
    rlve.mkdir()
    for relative in ("Gym", "slime", "tools/lora_teachers"):
        shutil.copytree(
            ROOT / relative, rlve / relative, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache")
        )
    for relative in (
        "tools/submit_data_scaling_lora.py",
        "scripts/slurm/port_slot.sh",
        "configs/1gpu-data-scaling-teacher-v1.yaml",
        *[t["eval"] for t in plan["teachers"]],
    ):
        path = rlve / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, path)
    (experiment / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (experiment / "provenance.json").write_text(
        json.dumps(
            {
                "miles_commit": subprocess.check_output(
                    ["git", "-C", str(miles_repo), "rev-parse", "HEAD"], text=True
                ).strip(),
                "miles_status": subprocess.check_output(
                    ["git", "-C", str(miles_repo), "status", "--short"], text=True
                ),
                "source_rlve": str(ROOT),
                "source_miles": str(miles_repo),
                "image": plan["image"],
            },
            indent=2,
        )
        + "\n"
    )
    paths = [
        experiment / "plan.json",
        *destination.rglob("*.py"),
        *destination.rglob("*.json"),
        *destination.rglob("*.yaml"),
        rlve / "scripts/slurm/port_slot.sh",
    ]
    hashes = {str(path.relative_to(experiment)): sha256(path) for path in sorted(set(paths)) if path.is_file()}
    (experiment / "input_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


def container_argv(experiment, *, scratch, image, model_path):
    container_home = experiment / "container-home"
    container_home.mkdir(exist_ok=True)
    runtime = experiment / "runtime"
    pythonpath = os.pathsep.join(
        map(
            str,
            (
                runtime,
                experiment / "source/rlve/tools",
                experiment / "source/rlve",
                experiment / "source/miles",
                Path("/root/Megatron-LM"),
            ),
        )
    )
    return [
        "apptainer",
        "exec",
        "--nv",
        "--pid",
        "--home",
        str(container_home),
        "--pwd",
        str(experiment / "source/miles"),
        "--bind",
        str(scratch),
        "--bind",
        str(experiment),
        "--env",
        f"PYTHONPATH={pythonpath}",
        "--env",
        "PYTHONUNBUFFERED=1",
        "--env",
        f"HF_HOME={scratch / 'cache/hf'}",
        "--env",
        "TOKENIZERS_PARALLELISM=false",
        str(image),
        "python3",
        str(experiment / "source/rlve/tools/submit_data_scaling_lora.py"),
        "--inside",
        "--experiment",
        str(experiment),
        "--model-path",
        str(model_path),
    ]


def gpu_environment(gpu_rows, visible_devices=None, *, expected_count=4, minimum_gpu_mib=45000):
    records = [tuple(field.strip() for field in row.split(",")) for row in gpu_rows]
    if len(records) != expected_count or any(len(row) != 3 or int(row[2]) < minimum_gpu_mib for row in records):
        raise RuntimeError(f"Expected {expected_count} allocated GPUs with at least {minimum_gpu_mib:,} MiB each")
    indices = [row[0] for row in records]
    if any(not index.isdecimal() for index in indices) or len(set(indices)) != expected_count:
        raise RuntimeError(f"Expected {expected_count} distinct numeric GPU indices")
    inherited = visible_devices.split(",") if visible_devices else []
    if (
        len(inherited) == expected_count
        and len(set(inherited)) == expected_count
        and all(index.isdecimal() for index in inherited)
    ):
        # Slurm's numeric mask already accounts for its device cgroup namespace;
        # NVML's physical indices can differ from those CUDA ordinals.
        indices = inherited
    # Miles' Ray worker placement uses integer GPU IDs. UUIDs are retained for
    # an independent CUDA visibility check before any trainer starts.
    return {
        "APPTAINERENV_CUDA_VISIBLE_DEVICES": ",".join(indices),
        "APPTAINERENV_CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "APPTAINERENV_LORA_EXPECTED_GPU_UUIDS": ",".join(row[1] for row in records),
    }


def worker(args):
    experiment = args.experiment.resolve()
    if (experiment / "clients.start").exists():
        raise RuntimeError("This final-only experiment already started; use a fresh submission after interruption")
    from lora_teachers.runner import verify_inputs

    verify_inputs(experiment)
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.total", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    plan = json.loads((experiment / "plan.json").read_text())
    os.environ.update(
        gpu_environment(
            gpu_rows,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            expected_count=plan["train_gpus"] + plan["sampling_gpus"],
        )
    )
    command = container_argv(experiment, scratch=args.scratch, image=args.image, model_path=args.model_path)
    # Reuse the cluster's shared port lock, including against existing Slime jobs.
    # Keep the Bash descriptor alive through exec so the lock covers the job.
    script = (
        'source "$1"; rlve_reserve_port_slot "$2"; '
        'export LORA_PORT_SLOT="$RLVE_RESERVED_PORT_SLOT"; shift 2; exec "$@"'
    )
    lock_dir = args.scratch / "locks/rlve-ports" / os.uname().nodename.split(".")[0]
    subprocess.run(
        [
            "bash",
            "-euc",
            script,
            "lora-port-lock",
            str(experiment / "source/rlve/scripts/slurm/port_slot.sh"),
            str(lock_dir),
            *command,
        ],
        check=True,
    )


def inside(args):
    # Tinker is deliberately isolated from the original Slime container/runtime.
    runtime = args.experiment / "runtime"
    if not (runtime / "tinker").is_dir():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(runtime),
                "tinker==0.26.2",
                "pyqwest==0.4.1",
            ],
            check=True,
        )
    from lora_teachers.runner import run

    run(args.experiment, args.model_path)


def submit(args, plan, *, afterok=None):
    if not args.image.is_file():
        raise FileNotFoundError(f"Finish pulling {plan['image']} to {args.image} before submitting")
    if not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing original HF base: {args.model_path}")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be exported for the requested project")
    experiment = args.experiment or args.scratch / "runs" / (
        plan["project"].removeprefix("rl-") + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    experiment = experiment.resolve()
    snapshot(experiment, args.miles_repo, plan)
    queue_experiment(args, plan, experiment, afterok=afterok)


def queue_experiment(args, plan, experiment, *, afterok=None):
    command = [
        "python3",
        str(experiment / "source/rlve/tools/submit_data_scaling_lora.py"),
        "--worker",
        "--experiment",
        str(experiment),
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
        "--job-name=scaling-lora-teachers-n16",
        "--account=miso",
        "--partition=miso,sphinx,jag-standard,miso-lo,sphinx-lo,jag-lo,sc-loprio",
        "--nodes=1",
        "--ntasks=1",
        f"--gres=gpu:{plan['train_gpus'] + plan['sampling_gpus']}",
        "--cpus-per-task=24",
        f"--mem={plan.get('host_memory_gib', 192)}G",
        f"--time={plan.get('time_limit', '1-00:00:00')}",
        "--constraint=ada|48G|80G|141G",
        "--exclude=pasteur8,jagupard30",
        "--no-requeue",
        f"--chdir={experiment}",
        f"--output={experiment}/slurm-%j.out",
        "--wrap",
        "exec " + shlex.join(command),
    ]
    if afterok is not None:
        argv.insert(1, f"--dependency=afterok:{int(afterok)}")
    (experiment / "submit_argv.json").write_text(json.dumps(argv, indent=2) + "\n")
    job_id = subprocess.check_output(argv, text=True).strip().split(";")[0]
    (experiment / "job.json").write_text(
        json.dumps({"job_id": job_id, "experiment": str(experiment)}, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "job_id": job_id,
                "experiment": str(experiment),
                "wandb_project": f"https://wandb.ai/{plan['entity']}/{plan['project']}",
            },
            indent=2,
        )
    )


def main():
    scratch = Path(os.environ.get("RLVE_SCRATCH", f"/nlp/scr/{os.environ.get('USER', 'dhei')}")).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("check-only", "submit", "worker", "inside"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--scratch", type=Path, default=scratch)
    parser.add_argument("--miles-repo", type=Path, default=ROOT.parent / "miles")
    parser.add_argument("--image", type=Path, default=scratch / f"images/miles-{IMAGE_TAG}.sif")
    parser.add_argument("--model-path", type=Path, default=scratch / "models/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--host-memory-gib", type=int, default=96)
    parser.add_argument("--time-limit", default="1-00:00:00")
    args = parser.parse_args()
    if args.worker:
        worker(args)
    elif args.inside:
        inside(args)
    else:
        plan = build_plan(project=args.project, learning_rate=args.learning_rate)
        if args.host_memory_gib <= 0:
            parser.error("--host-memory-gib must be positive")
        plan.update(host_memory_gib=args.host_memory_gib, time_limit=args.time_limit)
        if args.submit:
            submit(args, plan)
        else:
            print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
