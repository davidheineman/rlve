"""Freeze and launch causal N16/N32 async MOPD in one four-GPU Miles job."""

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from async_lora.plan import build_async_plan, experiment_stem
from lora_teachers.plan import IMAGE_TAG, ROOT, sha256
from submit_data_scaling_lora import container_argv, gpu_environment, snapshot


def freeze(experiment, miles_repo, plan):
    snapshot(experiment, miles_repo, plan)
    destination = experiment / "source/rlve"
    for relative in ("tools/async_lora", "async_branching"):
        shutil.copytree(
            ROOT / relative, destination / relative, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache")
        )
    for relative in (
        "tools/submit_async_lora.py",
        "tools/prepare_async_branching.py",
        "tools/prepare_async_lora_questions.py",
        "tools/make_rlve_eval_set.py",
        "configs/data-scaling-mopd-v1.yaml",
        plan["source_student_config"],
        f"data/DATA-SCALING-ASYNC-V1/n{plan['size']}.json",
        plan["id_eval"],
        plan["test_eval"],
        "data/DATA-SCALING-V1/manifest.json",
        "data/DATA-SCALING-V1/run_plan.json",
        "data/DATA-SCALING-V1/n16.txt",
        *([plan["single_question"]["data"]] if plan.get("single_question") else []),
    ):
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, path)
    paths = [
        experiment / "plan.json",
        *[
            p
            for p in (experiment / "source").rglob("*")
            if p.is_file() and p.suffix in {".py", ".json", ".yaml", ".sh", ".txt"}
        ],
    ]
    write_json(experiment / "input_sha256.json", {str(p.relative_to(experiment)): sha256(p) for p in sorted(paths)})


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def worker(args):
    from async_lora.flex_manager import FlexManager
    from lora_teachers.runner import verify_inputs

    experiment = args.experiment.resolve()
    verify_inputs(experiment)
    if (experiment / "flex").is_dir():
        (experiment / "flex/core.json").unlink(missing_ok=True)
        (experiment / "flex/lease.json").unlink(missing_ok=True)
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.total", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    os.environ.update(gpu_environment(rows, os.environ.get("CUDA_VISIBLE_DEVICES")))
    command = container_argv(experiment, scratch=args.scratch, image=args.image, model_path=args.model_path)
    original = str(experiment / "source/rlve/tools/submit_data_scaling_lora.py")
    command[command.index(original)] = str(experiment / "source/rlve/tools/submit_async_lora.py")
    script = (
        'source "$1"; rlve_reserve_port_slot "$2"; export LORA_PORT_SLOT="$RLVE_RESERVED_PORT_SLOT"; shift 2; exec "$@"'
    )
    lock_dir = args.scratch / "locks/rlve-ports" / os.uname().nodename.split(".")[0]
    process = subprocess.Popen(
        [
            "bash",
            "-euc",
            script,
            "async-lora-port-lock",
            str(experiment / "source/rlve/scripts/slurm/port_slot.sh"),
            str(lock_dir),
            *command,
        ],
        start_new_session=True,
    )
    requeue = False

    def terminate(signum, _frame):
        nonlocal requeue
        requeue = signum == signal.SIGUSR1
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGUSR1, terminate)
    signal.signal(signal.SIGTERM, terminate)
    manager = FlexManager(args)
    try:
        while process.poll() is None:
            manager.tick()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        status = process.returncode
    finally:
        manager.close()
    if requeue:
        write_json(
            experiment / "requeue.json",
            {"job_id": os.environ["SLURM_JOB_ID"], "reason": "walltime_signal_durable_resume"},
        )
        subprocess.run(["scontrol", "requeue", os.environ["SLURM_JOB_ID"]], check=True)
    elif status:
        raise SystemExit(status)


def inside(args):
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
    from async_lora.runner import run_clients
    from lora_teachers.runner import run

    run(args.experiment, args.model_path, clients=run_clients)


def prepare(args, plan):
    if not args.image.is_file() or not (args.model_path / "config.json").is_file():
        raise FileNotFoundError("The pinned Miles image and original HF model must exist")
    experiment = (
        args.experiment
        or args.scratch / "runs" / f"{experiment_stem(plan)}-{datetime.now().astimezone():%Y%m%d-%H%M%S}"
    ).resolve()
    freeze(experiment, args.miles_repo, plan)
    command = [
        "python3",
        str(experiment / "source/rlve/tools/submit_async_lora.py"),
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
        f"--job-name={experiment_stem(plan)}",
        "--account=miso",
        "--partition=miso,sphinx,jag-standard,miso-lo,sphinx-lo,jag-lo,sc-loprio",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:4",
        "--cpus-per-task=24",
        "--mem=96G",
        "--time=1-00:00:00",
        "--constraint=ada|48G|80G|141G",
        "--exclude=pasteur8,jagupard30",
        "--requeue",
        "--signal=B:USR1@180",
        f"--chdir={experiment}",
        f"--output={experiment}/slurm-%j.out",
        "--wrap",
        "exec " + shlex.join(command),
    ]
    write_json(experiment / "submit_argv.json", argv)
    return experiment, argv


def submit(args, plan):
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be exported")
    experiment, argv = prepare(args, plan)
    job = subprocess.check_output(argv, text=True).strip().split(";")[0]
    result = {
        "job_id": job,
        "experiment": str(experiment),
        "student_project": plan["student_project"],
        "teacher_project": plan["teacher_project"],
    }
    write_json(experiment / "job.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("check-only", "prepare-only", "submit", "worker", "inside", "flex-worker", "flex-inside"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--size", type=int, choices=(16, 32), default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--single-question", action="store_true")
    parser.add_argument("--single-question-data", type=Path)
    parser.add_argument("--replay-percent", type=int, choices=(0, 10, 30, 50), default=0)
    parser.add_argument(
        "--speed-profile",
        choices=("original", "v2", "v3"),
        default="original",
        help="v2: 1/3 GPU split; v3: 2/2 split. Both use cheaper teacher eval and eight eval prompts in flight",
    )
    parser.add_argument(
        "--flex-gpus",
        type=int,
        choices=range(9),
        default=0,
        help="Additional one-GPU generation jobs, beyond the four-GPU core",
    )
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--generation")
    parser.add_argument("--scratch", type=Path, default=Path("/juice5b/scr5b/dhei"))
    parser.add_argument("--image", type=Path, default=Path(f"/juice5b/scr5b/dhei/images/miles-{IMAGE_TAG}.sif"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("/juice5b/scr5b/dhei/models/DeepSeek-R1-Distill-Qwen-1.5B")
    )
    parser.add_argument("--miles-repo", type=Path, default=ROOT.parent / "miles")
    args = parser.parse_args()
    if args.flex_worker or args.flex_inside:
        from async_lora import flex_worker

        (flex_worker.worker if args.flex_worker else flex_worker.inside)(args)
    elif args.worker:
        worker(args)
    elif args.inside:
        inside(args)
    else:
        plan = build_async_plan(
            args.size,
            learning_rate=args.learning_rate,
            single_question=args.single_question,
            replay_percent=args.replay_percent,
            question_data=args.single_question_data,
            speed_profile=args.speed_profile,
            flex_gpus=args.flex_gpus,
        )
        if args.submit:
            submit(args, plan)
        elif args.prepare_only:
            experiment, _ = prepare(args, plan)
            result = {
                "status": "prepared_not_submitted",
                "experiment": str(experiment),
                "physical_gpus": plan["physical_gpus"],
                "adapter_slots": plan["adapter_slots"],
            }
            write_json(experiment / "prepared.json", result)
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
