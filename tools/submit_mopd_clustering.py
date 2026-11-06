"""Freeze and submit LoRA teachers followed by adapter transfer clustering.

python3 tools/submit_mopd_clustering.py --prepare-data  # pinned image, CPU
python3 tools/submit_mopd_clustering.py --submit
The four-GPU allocation survives preemption via durable teacher checkpoints;
clustering measurements are cached atomically and resumed independently.
"""

import argparse
import importlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from lora_clustering.plan import (
    build_clustering_plan,
    prepare_evaluation,
    validate_evaluation,
)
from lora_teachers.plan import IMAGE_TAG, ROOT, sha256
from submit_data_scaling_lora import container_argv, gpu_environment, snapshot


def freeze(experiment, miles_repo, plan):
    snapshot(experiment, miles_repo, plan)
    target = experiment / "source/rlve"
    for relative in ("tools/lora_clustering", "tools/async_lora", "async_branching"):
        shutil.copytree(
            ROOT / relative,
            target / relative,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
        )
    for relative in (
        "tools/submit_mopd_clustering.py",
        "tools/analyze_balanced_lora_clustering.py",
        "tools/monitor_mopd_clustering.py",
        "tools/make_rlve_eval_set.py",
        plan["clustering"]["selection"],
        plan["clustering"]["confirmation"],
        str(Path(plan["clustering"]["selection"]).with_name("manifest.json")),
        "data/DATA-SCALING-V1/manifest.json",
        "data/DATA-SCALING-V1/run_plan.json",
        "data/DATA-SCALING-V1/n16.txt",
        f"data/DATA-SCALING-V1/n{plan['size']}.txt",
    ):
        path = target / relative
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
    (experiment / "input_sha256.json").write_text(
        json.dumps(
            {str(p.relative_to(experiment)): sha256(p) for p in sorted(paths)}, indent=2
        )
        + "\n"
    )


def submit(args):
    if (
        not os.environ.get("WANDB_API_KEY")
        or not args.image.is_file()
        or not (args.model_path / "config.json").is_file()
    ):
        raise RuntimeError(
            "W&B credential, pinned image and original base model must be available"
        )
    plan = validate_evaluation(build_clustering_plan(size=args.size))
    plan["plot_output"] = str(args.plot_output.resolve())
    if args.reuse_experiment:
        from lora_clustering.reuse import describe_reuse

        plan["reuse"] = describe_reuse(args.reuse_experiment.resolve(), plan)
        plan["adapter_slots"] = len(plan["teachers"]) - plan["reuse"]["teachers"] + 1
    experiment = (
        args.experiment
        or args.scratch
        / "runs"
        / f"mopd-clustering-n{args.size}-{datetime.now():%Y%m%d-%H%M%S}"
    ).resolve()
    freeze(experiment, args.miles_repo, plan)
    if args.reuse_experiment:
        from lora_clustering.reuse import import_reuse

        import_reuse(experiment, args.reuse_experiment.resolve(), plan)
    command = [
        "python3",
        str(experiment / "source/rlve/tools/submit_mopd_clustering.py"),
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
        f"--job-name=mopd-clustering-n{args.size}",
        "--account=miso",
        "--partition=miso,sphinx,jag-standard,miso-lo,sphinx-lo,jag-lo,sc-loprio",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:4",
        "--cpus-per-task=24",
        "--mem=96G",
        "--time=3-00:00:00" if args.size == 64 else "--time=1-00:00:00",
        "--constraint=ada|48G|80G|141G",
        "--exclude=pasteur8,jagupard30,viscam10",
        "--requeue",
        "--signal=B:USR1@180",
        f"--chdir={experiment}",
        f"--output={experiment}/slurm-%j.out",
        "--open-mode=append",
        "--wrap",
        "exec " + shlex.join(command),
    ]
    (experiment / "submit_argv.json").write_text(json.dumps(argv, indent=2) + "\n")
    job_id = subprocess.check_output(argv, text=True).strip().split(";")[0]
    record = {
        "job_id": job_id,
        "experiment": str(experiment),
        "project": plan["project"],
        "teachers": args.size,
        "reused_teachers": plan.get("reuse", {}).get("teachers", 0),
        "teacher_rollouts": 8000,
        "plot_output": plan["plot_output"],
    }
    (experiment / "job.json").write_text(json.dumps(record, indent=2) + "\n")
    args.plot_output.mkdir(parents=True, exist_ok=True)
    (args.plot_output / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def worker(args):
    from lora_teachers.runner import verify_inputs

    experiment = args.experiment.resolve()
    verify_inputs(experiment)
    source_root = Path(__file__).resolve().parents[2]
    if source_root != experiment / "source":
        # A new immutable runtime revision can resume the same logical run,
        # retaining its original data, checkpoints, tenant and W&B identities.
        verify_inputs(source_root.parent)
    # Never clear arbitrary /tmp content. Avoid nodes that cannot hold this run's
    # exports; no teacher progress is lost beyond its last evaluation checkpoint.
    if shutil.disk_usage("/tmp").free < 16 * 2**30:
        job, node = os.environ["SLURM_JOB_ID"], os.uname().nodename.split(".")[0]
        record = experiment / "full-tmp-nodes.json"
        excluded = (
            json.loads(record.read_text())
            if record.exists()
            else ["pasteur8", "jagupard30", "viscam10"]
        )
        excluded = sorted(set([*excluded, node]))
        record.write_text(json.dumps(excluded) + "\n")
        subprocess.run(
            ["scontrol", "update", f"JobId={job}", "ExcNodeList=" + ",".join(excluded)],
            check=True,
        )
        subprocess.run(["scontrol", "requeue", job], check=True)
        return
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    os.environ.update(gpu_environment(rows, os.environ.get("CUDA_VISIBLE_DEVICES")))
    command = container_argv(
        experiment, scratch=args.scratch, image=args.image, model_path=args.model_path
    )
    original = str(experiment / "source/rlve/tools/submit_data_scaling_lora.py")
    command[command.index(original)] = str(
        source_root / "rlve/tools/submit_mopd_clustering.py"
    )
    for index, value in enumerate(command):
        if value.startswith("PYTHONPATH="):
            command[index] = value.replace(str(experiment / "source"), str(source_root))
    # The plots are deliberately written to the workspace, not only the snapshot.
    plot_output = Path(
        json.loads((experiment / "plan.json").read_text())["plot_output"]
    )
    command[command.index("--nv") : command.index("--nv")] = [
        "--bind",
        str(plot_output),
    ]
    script = 'source "$1"; rlve_reserve_port_slot "$2"; export LORA_PORT_SLOT="$RLVE_RESERVED_PORT_SLOT"; shift 2; exec "$@"'
    lock_dir = args.scratch / "locks/rlve-ports" / os.uname().nodename.split(".")[0]
    process = subprocess.Popen(
        [
            "bash",
            "-euc",
            script,
            "clustering-port-lock",
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
    while process.poll() is None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    if requeue:
        subprocess.run(["scontrol", "requeue", os.environ["SLURM_JOB_ID"]], check=True)
    elif process.returncode:
        raise SystemExit(process.returncode)


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
    # PYTHONPATH included this directory before it existed. Drop cached missing
    # directory importers now that pip has populated the SDK target.
    importlib.invalidate_caches()
    from lora_clustering.pipeline import run_pipeline
    from lora_teachers.runner import run

    run(args.experiment, args.model_path, clients=run_pipeline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare-data", "check-only", "submit", "worker", "inside"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--size", type=int, choices=(32, 64), default=32)
    parser.add_argument("--reuse-experiment", type=Path)
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
    parser.add_argument("--miles-repo", type=Path, default=ROOT.parent / "miles")
    parser.add_argument("--plot-output", type=Path)
    args = parser.parse_args()
    if args.plot_output is None:
        args.plot_output = ROOT / "plot/mopd-clustering"
        if args.size != 32:
            args.plot_output /= f"n{args.size}"
    if args.prepare_data:
        print(json.dumps(prepare_evaluation(size=args.size), indent=2))
    elif args.check_only:
        print(
            json.dumps(
                validate_evaluation(build_clustering_plan(size=args.size)), indent=2
            )
        )
    elif args.submit:
        submit(args)
    elif args.worker:
        worker(args)
    else:
        inside(args)


if __name__ == "__main__":
    main()
