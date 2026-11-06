"""Prepare a frozen sync transfer-regime sweep; submission is explicit and separate."""

import argparse
import fcntl
import hashlib
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

from lora_teachers.runner import verify_inputs
from submit_clustering_mopd import copy_base
from submit_data_scaling_lora import container_argv, gpu_environment
from transfer_regimes.estimate import calibration, estimate
from transfer_regimes.placement import allocated_rows
from transfer_regimes.plan import build_plans
from transfer_regimes.selection import select_groups

from async_branching.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path("/juice5b/scr5b/dhei/runs/async-lora-n32-20261025-141924")
DISCOVERY = Path("/juice5b/scr5b/dhei/runs/mopd-clustering-n64-20261031-165345")
RUNTIME = Path("/juice5b/scr5b/dhei/runs/clustering-mopd-n32-d8-20261101-174231")
PARTITIONS = "jag-standard,miso,sphinx,miso-lo,sphinx-lo,jag-lo,sc-loprio"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    sweep = (
        args.sweep
        or args.scratch
        / "runs"
        / f"clustering-simple-sync-v1-{datetime.now():%Y%m%d-%H%M%S}"
    ).resolve()
    if sweep.exists():
        raise FileExistsError("Preparation requires a new directory")
    if not args.image.is_file() or not (args.model_path / "config.json").is_file():
        raise FileNotFoundError("Pinned image or base model is missing")
    selection = select_groups(
        ROOT / "plot/mopd-clustering/n64/balanced/results.json",
        ROOT / "plot/mopd-clustering/n64/low-transfer/results.json",
        DISCOVERY / "clustering/transfer.json",
        seed=args.seed,
        controls=args.controls,
    )
    plans, generated = build_plans(
        selection,
        REFERENCE,
        DISCOVERY,
        rounds=args.rounds,
        seed=args.seed,
        student_project=args.student_project,
        teacher_project=args.teacher_project,
    )
    rates = calibration()
    sweep.mkdir(parents=True)
    write_json(sweep / "selection.json", selection)
    write_json(sweep / "calibration.json", rates)
    # Share one private, immutable runtime across only these new experiments.
    # Existing running experiments and their source manifests are untouched.
    shutil.copytree(
        RUNTIME / "source",
        sweep / "source",
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    shutil.copytree(
        RUNTIME / "runtime",
        sweep / "runtime",
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(
        ROOT / "tools/transfer_regimes",
        sweep / "source/rlve/tools/transfer_regimes",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(
        Path(__file__), sweep / "source/rlve/tools/prepare_transfer_regimes.py"
    )
    for relative, rows in generated.items():
        write_json(sweep / "source/rlve" / relative, rows)
    # Ensure the selected original 64-task teacher gates are copied verbatim.
    for teacher in {t["environment"]: t for p in plans for t in p["teachers"]}.values():
        destination = sweep / "source/rlve" / teacher["eval"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DISCOVERY / "source/rlve" / teacher["eval"], destination)
        if sha256(destination) != teacher["eval_sha256"]:
            raise ValueError("Teacher gate bytes changed")
    initial = Path(plans[0]["initial_student_reference"]["source_bundle"])
    shutil.copytree(initial, sweep / "common-base")
    hashes = {
        str(p.relative_to(sweep)): sha256(p)
        for p in (sweep / "source").rglob("*")
        if p.is_file()
    }
    write_json(sweep / "source-sha256.json", hashes)
    experiments, estimates = [], []
    # Exercise both methods immediately on the extremes, then schedule the
    # remaining longer JointRL jobs before the remaining MOPD jobs.
    group_order = [
        "worst-1",
        "best-1",
        "neutral-1",
        "random-1",
        "worst-2",
        "best-2",
        "neutral-2",
        "random-2",
    ]
    early = [
        ("worst-1", "joint-rl"),
        ("worst-1", "sync-mopd"),
        ("best-1", "joint-rl"),
        ("best-1", "sync-mopd"),
    ]

    def order(plan):
        pair = (plan["group"], plan["algorithm"])
        return (
            (0, early.index(pair))
            if pair in early
            else (
                1 if plan["algorithm"] == "joint-rl" else 2,
                group_order.index(plan["group"]),
            )
        )

    plans.sort(key=order)
    for index, plan in enumerate(plans):
        label = f"{plan['group']}-{plan['algorithm']}"
        experiment = sweep / "experiments" / f"{sweep.name}-{label}"
        experiment.mkdir(parents=True)
        (experiment / "source").symlink_to(sweep / "source", target_is_directory=True)
        (experiment / "runtime").symlink_to(sweep / "runtime", target_is_directory=True)
        copy_base(sweep / "common-base", experiment / "initial-base", experiment.name)
        plan.update(sweep=str(sweep), selection_sha256=sha256(sweep / "selection.json"))
        write_json(experiment / "plan.json", plan)
        owned = {**hashes, "plan.json": sha256(experiment / "plan.json")}
        owned.update(
            {
                str(p.relative_to(experiment)): sha256(p)
                for p in (experiment / "initial-base").rglob("*")
                if p.is_file()
            }
        )
        write_json(experiment / "input_sha256.json", owned)
        experiments.append(
            {
                "index": index,
                "label": label,
                "algorithm": plan["algorithm"],
                "group": plan["group"],
                "experiment": str(experiment),
            }
        )
        estimates.append(estimate(plan, rates))
    output = ROOT / "plot/mopd-clustering/transfer-regimes" / sweep.name
    manifest = {
        "status": "prepared_not_launched",
        "prepared_at": time.time(),
        "experiments": experiments,
        "physical_gpus_per_run": 4,
        "maximum_concurrent_runs": args.max_concurrent,
        "peak_gpus": 4 * args.max_concurrent,
        "student_project": args.student_project,
        "teacher_project": args.teacher_project,
        "controls": args.controls,
        "scratch": str(args.scratch),
        "image": str(args.image),
        "model_path": str(args.model_path),
        "analysis_output": str(output),
        "required_free_scratch_gib": 8 * args.max_concurrent + 16,
    }
    write_json(sweep / "sweep.json", manifest)
    write_json(sweep / "submit_argv.json", submission_command(sweep, manifest))
    write_json(
        sweep / "estimates.json",
        {
            "calibration": rates,
            "runs": estimates,
            "six_slot_ideal_central_hours": round(
                sum(e["central_hours"] for e in estimates) / args.max_concurrent, 1
            ),
            "queue_and_preemption_included": False,
        },
    )
    table = [
        "# JointRL versus synchronous MOPD",
        "",
        "Prepared only; no Slurm submissions or W&B runs have been created.",
        "",
        "Six four-task groups × two methods. Each run uses four GPUs; the default six concurrent runs use 24 GPUs.",
        "",
        "| Group | Discovery cross-task transfer | Tasks |",
        "|---|---:|---|",
    ]
    for group in selection["groups"]:
        table.append(
            f"| {group['label']} | {100 * group['mean_transfer']:+.2f} pp | {', '.join(group['environments'])} |"
        )
    table += [
        "",
        "Extremes rank the previously fitted positive/negative four-task clusters. "
        + (
            "The two neutral controls are the fitted quartets closest to zero mean net discovery transfer; individual directional effects may cancel."
            if args.controls == "near-zero"
            else "The two random groups are unfiltered seeded draws, not assumed to have zero transfer."
        ),
        "",
        "Two rounds: up to four specialist teachers × 4k completions (or 0.98 validation), followed by 4k mixed-task student completions. Successful specialists are retained in round two. JointRL trains one LoRA across the same four tasks for up to 40k completions. Both share the same maximum total completion budget; all MOPD teacher work counts. MOPD early stopping can use less. Methods share the untrained adapter, rank 32/alpha 64, base model, and LR 2e-5.",
        "",
        "Monitoring uses eight disjoint confirmation prompts per task. The final endpoint uses the remaining 24 per task, isolated from monitoring and teacher stopping. Both use eight completions per prompt and the same 32-environment unseen panel.",
        "",
        "| Method | Four-GPU planning wall time |",
        "|---|---|",
    ]
    for algorithm in ["joint-rl", "sync-mopd"]:
        row = next(e for e in estimates if e["algorithm"] == algorithm)
        table.append(
            f"| {algorithm} | {row['hours_range'][0]}–{row['hours_range'][1]} hours |"
        )
    table += [
        "",
        "These ranges use measured GRPO, low-contention OPD, and evaluation rates from prior runs. The new synchronous/mixed-teacher implementation has CPU validation only until a GPU preflight is explicitly launched. They exclude queue and preemption delays; task length and GPU type can change throughput.",
        "",
        "The hypothesis is a larger MOPD−JointRL gain in the negative-transfer groups. Two groups per condition and one training seed per group make this an exploratory comparison, not a powered causal test. Plot per-task and macro learning curves, actual training completions and physical elapsed time, and the separate final held-out endpoint.",
        "",
        "Launch only after review:",
        "",
        f"```bash\npython3 {sweep}/source/rlve/tools/prepare_transfer_regimes.py --submit --sweep {sweep}\n```",
        "",
    ]
    (sweep / "README.md").write_text("\n".join(table))
    shutil.copy2(ROOT / "AGENTS.md", sweep / "PROTOCOL.md")
    print(sweep)
    return sweep


def submission_command(sweep, manifest):
    script = sweep / "source/rlve/tools/prepare_transfer_regimes.py"
    command = ["python3", str(script), "--worker", "--sweep", str(sweep)]
    argv = [
        "sbatch",
        "--parsable",
        "--account=miso",
        f"--partition={PARTITIONS}",
        "--job-name=transfer-regimes-sync",
        "--nodes=1",
        "--ntasks=1",
        f"--chdir={sweep}",
        "--gres=gpu:4",
        "--cpus-per-task=24",
        "--mem=96G",
        "--constraint=ampere|ada|hopper",
        "--exclude=jagupard28,jagupard30,pasteur8,tiger6,tiger8,viscam5,viscam10,oriong12,arm1",
        "--time=06:00:00",
        "--time-min=00:30:00",
        "--requeue",
        "--signal=B:USR1@180",
        f"--array=0-{len(manifest['experiments']) - 1}%{manifest['maximum_concurrent_runs']}",
        f"--output={sweep}/slurm-%A_%a.out",
        "--open-mode=append",
        "--wrap",
        "exec " + shlex.join(command),
    ]
    return argv


def submit(args):
    with (args.sweep / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _submit(args)


def _submit(args):
    sweep = args.sweep.resolve()
    manifest = read_json(sweep / "sweep.json")
    if manifest["status"] != "prepared_not_launched":
        raise ValueError("This sweep has already been submitted")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required")
    if (
        shutil.disk_usage(manifest["scratch"]).free
        < manifest["required_free_scratch_gib"] * 2**30
    ):
        raise RuntimeError(
            "Insufficient shared scratch for the bounded checkpoint workload"
        )
    for item in manifest["experiments"]:
        verify_inputs(Path(item["experiment"]))
    argv = read_json(sweep / "submit_argv.json")
    if argv != submission_command(sweep, manifest):
        raise ValueError("Prepared launch arguments changed")
    script = sweep / "source/rlve/tools/prepare_transfer_regimes.py"
    job = subprocess.check_output(argv, text=True).strip().split(";")[0]
    manifest.update(status="submitted", array_job_id=job)
    write_json(sweep / "sweep.json", manifest)
    watch = ["python3", str(script), "--monitor", "--sweep", str(sweep)]
    watch_id = (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--account=miso",
                "--partition=sc-freecpu",
                "--job-name=transfer-regimes-watch",
                "--cpus-per-task=1",
                "--mem=1G",
                "--time=7-00:00:00",
                f"--output={sweep}/monitor-%j.out",
                "--wrap",
                "exec " + shlex.join(watch),
            ],
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    manifest["watchdog_job_id"] = watch_id
    write_json(sweep / "sweep.json", manifest)
    readme = sweep / "README.md"
    readme.write_text(
        readme.read_text()
        .replace(
            "Prepared only; no Slurm submissions or W&B runs have been created.",
            f"Submitted Slurm array `{job}`; recovery monitor `{watch_id}`. "
            f"Main policies log to `{manifest['student_project']}` and specialists to `{manifest['teacher_project']}`.",
        )
        .replace("Launch only after review:", "Recorded launch command:")
    )
    print(json.dumps(manifest, indent=2))


def worker(args):
    manifest = read_json(args.sweep / "sweep.json")
    experiment = Path(
        manifest["experiments"][int(os.environ["SLURM_ARRAY_TASK_ID"])]["experiment"]
    )
    print(
        f"TRANSFER_REGIMES_ATTEMPT {os.uname().nodename} {os.environ.get('SLURM_JOB_ID')} {time.time()}",
        flush=True,
    )
    verify_inputs(experiment)
    if (experiment / "completed.json").exists():
        return
    archive = experiment / "attempt-logs" / str(time.time_ns())
    for previous in [
        experiment / "gateway.log",
        *list((experiment / "workers").glob("*/error.json")),
    ]:
        if previous.exists():
            destination = archive / previous.relative_to(experiment)
            destination.parent.mkdir(parents=True, exist_ok=True)
            previous.rename(destination)
    if shutil.disk_usage("/tmp").free < 12 * 2**30:
        raise RuntimeError("Need 12 GiB of free node-local /tmp for adapter snapshots")
    if not (experiment / "physical-start.json").exists():
        write_json(
            experiment / "physical-start.json",
            {"time": time.time(), "job_id": os.environ["SLURM_JOB_ID"]},
        )
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    inventory = {
        "node": os.uname().nodename,
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_rows": rows,
        "time": time.time(),
    }
    write_json(experiment / "gpu-inventory.json", inventory)
    rows = allocated_rows(rows, os.environ.get("SLURM_JOB_GPUS", ""))
    write_json(experiment / "gpu-inventory.json", {**inventory, "selected_rows": rows})
    plan = read_json(experiment / "plan.json")
    os.environ.update(
        gpu_environment(
            rows,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            minimum_gpu_mib=plan.get("minimum_gpu_mib", 45000),
        )
    )
    command = container_argv(
        experiment,
        scratch=Path(manifest["scratch"]),
        image=Path(manifest["image"]),
        model_path=Path(manifest["model_path"]),
    )
    command[
        command.index(str(experiment / "source/rlve/tools/submit_data_scaling_lora.py"))
    ] = str(experiment / "source/rlve/tools/prepare_transfer_regimes.py")
    lock_dir = (
        Path(manifest["scratch"])
        / "locks/rlve-ports"
        / os.uname().nodename.split(".")[0]
    )
    shell = 'source "$1"; rlve_reserve_port_slot "$2"; export LORA_PORT_SLOT="$RLVE_RESERVED_PORT_SLOT"; shift 2; exec "$@"'
    process = subprocess.Popen(
        [
            "bash",
            "-euc",
            shell,
            "transfer-regimes-lock",
            str(experiment / "source/rlve/scripts/slurm/port_slot.sh"),
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
    for name in ["prepare", "submit", "worker", "inside", "monitor", "check"]:
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument("--sweep", type=Path)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--scratch", type=Path, default=Path("/juice5b/scr5b/dhei"))
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("/juice5b/scr5b/dhei/images/miles-dev-202610231907.sif"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/juice5b/scr5b/dhei/models/DeepSeek-R1-Distill-Qwen-1.5B"),
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-concurrent", type=int, default=6)
    parser.add_argument(
        "--controls", choices=["near-zero", "random"], default="near-zero"
    )
    parser.add_argument("--student-project", default="rl-clustering-simple-students")
    parser.add_argument("--teacher-project", default="rl-clustering-simple-teachers")
    args = parser.parse_args()
    if not 1 <= args.max_concurrent <= 12:
        parser.error("--max-concurrent must be between 1 and 12")
    if args.prepare:
        prepare(args)
    elif args.submit:
        submit(args)
    elif args.worker:
        worker(args)
    elif args.inside:
        from lora_teachers.runner import run
        from transfer_regimes.runtime import run_clients

        run(args.experiment, args.model_path, clients=run_clients)
    elif args.monitor:
        from transfer_regimes.monitor import monitor

        monitor(args.sweep)
    elif args.check:
        for item in read_json(args.sweep / "sweep.json")["experiments"]:
            verify_inputs(Path(item["experiment"]))
        print("All 12 frozen experiments verified; no jobs launched")


if __name__ == "__main__":
    main()
