"""Prepare, estimate, or submit a frozen clustered async MOPD sweep."""

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Slurm starts this script from the experiment directory, outside the source
# checkout. Its sibling tools are on sys.path; the repository package root is
# also required for async_branching before entering the container.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clustering_mopd.composed import assignments_from_composed_search
from clustering_mopd.estimate import (
    COMPLETED_K8_SWEEP,
    calibrate,
    calibrate_completed,
    estimate,
)
from clustering_mopd.plan import (
    ASSIGNMENTS,
    ASSIGNMENTS32,
    CLUSTERING,
    REFERENCE,
    build_plan,
    configurations,
    encoded,
)
from lora_teachers.plan import IMAGE_TAG, ROOT, sha256
from submit_data_scaling_lora import container_argv, gpu_environment

from async_branching.io import read_json, write_json


def submission_command(experiment, args):
    script = experiment / "source/rlve/tools/submit_clustering_mopd.py"
    command = [
        "python3",
        str(script),
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
    return [
        "sbatch",
        "--parsable",
        "--account=miso",
        "--partition="
        + getattr(
            args,
            "partitions",
            "jag-standard,miso,sphinx,miso-lo,sphinx-lo,jag-lo,sc-loprio",
        ),
        f"--job-name={experiment.name}",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:4",
        "--cpus-per-task=24",
        "--mem=96G",
        "--time=06:00:00",
        "--time-min=02:00:00",
        "--constraint=48G|80G|141G",
        "--exclude=pasteur8,jagupard30,viscam10,tiger8",
        "--requeue",
        "--signal=B:USR1@180",
        f"--chdir={experiment}",
        f"--output={experiment}/slurm-%j.out",
        "--open-mode=append",
        "--wrap",
        "exec " + shlex.join(command),
    ]


def copy_base(source, destination, experiment_name):
    """Share immutable tensor storage; give every run a separate ownership envelope."""

    def copy_file(src, dst):
        if Path(src).name == "META.json":
            return shutil.copy2(src, dst)
        os.link(src, dst)
        return dst

    shutil.copytree(source, destination, copy_function=copy_file)
    digest = hashlib.sha256(f"tml-async-{experiment_name}".encode()).hexdigest()
    for path in (destination / "adapter/META.json", destination / "state/META.json"):
        value = read_json(path)
        value["tenant_digest"] = digest
        write_json(path, value)


def prepare(args):
    if not args.image.is_file() or not (args.model_path / "config.json").is_file():
        raise FileNotFoundError("Pinned image and base model are required")
    sweep = (
        args.experiment
        or args.scratch
        / "runs"
        / f"clustering-mopd-n{args.size}-d{args.simulated_devices}-{datetime.now():%Y%m%d-%H%M%S}"
    ).resolve()
    if sweep.exists():
        raise FileExistsError("Preparation never edits an existing sweep")
    composed = getattr(args, "composed_search", None)
    composed_counts = getattr(args, "composed_cluster_counts", None) or (8,)
    selection = None
    if composed:
        selection = assignments_from_composed_search(
            composed, counts=composed_counts, random_replicate=args.random_replicate
        )
        args.assignments = sweep / "assignment-source.json"
    calibration_source = getattr(args, "calibration_sweep", None)
    if composed and calibration_source is None:
        calibration_source = COMPLETED_K8_SWEEP
    calibration = (
        calibrate_completed(calibration_source)
        if calibration_source
        else calibrate(args.reference, args.clustering)
    )
    student_project = getattr(args, "student_project", None) or (
        "rl-clustering-students-v2" if composed else "rl-clustering-students"
    )
    teacher_project = getattr(args, "teacher_project", None) or (
        "rl-clustering-teachers-v2" if composed else "rl-clustering-teachers"
    )
    settings, inputs = [], {}
    for count, assignment in configurations(
        args.size, composed=bool(composed), composed_counts=composed_counts
    ):
        plan, generated = build_plan(
            count,
            assignment,
            devices=args.simulated_devices,
            horizon=args.simulated_horizon,
            seed=args.seed,
            reference=args.reference,
            clustering=args.clustering,
            assignments=args.assignments,
            size=args.size,
            assignment_selection=selection,
            student_project=student_project,
            teacher_project=teacher_project,
        )
        for path, value in generated.items():
            if path in inputs and inputs[path] != value:
                raise ValueError(f"Generated inputs disagree: {path}")
            inputs[path] = value
        settings.append(plan)
    sweep.mkdir(parents=True)
    # Start from the known-working n64 source snapshot, then freeze only the
    # intended async/cluster implementation and its direct dependencies.
    shutil.copytree(
        args.clustering / "source",
        sweep / "source",
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    source = sweep / "source/rlve"
    for relative in ("tools/async_lora", "async_branching", "tools/clustering_mopd"):
        shutil.copytree(
            ROOT / relative,
            source / relative,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
        )
    for relative in (
        "tools/submit_clustering_mopd.py",
        "tools/submit_async_lora.py",
        "tools/submit_data_scaling_lora.py",
        "tools/prepare_async_branching.py",
        "tools/prepare_async_lora_questions.py",
        "tools/make_rlve_eval_set.py",
        "tools/lora_teachers/runner.py",
        "tools/lora_teachers/plan.py",
        "scripts/slurm/port_slot.sh",
    ):
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    for relative in (
        settings[0]["test_eval"],
        "configs/data-scaling-mopd-v1.yaml",
        "configs/1gpu-data-scaling-teacher-v1.yaml",
    ):
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.reference / "source/rlve" / relative, destination)
    for relative in (
        settings[0]["legacy_id_eval"],
        "data/DATA-SCALING-V1/manifest.json",
        "data/DATA-SCALING-V1/run_plan.json",
    ):
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    for relative, value in inputs.items():
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded(value))
    if selection is None:
        shutil.copy2(args.assignments, sweep / "assignment-source.json")
    else:
        write_json(sweep / "assignment-source.json", selection)
    initial = Path(settings[0]["initial_student_reference"]["source_bundle"])
    shutil.copytree(initial, sweep / "common-base")
    # A private SDK copy avoids modifying or depending on the old run's runtime.
    shutil.copytree(
        args.clustering / "runtime",
        sweep / "runtime",
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    common_hashes = {
        str(path.relative_to(sweep)): sha256(path)
        for path in (sweep / "source").rglob("*")
        if path.is_file()
    }
    common_hashes["assignment-source.json"] = sha256(sweep / "assignment-source.json")
    write_json(sweep / "source-sha256.json", common_hashes)
    experiments, estimates = [], []
    for plan in settings:
        label = f"k{plan['cluster_count']:02d}-{plan['assignment']}"
        experiment = sweep / "experiments" / f"{sweep.name}-{label}"
        experiment.mkdir(parents=True)
        (experiment / "source").symlink_to(sweep / "source", target_is_directory=True)
        (experiment / "runtime").symlink_to(sweep / "runtime", target_is_directory=True)
        (experiment / "assignment-source.json").symlink_to(
            sweep / "assignment-source.json"
        )
        copy_base(sweep / "common-base", experiment / "initial-base", experiment.name)
        plan.update(
            sweep=str(sweep),
            assignment_frozen_sha256=sha256(sweep / "assignment-source.json"),
        )
        write_json(experiment / "plan.json", plan)
        hashes = {k: v for k, v in common_hashes.items() if k.startswith("source/")}
        hashes["assignment-source.json"] = sha256(sweep / "assignment-source.json")
        hashes["plan.json"] = sha256(experiment / "plan.json")
        hashes.update(
            {
                str(path.relative_to(experiment)): sha256(path)
                for path in (experiment / "initial-base").rglob("*")
                if path.is_file()
            }
        )
        write_json(experiment / "input_sha256.json", hashes)
        argv = submission_command(experiment, args)
        write_json(experiment / "submit_argv.json", argv)
        experiments.append(
            {
                "label": label,
                "cluster_count": plan["cluster_count"],
                "assignment": plan["assignment"],
                "experiment": str(experiment),
                "physical_gpus": 4,
                "adapter_slots": plan["adapter_slots"],
            }
        )
        estimates.append(estimate(plan, calibration))
    report = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_not_launched",
        "size": args.size,
        "student_project": student_project,
        "teacher_project": teacher_project,
        "assignment_method": settings[0]["assignment_method"],
        "composed_search": str(Path(composed).resolve()) if composed else None,
        "simulated_devices": args.simulated_devices,
        "simulated_clock_budget": args.simulated_horizon,
        "physical_gpus_per_experiment": 4,
        "physical_gpus_for_all_runs": 4 * len(experiments),
        "experiments": experiments,
    }
    report["required_free_scratch_gib"] = math.ceil(
        0.6
        * sum(
            r["adapter_slots"] + math.ceil(args.simulated_horizon / 400) + 2
            for r in experiments
        )
        + 16
    )
    panel_plot = ROOT / "plot/mopd-clustering"
    if args.size == 64:
        panel_plot = panel_plot / "n64"
    report["analysis_output"] = str(panel_plot / "mopd" / sweep.name)
    output = Path(report["analysis_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    report["analysis_argv"] = [
        "apptainer",
        "exec",
        "--no-home",
        "--bind",
        str(args.scratch),
        "--bind",
        str(output.parent),
        "--env",
        f"PYTHONPATH={sweep}/source/rlve/tools:{sweep}/source/rlve",
        "--env",
        f"MPLCONFIGDIR={sweep}/plot-cache",
        str(args.image),
        "python3",
        "-m",
        "clustering_mopd.analysis",
        str(sweep),
        "--output",
        str(output),
    ]
    report["available_scratch_gib_at_preparation"] = round(
        shutil.disk_usage(sweep).free / 2**30, 1
    )
    report["storage_ready_for_submission"] = (
        report["available_scratch_gib_at_preparation"]
        >= report["required_free_scratch_gib"]
    )
    write_json(sweep / "sweep.json", report)
    write_json(
        sweep / "estimates.json",
        {
            "calibration": calibration,
            "runs": estimates,
            "all_runs_parallel_hours_range": [
                max(e["physical_hours_range"][i] for e in estimates) for i in (0, 1)
            ],
            "queue_time_included": False,
        },
    )
    lines = [
        "# Prepared clustered async MOPD sweep",
        "",
        "Status: prepared; no experiments submitted.",
        "",
        "| Run | Clusters | Envs/cluster | Teacher lanes | Four-GPU hours estimate |",
        "|---|---:|---:|---:|---:|",
    ]
    for e in estimates:
        lines.append(
            f"| {e['label']} | {e['cluster_count']} | {args.size // e['cluster_count']} | {e['simulated_teacher_lanes']} | {e['physical_hours_range'][0]}–{e['physical_hours_range'][1]} |"
        )
    lines.extend(
        [
            "",
            "Ranges are conditional planning estimates, not measurements or hard bounds. Initial Slurm queue time is additional.",
            "The simulated clock matches async-lora-n32's training-rollout proxy and excludes evaluations; actual four-GPU elapsed time is logged separately.",
            f"Primary ID evaluation equally weights all {args.size} tasks; the original ID subset is also reported from the same samples.",
            "Original settings retained: 4000 teacher rollouts or 0.98 cluster validation; 1000 student rollouts/version or 90% recovery; 400-rollout fair merge quanta; 2e-5 learning rate, rank32/alpha64.",
            "Each policy has 32 gate prompts, except the 64-task policy has 64 to cover every member; all use eight samples/prompt.",
            "The same untrained LoRA factors seed every student. Optimized/random assignments are frozen before new training. Discovery cost is excluded from the online curves.",
            f"One student and up to {args.simulated_devices - 1} teacher devices share the simulated {args.simulated_devices}-device budget; excess clusters queue fairly, and spare capacity remains idle for small k.",
        ]
    )
    if args.size == 32:
        lines.append(
            "The two k=1 runs have identical assignments and distinct training seeds; they are joint-training repeats. The k=32 baseline is newly trained under the same device budget."
        )
    if composed:
        lines.extend(
            [
                "",
                f"This v2 comparison uses k={','.join(map(str, composed_counts))} partitions over 64 environments, with high transfer, random, and low transfer at each k. Groups are imported from the selection-frozen composed-adapter search, without consulting confirmation scores. The single random baseline uses predeclared replicate "
                f"{selection['provenance']['random_replicate']} (seed {selection['provenance']['random_seed']}); the discovery plot averages three random partitions.",
                "Only the assignments are transferred. Teachers train fresh group adapters branched from the current student; discovery adapters and their composed weights are not used to initialize training.",
                f"Students log to {student_project}; teacher traces log to {teacher_project}.",
            ]
        )
    lines.extend(
        [
            "",
            f"Storage reservation: {report['required_free_scratch_gib']} GiB free required at launch; {report['available_scratch_gib_at_preparation']} GiB available after preparation. Submission enforces this guard before creating any jobs.",
        ]
    )
    (sweep / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"sweep": str(sweep), **report}, indent=2))
    return sweep


def verify_sweep(sweep):
    from lora_teachers.runner import verify_inputs

    manifest = read_json(sweep / "sweep.json")
    for record in manifest["experiments"]:
        verify_inputs(Path(record["experiment"]))
    return manifest


def submit(sweep):
    manifest = verify_sweep(sweep)
    if manifest.get("job_ids") or any(
        (Path(r["experiment"]) / "job.json").exists() for r in manifest["experiments"]
    ):
        raise ValueError("This sweep already has submitted jobs")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be exported at submission")
    # Keep the jobs from jointly overcommitting the shared checkpoint store.
    required_gib = manifest["required_free_scratch_gib"]
    free_gib = shutil.disk_usage(sweep).free / 2**30
    if free_gib < required_gib:
        raise RuntimeError(
            f"Need {required_gib} GiB free for this resumable sweep; found {free_gib:.1f} GiB"
        )
    submitted = []
    manifest.update(status="submitting", job_ids=[])
    write_json(sweep / "sweep.json", manifest)
    try:
        for record in manifest["experiments"]:
            directory = Path(record["experiment"])
            argv = read_json(directory / "submit_argv.json")
            job = subprocess.check_output(argv, text=True).strip().split(";")[0]
            submitted.append(job)
            write_json(
                directory / "job.json",
                {
                    "job_id": job,
                    "experiment": str(directory),
                    "student_project": manifest["student_project"],
                    "teacher_project": manifest["teacher_project"],
                },
            )
            manifest["job_ids"] = list(submitted)
            write_json(sweep / "sweep.json", manifest)
    except BaseException:
        if submitted:
            subprocess.run(["scancel", *submitted], check=True)
        manifest.update(status="submission_failed_cancelled", job_ids=submitted)
        write_json(sweep / "sweep.json", manifest)
        raise
    manifest.update(status="submitted", job_ids=submitted)
    write_json(sweep / "sweep.json", manifest)
    watcher = [
        "python3",
        str(sweep / "source/rlve/tools/submit_clustering_mopd.py"),
        "--monitor",
        "--recover",
        "--experiment",
        str(sweep),
    ]
    watch_id = (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--account=miso",
                "--partition=sc-freecpu",
                "--job-name=clustering-mopd-watch",
                "--cpus-per-task=1",
                "--mem=1G",
                "--time=7-00:00:00",
                "--requeue",
                f"--output={sweep}/monitor-%j.out",
                "--wrap",
                "exec " + shlex.join(watcher),
            ],
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    manifest["watchdog_job_id"] = watch_id
    write_json(sweep / "sweep.json", manifest)
    print(json.dumps(manifest, indent=2))


def worker(args):
    from lora_teachers.runner import verify_inputs

    experiment = args.experiment.resolve()
    start = experiment / "physical-start.json"
    if not start.exists():
        write_json(start, {"time": time.time(), "job_id": os.environ["SLURM_JOB_ID"]})
    verify_inputs(experiment)
    if shutil.disk_usage("/tmp").free < 12 * 2**30:
        raise RuntimeError(
            "Need 12 GiB of free node-local /tmp for async LoRA snapshots"
        )
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    plan = read_json(experiment / "plan.json")
    os.environ.update(
        gpu_environment(
            rows,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            minimum_gpu_mib=plan.get("minimum_gpu_mib", 45000),
        )
    )
    command = container_argv(
        experiment, scratch=args.scratch, image=args.image, model_path=args.model_path
    )
    original = str(experiment / "source/rlve/tools/submit_data_scaling_lora.py")
    command[command.index(original)] = str(
        experiment / "source/rlve/tools/submit_clustering_mopd.py"
    )
    lock_directory = (
        args.scratch / "locks/rlve-ports" / os.uname().nodename.split(".")[0]
    )
    script = 'source "$1"; rlve_reserve_port_slot "$2"; export LORA_PORT_SLOT="$RLVE_RESERVED_PORT_SLOT"; shift 2; exec "$@"'
    process = subprocess.Popen(
        [
            "bash",
            "-euc",
            script,
            "clustering-mopd-lock",
            str(experiment / "source/rlve/scripts/slurm/port_slot.sh"),
            str(lock_directory),
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
    for name in ("prepare-only", "check-only", "submit", "worker", "inside", "monitor"):
        modes.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--clustering", type=Path, default=CLUSTERING)
    parser.add_argument("--size", type=int, choices=(32, 64), default=64)
    parser.add_argument("--assignments", type=Path)
    parser.add_argument(
        "--composed-search",
        type=Path,
        help="Prepare high/random/low triplets from a selection-frozen composed search",
    )
    parser.add_argument(
        "--composed-cluster-counts",
        type=int,
        nargs="+",
        choices=(2, 4, 8, 16, 32),
        help="Numbers of groups to train; defaults to 8 (use 4 16 for 16/4 envs per group)",
    )
    parser.add_argument("--random-replicate", type=int, default=0)
    parser.add_argument("--student-project")
    parser.add_argument("--teacher-project")
    parser.add_argument("--calibration-sweep", type=Path)
    parser.add_argument("--simulated-devices", type=int)
    parser.add_argument("--simulated-horizon", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--partitions",
        default="jag-standard,miso,sphinx,miso-lo,sphinx-lo,jag-lo,sc-loprio",
    )
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
    if args.simulated_devices is None:
        args.simulated_devices = 8 if args.composed_search else 16
    if args.composed_search and args.assignments is not None:
        parser.error("--composed-search and --assignments are mutually exclusive")
    if args.composed_search and args.size != 64:
        parser.error("--composed-search requires --size 64")
    if args.composed_cluster_counts and not args.composed_search:
        parser.error("--composed-cluster-counts requires --composed-search")
    if args.assignments is None and not args.composed_search:
        args.assignments = ASSIGNMENTS32 if args.size == 32 else ASSIGNMENTS
    if args.prepare_only:
        prepare(args)
    elif args.check_only:
        print(json.dumps(verify_sweep(args.experiment), indent=2))
    elif args.submit:
        if args.experiment is None:
            parser.error("--submit requires a previously prepared --experiment")
        submit(args.experiment)
    elif args.worker:
        worker(args)
    elif args.monitor:
        from clustering_mopd.monitor import monitor

        monitor(args.experiment, repair=args.recover, once=args.once)
    else:
        from clustering_mopd.runner import run_clients
        from lora_teachers.runner import run

        run(args.experiment, args.model_path, clients=run_clients)


if __name__ == "__main__":
    main()
