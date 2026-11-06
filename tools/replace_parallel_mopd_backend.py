"""Broaden hardware for one queued shard, leaving running shards untouched."""

import argparse
import copy
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lora_teachers.plan import IMAGE_TAG, ROOT, sha256
from lora_teachers.runner import verify_inputs
from scale_mopd_clustering import OVERLAY

from async_branching.io import read_json, write_json


def replace(args):
    experiment = args.experiment.resolve()
    job = read_json(experiment / "job.json")
    if job.get("mode") != "parallel":
        raise ValueError("Expected an existing parallel execution")
    manifest = read_json(job["parallel_manifest"])
    old = next(b for b in manifest["backends"] if b["index"] == args.index)
    state = subprocess.check_output(
        ["squeue", "-h", "-j", old["job_id"], "-o", "%T"], text=True
    ).strip()
    if state != "PENDING":
        raise ValueError(
            "Only a queued shard can be replaced; running progress is retained"
        )
    verify_inputs(Path(old["revision"]))
    revision = (
        experiment
        / "runtime-revisions"
        / f"hardware-shard{args.index}-{datetime.now():%Y%m%dT%H%M%S}"
    )
    revision.mkdir(parents=True)
    shutil.copytree(
        Path(old["revision"]) / "source",
        revision / "source",
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    for relative in OVERLAY:
        shutil.copy2(ROOT / relative, revision / "source/rlve" / relative)
    shutil.copy2(
        Path(__file__), revision / "source/rlve/tools/replace_parallel_mopd_backend.py"
    )
    logical_plan = read_json(experiment / "plan.json")
    write_json(revision / "plan.json", logical_plan)
    hashes = {
        str(path.relative_to(revision)): sha256(path)
        for path in (revision / "source").rglob("*")
        if path.is_file()
    }
    hashes["plan.json"] = sha256(revision / "plan.json")
    write_json(revision / "input_sha256.json", hashes)
    backend = revision / "backends" / f"shard-{args.index}" / experiment.name
    backend.mkdir(parents=True)
    for name, target in (
        ("source", revision / "source"),
        ("runtime", experiment / "runtime"),
        ("teacher", experiment / "teacher"),
    ):
        (backend / name).symlink_to(target, target_is_directory=True)
    physical = read_json(Path(old["backend"]) / "plan.json")
    physical.update(
        minimum_gpu_mib=args.minimum_gpu_mib, max_tokens_per_gpu=args.packing_tokens
    )
    if physical["teacher_config"] != logical_plan["teacher_config"]:
        raise ValueError("Hardware replacement changed training hyperparameters")
    physical["execution_changes"]["max_tokens_per_gpu"] = (
        "4096 packing budget for optional 24GB cards; effective batch and GRPO settings unchanged"
    )
    write_json(backend / "plan.json", physical)
    record = {k: v for k, v in old.items() if k != "job_id"}
    record.update(
        backend=str(backend),
        revision=str(revision),
        minimum_gpu_mib=args.minimum_gpu_mib,
        packing_tokens=args.packing_tokens,
        replaced_job_id=old["job_id"],
    )
    write_json(backend / "backend.json", record)
    write_json(
        backend / "input_sha256.json",
        {name: sha256(backend / name) for name in ("plan.json", "backend.json")},
    )
    write_json(backend / "pipeline-status.json", {"stage": "queued"})
    command = [
        "python3",
        str(revision / "source/rlve/tools/scale_mopd_clustering.py"),
        "--worker",
        "--experiment",
        str(backend),
        "--scratch",
        str(args.scratch),
        "--image",
        str(args.image),
        "--model-path",
        str(args.model_path),
    ]
    gres = f"gpu:{args.gpu_type}:4" if args.gpu_type else "gpu:4"
    argv = [
        "sbatch",
        "--parsable",
        "--account=miso",
        "--partition=miso-lo,sphinx-lo,jag-lo,sc-loprio",
        f"--job-name=mopd-n64-shard-{args.index}",
        "--nodes=1",
        "--ntasks=1",
        f"--gres={gres}",
        "--cpus-per-task=24",
        "--mem=96G",
        "--time=06:00:00",
        "--time-min=02:00:00",
        f"--constraint={args.constraint}",
        "--exclude=pasteur8,jagupard30,viscam10,tiger8",
        "--requeue",
        "--signal=B:USR1@180",
        f"--dependency=afterany:{old['job_id']}",
        f"--chdir={backend}",
        f"--output={backend}/slurm-%j.out",
        "--open-mode=append",
        "--wrap",
        "exec " + shlex.join(command),
    ]
    new_id = subprocess.check_output(argv, text=True).strip().split(";")[0]
    record["job_id"] = new_id
    write_json(backend / "job.json", {"job_id": new_id, "indices": record["indices"]})
    write_json(backend / "submit_argv.json", argv)
    subprocess.run(["scancel", job["watchdog_job_id"]], check=True)
    subprocess.run(["scancel", old["job_id"]], check=True)
    updated = copy.deepcopy(manifest)
    updated["backends"] = [
        record if b["index"] == args.index else b for b in manifest["backends"]
    ]
    updated["revision"] = str(revision)
    updated["previous_manifest"] = job["parallel_manifest"]
    write_json(revision / "execution.json", updated)
    job.update(
        job_id=updated["backends"][0]["job_id"],
        job_ids=[b["job_id"] for b in updated["backends"]],
        runtime_revision=str(revision),
        parallel_manifest=str(revision / "execution.json"),
    )
    write_json(experiment / "job.json", job)
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
    job["watchdog_job_id"] = watchdog
    write_json(experiment / "job.json", job)
    write_json(Path(logical_plan["plot_output"]) / "launch.json", job)
    print(
        f"Replaced queued shard {args.index}: {old['job_id']} -> {new_id}; running shards unchanged",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--minimum-gpu-mib", type=int, default=23000)
    parser.add_argument("--packing-tokens", type=int, default=4096)
    parser.add_argument("--constraint", default="ampere|ada|hopper")
    parser.add_argument(
        "--gpu-type", choices=("a5000", "3090", "a100", "a6000", "h200")
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
    replace(parser.parse_args())


if __name__ == "__main__":
    main()
