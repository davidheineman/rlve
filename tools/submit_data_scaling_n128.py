"""Queue 64 additional teachers and two N128 heroes as independent Slurm jobs."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from prepare_data_scaling_n128 import ROOT, OUTPUT


PARTITIONS = "miso,sphinx,jag-standard,miso-lo,sphinx-lo,jag-lo,sc-loprio"


def write_rows(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-experiment", type=Path, required=True)
    parser.add_argument("--experiment", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    parent = args.parent_experiment.resolve()
    plan = json.loads((OUTPUT / "run_plan.json").read_text())
    parent_plan = json.loads((parent / "run_plan.json").read_text())
    assert plan["runs"][:64] == parent_plan["runs"][:64]
    assert len(plan["runs"]) == 130 and plan["new_jobs"] == 66
    subprocess.run(["sha256sum", "--check", "--status", str(parent / "input_sha256.txt")], check=True)
    frozen = json.loads((OUTPUT / "manifest.json").read_text())
    for name, digest in frozen["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    scratch = parent.parent.parent
    model = scratch / "models/DeepSeek-R1-Distill-Qwen-1.5B"
    for path in (model / "config.json", model / "tokenizer.json",
                 model.with_name(model.name + "_torch_dist") / "latest_checkpointed_iteration.txt",
                 ROOT / "scripts/slurm/data-scaling-n128-v1.sbatch"):
        assert path.is_file() and path.stat().st_size, path
    assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY is not set"
    assert shutil.disk_usage(scratch).free >= 500 * 1024**3, "Need 500 GiB free for final weights and concurrent exports"
    with (parent / "jobs.tsv").open() as handle:
        parents = {r["label"]: r for r in csv.DictReader(handle, delimiter="\t") if r["kind"] == "teacher"}
    assert len(parents) == 64
    parent_wait = [r["job"] for r in parents.values()
                   if not (parent / "teacher" / r["label"] / "weights_only_complete.json").exists()]
    if parent_wait:
        output = subprocess.check_output(["squeue", "-h", "-j", ",".join(parent_wait), "-o", "%i|%T"], text=True)
        states = dict(line.split("|") for line in output.splitlines())
        for job in parent_wait:
            assert states.get(job) in {"RUNNING", "PENDING", "COMPLETING"}, f"Parent completion job {job} is not active; inspect before extending"
    if args.check_only:
        print(f"Preflight passed: 66 new standalone jobs, {len(parent_wait)} unfinished shared teachers; no arrays/caps; existing inputs unchanged")
        return
    experiment = (args.experiment or parent.parent / ("data-scaling-v1-n128-" + datetime.now().strftime("%Y%m%d-%H%M%S"))).resolve()
    experiment.mkdir(exist_ok=False)
    (experiment / "teacher").mkdir()
    (experiment / "run_plan.json").write_bytes((OUTPUT / "run_plan.json").read_bytes())
    (experiment / "parent_experiment.json").write_text(json.dumps({
        "path": str(parent), "utc": datetime.now(timezone.utc).isoformat(),
        "shared_teachers": 64, "additional_teachers": 64,
    }, indent=2) + "\n")
    paths = {line.split("  ", 1)[1] for line in (parent / "input_sha256.txt").read_text().splitlines()}
    paths.update(str(p.relative_to(ROOT)) for p in OUTPUT.glob("*") if p.is_file())
    paths.update({"tools/prepare_data_scaling_n128.py", "tools/submit_data_scaling_n128.py",
                  "scripts/slurm/data-scaling-n128-v1.sbatch", "scripts/slurm/common.sh"})
    (experiment / "input_sha256.txt").write_text("".join(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() + "  " + name + "\n" for name in sorted(paths)))
    rows = []
    for spec in plan["runs"][:64]:
        old = parents[spec["label"]]
        (experiment / "teacher" / spec["label"]).symlink_to(parent / "teacher" / spec["label"], target_is_directory=True)
        rows.append({"kind": "teacher", "n": 1, "label": spec["label"], "job": old["job"],
                     "gpus": old["gpus"], "dependency": old["dependency"],
                     "training_job": old.get("training_job", old["job"]),
                     "training_gpus": 1, "phase": "shared", "source_experiment": str(parent)})
    manifest = experiment / "jobs.tsv"
    write_rows(manifest, rows)
    # Freeze the exact launch specification before any submissions.
    exports = ",".join(["ALL", f"RLVE_REPO={ROOT}", f"RLVE_SCRATCH={scratch}",
                        f"RLVE_SCALING_EXPERIMENT={experiment}",
                        f"RLVE_WANDB_TEAM={os.environ.get('RLVE_WANDB_TEAM', 'david-heineman')}",
                        f"RLVE_HF_CKPT={model}", f"RLVE_REF_LOAD={model}_torch_dist"])
    new_teacher_jobs = []
    for index in [128, *range(64, 128), 129]:
        spec = plan["runs"][index]
        kind = spec["kind"]
        dependency = "none"
        if kind == "mopd":
            dependency = "afterok:" + ":".join(parent_wait + new_teacher_jobs)
        command = ["sbatch", "--parsable", "--account=miso", f"--partition={PARTITIONS}",
                   f"--gres=gpu:{spec['gpus']}", f"--cpus-per-task={6 if kind == 'teacher' else 24}",
                   f"--mem={spec['memory_gib']}G", f"--constraint={spec['constraint']}",
                   "--exclude=viscam9,viscam10,iliad6,pasteur8",
                   "--time=" + {"teacher": "08:00:00", "rl": "12:00:00", "mopd": "1-00:00:00"}[kind],
                   f"--job-name=scaling-{spec['label']}",
                   f"--export={exports},RLVE_SCALING_RUN_INDEX={index}"]
        if dependency != "none":
            command.append("--dependency=" + dependency)
        command.append("scripts/slurm/data-scaling-n128-v1.sbatch")
        job = subprocess.check_output(command, text=True).strip().split(";")[0]
        assert job.isdigit(), job
        rows.append({"kind": kind, "n": spec["size"], "label": spec["label"], "job": job,
                     "gpus": spec["gpus"], "dependency": dependency, "training_job": job,
                     "training_gpus": spec["gpus"], "phase": "train", "source_experiment": str(experiment)})
        write_rows(manifest, rows)
        if kind == "teacher":
            new_teacher_jobs.append(job)
        print(f"Submitted {spec['label']}: {job} ({spec['gpus']} GPU)", flush=True)
    assert len(rows) == 130 and len(new_teacher_jobs) == 64
    print(f"Submitted 66 new jobs; manifest: {manifest}")


if __name__ == "__main__":
    main()
