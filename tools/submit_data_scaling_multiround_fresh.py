"""Queue a fresh 4k-rollout N16 MOPD round one followed by selective round two."""

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from data_scaling_multiround import ROOT
from submit_data_scaling_multiround import PARTITIONS, write_rows
from verify_opd_teachers import verify


def fresh_plan(parent):
    parent = parent.resolve()
    original = json.loads((parent / "run_plan.json").read_text())
    teachers = copy.deepcopy(original["runs"][:16])
    names = [run["environments"][0] for run in teachers]
    assert names == (ROOT / "data/DATA-SCALING-V1/n16.txt").read_text().split()
    student = copy.deepcopy(next(run for run in original["runs"] if run["label"] == "n16-mopd"))
    student.update(config="configs/data-scaling-multiround-mopd-v1.yaml", updates=500,
                   training_rollouts=4000, initialization="fresh-original-R1-Distill-base")
    scratch = parent.parent.parent
    return {"method": "multi-round", "round": 1, "size": 16,
            "teacher_source_experiment": str(parent),
            "initial_hf": str(scratch / "models/DeepSeek-R1-Distill-Qwen-1.5B"),
            "initial_ref": str(scratch / "models/DeepSeek-R1-Distill-Qwen-1.5B_torch_dist"),
            "runs": teachers + [student], "teacher_bank": "Use the current 16 completed round-1 teachers",
            "student": "Fresh original base initialization; 500 updates x 8 prompts x 1 sample = 4k rollouts; its final export initializes round two"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-experiment", type=Path, required=True)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--round2-experiment", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    parent = args.parent_experiment.resolve()
    subprocess.run(["sha256sum", "--check", "--status", str(parent / "input_sha256.txt")], check=True)
    plan = fresh_plan(parent)
    verify(parent / "teacher", ROOT / "data/DATA-SCALING-V1/n16.txt", Path(plan["initial_hf"]), 499)
    assert (Path(plan["initial_ref"]) / "latest_checkpointed_iteration.txt").is_file()
    assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY is not set"
    assert shutil.disk_usage(parent).free >= 150 * 1024**3
    assert "num-rollout: 500\n" in (ROOT / plan["runs"][-1]["config"]).read_text()
    if args.check_only:
        print("Preflight passed: fresh 4-GPU round 1 from the original R1 base, 4k rollouts; its final model initializes round-2 teachers and 4k-rollout MOPD")
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment = (args.experiment or parent.parent / ("data-scaling-multiround-r1-v1-" + stamp)).resolve()
    round2 = (args.round2_experiment or parent.parent / ("data-scaling-multiround-r2-v1-" + stamp)).resolve()
    if round2.exists():
        raise RuntimeError(f"Round-2 destination already exists: {round2}")
    experiment.mkdir(exist_ok=False)
    (experiment / "teacher").mkdir()
    (experiment / "run_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    paths = {line.split("  ", 1)[1] for line in (parent / "input_sha256.txt").read_text().splitlines()}
    paths.update({"tools/submit_data_scaling_multiround_fresh.py", "configs/data-scaling-multiround-mopd-v1.yaml",
                  "scripts/slurm/data-scaling-multiround-r1-v1.sbatch", "scripts/slurm/common.sh",
                  str(experiment / "run_plan.json")})
    (experiment / "input_sha256.txt").write_text("".join(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() + "  " + name + "\n" for name in sorted(paths)))
    with (parent / "jobs.tsv").open() as handle:
        parents = {row["label"]: row for row in csv.DictReader(handle, delimiter="\t") if row["kind"] == "teacher"}
    rows = []
    for spec in plan["runs"][:16]:
        old = parents[spec["label"]]
        (experiment / "teacher" / spec["label"]).symlink_to(parent / "teacher" / spec["label"], target_is_directory=True)
        rows.append({"kind": "teacher", "n": 1, "label": spec["label"], "job": old["job"],
                     "gpus": 0, "dependency": "none", "training_job": old.get("training_job", old["job"]),
                     "training_gpus": 1, "phase": "shared", "source_experiment": str(parent)})
    manifest = experiment / "jobs.tsv"
    write_rows(manifest, rows)
    scratch = parent.parent.parent
    exports = ",".join(["ALL", f"RLVE_REPO={ROOT}", f"RLVE_SCRATCH={scratch}",
                        f"RLVE_SCALING_EXPERIMENT={experiment}",
                        f"RLVE_WANDB_TEAM={os.environ.get('RLVE_WANDB_TEAM', 'david-heineman')}",
                        f"RLVE_HF_CKPT={plan['initial_hf']}", f"RLVE_REF_LOAD={plan['initial_ref']}",
                        "RLVE_SCALING_RUN_INDEX=16"])
    job = subprocess.check_output([
        "sbatch", "--parsable", "--account=miso", "--partition=" + PARTITIONS,
        "--gres=gpu:4", "--cpus-per-task=24", "--mem=192G", "--time=12:00:00",
        "--constraint=ada|48G|80G|141G", "--exclude=viscam9,viscam10,iliad6,pasteur8",
        "--job-name=multi-round-r1-n16-mopd", "--export=" + exports,
        "scripts/slurm/data-scaling-multiround-r1-v1.sbatch",
    ], text=True).strip().split(";")[0]
    assert job.isdigit(), job
    rows.append({"kind": "mopd", "n": 16, "label": "n16-mopd", "job": job,
                 "gpus": 4, "dependency": "none", "training_job": job, "training_gpus": 4,
                 "phase": "train", "source_experiment": str(experiment)})
    write_rows(manifest, rows)
    sequence = {"utc": datetime.now(timezone.utc).isoformat(), "method": "multi-round",
                "round1_experiment": str(experiment), "round1_job": job, "round1_training_rollouts": 4000,
                "round2_experiment": str(round2), "round2_training_rollouts": 4000,
                "teacher_source_experiment": str(parent)}
    (experiment / "sequence.json").write_text(json.dumps(sequence, indent=2) + "\n")
    print(f"Queued fresh 4k-rollout round-1 MOPD: {job}; {manifest}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "tools/submit_data_scaling_multiround.py"),
                    "--parent-experiment", str(experiment), "--experiment", str(round2), "--submit"], check=True)
    print(f"Queued two-round 4k + 4k MOPD sequence: {experiment} -> {round2}")


if __name__ == "__main__":
    main()
