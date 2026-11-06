"""Repeat round two with fresh teachers and student using 2k training generations."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from data_scaling_multiround import ROOT, build_plan as original_plan
from verify_opd_teachers import verify

PARTITIONS = "miso,sphinx,jag-standard,miso-lo,sphinx-lo,jag-lo,sc-loprio"


def write_rows(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_plan(parent):
    plan = original_plan(parent)
    plan.update(variant="gen-2k", rollout_max_response_len=2048,
                eval_max_response_len=1024, answer_forcing_extra_tokens=128)
    for run in plan["runs"]:
        if run["action"] == "train":
            run["config"] = ("configs/1gpu-data-scaling-multiround-teacher-2k-v1.yaml"
                             if run["kind"] == "teacher"
                             else "configs/data-scaling-multiround-mopd-2k-v1.yaml")
    return plan


def validate_configs():
    import yaml
    pairs = [("configs/1gpu-data-scaling-teacher-v1.yaml", "configs/1gpu-data-scaling-multiround-teacher-2k-v1.yaml"),
             ("configs/data-scaling-multiround-mopd-v1.yaml", "configs/data-scaling-multiround-mopd-2k-v1.yaml")]
    for baseline, longer in pairs:
        before = yaml.safe_load((ROOT / baseline).read_text())
        after = yaml.safe_load((ROOT / longer).read_text())
        assert after == {**before, "rollout-max-response-len": 2048}, longer
        assert after["eval-max-response-len"] == 1024
        assert after["num-rollout"] * after["rollout-batch-size"] * after["n-samples-per-prompt"] == 4000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-experiment", type=Path, required=True)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--reference-round2-experiment", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    parent = args.parent_experiment.resolve()
    subprocess.run(["sha256sum", "--check", "--status", str(parent / "input_sha256.txt")], check=True)
    plan = build_plan(parent)
    validate_configs()
    if args.reference_round2_experiment:
        reference = args.reference_round2_experiment.resolve()
        subprocess.run(["sha256sum", "--check", "--status", str(reference / "input_sha256.txt")], check=True)
        prior = json.loads((reference / "run_plan.json").read_text())
        assert prior["initial_hf"] == plan["initial_hf"], "Comparison must start from the same round-1 model"
        assert prior["teacher_selection"] == plan["teacher_selection"], "Comparison must retrain the same teachers"
        assert (reference / "mopd/n16-mopd/weights_only_complete.json").exists()
        plan["comparison_round2_experiment"] = str(reference)
    scratch = parent.parent.parent
    base = scratch / "models/DeepSeek-R1-Distill-Qwen-1.5B"
    verify(parent / "teacher", ROOT / "data/DATA-SCALING-V1/n16.txt", base, 499)
    with (parent / "jobs.tsv").open() as handle:
        parent_rows = list(csv.DictReader(handle, delimiter="\t"))
    student = next(row for row in parent_rows if row["label"] == "n16-mopd")
    parents = {row["label"]: row for row in parent_rows if row["kind"] == "teacher"}
    source = Path(plan["initial_hf"]).parent
    source_dependency = "none"
    if not (source / "weights_only_complete.json").exists():
        output = subprocess.check_output(["squeue", "-h", "-j", student["job"], "-o", "%i|%T"], text=True)
        states = dict(line.split("|") for line in output.splitlines())
        if states.get(student["job"]) not in {"PENDING", "RUNNING", "COMPLETING"}:
            raise RuntimeError("Round-1 MOPD has no final export or active completion job")
        source_dependency = "afterok:" + student["job"]
    assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY is not set"
    assert shutil.disk_usage(scratch).free >= 150 * 1024**3, "Need 150 GiB free for round-2 final models and exports"
    for name in ("configs/data-scaling-multiround-mopd-2k-v1.yaml", "configs/1gpu-data-scaling-multiround-teacher-2k-v1.yaml", "scripts/slurm/data-scaling-multiround-2k-v1.sbatch",
                 "scripts/slurm/data-scaling-multiround-convert.sbatch"):
        assert (ROOT / name).is_file(), name
    retrain = [run for run in plan["runs"] if run["kind"] == "teacher" and run["action"] == "train"]
    retained = [run for run in plan["runs"] if run["action"] == "retain"]
    report = {"method": "multi-round", "round": 2, "new_teachers": len(retrain),
              "retained_teachers": [run["environments"][0] for run in retained],
              "teacher_rollout_cap": 4000, "student_additional_rollout_cap": 4000,
              "source_dependency": source_dependency, "new_jobs": len(retrain) + 2, "rollout_max_response_len": 2048, "eval_max_response_len": 1024}
    print(json.dumps(report, indent=2), flush=True)
    if args.check_only:
        print("Preflight passed; no jobs submitted")
        return
    experiment = (args.experiment or parent.parent / ("data-scaling-multiround-r2-2k-v1-" + datetime.now().strftime("%Y%m%d-%H%M%S"))).resolve()
    experiment.mkdir(exist_ok=False)
    (experiment / "teacher").mkdir()
    plan["round1_completion_job"] = student["job"]
    (experiment / "run_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (experiment / "teacher_selection.json").write_text(json.dumps(plan["teacher_selection"], indent=2) + "\n")
    paths = {line.split("  ", 1)[1] for line in (parent / "input_sha256.txt").read_text().splitlines()}
    paths.update({"tools/data_scaling_multiround.py", "tools/submit_data_scaling_multiround_2k.py",
                  "tools/convert_hf_to_torch_dist.py", "configs/data-scaling-multiround-mopd-2k-v1.yaml", "configs/1gpu-data-scaling-multiround-teacher-2k-v1.yaml",
                  "scripts/slurm/data-scaling-multiround-2k-v1.sbatch",
                  "scripts/slurm/data-scaling-multiround-convert.sbatch", "scripts/slurm/common.sh",
                  str(experiment / "run_plan.json"), str(experiment / "teacher_selection.json")})
    (experiment / "input_sha256.txt").write_text("".join(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() + "  " + name + "\n" for name in sorted(paths)))
    rows = []
    for run in retained:
        old = parents[run["label"]]
        (experiment / "teacher" / run["label"]).symlink_to(parent / "teacher" / run["label"], target_is_directory=True)
        rows.append({"kind": "teacher", "n": 1, "label": run["label"], "job": old["job"],
                     "gpus": 0, "dependency": "none", "training_job": old.get("training_job", old["job"]),
                     "training_gpus": 1, "phase": "shared", "source_experiment": str(parent)})
    manifest = experiment / "jobs.tsv"
    if rows:
        write_rows(manifest, rows)
    exports = ",".join(["ALL", f"RLVE_REPO={ROOT}", f"RLVE_SCRATCH={scratch}",
                        f"RLVE_SCALING_EXPERIMENT={experiment}",
                        f"RLVE_WANDB_TEAM={os.environ.get('RLVE_WANDB_TEAM', 'david-heineman')}",
                        f"RLVE_HF_CKPT={plan['initial_hf']}",
                        f"RLVE_REF_LOAD={experiment / plan['initial_ref']}"])

    def submit(kind, label, index, gpus, memory, cpus, time_limit, dependency, script):
        command = ["sbatch", "--parsable", "--account=miso", "--partition=" + PARTITIONS,
                   f"--gres=gpu:{gpus}", f"--cpus-per-task={cpus}", f"--mem={memory}G",
                   "--constraint=ada|48G|80G|141G", "--exclude=viscam9,viscam10,iliad6,pasteur8",
                   "--time=" + time_limit, "--job-name=multi-round-r2-2k-" + label,
                   "--export=" + exports + (f",RLVE_SCALING_RUN_INDEX={index}" if index is not None else "")]
        if dependency != "none":
            command.append("--dependency=" + dependency)
        command.append(script)
        job = subprocess.check_output(command, text=True).strip().split(";")[0]
        assert job.isdigit(), job
        rows.append({"kind": kind, "n": 16 if kind == "mopd" else 1, "label": label,
                     "job": job, "gpus": gpus, "dependency": dependency, "training_job": job,
                     "training_gpus": gpus, "phase": "prepare" if kind == "prepare" else "train",
                     "source_experiment": str(experiment)})
        write_rows(manifest, rows)
        print(f"Submitted {label}: {job} ({gpus} GPU)", flush=True)
        return job

    conversion = submit("prepare", "round1-warmstart", None, 1, 96, 8, "02:00:00", source_dependency,
                        "scripts/slurm/data-scaling-multiround-convert.sbatch")
    teacher_jobs = []
    for index, run in enumerate(plan["runs"]):
        if run["kind"] == "teacher" and run["action"] == "train":
            teacher_jobs.append(submit("teacher", run["label"], index, 1, 96, 6, "08:00:00", "afterok:" + conversion,
                                       "scripts/slurm/data-scaling-multiround-2k-v1.sbatch"))
    submit("mopd", "n16-mopd", 16, 4, 192, 24, "12:00:00", "afterok:" + ":".join(teacher_jobs or [conversion]),
           "scripts/slurm/data-scaling-multiround-2k-v1.sbatch")
    (experiment / "submission.json").write_text(json.dumps({**report, "utc": datetime.now(timezone.utc).isoformat(),
                                                           "experiment": str(experiment)}, indent=2) + "\n")
    print(f"Queued fresh round-2 teachers and student with 2k rollouts; manifest: {manifest}")


if __name__ == "__main__":
    main()
