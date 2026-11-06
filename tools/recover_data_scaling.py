"""Finalize intact training checkpoints on CPU and replace MOPD completion gates."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from repair_data_scaling import ROOT, query_states, write_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    experiment = args.experiment.resolve()
    manifest = experiment / "jobs.tsv"
    rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
    plan = json.loads((experiment / "run_plan.json").read_text())["runs"]
    by_label = {run["label"]: (index, run) for index, run in enumerate(plan)}
    states = query_states(rows)
    finalize = []
    for row in rows:
        if row["kind"] == "mopd" or row.get("phase") == "finalize":
            continue
        run = experiment / row["kind"] / row["label"]
        if (run / "weights_only_complete.json").exists():
            continue
        has_checkpoint = (run / "final-ckpt/latest_checkpointed_iteration.txt").exists()
        state = states.get(row["job"])
        if ((state in {"FAILED", "COMPLETED"} and has_checkpoint)
                # Protect every active trainer. On NFS, replacing a launcher
                # can invalidate an older shell's open file handle, including
                # retry jobs submitted after the original sweep.
                or state == "RUNNING"):
            finalize.append(row)
    mopd = [row for row in rows if row["kind"] == "mopd" and states.get(row["job"]) == "PENDING"
            and any(by_label[item["label"]][0] < int(row["n"]) for item in finalize if item["kind"] == "teacher")]
    print(json.dumps({"cpu_finalizers": [(row["label"], row["job"]) for row in finalize],
                      "replace_mopd": [row["label"] for row in mopd]}, indent=2))
    if not args.submit or not finalize:
        return
    finalizer_inputs = [
        "tools/finalize_data_scaling.py", "tools/export_data_scaling_checkpoint.py",
        "tools/convert_torch_dist_to_hf.py", "tools/archive_completed_mixing_checkpoints.py",
        "scripts/slurm/finalize-data-scaling-v1.sbatch",
    ]
    hashes = "".join(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() + "  " + name + "\n" for name in finalizer_inputs)
    (experiment / "finalizer_sha256.txt").write_text(hashes)
    backup = experiment / "jobs.before-finalization-recovery.tsv"
    if not backup.exists():
        backup.write_bytes(manifest.read_bytes())
    for row in rows:
        row.setdefault("training_job", row["job"])
        row.setdefault("training_gpus", row["gpus"])
        row.setdefault("phase", "train")
    scratch = experiment.parent.parent
    exports = ",".join(["ALL", f"RLVE_REPO={ROOT}", f"RLVE_SCRATCH={scratch}", f"RLVE_SCALING_EXPERIMENT={experiment}"])
    recovery = experiment / "finalization_recovery.tsv"
    if not recovery.exists():
        recovery.write_text("utc\tlabel\ttraining_job\tcompletion_job\tphase\n")

    def record(row, previous_job, phase):
        write_rows(manifest, rows)
        with recovery.open("a") as output:
            csv.writer(output, delimiter="\t").writerow([
                datetime.now(timezone.utc).isoformat(), row["label"], previous_job, row["job"], phase,
            ])
        print(f"{row['label']}: completion gate {row['job']} ({phase})", flush=True)

    for row in finalize:
        previous = row["job"]
        dependency = f"afterany:{previous}"
        job = subprocess.check_output([
            "sbatch", "--parsable", f"--dependency={dependency}", f"--job-name=scaling-finalize-{row['label']}",
            f"--export={exports},RLVE_SCALING_FINALIZE_LABEL={row['label']}",
            "scripts/slurm/finalize-data-scaling-v1.sbatch",
        ], text=True).strip().split(";")[0]
        assert job.isdigit(), job
        row.update(job=job, gpus="0", dependency=dependency, phase="finalize")
        record(row, previous, "cpu_export")
    teachers = {row["label"]: row["job"] for row in rows if row["kind"] == "teacher"}
    for row in mopd:
        index, spec = by_label[row["label"]]
        dependency = "afterok:" + ":".join(teachers[run["label"]] for run in plan[:int(row["n"])] )
        previous = row["job"]
        job = subprocess.check_output([
            "sbatch", "--parsable", f"--dependency={dependency}", "--gres=gpu:4", "--cpus-per-task=24",
            f"--mem={spec['memory_gib']}G", "--time=1-00:00:00", f"--constraint={spec['constraint']}",
            "--exclude=viscam9,viscam10,iliad6,pasteur8", f"--job-name=scaling-{spec['label']}",
            f"--export={exports},RLVE_SCALING_RUN_INDEX={index},RLVE_HF_CKPT={scratch}/models/DeepSeek-R1-Distill-Qwen-1.5B,RLVE_REF_LOAD={scratch}/models/DeepSeek-R1-Distill-Qwen-1.5B_torch_dist",
            "scripts/slurm/data-scaling-v1.sbatch",
        ], text=True).strip().split(";")[0]
        assert job.isdigit(), job
        row.update(job=job, dependency=dependency, training_job=job)
        record(row, previous, "mopd_dependency_repair")
        subprocess.run(["scancel", previous], check=True)


if __name__ == "__main__":
    main()
