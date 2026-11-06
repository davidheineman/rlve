"""Retry failed scaling runs and rebuild only affected MOPD dependencies."""

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def query_states(rows):
    ids = ",".join(row["job"] for row in rows)
    output = subprocess.check_output([
        "sacct", "-X", "-j", ids, "-n", "-P", "-o", "JobIDRaw,State,ExitCode",
    ], text=True)
    states = {parts[0]: parts[1] for line in output.splitlines() if len(parts := line.split("|")) >= 3}
    queued = subprocess.check_output(["squeue", "-j", ids, "-h", "-o", "%i|%T"], text=True)
    states.update(line.split("|") for line in queued.splitlines())
    return states


def write_rows(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


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
    failed_states = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL"}
    failed = [row for row in rows if row["kind"] in ("teacher", "rl")
              and row.get("phase") != "finalize"
              and states.get(row["job"]) in failed_states
              and not (experiment / row["kind"] / row["label"] / "final-ckpt/latest_checkpointed_iteration.txt").exists()]
    failed_teacher_indices = {by_label[row["label"]][0] for row in failed if row["kind"] == "teacher"}
    affected_mopd = [row for row in rows if row["kind"] == "mopd"
                     and states.get(row["job"]) == "PENDING"
                     and any(index < int(row["n"]) for index in failed_teacher_indices)]
    print(json.dumps({"retry": [(row["label"], row["job"]) for row in failed],
                      "replace_mopd": [(row["label"], row["job"]) for row in affected_mopd]}, indent=2))
    if not args.submit or not failed:
        return
    subprocess.run(["sha256sum", "--check", "--status", str(experiment / "input_sha256.txt")], check=True)
    initial = experiment / "jobs.initial.tsv"
    if not initial.exists():
        initial.write_bytes(manifest.read_bytes())
    scratch = experiment.parent.parent
    model = "DeepSeek-R1-Distill-Qwen-1.5B"
    exports = ",".join([
        "ALL", f"RLVE_REPO={ROOT}", f"RLVE_SCRATCH={scratch}",
        f"RLVE_SCALING_EXPERIMENT={experiment}",
        f"RLVE_WANDB_TEAM={os.environ.get('RLVE_WANDB_TEAM', 'david-heineman')}",
        f"RLVE_HF_CKPT={scratch / 'models' / model}",
        f"RLVE_REF_LOAD={scratch / 'models' / (model + '_torch_dist')}",
    ])
    repairs = experiment / "repairs.tsv"
    if not repairs.exists():
        repairs.write_text("utc\tkind\tlabel\tprevious_job\tnew_job\treason\n")

    def submit(row, dependency="none"):
        index, run = by_label[row["label"]]
        time_limit = {"teacher": "08:00:00", "rl": "12:00:00", "mopd": "1-00:00:00"}[run["kind"]]
        command = [
            "sbatch", "--parsable", f"--gres=gpu:{run['gpus']}",
            f"--cpus-per-task={6 if run['kind'] == 'teacher' else 24}",
            f"--mem={run['memory_gib']}G", f"--time={time_limit}",
            f"--constraint={run['constraint']}", "--exclude=viscam9,viscam10,iliad6,pasteur8",
            f"--job-name=scaling-{run['label']}",
            f"--export={exports},RLVE_SCALING_RUN_INDEX={index}",
        ]
        if dependency != "none":
            command.append(f"--dependency={dependency}")
        command.append("scripts/slurm/data-scaling-v1.sbatch")
        previous_job = row["job"]
        job = subprocess.check_output(command, text=True).strip().split(";")[0]
        assert job.isdigit(), job
        row.update(job=job, dependency=dependency)
        write_rows(manifest, rows)
        with repairs.open("a") as output:
            csv.writer(output, delimiter="\t").writerow([
                datetime.now(timezone.utc).isoformat(), run["kind"], run["label"], previous_job, job,
                "rebuild_after_teacher_retry" if run["kind"] == "mopd" else "startup_failure_retry_with_shared_port_reservation",
            ])
        print(f"{run['label']}: {previous_job} -> {job}", flush=True)
        return previous_job

    for row in sorted(failed, key=lambda row: row["kind"] != "rl"):
        submit(row)
    teachers = {row["label"]: row["job"] for row in rows if row["kind"] == "teacher"}
    for row in affected_mopd:
        matching = [teachers[run["label"]] for run in plan[:int(row["n"])]]
        previous_job = submit(row, "afterok:" + ":".join(matching))
        subprocess.run(["scancel", previous_job], check=True)
    print(f"Repaired job manifest: {manifest}")


if __name__ == "__main__":
    main()
