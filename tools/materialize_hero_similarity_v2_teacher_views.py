"""Create zero-copy per-variant views of the completed v3 R1 teachers."""

import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/HERO-SIMILARITY-V2/manifest.json"
ORDER = ROOT / "data/MOPD-SWEEPS-V2/n16.txt"


def teacher_costs(array_job):
    result = subprocess.run(
        ["sacct", "-n", "-P", "--duplicates", "-j", str(array_job), "--format=JobID,ElapsedRaw,State,AllocTRES", "--allocations"],
        check=True, capture_output=True, text=True,
    )
    allocations = {index: [] for index in range(16)}
    completed = set()
    for line in result.stdout.splitlines():
        job, elapsed, state, tres = line.split("|", 3)
        match = re.fullmatch(rf"{array_job}_(\d+)", job)
        if match:
            index = int(match.group(1))
            if index not in allocations or "gres/gpu=1" not in tres:
                raise RuntimeError(f"Teacher task {job} is not a one-GPU allocation: {state} {tres}")
            if state == "COMPLETED":
                completed.add(index)
            allocations[index].append({"job": job, "state": state, "gpu_hours": int(elapsed) / 3600})
    if completed != set(range(16)):
        raise RuntimeError(f"Expected 16 completed teacher tasks, found {sorted(completed)}")
    return allocations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-teacher-dir", type=Path, required=True)
    parser.add_argument("--source-teacher-array", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--refresh-costs", action="store_true", help="Replace an existing generated cost manifest after auditing Slurm allocations")
    args = parser.parse_args()

    source = args.source_teacher_dir.resolve(strict=True)
    names = ORDER.read_text().split()
    selections = json.loads(DATA.read_text())["selections"]
    allocations = teacher_costs(args.source_teacher_array)
    if len(names) != 16 or len(set(names)) != 16:
        raise RuntimeError("Invalid source teacher order")
    records = {}
    for index, environment in enumerate(names):
        run = source / f"{index:02d}-{environment}"
        marker = json.loads((run / "weights_only_complete.json").read_text())
        run_id = (run / "wandb_run_id.txt").read_text().strip()
        if marker.get("format") != "hf-safetensors" or not 0 <= marker["checkpoint_step"] <= 499:
            raise RuntimeError(f"Invalid v3 teacher marker for {environment}")
        if len(run_id) != 8 or not (run / "hf-final/model.safetensors.index.json").is_file():
            raise RuntimeError(f"Incomplete v3 teacher export for {environment}")
        records[environment] = {
            "source_run": str(run),
            "source_wandb_run": f"https://wandb.ai/david-heineman/rl-data-opd-sweeps-teachers-v3/runs/{run_id}",
            "checkpoint_step": marker["checkpoint_step"],
            "teacher_gpu_hours": sum(item["gpu_hours"] for item in allocations[index]),
            "source_allocations": allocations[index],
        }

    for variant, selection in selections.items():
        view = args.output_dir / "teacher-views" / variant / "resumable"
        view.mkdir(parents=True, exist_ok=True)
        for index, environment in enumerate(selection):
            link = view / f"{index:02d}-{environment}"
            target = Path(records[environment]["source_run"])
            if link.is_symlink():
                if link.resolve() != target:
                    raise RuntimeError(f"Wrong existing teacher link: {link}")
            elif link.exists():
                raise RuntimeError(f"Teacher link path already occupied: {link}")
            else:
                link.symlink_to(target, target_is_directory=True)
    cost_by_variant = {
        variant: sum(records[name]["teacher_gpu_hours"] for name in selection)
        for variant, selection in selections.items()
    }
    output = {
        "source_teacher_array": args.source_teacher_array,
        "source_teacher_dir": str(source),
        "source_teacher_project": "rl-data-opd-sweeps-teachers-v3",
        "registry_project": "rl-data-hero-similarity-teachers-v2",
        "teachers": records,
        "per_variant_teacher_gpu_hours": cost_by_variant,
        "shared_all16_teacher_gpu_hours": sum(record["teacher_gpu_hours"] for record in records.values()),
        "views": {variant: str(args.output_dir / "teacher-views" / variant / "resumable") for variant in selections},
    }
    path = args.output_dir / "teacher_costs.json"
    content = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != content and not args.refresh_costs:
        raise RuntimeError(f"Existing teacher cost manifest differs: {path}")
    path.write_text(content)
    print(f"Linked {len(records)} v3 teachers into {len(selections)} zero-copy views")
    for variant, hours in cost_by_variant.items():
        print(f"{variant}: standalone teacher cost {hours:.2f} GPU-hours")


if __name__ == "__main__":
    main()
