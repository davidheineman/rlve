"""Record reused v3 teacher weights and their original GPU time in W&B."""

import argparse
import hashlib
import json
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--entity", default="david-heineman")
    args = parser.parse_args()
    root = args.experiment_dir
    costs = json.loads((root / "teacher_costs.json").read_text())
    selections = json.loads((Path(__file__).resolve().parent.parent / "data/HERO-SIMILARITY-V2/manifest.json").read_text())["selections"]
    output = root / "teacher_registry_runs.json"
    expected = {}
    for environment, record in sorted(costs["teachers"].items()):
        variants = [name for name, names in selections.items() if environment in names]
        run_id = hashlib.sha256(f"hero-similarity-v2-reused-v3-r1-{environment}".encode()).hexdigest()[:8]
        with wandb.init(
            entity=args.entity,
            project=costs["registry_project"],
            id=run_id,
            resume="allow",
            name=f"REUSED-v3-R1-p1r8-{environment}",
            group="hero-similarity-v2-r1-reused-v3-teachers",
            job_type="reused-teacher-reference",
            tags=["reused-v3", "R1", "p1r8", environment],
            config={
                "environment": environment,
                "model": "DeepSeek-R1-Distill-Qwen-1.5B",
                "training_performed_in_this_project": False,
                "source_project": costs["source_teacher_project"],
                "source_run_url": record["source_wandb_run"],
                "source_checkpoint_step": record["checkpoint_step"],
                "teacher_gpu_hours": record["teacher_gpu_hours"],
                "source_allocations": record["source_allocations"],
                "used_by_variants": variants,
            },
        ) as run:
            run.config.update({"teacher_gpu_hours": record["teacher_gpu_hours"], "source_allocations": record["source_allocations"]}, allow_val_change=True)
            run.summary["david/compute/teacher_gpu_hours"] = record["teacher_gpu_hours"]
            run.summary["david/reuse/source_run_url"] = record["source_wandb_run"]
            run.summary["david/reuse/checkpoint_step"] = record["checkpoint_step"]
            expected[environment] = run.url
            print(f"{environment}: {run.url}", flush=True)
    content = json.dumps(expected, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != content:
        raise RuntimeError(f"Existing registry differs: {output}")
    output.write_text(content)


if __name__ == "__main__":
    main()
