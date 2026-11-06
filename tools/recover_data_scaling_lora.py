"""Restart interrupted teachers from their frozen source; preserve completed teachers.

Example: python3 tools/recover_data_scaling_lora.py --experiment /runs/old --submit
The replacement retains adapter exports every 50 updates and requests 24 hours.
It starts incomplete teachers fresh, with fresh W&B runs and fresh optimizers.
"""

import argparse
import json
import os
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from lora_teachers.plan import IMAGE_TAG, ROOT, sha256
from lora_teachers.runner import preserved_results, verify_inputs
from submit_data_scaling_lora import queue_experiment


def recovery_plan(experiment):
    plan = deepcopy(json.loads((experiment / "plan.json").read_text()))
    if plan.get("recovery"):
        raise ValueError("Recover the original bank; nested recovery requires an explicit new manifest")
    preserved, unfinished = [], []
    for teacher in plan["teachers"]:
        directory = experiment / "teacher" / teacher["source_label"]
        marker = directory / "completed.json"
        if not marker.exists():
            unfinished.append(teacher)
            continue
        result = json.loads(marker.read_text())
        if result["environment"] != teacher["environment"]:
            raise ValueError(f"Completion identity disagrees with plan: {directory}")
        if Path(result["adapter_path"]).resolve() != (directory / "adapter-final").resolve():
            raise ValueError(f"Completion points outside its final adapter directory: {directory}")
        preserved.append(
            {
                "source_label": teacher["source_label"],
                "environment": teacher["environment"],
                "completion_path": str(marker),
                "completion_sha256": sha256(marker),
                "adapter_sha256": {
                    name: sha256(directory / "adapter-final" / name)
                    for name in ("adapter_config.json", "adapter_model.safetensors")
                },
            }
        )
    if not unfinished:
        raise ValueError("All teachers already completed")
    plan.update(teachers=unfinished, time_limit="1-00:00:00", host_memory_gib=96, progress_adapter_interval=50)
    plan["recovery"] = {
        "original_experiment": str(experiment),
        "original_job_id": json.loads((experiment / "job.json").read_text())["job_id"],
        "reason": "Eight-hour allocation timed out; incomplete node-local adapter exports were removed",
        "restart_mode": "Incomplete teachers restart from the base with fresh optimizers and W&B identities",
        "preserved_teachers": preserved,
        "bank_size": len(preserved) + len(unfinished),
        "source_overrides": ["tools/lora_teachers/client.py", "tools/lora_teachers/runner.py"],
    }
    preserved_results(plan)
    return plan


def freeze_recovery(original, destination, plan):
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(original / "source", destination / "source", symlinks=True)
    hashes = json.loads((original / "input_sha256.json").read_text())
    changes = {}
    for relative in plan["recovery"]["source_overrides"]:
        target = destination / "source/rlve" / relative
        previous = sha256(target)
        shutil.copy2(ROOT / relative, target)
        hashes[str(target.relative_to(destination))] = sha256(target)
        changes[relative] = {"original": previous, "recovery": sha256(target)}
    (destination / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    hashes["plan.json"] = sha256(destination / "plan.json")
    (destination / "input_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    provenance = json.loads((original / "provenance.json").read_text())
    provenance.update(parent_experiment=str(original), source_changes=changes)
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    verify_inputs(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    original = args.experiment.resolve()
    verify_inputs(original)
    plan = recovery_plan(original)
    job = str(plan["recovery"]["original_job_id"])
    queued = subprocess.check_output(["squeue", "-h", "-j", job, "-o", "%T"], text=True).strip()
    if queued:
        raise RuntimeError(f"Original job {job} is still queued or running: {queued}")
    pointer = original / "recovery_submission.json"
    if pointer.exists():
        raise RuntimeError(f"A recovery is already recorded in {pointer}")
    print(
        json.dumps(
            {
                "restart": [t["source_label"] for t in plan["teachers"]],
                "preserved": [t["source_label"] for t in plan["recovery"]["preserved_teachers"]],
                "time_limit": plan["time_limit"],
                "learning_rate": plan["teacher_config"]["lr"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.submit:
        return
    scratch = original.parent.parent
    image = scratch / f"images/miles-{IMAGE_TAG}.sif"
    model = scratch / "models/DeepSeek-R1-Distill-Qwen-1.5B"
    if not image.is_file() or not (model / "config.json").is_file() or not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Recovery needs the pinned image, original base model and W&B credentials")
    destination = original.parent / (
        plan["project"].removeprefix("rl-") + "-recovery-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    freeze_recovery(original, destination, plan)
    queue_experiment(SimpleNamespace(scratch=scratch, image=image, model_path=model), plan, destination)
    pointer.write_text((destination / "job.json").read_text())


if __name__ == "__main__":
    main()
