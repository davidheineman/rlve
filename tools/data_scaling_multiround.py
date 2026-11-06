"""Plan and validate a second N16 MOPD round initialized from round one."""

import argparse
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = .98


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def teacher_result(run):
    marker_path = run / "weights_only_complete.json"
    marker = json.loads(marker_path.read_text())
    if marker.get("format") != "hf-safetensors" or not 0 <= marker["checkpoint_step"] <= 499:
        raise ValueError(f"Incomplete round-1 teacher: {run}")
    target = marker["checkpoint_step"]
    accuracy = None
    # Select the final evaluation matching the exported checkpoint, rather
    # than an earlier attempt's best score or any unseen-test metric.
    with (run / "train.log").open(errors="replace") as log:
        for line in log:
            if "eval " not in line:
                continue
            step = re.search(r"\beval (-?\d+):", line)
            value = re.search(r"'eval/val_envs': ([0-9.e+-]+)", line)
            if step and int(step.group(1)) == target and value:
                accuracy = float(value.group(1))
    if accuracy is None or not 0 <= accuracy <= 1:
        raise ValueError(f"Missing terminal teacher validation: {run}")
    return {"label": run.name, "checkpoint_step": target,
            "validation_pass_at_1": accuracy, "retrain": accuracy < THRESHOLD,
            "marker_sha256": digest(marker_path)}


def build_plan(parent):
    parent = parent.resolve()
    source = json.loads((parent / "run_plan.json").read_text())
    teachers = source["runs"][:16]
    names = [run["environments"][0] for run in teachers]
    if names != (ROOT / "data/DATA-SCALING-V1/n16.txt").read_text().split():
        raise ValueError("Round one does not use the current frozen N16 panel")
    selection = [teacher_result(parent / "teacher" / run["label"]) for run in teachers]
    student = next(run for run in source["runs"] if run["label"] == "n16-mopd")
    initial = parent / "mopd/n16-mopd/hf-final"
    runs = []
    for spec, result in zip(teachers, selection, strict=True):
        spec = copy.deepcopy(spec)
        spec.update(project="rl-data-scaling-multiround-teachers-v1",
                    action="train" if result["retrain"] else "retain",
                    initialization="final-round1-mopd" if result["retrain"] else "retain-round1-teacher")
        runs.append(spec)
    student = copy.deepcopy(student)
    student.update(config="configs/data-scaling-multiround-mopd-v1.yaml", updates=500,
                   training_rollouts=4000, project="rl-data-scaling-students-v1",
                   action="train", initialization="final-round1-mopd")
    runs.append(student)
    return {"method": "multi-round", "round": 2, "size": 16,
            "parent_experiment": str(parent), "initial_hf": str(initial),
            "initial_ref": "warmstart/round1-torch-dist", "teacher_threshold": THRESHOLD,
            "teacher_selection": selection, "runs": runs,
            "teacher_bank": "Retain successful round-1 teachers; replace all remaining teachers with specialists initialized from the final round-1 MOPD actor",
            "student": "Initialize from the final round-1 MOPD actor and train across all 16 environments for up to 500 updates x 8 prompts x 1 sample",
            "storage": "Final weights only; no optimizer/RNG/dataloader/activation checkpoints; delete derived initialization checkpoint after round-2 export"}


def plan_at(experiment):
    return json.loads((experiment / "run_plan.json").read_text())


def validate_source(experiment):
    from archive_completed_mixing_checkpoints import tensor_shapes, validate_export

    plan = plan_at(experiment)
    hf = Path(plan["initial_hf"])
    marker = json.loads((hf.parent / "weights_only_complete.json").read_text())
    if not (hf.parent / "completed").exists() or marker.get("format") != "hf-safetensors":
        raise RuntimeError("Round-1 MOPD does not have a completed final export")
    if not 0 <= marker["checkpoint_step"] < 2000:
        raise RuntimeError("Invalid round-1 MOPD checkpoint step")
    origin = Path(plan["parent_experiment"]).parent.parent / "models/DeepSeek-R1-Distill-Qwen-1.5B"
    validate_export(hf, origin, tensor_shapes(origin)[0])
    return hf


def checkpoint_files(reference):
    if (reference / "latest_checkpointed_iteration.txt").read_text().strip() != "release":
        raise RuntimeError("Expected a fresh model-only release checkpoint, not a training resume")
    model = reference / "release"
    required = [model / "common.pt", model / ".metadata"]
    shards = list(model.glob("*.distcp"))
    if not shards or any(not p.is_file() or not p.stat().st_size for p in required + shards):
        raise RuntimeError("Incomplete round-1 initialization conversion")
    return required


def record_warmstart(experiment):
    plan = plan_at(experiment)
    reference = experiment / plan["initial_ref"]
    files = checkpoint_files(reference)
    hf = Path(plan["initial_hf"])
    identity = {"method": "multi-round", "round": 2, "source_hf": str(hf),
                "source_marker_sha256": digest(hf.parent / "weights_only_complete.json"),
                "source_index_sha256": digest(hf / "model.safetensors.index.json"),
                "round1_wandb_run_id": (hf.parent / "wandb_run_id.txt").read_text().strip(),
                "converted_checkpoint_metadata": {str(p.relative_to(reference)): digest(p) for p in files}}
    (experiment / "initialization.json").write_text(json.dumps(identity, indent=2) + "\n")
    check_warmstart(experiment)


def check_warmstart(experiment):
    plan = plan_at(experiment)
    reference = experiment / plan["initial_ref"]
    checkpoint_files(reference)
    identity = json.loads((experiment / "initialization.json").read_text())
    hf = Path(plan["initial_hf"])
    if (identity["source_hf"] != str(hf)
            or identity["source_marker_sha256"] != digest(hf.parent / "weights_only_complete.json")
            or identity["source_index_sha256"] != digest(hf / "model.safetensors.index.json")):
        raise RuntimeError("Warm start does not match the final round-1 MOPD model")
    for relative, expected in identity["converted_checkpoint_metadata"].items():
        if digest(reference / relative) != expected:
            raise RuntimeError("Converted warm-start checkpoint changed")


def cleanup_warmstart(experiment):
    plan = plan_at(experiment)
    for run in plan["runs"]:
        directory = experiment / run["kind"] / run["label"]
        if not (directory / "weights_only_complete.json").exists():
            raise RuntimeError("Cannot remove initialization while a round-2 learner still needs it")
    reference = experiment / plan["initial_ref"]
    if reference != experiment / "warmstart/round1-torch-dist" or reference.is_symlink():
        raise RuntimeError("Unexpected derived initialization directory")
    if reference.exists():
        check_warmstart(experiment)
        shutil.rmtree(reference)
    (experiment / "warmstart_pruned.json").write_text(json.dumps({
        "path": plan["initial_ref"], "reason": "All round-2 final exports are complete; retained initialization lineage only",
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "validate-source", "record-warmstart", "check-warmstart", "cleanup-warmstart"])
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--parent-experiment", type=Path)
    args = parser.parse_args()
    if args.action == "plan":
        print(json.dumps(build_plan(args.parent_experiment), indent=2))
    else:
        if args.experiment is None:
            parser.error("--experiment is required")
        {"validate-source": validate_source, "record-warmstart": record_warmstart,
         "check-warmstart": check_warmstart, "cleanup-warmstart": cleanup_warmstart}[args.action](args.experiment.resolve())


if __name__ == "__main__":
    main()
