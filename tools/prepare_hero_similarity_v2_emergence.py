"""Freeze two emergence-based hero similarity v2 comparisons and teacher views."""

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data/MOPD-SWEEPS-V2/id_n16_d0_x4.json"
OOD = ROOT / "data/MOPD-SWEEPS/ood_d0_16env_x24.json"
OUTPUT = ROOT / "data/HERO-SIMILARITY-V2-EMERGENCE"
SOURCE_COSTS = "hero-similarity-v2-20261004"
SELECTIONS = {
    "n4-most-emergent": ["Differentiate", "GoldWashing", "MafMafia", "KloBlocks"],
    "n4-least-emergent": ["Integral", "DeltaMinPopcount", "FractionalProgramming", "CantorExpansion"],
}


def environment(row):
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return metadata["environment"]


def write_or_verify(path, content):
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"Frozen input differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def materialize_teachers(experiment_dir, runs_dir):
    source_costs = runs_dir / SOURCE_COSTS / "teacher_costs.json"
    costs = json.loads(source_costs.read_text())
    teachers = costs["teachers"]
    if not {name for names in SELECTIONS.values() for name in names} <= teachers.keys():
        raise RuntimeError("A selected environment has no registered v3 teacher")
    views = {}
    for variant, names in SELECTIONS.items():
        view = experiment_dir / "teacher-views" / variant / "resumable"
        view.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(names):
            target = Path(teachers[name]["source_run"]).resolve(strict=True)
            link = view / f"{index:02d}-{name}"
            if link.is_symlink():
                if link.resolve() != target:
                    raise RuntimeError(f"Wrong teacher link: {link}")
            elif link.exists():
                raise RuntimeError(f"Teacher link path occupied: {link}")
            else:
                link.symlink_to(target, target_is_directory=True)
        views[variant] = str(view)
    costs["per_variant_teacher_gpu_hours"] = {
        variant: sum(teachers[name]["teacher_gpu_hours"] for name in names)
        for variant, names in SELECTIONS.items()
    }
    costs["views"] = views
    costs["derived_from"] = str(source_costs)
    write_or_verify(experiment_dir / "teacher_costs.json", json.dumps(costs, indent=2, sort_keys=True) + "\n")
    for variant, hours in costs["per_variant_teacher_gpu_hours"].items():
        print(f"{variant}: {hours:.3f} original teacher GPU-hours")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    args = parser.parse_args()
    if bool(args.experiment_dir) != bool(args.runs_dir):
        parser.error("--experiment-dir and --runs-dir must be supplied together")
    source = json.loads(SOURCE.read_text())
    source_counts = Counter(environment(row) for row in source)
    ood = json.loads(OOD.read_text())
    ood_counts = Counter(environment(row) for row in ood)
    selected = {name for names in SELECTIONS.values() for name in names}
    if len(source) != 64 or any(source_counts[name] != 4 for name in selected):
        raise RuntimeError("Expected four existing ID prompts per selected environment")
    if len(ood) != 384 or len(ood_counts) != 16 or set(ood_counts.values()) != {24}:
        raise RuntimeError("Expected 16 OOD environments with 24 prompts each")
    if selected & ood_counts.keys():
        raise RuntimeError("Training and OOD environments overlap")

    for variant, names in SELECTIONS.items():
        if len(names) != 4 or len(set(names)) != 4:
            raise RuntimeError(f"Expected four distinct environments for {variant}")
        records = [row for name in names for row in source if environment(row) == name]
        if len(records) != 16 or len({row["user_prompt"] for row in records}) != 16:
            raise RuntimeError(f"Invalid ID evaluation prompts for {variant}")
        write_or_verify(OUTPUT / f"{variant}.txt", " ".join(names) + "\n")
        write_or_verify(OUTPUT / f"id_{variant}_d0_x4.json", json.dumps(records, indent=1) + "\n")
    write_or_verify(OUTPUT / "manifest.json", json.dumps({
        "source_id": str(SOURCE.relative_to(ROOT)),
        "source_ood": str(OOD.relative_to(ROOT)),
        "source_teacher_costs_experiment": SOURCE_COSTS,
        "selections": SELECTIONS,
        "id_prompts_per_environment": 4,
        "ood_prompts_per_environment": 24,
    }, indent=2, sort_keys=True) + "\n")
    if args.experiment_dir:
        materialize_teachers(args.experiment_dir, args.runs_dir)


if __name__ == "__main__":
    main()
