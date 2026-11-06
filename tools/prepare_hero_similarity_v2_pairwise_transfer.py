"""Freeze four two-environment comparisons selected from the R1 transfer plot."""

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data/MOPD-SWEEPS-V2/id_n16_d0_x4.json"
OOD = ROOT / "data/MOPD-SWEEPS/ood_d0_16env_x24.json"
OUTPUT = ROOT / "data/HERO-SIMILARITY-V2-PAIRWISE-TRANSFER"
SOURCE_COSTS = "hero-similarity-v2-20261004"
SNAPSHOT = "2026-10-06T05:51:54.711166+00:00"

# Values are smoothed, row-normalized pass@1 changes in percentage points.
# The directions and checkpoint steps are frozen from the 16-by-32 snapshot.
SELECTIONS = {
    "pairwise_transfer_positive": {
        "environments": ["Circuit", "MafMafia"],
        "a_to_b_pp": 37.84,
        "b_to_a_pp": 37.02,
        "a_step": 269,
        "b_step": 429,
    },
    "pairwise_transfer_negative": {
        "environments": ["DiscreteLogarithm", "GaussianElimination"],
        "a_to_b_pp": -47.43,
        "b_to_a_pp": -9.02,
        "a_step": 59,
        "b_step": 59,
    },
    "pairwise_transfer_oneway": {
        "environments": ["GaussianElimination", "MafMafia"],
        "a_to_b_pp": -0.47,
        "b_to_a_pp": 50.17,
        "a_step": 59,
        "b_step": 429,
    },
    "pairwise_transfer_none": {
        "environments": ["CongruentEquation", "Sorting"],
        "a_to_b_pp": 1.50,
        "b_to_a_pp": -1.51,
        "a_step": 9,
        "b_step": 29,
    },
}


def environment(row):
    metadata = row["metadata"]
    return (json.loads(metadata) if isinstance(metadata, str) else metadata)["environment"]


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
    selected = {name for choice in SELECTIONS.values() for name in choice["environments"]}
    if not selected <= teachers.keys():
        raise RuntimeError(f"Missing v3 teachers: {selected - teachers.keys()}")
    views = {}
    for variant, choice in SELECTIONS.items():
        view = experiment_dir / "teacher-views" / variant / "resumable"
        view.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(choice["environments"]):
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
        variant: sum(teachers[name]["teacher_gpu_hours"] for name in choice["environments"])
        for variant, choice in SELECTIONS.items()
    }
    costs["views"] = views
    costs["derived_from"] = str(source_costs)
    write_or_verify(experiment_dir / "teacher_costs.json", json.dumps(costs, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    args = parser.parse_args()
    if bool(args.experiment_dir) != bool(args.runs_dir):
        parser.error("--experiment-dir and --runs-dir must be supplied together")

    source = json.loads(SOURCE.read_text())
    groups = {name: [row for row in source if environment(row) == name]
              for name in {n for choice in SELECTIONS.values() for n in choice["environments"]}}
    if any(len(rows) != 4 or len({row["user_prompt"] for row in rows}) != 4
           for rows in groups.values()):
        raise RuntimeError("Expected four distinct frozen ID prompts per selected environment")
    ood_counts = Counter(environment(row) for row in json.loads(OOD.read_text()))
    if len(ood_counts) != 16 or set(ood_counts.values()) != {24} or groups.keys() & ood_counts.keys():
        raise RuntimeError("Expected 16 disjoint OOD environments with 24 prompts each")

    for variant, choice in SELECTIONS.items():
        names = choice["environments"]
        if len(names) != 2 or len(set(names)) != 2:
            raise RuntimeError(f"Expected two distinct environments for {variant}")
        records = [row for name in names for row in groups[name]]
        write_or_verify(OUTPUT / f"{variant}.txt", " ".join(names) + "\n")
        write_or_verify(OUTPUT / f"id_{variant}_d0_x4.json", json.dumps(records, indent=1) + "\n")
        print(f"{variant}: {', '.join(names)}; transfer {choice['a_to_b_pp']:+.2f}/{choice['b_to_a_pp']:+.2f} pp")
    write_or_verify(OUTPUT / "manifest.json", json.dumps({
        "source_id": str(SOURCE.relative_to(ROOT)),
        "source_ood": str(OOD.relative_to(ROOT)),
        "source_transfer_plot": "plot/initial-pairwise-ood-r1/pairwise/delta_curves_normalized_smoothed_pairs.png",
        "source_transfer_snapshot_fetched_at": SNAPSHOT,
        "source_teacher_costs_experiment": SOURCE_COSTS,
        "selections": SELECTIONS,
        "id_prompts_per_environment": 4,
        "ood_prompts_per_environment": 24,
    }, indent=2, sort_keys=True) + "\n")
    if args.experiment_dir:
        materialize_teachers(args.experiment_dir, args.runs_dir)


if __name__ == "__main__":
    main()
