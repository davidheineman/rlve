"""Freeze mixed-difficulty evaluation sets for the horizon-difficulty comparison."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_rlve_eval_set import _records_for_environment


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data/HERO-SIMILARITY-V2/manifest.json"
SOURCE_OOD = ROOT / "data/MOPD-SWEEPS/ood_d0_16env_x24.json"
SPEEDUPS = ROOT / "plot/mopd-speedup/speedups.json"
OUTPUT = ROOT / "data/HORIZON-DIFFICULTY"
SELECTIONS = {"high": "n4-similar1", "low": "n4-dissimilar1"}
DIFFICULTIES = (0, 1, 2, 3)


def metadata(record):
    value = record["metadata"]
    return json.loads(value) if isinstance(value, str) else value


def freeze(path, content):
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"Frozen file differs from regenerated data: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def generate(environment, count, seed_base):
    assert count % len(DIFFICULTIES) == 0
    records = []
    for difficulty in DIFFICULTIES:
        records.extend(_records_for_environment(
            environment, difficulty, count // len(DIFFICULTIES),
            seed_base + difficulty * 1000, 20261008 + difficulty,
        ))
    distribution = Counter(metadata(record)["config"]["seed"] // 1000 for record in records)
    if len(records) != count or len(distribution) != len(DIFFICULTIES):
        raise RuntimeError(f"Incomplete difficulty coverage for {environment}")
    return records


def main():
    selections = json.loads(SOURCE.read_text())["selections"]
    gains = json.loads(SPEEDUPS.read_text())
    groups = {label: selections[variant] for label, variant in SELECTIONS.items()}
    if any(len(names) != 4 or len(set(names)) != 4 for names in groups.values()):
        raise RuntimeError("Expected two distinct four-environment selections")
    if set(groups["high"]) & set(groups["low"]):
        raise RuntimeError("High and low groups must be disjoint")
    if gains[SELECTIONS["high"]]["val"]["speedup"] <= 1 or gains[SELECTIONS["low"]]["val"]["speedup"] >= 1:
        raise RuntimeError("Source plot no longer supports high/low selection")

    ood_names = sorted({metadata(row)["environment"] for row in json.loads(SOURCE_OOD.read_text())})
    if len(ood_names) != 16 or set(ood_names) & set(sum(groups.values(), [])):
        raise RuntimeError("Expected 16 disjoint OOD environments")

    paths = []
    all_training = sorted(set(sum(groups.values(), [])))
    for index, environment in enumerate(all_training):
        path = OUTPUT / f"{environment}_d0123.json"
        freeze(path, json.dumps(generate(environment, 32, 30_000_000 + index * 100_000), indent=1) + "\n")
        paths.append(path)

    for label, names in groups.items():
        env_path = OUTPUT / f"{label}.txt"
        freeze(env_path, " ".join(names) + "\n")
        paths.append(env_path)
        rows = []
        for environment in names:
            index = all_training.index(environment)
            rows.extend(generate(environment, 16, 40_000_000 + index * 100_000))
        id_path = OUTPUT / f"id_{label}_d0123_x16.json"
        freeze(id_path, json.dumps(rows, indent=1) + "\n")
        paths.append(id_path)

    ood_rows = []
    for index, environment in enumerate(ood_names):
        ood_rows.extend(generate(environment, 24, 50_000_000 + index * 100_000))
    ood_path = OUTPUT / "ood_d0123_16env_x24.json"
    freeze(ood_path, json.dumps(ood_rows, indent=1) + "\n")
    paths.append(ood_path)

    manifest = {
        "hypothesis": "Lower teacher learning slopes predict higher MOPD compute gain.",
        "source_plot": str(SPEEDUPS.relative_to(ROOT)),
        "selection_metric": "validation rollout speedup at the second local MOPD peak; actual GPU-hour gain will be measured after these runs",
        "source_ood": str(SOURCE_OOD.relative_to(ROOT)),
        "groups": {label: {
            "source_variant": SELECTIONS[label],
            "environments": names,
            "source_validation_rollout_speedup": gains[SELECTIONS[label]]["val"]["speedup"],
        } for label, names in groups.items()},
        "difficulty_levels": DIFFICULTIES,
        "teacher_validation_prompts_per_environment_per_level": 8,
        "student_validation_prompts_per_environment_per_level": 4,
        "test_prompts_per_environment_per_level": 6,
        "teacher_updates": 2000,
        "teacher_train_rollouts_each": 16000,
        "mopd_updates": 1000,
        "mopd_train_rollouts": 8000,
        "hero_rl_updates": 2000,
        "hero_rl_train_rollouts": 256000,
        "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    freeze(OUTPUT / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Froze {len(all_training)} teacher validation sets, two student validation sets, and {len(ood_rows)} OOD test prompts")


if __name__ == "__main__":
    main()
