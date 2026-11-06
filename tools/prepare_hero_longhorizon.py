"""Freeze the two matched long-horizon training/validation conditions."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_rlve_eval_set import _records_for_environment


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data/HERO-LONGHORIZON"
OOD_SOURCE = ROOT / "data/HORIZON-DIFFICULTY/ood_d0123_16env_x24.json"
ENVIRONMENTS = ("CantorExpansion", "DeltaMinPopcount", "GoldWashing", "HalvingChainCounting")
DIFFICULTIES = (0, 1, 2, 3)
UPPER_BOUNDS = (1, 3)
PROMPTS_PER_LEVEL = 8
SHARED_PROMPTS_PER_LEVEL = 3


def freeze(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Frozen input differs from regenerated data: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def metadata(record):
    value = record["metadata"]
    return json.loads(value) if isinstance(value, str) else value


def generate(environments, difficulties, per_level, seed_base):
    rows = []
    for environment_index, environment in enumerate(environments):
        seen_prompts = set()
        for difficulty in difficulties:
            rows.extend(_records_for_environment(
                environment, difficulty, per_level,
                seed_base + environment_index * 100_000 + difficulty * 1000,
                20261014 + difficulty,
                seen_prompts=seen_prompts,
            ))
    counts = Counter((metadata(row)["environment"], metadata(row)["config"]["seed"] // 1000 % 100)
                     for row in rows)
    expected = Counter({(environment, difficulty): per_level
                        for environment in environments for difficulty in difficulties})
    if counts != expected:
        raise RuntimeError(f"Unbalanced frozen set: {counts} != {expected}")
    return rows


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def main():
    paths = []
    for upper in UPPER_BOUNDS:
        tag = "d" + "".join(str(difficulty) for difficulty in range(upper + 1))
        directory = OUTPUT / tag
        environment_path = directory / "environments.txt"
        freeze(environment_path, (" ".join(ENVIRONMENTS) + "\n").encode())
        paths.append(environment_path)
        levels = tuple(range(upper + 1))
        for environment_index, environment in enumerate(ENVIRONMENTS):
            path = directory / f"{environment}_{tag}.json"
            rows = generate([environment], levels, PROMPTS_PER_LEVEL,
                            60_000_000 + environment_index * 100_000)
            freeze(path, json_bytes(rows))
            paths.append(path)
        student_path = directory / "student_val.json"
        freeze(student_path, json_bytes(generate(ENVIRONMENTS, levels, PROMPTS_PER_LEVEL, 70_000_000)))
        paths.append(student_path)

    shared = OUTPUT / "shared_id_d0123.json"
    freeze(shared, json_bytes(generate(ENVIRONMENTS, DIFFICULTIES, SHARED_PROMPTS_PER_LEVEL, 80_000_000)))
    paths.append(shared)

    ood = OUTPUT / "ood_d0123_16env_x24.json"
    source_ood = OOD_SOURCE if OOD_SOURCE.exists() else ood
    if not source_ood.exists():
        raise RuntimeError(f"Missing frozen OOD questions: {OOD_SOURCE} or {ood}")
    source_rows = json.loads(source_ood.read_text())
    ood_names = {metadata(row)["environment"] for row in source_rows}
    if len(ood_names) != 16 or ood_names.intersection(ENVIRONMENTS):
        raise RuntimeError("Expected 16 OOD environments disjoint from the four training environments")
    freeze(ood, source_ood.read_bytes())
    paths.append(ood)

    manifest = {
        "environments": ENVIRONMENTS,
        "difficulty_ranges": {"d" + "".join(str(level) for level in range(upper + 1)): list(range(upper + 1))
                              for upper in UPPER_BOUNDS},
        "teacher_validation_prompts_per_environment_per_difficulty": PROMPTS_PER_LEVEL,
        "student_validation_prompts_per_environment_per_difficulty": PROMPTS_PER_LEVEL,
        "shared_id_prompts_per_environment_per_difficulty": SHARED_PROMPTS_PER_LEVEL,
        "ood_prompts_per_environment_per_difficulty": 6,
        "teacher_max_updates": 2500,
        "teacher_max_rollouts": 20000,
        "mopd_max_updates": 2000,
        "mopd_max_rollouts": 16000,
        "hero_rl_max_updates": 125,
        "hero_rl_max_rollouts": 16000,
        "student_eval_samples_per_prompt": 4,
        "mopd_eval_interval_updates": 50,
        "hero_rl_eval_interval_updates": 20,
        "validation_accuracy_stop": 0.95,
        "ood_source": str(OOD_SOURCE.relative_to(ROOT)),
        "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    freeze(OUTPUT / "manifest.json", json_bytes(manifest))
    print(f"Froze {len(UPPER_BOUNDS) * len(ENVIRONMENTS)} teacher validation sets, "
          f"{len(UPPER_BOUNDS)} student validation sets, "
          f"{len(json.loads(shared.read_text()))} shared ID prompts, and {len(source_rows)} OOD prompts")


if __name__ == "__main__":
    main()
