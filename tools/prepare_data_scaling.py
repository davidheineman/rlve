"""Freeze nested RLVE scaling panels and ID validation at difficulties 0 and 1."""

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

from make_rlve_eval_set import _records_for_environment

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/DATA-SCALING-V1"
SOURCE = ROOT / "scripts/slurm/sweeps/mopd-v2-32.txt"
SEED = 20261023
SIZES = (16, 32, 64)


def freeze(path, value):
    content = value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"Frozen input changed: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def generate(environment, index):
    records = []
    for difficulty in (0, 1):
        level = _records_for_environment(
            environment, difficulty, 16,
            60_000_000 + index * 100_000 + difficulty * 1000,
            SEED + difficulty,
        )
        for row in level:
            metadata = json.loads(row["metadata"])
            metadata["difficulty"] = difficulty
            row["metadata"] = json.dumps(metadata)
        records.extend(level)
    return records


def check(directory=OUTPUT):
    from collections import Counter

    manifest = json.loads((directory / "manifest.json").read_text())
    names = manifest["environments"]
    assert len(names) == len(set(names)) == 64
    test_names = manifest["test_environments"]
    assert len(test_names) == len(set(test_names)) == 32
    assert not set(names).intersection(test_names)
    assert not set(names + test_names).intersection(SOURCE.read_text().split())
    for relative, expected in manifest["sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    for size in SIZES:
        assert (directory / f"n{size}.txt").read_text().split() == names[:size]
        rows = json.loads((directory / f"id_n{size}_d01.json").read_text())
        assert len(rows) == 32
        metadata = [json.loads(row["metadata"]) for row in rows]
        assert set(row["environment"] for row in metadata) <= set(names[:size])
        assert Counter(row["difficulty"] for row in metadata) == {0: 16, 1: 16}
    for name in names:
        metadata = [json.loads(row["metadata"]) for row in json.loads((directory / f"{name}_d01.json").read_text())]
        assert Counter((row["environment"], row["difficulty"]) for row in metadata) == {(name, 0): 16, (name, 1): 16}
    test_rows = json.loads((directory / "test_32env_d01.json").read_text())
    assert len(test_rows) == 32
    assert Counter(json.loads(row["metadata"])["environment"] for row in test_rows) == {name: 1 for name in test_names}
    assert Counter(json.loads(row["metadata"])["difficulty"] for row in test_rows) == {0: 16, 1: 16}
    print("Verified fresh nested 16/32/64 train envs; 32 total ID prompts/hero; 32 disjoint test envs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--worker", nargs=2, metavar=("ENVIRONMENT", "INDEX"))
    parser.add_argument("--candidate-timeout", type=int, default=90)
    args = parser.parse_args()
    if args.worker:
        environment, index = args.worker
        print(json.dumps(generate(environment, int(index))))
        return
    if args.check_only or (OUTPUT / "manifest.json").exists():
        check()
        return

    from Gym.environments import identifier2environment
    from Gym.parameter_controllers import identifier2controller

    original = SOURCE.read_text().split()
    assert len(original) == len(set(original)) == 32
    candidates = sorted(set(identifier2environment).intersection(identifier2controller) - set(original))
    random.Random(SEED).shuffle(candidates)
    selected, excluded, rows_by_name = [], [], {}
    for environment in candidates:
        index = len(selected)
        try:
            result = subprocess.run(
                [sys.executable, __file__, "--worker", environment, str(index)],
                capture_output=True, text=True, check=True, timeout=args.candidate_timeout,
            )
            # Some generators emit diagnostics before the JSON result.
            rows = json.loads(result.stdout.splitlines()[-1])
            assert len(rows) == 32
        except (subprocess.SubprocessError, ValueError, AssertionError) as error:
            excluded.append({"environment": environment, "reason": str(error)[-1000:]})
            print(f"Skipped {environment}: cannot freeze scoreable d0/d1 questions", flush=True)
            continue
        selected.append(environment)
        rows_by_name[environment] = rows
        print(f"Prepared {len(selected)}/96 {environment}", flush=True)
        if len(selected) == 96:
            break
    if len(selected) != 96:
        raise RuntimeError("Insufficient scoreable RLVE environments")

    training, testing = selected[:64], selected[64:]
    for environment in training:
        freeze(OUTPUT / f"{environment}_d01.json", rows_by_name[environment])
    for size in SIZES:
        freeze(OUTPUT / f"n{size}.txt", " ".join(training[:size]) + "\n")
        # Draw a fixed 32-question ID panel across the uniform environment/difficulty distribution.
        rng = random.Random(SEED + size)
        env_draws = (training[:size] * 2 if size == 16 else rng.sample(training[:size], 32))
        rng.shuffle(env_draws)
        levels = [0] * 16 + [1] * 16
        rng.shuffle(levels)
        rows = []
        for draw_index, (environment, difficulty) in enumerate(zip(env_draws, levels, strict=True)):
            generated = _records_for_environment(environment, difficulty, 1,
                80_000_000 + size * 100_000 + draw_index * 1000, SEED + size + draw_index)
            meta = json.loads(generated[0]["metadata"])
            meta["difficulty"] = difficulty
            generated[0]["metadata"] = json.dumps(meta)
            rows.extend(generated)
        freeze(OUTPUT / f"id_n{size}_d01.json", rows)
    test_rows = []
    for index, environment in enumerate(testing):
        difficulty = index % 2
        generated = _records_for_environment(environment, difficulty, 1,
            90_000_000 + index * 1000, SEED + index)
        meta = json.loads(generated[0]["metadata"])
        meta["difficulty"] = difficulty
        generated[0]["metadata"] = json.dumps(meta)
        test_rows.extend(generated)
    freeze(OUTPUT / "test_32env_d01.json", test_rows)
    paths = [SOURCE, *OUTPUT.glob("*.txt"), *OUTPUT.glob("*.json")]
    manifest = {
        "seed": SEED,
        "source_project": "david-heineman/rl-data-opd-sweeps-teachers-v3",
        "excluded_original_pool": original,
        "selection": "Seeded shuffle of registered RLVE environments excluding the original 32-env pool; first 64 scoreable envs train fresh models, next 32 are unseen test envs",
        "environments": training,
        "test_environments": testing,
        "excluded_candidates": excluded,
        "difficulty_levels": [0, 1],
        "teacher_eval_prompts": 32,
        "hero_id_eval_prompts": 32,
        "hero_test_eval_prompts": 32,
        "eval_samples_per_prompt": 8,
        "eval_interval": 50,
        "eval_prompt_repetition": "Finite question spaces may repeat; all 32 generator draws and seeds are retained",
        "unique_teacher_eval_prompts": {name: len({row["user_prompt"] for row in rows_by_name[name]}) for name in training},
        "teacher_project": "rl-data-scaling-teachers-v1",
        "student_project": "rl-data-scaling-students-v1",
        "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)},
    }
    freeze(OUTPUT / "manifest.json", manifest)
    check()


if __name__ == "__main__":
    main()
