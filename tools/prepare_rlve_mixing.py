"""Freeze a shared difficulty-zero training pool, unseen OOD set, and 48 mixtures."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from make_rlve_eval_set import _records_for_environment

from Gym.environments import identifier2environment
from Gym.parameter_controllers import identifier2controller


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "MIXING"
OLD_ID = ROOT / "data" / "EVAL-pairwise32" / "targets_d0_x12.json"
TRAINING = ROOT / "scripts" / "slurm" / "sweeps" / "pairwise32.txt"
HELDOUT = ROOT / "data" / "HELD-OUT_ENVIRONMENTS" / "test.json"
SEED = 20260927


def write_json(path, value):
    path.write_text(json.dumps(value, indent=1) + "\n")


def metadata(record):
    value = record["metadata"]
    return json.loads(value) if isinstance(value, str) else value


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pool_path = OUTPUT / "train_pool_d0_x64.json"
    ood_path = OUTPUT / "ood_d0_16env_x12.json"
    manifest_path = OUTPUT / "mixes.json"
    if any(path.exists() for path in (pool_path, ood_path, manifest_path)):
        if not all(path.exists() for path in (pool_path, ood_path, manifest_path)):
            raise RuntimeError("mixing data is partially prepared; inspect it before retrying")
        manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(pool_path.read_bytes()).hexdigest() != manifest["pool_sha256"]:
            raise RuntimeError("prepared pool no longer matches the mixture manifest")
        print(f"already prepared: {manifest_path}")
        return

    environments = TRAINING.read_text().split()
    if len(environments) != 32 or len(set(environments)) != 32:
        raise ValueError("expected 32 distinct training environments")
    previous_id = json.loads(OLD_ID.read_text())
    if len(previous_id) != 32 * 12 or {metadata(row)["environment"] for row in previous_id} != set(environments):
        raise ValueError("previous pairwise OOD evaluation is not 12 prompts for each training environment")
    seen_by_environment = {environment: set() for environment in environments}
    for row in previous_id:
        seen_by_environment[metadata(row)["environment"]].add(row["user_prompt"])

    pool = []
    overlap_counts = {}
    for index, environment in enumerate(environments):
        seed_start = 10_000_000 + index * 100_000
        try:
            records = _records_for_environment(
                environment, 0, 64, seed_start, SEED,
                seen_prompts=set(seen_by_environment[environment]),
                allow_fewer=True,
            )
        except AssertionError as error:
            if str(error) != f"no unique prompts generated for {environment}":
                raise
            records = _records_for_environment(
                environment, 0, 64, seed_start, SEED, seen_prompts=None,
            )
            print(f"{environment}: unavoidable exact-prompt overlap with ID eval", flush=True)
        pool.extend(records)
        overlap_counts[environment] = sum(row["user_prompt"] in seen_by_environment[environment] for row in records)
        print(f"pool {index + 1}/32 {environment}: {len(records)} records, {overlap_counts[environment]} overlap ID", flush=True)

    registered = set(identifier2environment) & set(identifier2controller)
    heldout = {metadata(row)["environment"] for row in json.loads(HELDOUT.read_text())}
    eligible = sorted((heldout & registered) - set(environments))
    rng = random.Random(SEED)
    rng.shuffle(eligible)
    ood, ood_environments, rejected = [], [], {}
    for index, environment in enumerate(eligible):
        if len(ood_environments) == 16:
            break
        try:
            records = _records_for_environment(
                environment, 0, 12, 20_000_000 + index * 100_000, SEED,
                seen_prompts=set(),
            )
        except (Exception, SystemExit) as error:
            rejected[environment] = str(error)[:300]
            print(f"skip OOD {environment}: {error}", flush=True)
            continue
        ood_environments.append(environment)
        ood.extend(records)
        print(f"OOD {len(ood_environments)}/16 {environment}", flush=True)
    if len(ood_environments) != 16:
        raise RuntimeError("could not make 16 scoreable unseen OOD environments")

    mix_rng = np.random.default_rng(SEED)
    candidates = []
    for alpha in (0.1, 0.5, 2.0):
        for _ in range(16):
            weights = mix_rng.dirichlet(np.full(len(environments), alpha))
            candidates.append({
                "index": len(candidates),
                "alpha": alpha,
                "weights": dict(zip(environments, map(float, weights), strict=True)),
            })
    write_json(pool_path, pool)
    write_json(ood_path, ood)
    write_json(manifest_path, {
        "seed": SEED,
        "environments": environments,
        "pool_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
        "pool_prompts_max_per_environment": 64,
        "pool_counts": {environment: sum(metadata(row)["environment"] == environment for row in pool) for environment in environments},
        "pool_id_exact_prompt_overlap_counts": overlap_counts,
        "id_eval_path": str(OLD_ID.relative_to(ROOT)),
        "ood_environments": ood_environments,
        "ood_rejected": rejected,
        "candidates": candidates,
    })
    print(f"wrote {len(pool)} pool prompts, {len(ood)} unseen OOD prompts, and {len(candidates)} mixtures")


if __name__ == "__main__":
    main()
