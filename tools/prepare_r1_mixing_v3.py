"""Freeze the 24-environment, 73-run R1 mixing sweep and its evaluations."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/MIXING-R1-V3"
OLD = ROOT / "data/MIXING"
SEED = 20261007


def metadata(row):
    value = row["metadata"]
    return json.loads(value) if isinstance(value, str) else value


def dump(path, rows):
    path.write_text(json.dumps(rows, indent=1) + "\n")


def allocate(weights, count):
    """Largest-remainder allocation gives exactly count weighted ID questions."""
    raw = np.asarray(weights, dtype=float) * count
    result = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - result), kind="stable")[: count - result.sum()]:
        result[index] += 1
    assert result.sum() == count
    return result.tolist()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [OUT / "n24.txt", OUT / "train_pool_d0_x64.json",
             OUT / "ood_d0_16env_x12.json", OUT / "ood_d0_16env_x48.json",
             OUT / "mixes.json"]
    paths += [OUT / f"id_mix_{i:02d}_d0_x192.json" for i in range(73)]
    if any(path.exists() for path in paths):
        if not all(path.exists() for path in paths):
            raise RuntimeError("partially prepared data; inspect before retrying")
        verify()
        return

    environments = (ROOT / "data/MOPD-SWEEPS-V2/n32.txt").read_text().split()[:24]
    assert len(environments) == len(set(environments)) == 24
    old_ood = json.loads((OLD / "ood_d0_16env_x12.json").read_text())
    ood_environments = list(dict.fromkeys(metadata(row)["environment"] for row in old_ood))
    assert len(old_ood) == 192 and len(ood_environments) == 16
    assert not set(environments) & set(ood_environments)

    old_pool = json.loads((OLD / "train_pool_d0_x64.json").read_text())
    pool = [row for row in old_pool if metadata(row)["environment"] in environments]
    pool_prompts = defaultdict(set)
    for row in pool:
        pool_prompts[metadata(row)["environment"]].add(row["user_prompt"])
    assert set(pool_prompts) == set(environments)

    # Each candidate receives the same question budget, with environment counts
    # fixed by its training weights. The ordered per-environment pools are frozen.
    id_pools = {}
    overlap = {}
    for environment in environments:
        root = "data/MOPD-SWEEPS-V2" if (ROOT / f"data/MOPD-SWEEPS-V2/{environment}_d0.json").exists() else "data/EVAL-pairwise32"
        source = json.loads((ROOT / root / f"{environment}_d0.json").read_text())
        assert len(source) == 128 and all(metadata(row)["environment"] == environment for row in source)
        held_out = [row for row in source if row["user_prompt"] not in pool_prompts[environment]]
        # Some difficulty-zero environments have finite support and cannot be
        # fully disjoint. Preserve evaluation coverage where that is unavoidable.
        id_pools[environment] = held_out or source
        overlap[environment] = 0 if held_out else len({r["user_prompt"] for r in source} & pool_prompts[environment])

    rng = np.random.default_rng(SEED)
    candidates = []
    for alpha in (0.1, 0.5, 2.0):
        for _ in range(24):
            values = rng.dirichlet(np.full(24, alpha))
            candidates.append({"index": len(candidates), "alpha": alpha,
                               "weights": dict(zip(environments, map(float, values), strict=True))})
    candidates.append({"index": 72, "alpha": "uniform",
                       "weights": {environment: 1 / 24 for environment in environments}})

    id_rows = []
    for candidate in candidates:
        counts = allocate(list(candidate["weights"].values()), 192)
        rows = []
        local_rng = np.random.default_rng(SEED + candidate["index"] + 1)
        for environment, count in zip(environments, counts, strict=True):
            source = id_pools[environment]
            selected = local_rng.choice(len(source), size=count, replace=count > len(source))
            rows.extend(source[int(index)] for index in selected)
        local_rng.shuffle(rows)
        assert len(rows) == 192
        id_rows.append(rows)

    # Keep the original 12 questions for each OOD environment and add 36 new
    # questions with disjoint generation seeds. The first 192 rows remain exact.
    from make_rlve_eval_set import _records_for_environment

    expanded = list(old_ood)
    for index, environment in enumerate(ood_environments):
        seen = {row["user_prompt"] for row in old_ood if metadata(row)["environment"] == environment}
        added = _records_for_environment(environment, 0, 36,
                                         30_000_000 + index * 100_000, SEED,
                                         seen_prompts=seen, allow_fewer=True)
        if len(added) < 36:
            added.extend(_records_for_environment(environment, 0, 36 - len(added),
                                                  40_000_000 + index * 100_000, SEED,
                                                  seen_prompts=None))
        expanded.extend(added)
        unique = len({row["user_prompt"] for row in old_ood if metadata(row)["environment"] == environment}
                     | {row["user_prompt"] for row in added})
        print(f"OOD {index + 1}/16: {environment}, +{len(added)}, {unique}/48 unique", flush=True)

    (OUT / "n24.txt").write_text(" ".join(environments) + "\n")
    dump(OUT / "train_pool_d0_x64.json", pool)
    dump(OUT / "ood_d0_16env_x12.json", old_ood)
    dump(OUT / "ood_d0_16env_x48.json", expanded)
    for index, rows in enumerate(id_rows):
        dump(OUT / f"id_mix_{index:02d}_d0_x192.json", rows)
    dump(OUT / "mixes.json", {
        "seed": SEED, "environments": environments,
        "pool_sha256": hashlib.sha256((OUT / "train_pool_d0_x64.json").read_bytes()).hexdigest(),
        "ood_environments": ood_environments,
        "ood_normal_sha256": hashlib.sha256((OUT / "ood_d0_16env_x12.json").read_bytes()).hexdigest(),
        "ood_final_sha256": hashlib.sha256((OUT / "ood_d0_16env_x48.json").read_bytes()).hexdigest(),
        "id_questions_per_mix": 192, "ood_normal_questions": 192, "ood_final_questions": 768,
        "id_source_overlap_with_train_pool": overlap,
        "candidates": candidates,
    })
    verify()


def verify():
    manifest = json.loads((OUT / "mixes.json").read_text())
    environments = manifest["environments"]
    assert len(environments) == len(set(environments)) == 24
    assert (OUT / "n24.txt").read_text().split() == environments
    assert environments == (ROOT / "data/MOPD-SWEEPS-V2/n32.txt").read_text().split()[:24]
    assert len(manifest["candidates"]) == 73
    assert {candidate["index"] for candidate in manifest["candidates"]} == set(range(73))
    assert hashlib.sha256((OUT / "train_pool_d0_x64.json").read_bytes()).hexdigest() == manifest["pool_sha256"]
    pool = json.loads((OUT / "train_pool_d0_x64.json").read_text())
    assert {metadata(row)["environment"] for row in pool} == set(environments)
    normal = json.loads((OUT / "ood_d0_16env_x12.json").read_text())
    final = json.loads((OUT / "ood_d0_16env_x48.json").read_text())
    assert normal == json.loads((OLD / "ood_d0_16env_x12.json").read_text())
    assert hashlib.sha256((OUT / "ood_d0_16env_x12.json").read_bytes()).hexdigest() == manifest["ood_normal_sha256"]
    assert hashlib.sha256((OUT / "ood_d0_16env_x48.json").read_bytes()).hexdigest() == manifest["ood_final_sha256"]
    assert final[:len(normal)] == normal
    for rows, per_env in ((normal, 12), (final, 48)):
        counts = Counter(metadata(row)["environment"] for row in rows)
        assert set(counts) == set(manifest["ood_environments"])
        assert set(counts.values()) == {per_env}
        assert not set(counts) & set(environments)
    for candidate in manifest["candidates"]:
        index, weights = candidate["index"], candidate["weights"]
        assert set(weights) == set(environments) and abs(sum(weights.values()) - 1) < 1e-9
        rows = json.loads((OUT / f"id_mix_{index:02d}_d0_x192.json").read_text())
        counts = Counter(metadata(row)["environment"] for row in rows)
        expected = dict(zip(environments, allocate(list(weights.values()), 192), strict=True))
        assert len(rows) == 192 and all(counts[e] == expected[e] for e in environments)
    print("verified 24 environments, 73 mixtures, 192 weighted ID and normal OOD questions, 768 final OOD questions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verify() if args.check_only else prepare()
