"""Freeze nested, diverse MOPD training sets and a larger common OOD evaluation."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from make_rlve_eval_set import _records_for_environment


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data/MOPD-SWEEPS"
ENVIRONMENTS = ROOT / "scripts/slurm/sweeps/mopd32.txt"
OLD_OOD = ROOT / "data/MIXING/ood_d0_16env_x12.json"
HELDOUT = ROOT / "data/HELD-OUT_ENVIRONMENTS/test.json"
SIZES = (1, 2, 4, 8, 16, 32)


def metadata(row):
    value = row["metadata"]
    return json.loads(value) if isinstance(value, str) else value


def write_or_verify(path, records):
    content = json.dumps(records, indent=1) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"Frozen data differs from regenerated content: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def main():
    from Gym.environments import identifier2environment
    from Gym.parameter_controllers import identifier2controller

    environments = ENVIRONMENTS.read_text().split()
    if len(environments) != 32 or len(set(environments)) != 32:
        raise ValueError("Expected 32 distinct training environments")
    for environment in environments:
        if environment not in identifier2environment or environment not in identifier2controller:
            raise ValueError(f"Unregistered environment: {environment}")

    id_by_env = {}
    for index, environment in enumerate(environments):
        id_by_env[environment] = _records_for_environment(
            environment, 0, 4, 4_000_000 + index * 100_000, 20261002,
        )
        val_path = OUTPUT / f"{environment}_d0.json"
        source = ROOT / "data/EVAL-pairwise32" / f"{environment}_d0.json"
        if source.exists():
            val = json.loads(source.read_text())
            if len(val) != 128 or {metadata(row)["environment"] for row in val} != {environment}:
                raise ValueError(f"Unexpected previous validation set: {source}")
        else:
            val = _records_for_environment(
                environment, 0, 128, 1_000_000 + index * 100_000, 20261002,
            )
        write_or_verify(val_path, val)
        if len({metadata(row)["config"]["seed"] for row in val + id_by_env[environment]}) != 132:
            raise ValueError(f"Overlapping validation and ID seeds: {environment}")
        print(f"validated {index + 1}/32 {environment}", flush=True)

    old_ood = json.loads(OLD_OOD.read_text())
    old_ood_environments = list(dict.fromkeys(metadata(row)["environment"] for row in old_ood))
    if len(old_ood_environments) != 16 or set(old_ood_environments) & set(environments):
        raise ValueError("OOD environments must be disjoint from all 32 training environments")
    counts = Counter(metadata(row)["environment"] for row in old_ood)
    if set(counts.values()) != {12}:
        raise ValueError("Expected 12 existing OOD prompts per environment")
    heldout = {metadata(row)["environment"] for row in json.loads(HELDOUT.read_text())}
    candidates = old_ood_environments + sorted(heldout - set(old_ood_environments) - set(environments))
    ood = []
    ood_environments = []
    for index, environment in enumerate(candidates):
        if len(ood_environments) == 16:
            break
        if environment not in identifier2environment or environment not in identifier2controller:
            continue
        existing = [row for row in old_ood if metadata(row)["environment"] == environment]
        try:
            more = _records_for_environment(
                environment, 0, 12 if existing else 24,
                30_000_000 + index * 100_000, 20261002,
                seen_prompts={row["user_prompt"] for row in existing},
            )
        except (Exception, SystemExit) as error:
            print(f"skip OOD {environment}: {error}", flush=True)
            continue
        records = existing + more
        if len({metadata(row)["config"]["seed"] for row in records}) != 24:
            raise ValueError(f"Repeated OOD seed: {environment}")
        if len({row["user_prompt"] for row in records}) != 24:
            raise ValueError(f"Repeated OOD prompt: {environment}")
        ood.extend(records)
        ood_environments.append(environment)
        print(f"OOD {len(ood_environments)}/16 {environment}", flush=True)
    if len(ood_environments) != 16:
        raise RuntimeError(f"Only {len(ood_environments)} distinct OOD environments yielded 24 unique prompts")
    write_or_verify(OUTPUT / "ood_d0_16env_x24.json", ood)

    for size in SIZES:
        selected = environments[:size]
        env_file = OUTPUT / f"n{size}.txt"
        content = " ".join(selected) + "\n"
        if env_file.exists() and env_file.read_text() != content:
            raise RuntimeError(f"Frozen environment selection differs: {env_file}")
        env_file.write_text(content)
        id_records = [row for environment in selected for row in id_by_env[environment]]
        write_or_verify(OUTPUT / f"id_n{size}_d0_x4.json", id_records)

    manifest = {
        "sizes": SIZES,
        "environments": environments,
        "ood_environments": ood_environments,
        "ood_prompts_per_environment": 24,
        "sources": [str(OLD_OOD.relative_to(ROOT))],
        "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(OUTPUT.glob("*.json")) if path.name != "manifest.json"},
    }
    write_or_verify(OUTPUT / "manifest.json", manifest)
    print(f"Prepared {len(environments)} training environments and {len(ood)} OOD prompts", flush=True)


if __name__ == "__main__":
    main()
