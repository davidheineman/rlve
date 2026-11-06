"""Freeze nested v2 environment, validation, and ID data for both model families."""

import hashlib
import json
from pathlib import Path

from make_rlve_eval_set import _records_for_environment


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data/MOPD-SWEEPS-V2"
ENVIRONMENTS = ROOT / "scripts/slurm/sweeps/mopd-v2-32.txt"
R1_ENVIRONMENTS = ROOT / "scripts/slurm/sweeps/pairwise32.txt"
VALIDATION = ROOT / "data/EVAL-pairwise32"
OOD = ROOT / "data/MOPD-SWEEPS/ood_d0_16env_x24.json"
SIZES = (2, 4, 8, 16, 32)


def metadata(row):
    value = row["metadata"]
    return json.loads(value) if isinstance(value, str) else value


def write_or_verify(path, content):
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
    old_r1 = R1_ENVIRONMENTS.read_text().split()
    if len(environments) != 32 or len(set(environments)) != 32 or set(environments) != set(old_r1):
        raise ValueError("V2 N=32 must contain exactly the original R1 teacher environments")
    expected = {
        2: ["Circuit", "FractionalProgramming"],
        4: ["Circuit", "FractionalProgramming", "MafMafia", "CantorExpansion"],
        8: ["Circuit", "FractionalProgramming", "MafMafia", "CantorExpansion",
            "CRT", "DeltaMinPopcount", "Differentiate", "CapitalCityEffect"],
    }
    for size, selection in expected.items():
        if environments[:size] != selection:
            raise ValueError(f"N={size} differs from the requested environment selection")
    if any(env not in identifier2environment or env not in identifier2controller for env in environments):
        raise ValueError("An environment is not registered")
    ood = json.loads(OOD.read_text())
    ood_environments = {metadata(row)["environment"] for row in ood}
    if len(ood) != 384 or len(ood_environments) != 16 or ood_environments & set(environments):
        raise ValueError("OOD data must have 16 environments, 24 prompts each, disjoint from N=32")

    ids = {}
    for index, environment in enumerate(environments[:16]):
        source = VALIDATION / f"{environment}_d0.json"
        validation = json.loads(source.read_text())
        val_seeds = {metadata(row)["config"]["seed"] for row in validation}
        if len(validation) != 128 or len(val_seeds) != 128 or {
            metadata(row)["environment"] for row in validation
        } != {environment}:
            raise ValueError(f"Invalid frozen R1 validation set: {source}")
        write_or_verify(OUTPUT / source.name, source.read_text())
        ids[environment] = _records_for_environment(
            environment, 0, 4, 6_000_000 + index * 100_000, 20261003,
            seen_prompts=set(),
        )
        if (len(ids[environment]) != 4
                or len({row["user_prompt"] for row in ids[environment]}) != 4
                or {metadata(row)["config"]["seed"] for row in ids[environment]} & val_seeds):
            raise ValueError(f"Invalid ID data for {environment}")
        print(f"validated {index + 1}/16 {environment}", flush=True)

    for size in SIZES:
        selected = environments[:size]
        write_or_verify(OUTPUT / f"n{size}.txt", " ".join(selected) + "\n")
        if size <= 16:
            records = [row for env in selected for row in ids[env]]
            write_or_verify(OUTPUT / f"id_n{size}_d0_x4.json", json.dumps(records, indent=1) + "\n")

    manifest = {
        "variant": "v2",
        "sizes": SIZES,
        "environments": environments,
        "ood_source": str(OOD.relative_to(ROOT)),
        "validation_source": str(VALIDATION.relative_to(ROOT)),
        "sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted([ENVIRONMENTS, R1_ENVIRONMENTS, OOD, *OUTPUT.glob("*.json")])
            if path.name != "manifest.json"
        },
    }
    write_or_verify(OUTPUT / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    print("Prepared nested N=2,4,8,16,32 v2 sets and 384 shared OOD prompts", flush=True)


if __name__ == "__main__":
    main()
