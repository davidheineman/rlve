"""Freeze nested MOPD arms using environments that learned in the old R1 teacher runs."""

import hashlib
import json
from pathlib import Path

from make_rlve_eval_set import _records_for_environment


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data/MOPD-SWEEPS-LEARNED16"
ENVIRONMENTS = ROOT / "scripts/slurm/sweeps/mopd-learned16.txt"
OLD_VALIDATION = ROOT / "data/EVAL-pairwise32"
CANTOR_N1 = ROOT / "data/MOPD-SWEEPS-CANTOR-N1"
OOD_SOURCE = ROOT / "data/MOPD-SWEEPS/ood_d0_16env_x24.json"
SIZES = (1, 2, 4, 8, 16)


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
    if len(environments) != 16 or len(set(environments)) != 16 or environments[0] != "CantorExpansion":
        raise ValueError("Expected 16 distinct environments starting with CantorExpansion")
    if any(env not in identifier2environment or env not in identifier2controller for env in environments):
        raise ValueError("Selection contains an unregistered environment")

    ood = json.loads(OOD_SOURCE.read_text())
    ood_environments = {metadata(row)["environment"] for row in ood}
    if len(ood) != 384 or len(ood_environments) != 16 or ood_environments & set(environments):
        raise ValueError("Shared OOD set must have 16 disjoint environments and 384 prompts")

    ids = {}
    sources = []
    for index, environment in enumerate(environments):
        source = OLD_VALIDATION / f"{environment}_d0.json"
        sources.append(source)
        content = source.read_text()
        validation = json.loads(content)
        val_prompts = {row["user_prompt"] for row in validation}
        val_seeds = {metadata(row)["config"]["seed"] for row in validation}
        if (len(validation) != 128 or len(val_seeds) != 128
                or {metadata(row)["environment"] for row in validation} != {environment}):
            raise ValueError(f"Invalid old teacher validation set: {source}")
        write_or_verify(OUTPUT / f"{environment}_d0.json", content)

        if index == 0:
            # Reuse the running N=1 Cantor arm's exact ID prompts.
            ids[environment] = json.loads((CANTOR_N1 / "id_n1_d0_x4.json").read_text())
        else:
            ids[environment] = _records_for_environment(
                environment, 0, 4, 5_000_000 + index * 100_000, 20261002,
                seen_prompts=set(),
            )
        if (len(ids[environment]) != 4 or len({row["user_prompt"] for row in ids[environment]}) != 4
                or len({metadata(row)["config"]["seed"] for row in ids[environment]}) != 4
                or {metadata(row)["config"]["seed"] for row in ids[environment]} & val_seeds):
            raise ValueError(f"ID prompts or seeds are not unique: {environment}")
        prompt_overlap = len({row["user_prompt"] for row in ids[environment]} & val_prompts)
        print(f"validated {index + 1}/16 {environment}; {prompt_overlap}/4 ID prompts recur in old validation", flush=True)

    for size in SIZES:
        selected = environments[:size]
        write_or_verify(OUTPUT / f"n{size}.txt", " ".join(selected) + "\n")
        records = [row for environment in selected for row in ids[environment]]
        write_or_verify(OUTPUT / f"id_n{size}_d0_x4.json", json.dumps(records, indent=1) + "\n")

    if json.loads((OUTPUT / "id_n1_d0_x4.json").read_text()) != json.loads((CANTOR_N1 / "id_n1_d0_x4.json").read_text()):
        raise ValueError("Corrected N=1 ID data differs from the running Cantor arm")

    manifest = {
        "variant": "learned16",
        "sizes": SIZES,
        "environments": environments,
        "ood_source": str(OOD_SOURCE.relative_to(ROOT)),
        "sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted([ENVIRONMENTS, OOD_SOURCE, *sources, *OUTPUT.glob("*.json")])
            if path.name != "manifest.json"
        },
    }
    write_or_verify(OUTPUT / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    print("Prepared learned R1 MOPD arms: N=1,2,4,8,16; 384 shared OOD prompts", flush=True)


if __name__ == "__main__":
    main()
