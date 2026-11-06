"""Freeze a CantorExpansion-only N=1 arm of the MOPD sweep."""

import hashlib
import json
from pathlib import Path

from make_rlve_eval_set import _records_for_environment


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data/MOPD-SWEEPS-CANTOR-N1"
VALIDATION_SOURCE = ROOT / "data/EVAL-pairwise32/CantorExpansion_d0.json"
OOD_SOURCE = ROOT / "data/MOPD-SWEEPS/ood_d0_16env_x24.json"
ENVIRONMENT = "CantorExpansion"


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

    if ENVIRONMENT not in identifier2environment or ENVIRONMENT not in identifier2controller:
        raise ValueError(f"Unregistered environment: {ENVIRONMENT}")

    validation_content = VALIDATION_SOURCE.read_text()
    validation = json.loads(validation_content)
    if len(validation) != 128 or {metadata(row)["environment"] for row in validation} != {ENVIRONMENT}:
        raise ValueError("Unexpected CantorExpansion validation set")
    validation_prompts = {row["user_prompt"] for row in validation}
    validation_seeds = {metadata(row)["config"]["seed"] for row in validation}
    if len(validation_prompts) != 128 or len(validation_seeds) != 128:
        raise ValueError("CantorExpansion validation set has duplicate prompts or seeds")

    ood = json.loads(OOD_SOURCE.read_text())
    if len(ood) != 384 or any(metadata(row)["environment"] == ENVIRONMENT for row in ood):
        raise ValueError("Shared OOD set is malformed or overlaps CantorExpansion")

    id_rows = _records_for_environment(
        ENVIRONMENT, 0, 4, 4_500_000, 20261002,
        seen_prompts=set(validation_prompts),
    )
    id_seeds = {metadata(row)["config"]["seed"] for row in id_rows}
    if len(id_seeds) != 4 or id_seeds & validation_seeds:
        raise ValueError("CantorExpansion ID seeds overlap validation")

    write_or_verify(OUTPUT / "n1.txt", ENVIRONMENT + "\n")
    write_or_verify(OUTPUT / f"{ENVIRONMENT}_d0.json", validation_content)
    write_or_verify(OUTPUT / "id_n1_d0_x4.json", json.dumps(id_rows, indent=1) + "\n")

    manifest = {
        "variant": "cantor-n1",
        "environment": ENVIRONMENT,
        "validation_source": str(VALIDATION_SOURCE.relative_to(ROOT)),
        "ood_source": str(OOD_SOURCE.relative_to(ROOT)),
        "sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((VALIDATION_SOURCE, OOD_SOURCE, *OUTPUT.glob("*.json")))
            if path.name != "manifest.json"
        },
    }
    write_or_verify(OUTPUT / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    print("Prepared CantorExpansion N=1: 128 teacher validation, 4 hero ID, 384 shared OOD prompts", flush=True)


if __name__ == "__main__":
    main()
