"""Freeze the requested R1 hero similarity v2 environment and ID evaluation sets."""

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data/MOPD-SWEEPS-V2/id_n16_d0_x4.json"
TRAINED = ROOT / "data/MOPD-SWEEPS-V2/n16.txt"
OOD = ROOT / "data/MOPD-SWEEPS/ood_d0_16env_x24.json"
OUTPUT = ROOT / "data/HERO-SIMILARITY-V2"

SELECTIONS = {
    "n4-similar1": ("DiscreteLogarithm", "CongruentEquation", "CRT", "AdditionTable"),
    "n4-similar2": ("KloBlocks", "DeltaMinPopcount", "FractionalProgramming", "MafMafia"),
    "n4-dissimilar1": ("Integral", "CapitalCityEffect", "MafMafia", "Circuit"),
    "n4-dissimilar2": ("Differentiate", "CRT", "KloBlocks", "CantorExpansion"),
    "n8-similar": (
        "AdditionTable", "GaussianElimination", "DiscreteLogarithm", "CongruentEquation",
        "CRT", "Circuit", "Integral", "Differentiate",
    ),
    "n8-dissimilar": (
        "Integral", "CRT", "KloBlocks", "GoldWashing", "Sorting", "MafMafia", "Circuit", "DeltaMinPopcount",
    ),
}


def environment(row):
    value = row["metadata"]
    metadata = json.loads(value) if isinstance(value, str) else value
    return metadata["environment"]


def write_or_verify(path, content):
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"Frozen data differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def main():
    trained = TRAINED.read_text().split()
    if len(trained) != 16 or len(set(trained)) != 16:
        raise RuntimeError("Expected 16 distinct v3 teachers")
    requested = {name for names in SELECTIONS.values() for name in names}
    if requested != set(trained):
        raise RuntimeError(f"Requested teachers differ from v3: {requested ^ set(trained)}")
    source = json.loads(SOURCE.read_text())
    groups = {name: [row for row in source if environment(row) == name] for name in trained}
    if len(source) != 64 or any(len(rows) != 4 for rows in groups.values()):
        raise RuntimeError("Expected four frozen ID prompts per v3 teacher environment")
    if any(len({row["user_prompt"] for row in rows}) != 4 for rows in groups.values()):
        raise RuntimeError("Repeated frozen ID prompt")
    ood = json.loads(OOD.read_text())
    ood_counts = Counter(environment(row) for row in ood)
    if len(ood) != 384 or len(ood_counts) != 16 or set(ood_counts.values()) != {24} or requested & ood_counts.keys():
        raise RuntimeError("Expected 16 disjoint OOD environments with 24 prompts each")

    for label, names in SELECTIONS.items():
        expected = int(label[1])
        if len(names) != expected or len(set(names)) != expected or not set(names) <= set(trained):
            raise RuntimeError(f"Invalid selection: {label}")
        records = [row for name in names for row in groups[name]]
        write_or_verify(OUTPUT / f"{label}.txt", " ".join(names) + "\n")
        write_or_verify(OUTPUT / f"id_{label}_d0_x4.json", json.dumps(records, indent=1) + "\n")
        print(f"{label}: {', '.join(names)}; {len(records)} ID prompts", flush=True)
    write_or_verify(OUTPUT / "manifest.json", json.dumps({
        "source_id": str(SOURCE.relative_to(ROOT)),
        "source_ood": str(OOD.relative_to(ROOT)),
        "selections": SELECTIONS,
        "id_prompts_per_environment": 4,
        "ood_prompts_per_environment": 24,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
