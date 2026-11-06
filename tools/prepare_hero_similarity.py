"""Freeze N=4/8 similar and dissimilar subsets of the v2 R1 teacher environments."""

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data/MOPD-SWEEPS-V2/id_n16_d0_x4.json"
OOD = ROOT / "data/MOPD-SWEEPS/ood_d0_16env_x24.json"
OUTPUT = ROOT / "data/HERO-SIMILARITY"

# Arithmetic and number theory, then other mathematical problem types.
SIMILAR_4 = ("CRT", "CongruentEquation", "DiscreteLogarithm", "AdditionTable")
SIMILAR_8 = SIMILAR_4 + ("GaussianElimination", "Differentiate", "Integral", "CantorExpansion")

# Logic, optimization, planning, and digit DP; then four other algorithm families.
DISSIMILAR_4 = ("Circuit", "FractionalProgramming", "MafMafia", "CapitalCityEffect")
DISSIMILAR_8 = DISSIMILAR_4 + ("DeltaMinPopcount", "Sorting", "GoldWashing", "KloBlocks")
SELECTIONS = {
    "n4-similar": SIMILAR_4,
    "n4-dissimilar": DISSIMILAR_4,
    "n8-similar": SIMILAR_8,
    "n8-dissimilar": DISSIMILAR_8,
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
    trained = set((ROOT / "data/MOPD-SWEEPS-V2/n16.txt").read_text().split())
    if len(trained) != 16 or set(SIMILAR_8) | set(DISSIMILAR_8) != trained:
        raise RuntimeError("N=8 selections must partition all 16 v2 teacher environments")
    if set(SIMILAR_8) & set(DISSIMILAR_8):
        raise RuntimeError("N=8 selections must be disjoint")
    source = json.loads(SOURCE.read_text())
    groups = {name: [row for row in source if environment(row) == name] for name in trained}
    if len(source) != 64 or any(len(rows) != 4 for rows in groups.values()):
        raise RuntimeError("Expected four frozen v2 ID prompts per environment")
    if any(len({row["user_prompt"] for row in rows}) != 4 for rows in groups.values()):
        raise RuntimeError("Repeated frozen ID prompt")
    ood = json.loads(OOD.read_text())
    ood_counts = Counter(environment(row) for row in ood)
    if len(ood) != 384 or len(ood_counts) != 16 or set(ood_counts.values()) != {24} or trained & ood_counts.keys():
        raise RuntimeError("Expected 16 disjoint OOD environments with 24 prompts each")

    for label, names in SELECTIONS.items():
        if len(names) != int(label[1]) or len(set(names)) != len(names) or not set(names) <= trained:
            raise RuntimeError(f"Invalid selection: {label}")
        records = [row for name in names for row in groups[name]]
        write_or_verify(OUTPUT / f"{label}.txt", " ".join(names) + "\n")
        write_or_verify(OUTPUT / f"id_{label}_d0_x4.json", json.dumps(records, indent=1) + "\n")
        print(f"{label}: {', '.join(names)}; {len(records)} ID prompts", flush=True)
    write_or_verify(OUTPUT / "manifest.json", json.dumps({
        "source_id": str(SOURCE.relative_to(ROOT)),
        "source_ood": str(OOD.relative_to(ROOT)),
        "selections": SELECTIONS,
        "prompts_per_update": 8,
        "rollouts_per_prompt": 16,
        "updates": 157,
        "training_rollouts": 20096,
    }, indent=1) + "\n")


if __name__ == "__main__":
    main()
