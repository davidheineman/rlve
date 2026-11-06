"""Pin the natural and plotted proposed mixtures to the frozen training pool."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def prepare(output):
    mixing_dir = ROOT / "data/MIXING"
    original = json.loads((mixing_dir / "mixes.json").read_text())
    pool = mixing_dir / "train_pool_d0_x64.json"
    if hashlib.sha256(pool.read_bytes()).hexdigest() != original["pool_sha256"]:
        raise ValueError("Frozen prompt pool does not match the mixing manifest")

    model = json.loads((ROOT / "plot/mixing/model.json").read_text())
    if model["natural_definition"] != "uniform sampling over records in the frozen 1500-prompt training pool":
        raise ValueError("Unexpected natural-distribution definition")
    proposed = original["candidates"][model["proposed_mix_index"]]
    environments = original["environments"]
    counts = original["pool_counts"]
    total = sum(counts.values())
    natural_weights = {environment: counts[environment] / total for environment in environments}

    with (ROOT / "plot/mixing/mix_recommendation.csv").open(newline="") as file:
        plotted = {row["environment"]: row for row in csv.DictReader(file)}
    if set(plotted) != set(environments):
        raise ValueError("Plotted environments differ from the mixing manifest")
    for environment in environments:
        if not math.isclose(
            natural_weights[environment],
            float(plotted[environment]["natural_pool_probability"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Natural probability differs from the plot for {environment}")
        if not math.isclose(
            proposed["weights"][environment],
            float(plotted[environment]["proposed_probability"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Proposed probability differs from the plot for {environment}")

    manifest = {
        "environments": environments,
        "pool_sha256": original["pool_sha256"],
        "source_proposed_mix_index": model["proposed_mix_index"],
        "candidates": [
            {"name": "natural", "alpha": "pool-proportional", "weights": natural_weights},
            {"name": "proposed", "alpha": proposed["alpha"], "weights": proposed["weights"]},
        ],
    }
    output = Path(output)
    if output.exists():
        if json.loads(output.read_text()) != manifest:
            raise ValueError(f"Existing hero manifest differs: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/MIXING/hero_mixes.json")
    args = parser.parse_args()
    prepare(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
