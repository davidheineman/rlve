"""Pin the Qwen and R1 KL-proposed mixtures to the frozen mixing prompt pool."""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/MIXING/hero_kl_proposals.json"


def prepare(output):
    source = json.loads((ROOT / "data/MIXING/mixes.json").read_text())
    proposals = json.loads((ROOT / "plot/mixing-r1/mix_recommendation_combined.json").read_text())
    pool = ROOT / "data/MIXING/train_pool_d0_x64.json"
    if hashlib.sha256(pool.read_bytes()).hexdigest() != source["pool_sha256"]:
        raise ValueError("Frozen prompt pool hash differs from the source manifest")
    environments = source["environments"]
    candidates = []
    for model, project in (
        ("qwen", "david-heineman/rl-data-mixing"),
        ("r1", "david-heineman/rl-data-mixing-r1-distil"),
    ):
        record = proposals[model]
        if record["source_project"] != project:
            raise ValueError(f"Unexpected source project for {model}")
        weights = record["weights"]
        if set(weights) != set(environments) or any(weights[name] <= 0 for name in environments):
            raise ValueError(f"Invalid environment weights for {model}")
        if not math.isclose(sum(weights.values()), 1, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"Mixture weights do not sum to one for {model}")
        candidates.append({
            "name": f"kl-proposed-{model}",
            "alpha": "per-task-log-error-kl",
            "predicted_ood_pass1": record["predicted_ood_pass1"],
            "weights": {name: weights[name] for name in environments},
        })
    manifest = {
        "environments": environments,
        "pool_sha256": source["pool_sha256"],
        "source": "plot/mixing-r1/mix_recommendation_combined.json",
        "candidates": candidates,
    }
    if output.exists():
        if json.loads(output.read_text()) != manifest:
            raise ValueError(f"Pinned manifest differs from the current proposals: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()
