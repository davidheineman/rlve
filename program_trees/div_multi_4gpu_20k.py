"""Freeze comparable 20k-rollout multi-tree samplers and their eval lists."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from .experiment import RLVE_ROOT
from .next_experiments import SINGLE_CHAINS, multi_chains


ROOT = RLVE_ROOT / "data/PROGRAM-TREES-DIV-MULTI-4GPU-20K"
EVALS = [*(f"single_s{i}_id" for i in range(1, 5)), "single_pooled_id", "multi_id"]


def prepare() -> None:
    ROOT.joinpath("specs").mkdir(parents=True, exist_ok=True)
    runs = []
    for run_id, chains in (
        ("div-multi-random-4gpu-20k", multi_chains()),
        ("div-multi-single-trees-4gpu-20k", [[list(op) for op in chain] for chain in SINGLE_CHAINS]),
    ):
        spec = {
            "run_id": run_id,
            "family": "diversity",
            "kind": "diversity_multi",
            "seed": 4000,
            "chains": chains,
            "budget": 20000,
            "evals": EVALS,
        }
        path = ROOT / "specs" / f"{run_id}.json"
        path.write_text(json.dumps(spec, indent=2) + "\n")
        runs.append({**spec, "spec_path": str(path.relative_to(RLVE_ROOT)),
                     "spec_sha256": sha256(path.read_bytes()).hexdigest()})
    manifest = {
        "description": "Four GPU 20k rollout multi-tree comparison: 32 chains versus the four single-env chains",
        "source_manifest": "data/PROGRAM-TREES-NEXT/manifest.json",
        "eval_dir": "data/PROGRAM-TREES-NEXT/eval",
        "runs": runs,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
