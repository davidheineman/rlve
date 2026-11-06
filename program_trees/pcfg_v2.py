"""Prepare and verify the eight-run PCFG v2 sweep."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

from .experiment import RLVE_ROOT
from .next_experiments import (
    PCFG_M,
    PCFG_N,
    ProgramTreeSampler,
    eval_inputs,
    verified_rows,
    walk,
    write_json,
    write_jsonl,
)


SOURCE = RLVE_ROOT / "data/PROGRAM-TREES-NEXT"
DEST = RLVE_ROOT / "data/PROGRAM-TREES-PCFG-V2"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def prepare() -> dict:
    old = json.loads((SOURCE / "manifest.json").read_text())
    values = eval_inputs()
    eval_sets = {}
    for name, alpha in (("N_id", 0.0), ("M_id", 1.0)):
        old_path = SOURCE / "eval" / f"{name}.jsonl"
        rows = [json.loads(line) for line in old_path.read_text().splitlines()]
        # Keep every original eval prompt. Add ten new inputs per depth from the
        # existing held-out input pool, using a separate deterministic RNG stream.
        sampler = ProgramTreeSampler({
            "run_id": f"eval-{name}-v2", "kind": "pcfg", "seed": 12001,
            "alpha": alpha, "branch_probability": 0.35,
        })
        rows.extend(
            sampler.draw(depth=depth, value=values[30 + (depth - 1) * 10 + i])
            for depth in range(1, 4) for i in range(10)
        )
        path = DEST / "eval" / f"{name}.jsonl"
        write_jsonl(path, verified_rows(rows))
        eval_sets[name] = {
            "path": str(path.relative_to(RLVE_ROOT)), "prompts": len(rows),
            "sha256": digest(path), "original_sha256": old["eval_sets"][name]["sha256"],
        }

    runs = []
    for index in range(8):
        source = next(run for run in old["runs"] if run["run_id"] == f"pcfg-m{index}of7")
        spec = json.loads((RLVE_ROOT / source["spec_path"]).read_text())
        spec.update(run_id=f"pcfg-m{index}of7-v2", budget=16000)
        path = DEST / "specs" / f"{spec['run_id']}.json"
        write_json(path, spec)
        runs.append({
            **spec, "spec_path": str(path.relative_to(RLVE_ROOT)),
            "spec_sha256": digest(path),
        })
    manifest = {
        "source_manifest_sha256": digest(SOURCE / "manifest.json"),
        "model": old["model"], "source": old["source"],
        "source_sha256": old["source_sha256"],
        "max_initial_tokens": old["max_initial_tokens"],
        "answer_forcing_extra_tokens": old["answer_forcing_extra_tokens"],
        "rollouts_per_prompt": old["rollouts_per_prompt"],
        "eval_samples_per_prompt": old["eval_samples_per_prompt"],
        "eval_interval_updates": old["eval_interval_updates"],
        "early_stop_accuracy": old["early_stop_accuracy"],
        "eval_sets": eval_sets, "runs": runs,
    }
    write_json(DEST / "manifest.json", manifest)
    verify()
    return manifest


def verify() -> dict:
    old = json.loads((SOURCE / "manifest.json").read_text())
    manifest = json.loads((DEST / "manifest.json").read_text())
    assert manifest["source_manifest_sha256"] == digest(SOURCE / "manifest.json")
    assert len(manifest["runs"]) == 8
    assert {x["alpha"] for x in manifest["runs"]} == {i / 7 for i in range(8)}
    for name in ("N_id", "M_id"):
        entry = manifest["eval_sets"][name]
        path = RLVE_ROOT / entry["path"]
        assert digest(path) == entry["sha256"]
        original = (SOURCE / "eval" / f"{name}.jsonl").read_text().splitlines()
        rows = path.read_text().splitlines()
        assert len(rows) == entry["prompts"] == 60
        assert rows[:30] == original
        records = verified_rows([json.loads(line) for line in rows])
        assert Counter(row["metadata"]["depth"] for row in records) == {1: 20, 2: 20, 3: 20}
        allowed = {op[0] for op in (PCFG_N if name == "N_id" else PCFG_M)}
        assert all({part["fn"] for part in walk(row["metadata"]["tree"])} <= allowed for row in records)
        assert len({row["user_prompt"] for row in records}) == 60
        assert entry["original_sha256"] == old["eval_sets"][name]["sha256"]
    for run in manifest["runs"]:
        path = RLVE_ROOT / run["spec_path"]
        assert digest(path) == run["spec_sha256"]
        spec = json.loads(path.read_text())
        assert run == {**spec, "spec_path": run["spec_path"], "spec_sha256": run["spec_sha256"]}
        assert spec["family"] == spec["kind"] == "pcfg"
        assert spec["budget"] == 16000 and spec["seed"] >= 7000
        assert spec["evals"] == spec["stop_evals"] == ["N_id", "M_id"]
        old_spec = json.loads((SOURCE / "specs" / f"{spec['run_id'][:-3]}.json").read_text())
        assert spec == {**old_spec, "run_id": spec["run_id"], "budget": 16000}
        a, b = ProgramTreeSampler(old_spec), ProgramTreeSampler(spec)
        for _ in range(20):
            old_row, new_row = a.draw(), b.draw()
            assert old_row["user_prompt"] == new_row["user_prompt"]
            assert old_row["label"] == new_row["label"]
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    args = parser.parse_args()
    manifest = prepare() if args.action == "prepare" else verify()
    print(f"Verified {len(manifest['runs'])} PCFG v2 runs and two 60-prompt eval sets")


if __name__ == "__main__":
    main()
