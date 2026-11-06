"""Verify frozen prompts, labels, source identity, and reward extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .experiment import (EVAL_COUNT, EVAL_INTERVAL, EVAL_SAMPLES_PER_PROMPT,
                         RLVE_ROOT, SETS, SOURCE_ROOT, TRAIN_COUNT,
                         answer, functions, inputs, run_id, sha256)
from .reward import extract_output


def verify(data_dir: Path) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    config = yaml.safe_load((RLVE_ROOT / "configs/1gpu-500step-program-trees.yaml").read_text())
    assert (config["rollout-batch-size"], config["n-samples-per-prompt"],
            config["num-rollout"]) == (1, 8, 500)
    assert (config["eval-interval"], config["n-samples-per-eval-prompt"],
            config["rollout-max-response-len"], config["eval-max-response-len"]) == (
                EVAL_INTERVAL, EVAL_SAMPLES_PER_PROMPT, 1024, 1024)
    assert manifest["source_sha256"] == sha256(SOURCE_ROOT / "envs/strings.py")
    assert manifest["model"] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    assert (manifest["updates"], manifest["rollouts_per_prompt"],
            manifest["max_response_tokens"]) == (500, 8, 1024)
    assert (manifest["eval_kind"], manifest["eval_prompts_per_run"],
            manifest["eval_samples_per_prompt"], manifest["eval_interval"]) == (
                "id", EVAL_COUNT, EVAL_SAMPLES_PER_PROMPT, EVAL_INTERVAL)
    assert len(manifest["runs"]) == 24
    expected_inputs = inputs()
    seen_ids = set()
    for run in manifest["runs"]:
        identifier = run["id"]
        assert identifier == run_id(run["set"], run["family"], run["size"])
        assert identifier not in seen_ids
        seen_ids.add(identifier)
        assert run["functions"] == [fn.__name__ for fn, _ in functions(run["set"])[:run["size"]]]
        assert (run["depth"], run["breadth"]) == (
            (run["size"], 1) if run["family"] == "depth" else (1, run["size"]))
        for split, values in (("train", expected_inputs[:TRAIN_COUNT]),
                              ("eval", expected_inputs[TRAIN_COUNT:])):
            entry = run[split]
            path = RLVE_ROOT / entry["path"]
            assert path.resolve().is_relative_to(data_dir.resolve())
            assert entry["sha256"] == sha256(path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == entry["prompts"] == len(values)
            for row, value in zip(rows, values):
                assert row["metadata"]["input"] == value
                assert row["metadata"]["run_id"] == identifier
                assert row["label"] == answer(run["set"], run["family"], run["size"], value)
                assert f'main({json.dumps(value)})' in row["user_prompt"]
                assert extract_output(json.dumps({"output": row["label"]})) == row["label"]
    assert seen_ids == {run_id(set_name, family, n) for set_name in SETS
                        for family in ("depth", "breadth") for n in range(1, 7)}
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=RLVE_ROOT / "data/PROGRAM-TREES")
    args = parser.parse_args()
    manifest = verify(args.data_dir)
    print(f"Verified {len(manifest['runs'])} frozen runs and their train/eval labels")


if __name__ == "__main__":
    main()
