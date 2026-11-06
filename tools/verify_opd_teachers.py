"""Verify environment teachers have complete HF weight exports."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(teacher_dir, environment_file, base_hf_dir, max_checkpoint_step=149):
    environments = environment_file.read_text().split()
    if not environments or len(environments) != len(set(environments)):
        raise ValueError("Expected distinct environments")
    base_config = digest(base_hf_dir / "config.json")
    base_tokenizer = digest(base_hf_dir / "tokenizer.json")
    for index, environment in enumerate(environments):
        run = teacher_dir / f"{index:02d}-{environment}"
        export = run / "hf-final"
        marker = json.loads((run / "weights_only_complete.json").read_text())
        if not 0 <= marker["checkpoint_step"] <= max_checkpoint_step or marker["format"] != "hf-safetensors":
            raise ValueError(f"Incomplete teacher export: {environment}")
        if digest(export / "config.json") != base_config or digest(export / "tokenizer.json") != base_tokenizer:
            raise ValueError(f"Teacher architecture or tokenizer differs: {environment}")
        weight_map = json.loads((export / "model.safetensors.index.json").read_text())["weight_map"]
        weight_files = {export / filename for filename in weight_map.values()}
        if len(weight_map) != marker["tensor_count"] or not all(path.is_file() for path in weight_files):
            raise ValueError(f"Missing teacher weights: {environment}")
        if sum(path.stat().st_size for path in weight_files) != marker["weight_bytes"]:
            raise ValueError(f"Teacher weight size differs from validated export: {environment}")
    return environments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--environment-file", type=Path, required=True)
    parser.add_argument("--base-hf-dir", type=Path, required=True)
    parser.add_argument("--max-checkpoint-step", type=int, default=149)
    args = parser.parse_args()
    environments = verify(args.teacher_dir, args.environment_file, args.base_hf_dir, args.max_checkpoint_step)
    print(f"Verified {len(environments)} complete, tokenizer-compatible OPD teachers in {args.teacher_dir}")


if __name__ == "__main__":
    main()
