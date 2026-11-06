"""Build a stratified MATH holdout and reuse v1's frozen MBPP/Tulu splits."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from modular_rl.prepare import write_json

from .data import normalize_math, stratified_sample, stratum

REPOSITORY = "EleutherAI/hendrycks_math"
SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def prepare(root, previous, seed=42):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    root.mkdir(parents=True, exist_ok=True)
    directory = root / "math"
    directory.mkdir(exist_ok=True)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        revision = HfApi().dataset_info(REPOSITORY).sha
        examples, excluded, seen = [], [], set()
        for subject in SUBJECTS:
            rows = load_dataset(REPOSITORY, subject, revision=revision, split="train")
            for index, row in enumerate(rows):
                try:
                    example = normalize_math(row, subject, index)
                except ValueError as exc:
                    excluded.append({"id": f"{subject}:{index}", "reason": str(exc)})
                    continue
                key = example["metadata"]["split_group"]
                if key in seen:
                    excluded.append(
                        {"id": f"{subject}:{index}", "reason": "duplicate_problem"}
                    )
                    continue
                seen.add(key)
                examples.append(example)
        val = stratified_sample(examples, round(len(examples) * 0.1), seed)
        held_out = {row["metadata"]["key"] for row in val}
        train = [row for row in examples if row["metadata"]["key"] not in held_out]
        import random

        random.Random(seed).shuffle(train)
        checksums, distributions = {}, {}
        for split, rows in (("train", train), ("val", val)):
            path = directory / f"{split}.jsonl"
            temporary = path.with_suffix(".jsonl.tmp")
            temporary.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            )
            temporary.replace(path)
            checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            distributions[split] = dict(Counter("/".join(stratum(row)) for row in rows))
        write_json(directory / "excluded.json", excluded)
        write_json(
            manifest_path,
            {
                "dataset": REPOSITORY,
                "revision": revision,
                "source_split": "train",
                "official_test_used": False,
                "seed": seed,
                "validation_fraction": 0.1,
                "counts": {"train": len(train), "val": len(val)},
                "strata": distributions,
                "excluded": len(excluded),
                "sha256": checksums,
            },
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest["seed"] != seed:
        raise ValueError("Use a new directory to change split seeds")
    for name, checksum in manifest["sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"MATH split changed: {name}")
    for task in ("code", "chat"):
        source = (previous / task).resolve()
        destination = root / task
        if not (source / "manifest.json").is_file():
            raise ValueError(f"Missing v1 split: {source}")
        if destination.is_symlink():
            if destination.resolve() != source:
                raise ValueError(f"Different split already linked: {destination}")
        elif destination.exists():
            raise ValueError(f"Refusing to replace {destination}")
        else:
            destination.symlink_to(source, target_is_directory=True)
    write_json(
        root / "ready.json",
        {
            "version": 2,
            "math": manifest,
            "code": str((previous / "code").resolve()),
            "chat": str((previous / "chat").resolve()),
        },
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    prepare(**vars(parser.parse_args()))
