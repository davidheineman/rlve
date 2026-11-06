"""Pack eight validated environment pair sets into balanced DPO replay steps."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def pack(input_dir, environments_file, output_dir, steps=1000):
    environments = Path(environments_file).read_text().split()
    if len(environments) != 8 or len(set(environments)) != 8:
        raise ValueError("Expected eight distinct environments")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    input_dir = Path(input_dir)
    rows = {}
    digests = {}
    for environment in environments:
        path = input_dir / f"{environment}.jsonl"
        digest = hashlib.sha256()
        entries = []
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                entry = json.loads(line)
                chosen, rejected = entry["chosen"], entry["rejected"]
                if (chosen["prompt"] != rejected["prompt"] or
                        chosen["metadata"]["environment"] != environment or
                        rejected["metadata"]["environment"] != environment or
                        chosen["reward"]["accuracy"] != 1 or rejected["reward"]["accuracy"] != 0 or
                        chosen["status"] not in ("completed", "truncated") or
                        rejected["status"] not in ("completed", "truncated") or
                        chosen["tokens"][:-chosen["response_length"]] != rejected["tokens"][:-rejected["response_length"]] or
                        len(chosen["tokens"]) > 2048 or len(rejected["tokens"]) > 2048):
                    raise ValueError(f"Invalid DPO pair in {path} at row {len(entries)}")
                entries.append(entry)
        if len(entries) < steps:
            raise ValueError(f"Need {steps} pairs for {environment}; have {len(entries)}")
        if len({entry["prompt_id"] for entry in entries[:steps]}) != steps:
            raise ValueError(f"Duplicate training prompt for {environment}")
        rows[environment] = entries[:steps]
        digests[environment] = digest.hexdigest()

    temporary = output_dir.with_name(output_dir.name + ".tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    for step in range(steps):
        samples = []
        for environment in environments:
            entry = rows[environment][step]
            for side in ("chosen", "rejected"):
                sample = entry[side].copy()
                sample["index"] = step * 16 + len(samples)
                samples.append(sample)
        torch.save({"rollout_id": step, "samples": samples}, temporary / f"{step:06d}.pt")
    manifest = {
        "environment_file": str(environments_file),
        "environments": environments,
        "pairs_per_environment": steps,
        "num_batches": steps,
        "samples_per_batch": 16,
        "source_sha256": digests,
        "chosen": "correct high-validation teacher response",
        "rejected": "incorrect low-validation teacher response; CRT uses base-model exception",
    }
    with (temporary / "manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--environments-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(pack(args.input_dir, args.environments_file, args.output_dir, args.steps), sort_keys=True))


if __name__ == "__main__":
    main()
