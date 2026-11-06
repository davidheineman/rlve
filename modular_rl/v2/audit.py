"""Verify every frozen MATH reference can be graded before spending GPU time."""

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from modular_rl.prepare import write_json

from .math_reward import grade, initialize_worker


def check(row):
    metadata = row["metadata"]
    answer = metadata["target"]["answer"]
    try:
        if grade(r"\boxed{" + answer + "}", answer)["accuracy"] != 1:
            raise ValueError("Reference does not pass its own grader")
    except Exception as exc:  # noqa: BLE001 -- report every broken reference
        return {"id": metadata["id"], "error": str(exc)}
    return None


def audit(data, output):
    rows = [
        json.loads(line)
        for split in ("train", "val")
        for line in (data / "math" / f"{split}.jsonl").read_text().splitlines()
    ]
    with ProcessPoolExecutor(
        max_workers=4,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
    ) as pool:
        failed = [item for item in pool.map(check, rows, chunksize=16) if item]
    result = {
        "references": len(rows),
        "failed": failed,
        "passed": not failed,
        "sha256": {
            split: hashlib.sha256(
                (data / "math" / f"{split}.jsonl").read_bytes()
            ).hexdigest()
            for split in ("train", "val")
        },
    }
    write_json(output, result)
    print(json.dumps(result), flush=True)
    if failed:
        raise ValueError("MATH reference audit failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    audit(**vars(parser.parse_args()))
