"""Subsample and combine the frozen ID prompt sets used by the OPD teachers."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environments", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts-per-environment", type=int, default=4)
    args = parser.parse_args()

    if args.prompts_per_environment < 1 or 128 % args.prompts_per_environment:
        raise ValueError("prompts-per-environment must be a positive divisor of 128")

    environments = args.environments.read_text().split()
    if len(environments) != 32 or len(set(environments)) != 32:
        raise ValueError("Expected 32 distinct teacher environments")
    prompts = []
    for environment in environments:
        path = args.eval_dir / f"{environment}_d0.json"
        records = json.loads(path.read_text())
        if len(records) != 128:
            raise ValueError(f"Expected 128 ID prompts in {path}, got {len(records)}")
        for record in records:
            metadata = json.loads(record["metadata"])
            if metadata["environment"] != environment:
                raise ValueError(f"Incorrect environment in {path}")
        prompts.extend(records[::128 // args.prompts_per_environment])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(prompts, indent=1) + "\n")
    os.replace(temporary, args.output)
    print(f"Wrote {len(prompts)} frozen ID prompts across {len(environments)} environments to {args.output}")


if __name__ == "__main__":
    main()
