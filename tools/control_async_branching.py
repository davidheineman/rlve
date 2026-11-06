"""Change a live elastic experiment's GPU cap without changing its simulation."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from async_branching.io import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--max-gpus", type=int, choices=range(4, 17))
    action.add_argument("--stop-workers", action="store_true",
                        help="Cancel GPU workers; cancel the CPU coordinator first to prevent new submissions")
    args = parser.parse_args()
    plan = read_json(args.directory / "plan.json")
    if plan.get("backend") != "slurm":
        parser.error("This is not an elastic Slurm experiment")
    if args.stop_workers:
        from async_branching.slurm import SlurmBackend
        SlurmBackend.stop_workers(args.directory)
        print("GPU workers stopped; committed state/checkpoints retained")
    else:
        write_json(args.directory / "resources.json", {"max_gpus": args.max_gpus})
        print(f"GPU cap set to {args.max_gpus}, including two student GPUs; running jobs drain before reductions take effect")


if __name__ == "__main__":
    main()
