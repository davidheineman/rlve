"""Submit the established round-2 setup and attach bounded CPU recovery."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-experiment", type=Path, required=True)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--reference-round2-experiment", type=Path)
    parser.add_argument("--generation-length", type=int, choices=(1024, 2048), default=2048)
    parser.add_argument("--max-retries", type=int, default=2)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.max_retries < 0:
        parser.error("max-retries must be nonnegative")
    if args.reference_round2_experiment and args.generation_length != 2048:
        parser.error("reference-round2-experiment is supported by the 2k comparison")
    script = "submit_data_scaling_multiround_2k.py" if args.generation_length == 2048 else "submit_data_scaling_multiround.py"
    experiment = (args.experiment or args.parent_experiment.parent / (
        f"data-scaling-multiround-r2-{args.generation_length}-resilient-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    command = [sys.executable, str(ROOT / "tools" / script), "--parent-experiment", str(args.parent_experiment),
               "--experiment", str(experiment), "--submit" if args.submit else "--check-only"]
    if args.reference_round2_experiment:
        command.extend(["--reference-round2-experiment", str(args.reference_round2_experiment)])
    subprocess.run(command, check=True, cwd=ROOT)
    if args.submit:
        subprocess.run([sys.executable, str(ROOT / "tools/watch_data_scaling_multiround.py"),
                        "--experiment", str(experiment), "--max-retries", str(args.max_retries), "--submit"],
                       check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
