"""Recover final weights after a completed trainer's shell wrapper failed."""

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def validate_terminal_checkpoint(run, update_cap):
    checkpoint = run / "final-ckpt"
    iteration = int((checkpoint / "latest_checkpointed_iteration.txt").read_text())
    if not 0 <= iteration < update_cap:
        raise RuntimeError(f"Invalid terminal checkpoint iteration: {iteration}")
    model = checkpoint / f"iter_{iteration:07d}"
    required = [model / "common.pt", model / ".metadata"]
    shards = list(model.glob("*.distcp"))
    if not shards or any(not path.is_file() or path.stat().st_size == 0 for path in required + shards):
        raise RuntimeError(f"Incomplete final checkpoint: {model}")
    trained = False
    stopped = False
    with (run / "train.log").open(errors="replace") as log:
        for line in log:
            if re.search(rf"\bdavid {iteration}:", line):
                trained = True
            match = re.search(rf"Early stop at update {iteration + 1}: reason=accuracy, validation=([0-9.]+)", line)
            if match and float(match.group(1)) >= .98:
                stopped = True
    if not trained or (iteration + 1 != update_cap and not stopped):
        raise RuntimeError("Checkpoint lacks evidence of a terminal trained update")
    return iteration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--origin-hf-dir", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    runs = json.loads((args.experiment / "run_plan.json").read_text())["runs"]
    spec = next(run for run in runs if run["label"] == args.label)
    run = args.experiment / spec["kind"] / spec["label"]
    with (run / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (run / "completed").exists():
            iteration = validate_terminal_checkpoint(run, spec["updates"])
            temporary = run / "completed.recovered.tmp"
            temporary.write_text(f"final_model=final-ckpt/iter_{iteration:07d}\nrecovered_terminal_update={iteration + 1}\n")
            os.replace(temporary, run / "completed")
            print(f"Validated terminal trained update {iteration + 1}; recovering export", flush=True)
        subprocess.run([
            sys.executable, "tools/export_data_scaling_checkpoint.py",
            "--run-dir", str(run), "--origin-hf-dir", str(args.origin_hf_dir),
        ], check=True)


if __name__ == "__main__":
    main()
