#!/usr/bin/env python3
"""Show live convergence from the local fast-teacher worker logs."""

import argparse
import re
from pathlib import Path


EVAL = re.compile(r"eval (\d+): \{'eval/val_envs': ([0-9.]+)")
STEP = re.compile(r"david (\d+): \{'david/perf/step_time': ([0-9.]+)")
STOP = re.compile(r"Early stop at update (\d+):")


def summarize(log):
    evaluations = [(int(step), float(accuracy)) for step, accuracy in EVAL.findall(log)]
    steps = [(int(step), float(seconds)) for step, seconds in STEP.findall(log)]
    stop = STOP.search(log)
    first_95 = next((step + 1 for step, acc in evaluations if step > 0 and acc >= .95), None)
    return {
        "initial": evaluations[0][1] if evaluations else None,
        "best": max((acc for _, acc in evaluations), default=None),
        "latest": evaluations[-1][1] if evaluations else None,
        "updates": max((step for step, _ in steps), default=-1) + 1,
        "first_95": first_95,
        "stopped": int(stop.group(1)) if stop else None,
        "train_seconds": sum(seconds for _, seconds in steps),
        "error": "Traceback (most recent call last)" in log or "CUDA out of memory" in log,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("/nlp/scr/dhei/runs"))
    args = parser.parse_args()
    print("variant\tenvironment\tupdates\tinitial\tlatest\tbest\tfirst95\tstopped\ttrain_min\terror")
    for directory in sorted(args.runs_dir.glob("fast-r1-*-20*")):
        variant = directory.name.removeprefix("fast-r1-").rsplit("-20", 1)[0]
        for path in sorted(directory.glob("[0-9][0-9]-*.log")):
            env = path.stem.split("-", 1)[1]
            row = summarize(path.read_text(errors="replace"))
            def fmt(x):
                return "" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)
            print("\t".join(map(fmt, [
                variant, env, row["updates"], row["initial"], row["latest"], row["best"],
                row["first_95"], row["stopped"], row["train_seconds"] / 60, int(row["error"]),
            ])))


if __name__ == "__main__":
    main()
