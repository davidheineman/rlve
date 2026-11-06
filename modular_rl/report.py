"""Require completed, nonzero-gradient pilots and write reviewable local evidence."""

import argparse
import ast
import json
import re
from pathlib import Path

from .prepare import write_json


def report(root):
    rows = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / "train.log").exists():
            continue
        log = (directory / "train.log").read_text(errors="replace")
        # Slime emits dictionaries through Ray's prefixed logging.
        metrics = []
        for line in log.splitlines():
            if "'david/rollouts':" in line and "david " in line:
                metrics.append(
                    ast.literal_eval(line[line.index("{") : line.rindex("}") + 1])
                )
        signals = [
            json.loads(line)
            for line in (directory / "signal.jsonl").read_text().splitlines()
        ]
        gradients = [
            float(x) for x in re.findall(r"'train/grad_norm':\s*([\d.eE+-]+)", log)
        ]
        evaluations = []
        for line in log.splitlines():
            if "'david/eval/val_envs/pass@8':" in line and "eval " in line:
                evaluations.append(
                    ast.literal_eval(line[line.index("{") : line.rindex("}") + 1])
                )
        passed = (
            (directory / "completed").exists()
            and bool(metrics)
            and any(s["david/train/signal_groups"] > 0 for s in signals)
            and any(g > 0 for g in gradients)
            and len(evaluations) >= 2
        )
        rows.append(
            {
                "run": directory.name,
                "passed": passed,
                "steps": len(metrics),
                "train_rollouts": metrics[-1]["david/rollouts"] if metrics else 0,
                "max_grad_norm": max(gradients, default=0),
                "signal_groups": sum(s["david/train/signal_groups"] for s in signals),
                "initial_eval": evaluations[0] if evaluations else {},
                "final_eval": evaluations[-1] if evaluations else {},
                "wandb_run_id": (directory / "wandb_run_id.txt").read_text().strip()
                if (directory / "wandb_run_id.txt").exists()
                else None,
            }
        )
    value = {"passed": bool(rows) and all(r["passed"] for r in rows), "runs": rows}
    write_json(root / "summary.json", value)
    print(json.dumps(value, indent=2))
    if not value["passed"]:
        raise RuntimeError(
            "A smoke run has not demonstrated completion, evaluation, and learning signal"
        )
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    report(parser.parse_args().root)
