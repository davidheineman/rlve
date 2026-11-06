"""Validate a completed v2 budget or a target-accuracy early stop."""

import argparse
import json
import os
from pathlib import Path

from modular_rl.prepare import write_json

from .rollout import validation_prompt_count


def report(root):
    results = []
    for directory in sorted(p.parent for p in root.glob('*/completed.json')):
        if not directory.is_dir():
            continue
        completed = json.loads((directory / "completed.json").read_text())
        events = [
            json.loads(line)
            for line in (directory / "progress.jsonl").read_text().splitlines()
        ]
        evaluations = {
            e["completed_updates"]: e for e in events if e["phase"] == "eval"
        }
        count, maximum = completed["train_rollouts"], completed["max_train_rollouts"]
        if not completed["completed"] or count > maximum:
            raise RuntimeError("Invalid response budget completion")
        if completed["reason"] == "validation_accuracy":
            if evaluations[completed["updates"]]["accuracy"] < 0.98:
                raise RuntimeError("Early stop is below the requested pass@1 target")
        elif count != maximum:
            raise RuntimeError("Run ended before exhausting its budget")
        generation = [
            json.loads(line)
            for line in (directory / "generation.jsonl").read_text().splitlines()
        ]
        cap = completed.get("sampled_token_budget", 1024)
        train_batch = completed.get("prompts_per_update", 1) * completed.get("samples_per_prompt", 8)
        validation_batch = validation_prompt_count(os.environ.get('MODULAR_RL_TASKS', 'math')) * completed.get("eval_samples_per_prompt", 8)
        if cap <= 0 or any(
            e["max_sampled_tokens"] > cap or e.get("sampled_token_budget", 1024) != cap
            for e in generation
        ):
            raise RuntimeError("Sampled-token cap exceeded")
        if any(
            e["responses"] != (validation_batch if e["phase"] == "eval" else train_batch) for e in generation
        ):
            raise RuntimeError("Incorrect train or validation batch size")
        training = {e["rollout_id"]: e for e in generation if e["phase"] == "train"}
        if (
            sum(
                e["responses"]
                for key, e in training.items()
                if key < completed["updates"]
            )
            != count
        ):
            raise RuntimeError(
                "Generated responses disagree with completed optimizer updates"
            )
        results.append(
            {
                "run": directory.name,
                **completed,
                "validation_evaluations": len(evaluations),
                "final_pass@1": evaluations[completed["updates"]]["accuracy"],
                "wandb_run_id": (directory / "wandb_run_id.txt").read_text().strip(),
            }
        )
    if not results:
        raise RuntimeError("No completed v2 runs")
    value = {"passed": True, "runs": results}
    write_json(root / "summary.json", value)
    print(json.dumps(value, indent=2))
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    report(parser.parse_args().root)
