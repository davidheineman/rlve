"""Check GPU probe advantages against indexed, scored response exports."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from modular_rl.prepare import write_json

from .inspect_rollouts import read_events


def verify(experiment):
    plan = json.loads((experiment / "plan.json").read_text())
    model, = plan.get("models", ["Qwen3-0.6B"])
    results = []
    for task in plan["tasks"]:
        for variant in plan["variants"]:
            directory = experiment / task / variant / f"{model}-{task}"
            progress = read_events(directory / "progress.jsonl")
            updates = max((e["completed_updates"] for e in progress if e["phase"] == "train"), default=0)
            probes = {p["rollout_id"]: p for p in read_events(directory / "update-probes.jsonl")
                      if p["rollout_id"] < updates}
            checked = 0
            for step, probe in probes.items():
                groups = defaultdict(list)
                path = directory / "rollouts/train" / f"{step:04d}.jsonl"
                for row in read_events(path):
                    groups[json.loads(row["metadata_json"])["key"]].append(row)
                expected = {}
                for group in groups.values():
                    if len(group) != 8:
                        raise ValueError(f"Expected eight responses per prompt: {path}")
                    rewards = [float(json.loads(row["reward_json"])["reward"]) for row in group]
                    mean, std = statistics.mean(rewards), statistics.stdev(rewards)
                    for row, reward in zip(group, rewards, strict=True):
                        prefix = json.loads(row["metadata_json"]).get("_modular_v2_forcing", {}).get("prefix_tokens", 0)
                        expected[row["row_index"]] = (reward, row["response_tokens"] - prefix,
                                                       (reward - mean) / (std + 1e-6))
                if {p["sample"] for p in probe["samples"]} != set(expected):
                    raise ValueError(f"Probe/export sample index sets differ: {path}")
                for sample in probe["samples"]:
                    reward, tokens, advantage = expected[sample["sample"]]
                    if (abs(sample["reward"] - reward) > 1e-6 or sample["tokens"] != tokens
                            or abs(sample["advantage"] - advantage) > 2e-5):
                        raise ValueError(f"GPU reward/length/advantage alignment mismatch: {path}, {sample}")
                    checked += 1
            if (directory / "completed.json").exists() and updates and not checked:
                raise ValueError(f"Completed diagnostic run lacks update probes: {directory}")
            results.append({"task": task, "variant": variant, "updates_checked": len(probes),
                            "indexed_samples_checked": checked, "passed": True if checked else None})
    value = {"experiment": str(experiment), "runs": results}
    write_json(experiment / "probe-alignment.json", value)
    print(json.dumps(value), flush=True)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", type=Path, nargs="+")
    for directory in parser.parse_args().experiments:
        verify(directory)
