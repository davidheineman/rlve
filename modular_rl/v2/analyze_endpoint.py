"""Compare saved policies on the same independent, problem-level holdout panel."""

import argparse
import json
from pathlib import Path

from modular_rl.prepare import write_json

from .ablation_report import average, paired_interval, validation_panel
from .inspect_rollouts import read_events
from .snapshot import MANIFEST, validate_snapshot


def analyze(experiment):
    plan = json.loads((experiment / "plan.json").read_text())
    jobs = json.loads((experiment / "jobs.json").read_text())
    model = plan.get("model", "Qwen/Qwen3-0.6B").split("/")[-1]
    results = []
    for job in jobs:
        task = job["task"]
        directory = Path(job["output"]) / f"{model}-{task}"
        result = json.loads((directory / "evaluated.json").read_text())
        if not result["evaluation_only"] or result["optimizer_updates"] != 0:
            raise ValueError("An endpoint evaluation took training updates")
        scores = validation_panel(directory / "rollouts/eval/0000.jsonl", prompts=plan["prompts"][task])
        if not scores or abs(average(list(scores.values())) - result["pass@1"]) > 1e-9:
            raise ValueError("Exported grades disagree with the reported endpoint score")
        samples = read_events(directory / "rollouts/eval/0000.jsonl")
        behavior = {
            "truncated_fraction": average([s["status"] == "truncated" for s in samples]),
            "finished_correct_fraction": average([
                bool(s["correct"]) and s["status"] == "completed" for s in samples
            ]),
            "over_10_alternatively_fraction": average([
                s["response"].lower().count("alternatively") > 10 for s in samples
            ]),
        }
        checkpoint = Path(job["checkpoint"])
        completion = checkpoint.parent / "completed.json"
        if (checkpoint / MANIFEST).exists():
            budget = validate_snapshot(checkpoint)["train_rollouts"]
        else:
            budget = json.loads(completion.read_text())["train_rollouts"] if completion.exists() else 0
        results.append({**job, **result, "policy": job["name"], "directory": str(directory),
                        "training_responses": budget, "problems": scores, "behavior": behavior,
                        "pass@8": average([s > 0 for s in scores.values()])})
    comparisons = []
    for task in sorted({r["task"] for r in results}):
        policies = {r["policy"]: r for r in results if r["task"] == task}
        family_size = plan.get("primary_family_size", 1)
        if family_size > 1 and (len(policies) != 2 or "base" not in policies):
            raise ValueError("Adjusted primary confirmation requires one candidate and its Base baseline per task")
        primary_baseline = "base_a6000" if "base_a6000" in policies else "base"
        for reference in (primary_baseline, "native"):
            if reference not in policies:
                continue
            for name, candidate in policies.items():
                if name == reference or (reference == "native" and name in ("base", "base_a6000")):
                    continue
                control = policies[reference]
                if set(control["problems"]) != set(candidate["problems"]):
                    raise ValueError("Endpoint policies saw different problems")
                comparisons.append({"task": task, "candidate": name, "reference": reference,
                                    **paired_interval([candidate["problems"][key] - control["problems"][key]
                                                       for key in candidate["problems"]], family_size=family_size)})
    value = {"experiment": str(experiment), "policies": results, "comparisons": comparisons}
    write_json(experiment / "analysis.json", value)
    for result in results:
        print(json.dumps({key: result[key] for key in ("task", "policy", "training_responses", "pass@1", "pass@8")}))
    for comparison in comparisons:
        print(json.dumps(comparison))
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    analyze(parser.parse_args().experiment)
