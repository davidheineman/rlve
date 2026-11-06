"""Publish independent endpoint scores, paired uncertainty and complete responses."""

import argparse
import json
import re
import tarfile
import tempfile
from pathlib import Path

from modular_rl.prepare import write_json

from .analyze_endpoint import analyze
from .inspect_rollouts import annotate, read_events


def publish(experiment, review, namespace="endpoint", artifact_name=None):
    import wandb

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", namespace):
        raise ValueError("Use a simple, distinct endpoint namespace")
    plan = json.loads((experiment / "plan.json").read_text())
    model = plan.get("model", "Qwen/Qwen3-0.6B")
    if artifact_name is None:
        model_label = {"Qwen/Qwen3-0.6B": "qwen06b", "Qwen/Qwen3-1.7B": "qwen17b",
                       "Qwen/Qwen3-0.6B-Base": "qwen06b-base", "Qwen/Qwen3-1.7B-Base": "qwen17b-base"}[model]
        cap = plan.get("max_sampled_tokens", 2048)
        artifact_name = f"{model_label}-{cap}-independent-{namespace}"
    value = analyze(experiment)
    outcomes, comparisons, examples = [], [], []
    by_task = {}
    for policy in value["policies"]:
        by_task.setdefault(policy["task"], []).append(policy)
        outcomes.append([model, policy["task"], policy["policy"], policy["training_responses"],
                         policy["prompts"], policy["samples_per_prompt"], policy["pass@1"], policy["pass@8"],
                         policy["behavior"]["truncated_fraction"], policy["behavior"]["finished_correct_fraction"],
                         policy["behavior"]["over_10_alternatively_fraction"],
                         "https://forge.coreweave.com/wandb/david-heineman/modular-mopd-rl-initial/runs/" + policy["wandb_id"]])
    for item in value["comparisons"]:
        low, high = item["paired_problem_95_interval"]
        adjusted_low, adjusted_high = item.get("paired_problem_adjusted_interval", [low, high])
        comparisons.append([item["task"], item["candidate"], item["reference"],
                            item["difference"], low, high, item["problems"],
                            item.get("comparison_confidence_level", .95), adjusted_low, adjusted_high])
    for task, policies in by_task.items():
        lookup = {p["policy"]: p for p in policies}
        baseline = lookup.get("base_a6000") or lookup.get("base")
        if baseline is None:
            raise ValueError("Publish an explicit, matching baseline alongside endpoint candidates")
        base = baseline["problems"]
        candidate = lookup.get("fixed", policies[-1])["problems"]
        ordered = sorted(base, key=lambda key: candidate[key] - base[key])
        selected = set(ordered[:2] + ordered[-2:] + list(base)[:2])
        for policy in policies:
            path = Path(policy["directory"]) / "rollouts/eval/0000.jsonl"
            for raw in read_events(path):
                row = annotate(raw, path, policy["policy"])
                if row["dataset_key"] in selected:
                    examples.append([task, policy["policy"], row["dataset_id"],
                                     json.dumps(row["prompt"], ensure_ascii=False), row["response"],
                                     row["final_answer"], row["accuracy"], row["forced"],
                                     row["sampled_tokens"], json.dumps(row["target"], ensure_ascii=False)])
    with tempfile.TemporaryDirectory(prefix="endpoint-review-") as local:
        publication = review / ("base-plan-publication.json" if model.endswith("-Base")
                                else "initial-publication.json")
        run = wandb.init(entity="david-heineman", project="modular-mopd-rl-initial",
                         id=json.loads(publication.read_text())["run_id"],
                         resume="must", dir=local, settings=wandb.Settings(mode="online", init_timeout=120))
        run.log({
            f"{namespace}/outcomes": wandb.Table(columns=["model", "task", "policy", "training_responses", "prompts",
                                                       "samples_per_prompt", "pass@1", "pass@8", "truncated_fraction",
                                                       "finished_correct_fraction", "over_10_alternatively_fraction",
                                                       "source_run_url"], data=outcomes),
            f"{namespace}/paired_comparisons": wandb.Table(columns=["task", "candidate", "reference", "difference",
                                                                "lower_95", "upper_95", "problems",
                                                                "adjusted_comparison_confidence", "adjusted_lower",
                                                                "adjusted_upper"], data=comparisons),
            f"{namespace}/complete_selected_responses": wandb.Table(columns=["task", "policy", "problem_id", "prompt",
                                                                         "full_response", "final_answer", "accuracy", "forced",
                                                                         "sampled_tokens", "target"], data=examples),
        })
        artifact = wandb.Artifact(artifact_name, type="rl-diagnostics",
                                  metadata={"evaluation_only": True, "model": model, "namespace": namespace,
                                            "panel_sizes": plan["prompts"], "panels": list(by_task),
                                            "optimizer_updates": 0})
        for name in ("analysis.json", "plan.json", "jobs.json", "lane-adjustments.jsonl", "runtime-adjustments.jsonl",
                     "supplemental-baseline-plan.json", "jobs-original-three.json", "confirmation-protocol.json",
                     "development-panel-protocol.json"):
            if (experiment / name).exists():
                artifact.add_file(str(experiment / name), name=name)
        artifact.add_dir(str(experiment / "selection"), name="selection")
        archive_path = Path(local) / "complete-endpoint-responses.tar.gz"
        with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
            for policy in value["policies"]:
                directory = Path(policy["directory"])
                prefix = Path(policy["task"]) / policy["policy"]
                archive.add(directory / "rollouts/eval", arcname=str(prefix / "rollouts/eval"))
                for name in ("evaluated.json", "generation.jsonl", "train.log"):
                    archive.add(directory / name, arcname=str(prefix / name))
        artifact.add_file(str(archive_path), name=archive_path.name)
        artifact.add_file(str(Path(__file__)), name="publish_endpoint.py")
        logged = run.log_artifact(artifact)
        logged.wait()
        result = {"run_url": run.url, "artifact_url": logged.url, "policies": len(outcomes),
                  "selected_responses": len(examples), "model": model, "namespace": namespace}
        run.finish()
    receipt = "independent-endpoint-publication.json" if namespace == "endpoint" else f"{namespace}-publication.json"
    write_json(review / receipt, result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--namespace", default="endpoint", help="Distinct W&B keys and receipt for each final panel/model")
    parser.add_argument("--artifact-name", help="Override the model-specific diagnostic artifact name")
    args = parser.parse_args()
    publish(args.experiment, args.review, args.namespace, args.artifact_name)
