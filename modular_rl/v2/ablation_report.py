"""Inspect frozen short trials, with paired problem-level validation intervals."""

import argparse
import ast
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from modular_rl.prepare import write_json

from .inspect_rollouts import annotate, read_events


def average(values):
    return statistics.mean(values) if values else None


def paired_interval(differences, family_size=1):
    if not isinstance(family_size, int) or family_size < 1:
        raise ValueError("The comparison family must have a positive integer size")
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Use finite paired problem-level differences")
    rng = np.random.default_rng(42)
    draws = values[rng.integers(len(values), size=(20000, len(values)))].mean(axis=1)
    result = {"difference": float(values.mean()), "paired_problem_95_interval":
              np.quantile(draws, [0.025, 0.975]).tolist(), "problems": len(values)}
    if family_size > 1:
        tail = .025 / family_size
        result.update(primary_family_size=family_size, comparison_confidence_level=1 - 2 * tail,
                      familywise_confidence_level=.95,
                      paired_problem_adjusted_interval=np.quantile(draws, [tail, 1 - tail]).tolist())
    return result


def validation_panel(path, prompts=32):
    problems = defaultdict(list)
    for row in read_events(path):
        metadata = json.loads(row["metadata_json"])
        problems[metadata["key"]].append(float(row["correct"]))
    if problems and (len(problems) != prompts or any(len(v) != 8 for v in problems.values())):
        raise ValueError(f"Incomplete validation panel: {path}")
    return {key: average(value) for key, value in problems.items()}


def training_metric_groups(log, passes, updates):
    groups = {}
    for attempt in log.split("Resumable run "):
        current = defaultdict(list)
        for match in re.finditer(r"step (\d+): (?=\{)", attempt):
            # Ray workers can concatenate stdout records on one physical line.
            # Parse only the first complete dictionary, while respecting nested
            # dictionaries and braces inside quoted strings through literal_eval.
            newline = attempt.find("\n", match.end())
            line = attempt[match.end():newline if newline >= 0 else len(attempt)]
            for end in re.finditer("}", line):
                try:
                    row = ast.literal_eval(line[:end.end()])
                except (SyntaxError, ValueError):
                    continue
                break
            else:
                raise ValueError(f"Malformed training metric dictionary at step{match[1]}")
            step = int(match[1])
            rollout = step if row.get("train/optimizer_update_skipped") else step // passes
            if rollout < updates:
                current[rollout].append(row)
        groups.update(current)
    return groups


def inspect_run(task, variant, directory):
    experiment = directory.parents[2]
    plan = json.loads((experiment / "plan.json").read_text())
    prompts = plan.get("prompts_per_update", 1)
    cap, passes = plan.get("max_sampled_tokens", 2048), plan.get("objective_passes", 1)
    selected = {r["metadata"]["key"]: r["metadata"]
                for r in read_events(experiment / "selection" / task / "train.jsonl")}
    val_keys = {r["metadata"]["key"] for r in read_events(experiment / "selection" / task / "val.jsonl")}
    progress = read_events(directory / "progress.jsonl")
    updates = max((e["completed_updates"] for e in progress if e["phase"] == "train"), default=0)
    events = sorted({e["completed_updates"]: e for e in progress if e["phase"] == "eval"}.values(),
                    key=lambda e: e["completed_updates"])
    groups, panels = [], []
    prompt_scores = defaultdict(list)
    for path in sorted((directory / "rollouts/train").glob("*.jsonl")):
        if int(path.stem) >= updates:
            continue
        samples = [annotate(row, path, task + "-" + variant) for row in read_events(path)]
        by_prompt = defaultdict(list)
        for sample in samples:
            target = selected.get(sample["dataset_key"])
            if (target is None or sample["dataset_key"] in val_keys
                    or sample["target"] != target["target"] or sample["prompt"] != target["messages"]
                    or sample["rollout_id"] != int(path.stem)
                    or not 0 < sample["sampled_tokens"] <= cap):
                raise ValueError(f"Training prompt, target, split, index or token cap mismatch: {path}")
            by_prompt[sample["dataset_key"]].append(sample)
        if len(by_prompt) != prompts or any(len(g) != 8 for g in by_prompt.values()):
            raise ValueError(f"Training must retain {prompts} distinct prompts with eight responses each: {path}")
        for key, responses in by_prompt.items():
            prompt_scores[key].append(average([s["accuracy"] for s in responses]))
        groups.append({
            "rollout_id": int(path.stem), "responses": len(samples),
            "reward": average([s["reward"] for s in samples]),
            "accuracy": average([s["accuracy"] for s in samples]),
            "forced": average([s["forced"] for s in samples]),
            "tokens": average([s["sampled_tokens"] for s in samples]),
            "mixed_groups": sum(len({s["reward"] for s in g}) > 1 for g in by_prompt.values()),
            "prompt_groups": len(by_prompt),
        })
    if [g["rollout_id"] for g in groups] != list(range(updates)):
        raise ValueError(f"Committed training exports are incomplete: {directory}")
    baseline_panel = None
    for event in events:
        step = event["completed_updates"]
        path = directory / "rollouts/eval" / f"{max(0, step - 1):04d}.jsonl"
        panel = validation_panel(path)
        if baseline_panel is None:
            baseline_panel = panel
        if panel and set(panel) != set(baseline_panel):
            raise ValueError("Validation problems changed within a run")
        raw = read_events(path)
        metadata = [json.loads(row["metadata_json"]) for row in raw]
        behavior = {
            "finished_correct_fraction": average([
                bool(row["correct"]) and row["status"] == "completed" for row in raw
            ]),
            "truncated_correct_fraction": average([
                bool(row["correct"]) and row["status"] == "truncated" for row in raw
            ]),
            "mean_alternatively_count": average([row["response"].lower().count("alternatively") for row in raw]),
            "over_10_alternatively_fraction": average([row["response"].lower().count("alternatively") > 10 for row in raw]),
            "forced_fraction": average(["_modular_v2_forcing" in meta for meta in metadata]),
            "truncated_fraction": average([row["status"] == "truncated" for row in raw]),
            "forced_after_think_closed_fraction": average([
                "</think>" in row["response"][:meta["_modular_v2_forcing"]["answer_char_start"]]
                for row, meta in zip(raw, metadata, strict=True) if "_modular_v2_forcing" in meta
            ]),
            "mean_response_tokens": average([row["response_tokens"] for row in raw]),
        }
        panels.append({"updates": step, "pass@1": event["accuracy"],
                       "pass@8": average([score > 0 for score in panel.values()]),
                       "problems": panel, "behavior": behavior,
                       "vs_initial": paired_interval([panel[k] - baseline_panel[k] for k in panel])
                       if panel else None})
    log = (directory / "train.log").read_text(errors="replace") if (directory / "train.log").exists() else ""
    metric_groups = training_metric_groups(log, passes, updates)
    metric_rows = {step: rows[-1] for step, rows in metric_groups.items()}
    metrics = [row for rows in metric_groups.values() for row in rows]
    largest_gradient = max(((step, row.get("train/grad_norm", 0))
                            for step, rows in metric_groups.items() for row in rows),
                           key=lambda item: item[1], default=None)
    importance = []
    for group in groups:
        row = metric_rows.get(group["rollout_id"], {})
        if "train/tis" not in row:
            continue
        mass = (row.get("train/normalization_mass", group["tokens"] / cap)
                if variant.startswith("fixed") else 1)
        if mass <= 0:
            raise ValueError("Importance diagnostic normalization must be positive")
        importance.append({key: row[f"train/{key}"] / mass
                           for key in ("tis", "ois", "tis_clipfrac")})
    probes = list({p["rollout_id"]: p for p in read_events(directory / "update-probes.jsonl")
                   if p["rollout_id"] < updates}.values())
    measured = [s for p in probes for s in p["samples"]]
    active_probes = [p for p in probes if any(abs(s["advantage"]) > 1e-8 for s in p["samples"])]
    def objective_gain(p):
        return average([s["advantage_delta_logp"] * (s["tokens"] / cap if variant.startswith("fixed") else 1)
                        for s in p["samples"]])
    inactive_samples = [s for p in probes if all(abs(s["advantage"]) <= 1e-8 for s in p["samples"])
                        for s in p["samples"]]
    probe_summary = {
        "updates_measured": len(probes),
        "active_updates_measured": len(active_probes),
        "positive_advantage_delta_fraction": average([
            objective_gain(p) > 0 for p in active_probes
        ]),
        "objective_aligned_advantage_delta_logp": average([objective_gain(p) for p in probes]),
        "zero_advantage_mean_abs_change": average([s["mean_abs_delta_logp"] for s in inactive_samples]),
        "advantage_delta_logp": average([s["advantage_delta_logp"] for s in measured]),
        "mean_abs_delta_logp": average([s["mean_abs_delta_logp"] for s in measured]),
        "sampled_k3_old_to_new": average([s["sampled_k3_old_to_new"] for s in measured]),
        "rewarded_mean_delta_logp": average([s["mean_delta_logp"] for s in measured if s["reward"] == 1]),
        "failed_mean_delta_logp": average([s["mean_delta_logp"] for s in measured if s["reward"] == 0]),
    }
    windows = []
    for start in range(0, updates, 48):
        batch = [g for g in groups if start <= g["rollout_id"] < start + 48]
        windows.append({"first_update": start + 1, "updates": len(batch),
                        "mixed_group_fraction": sum(g["mixed_groups"] for g in batch) / sum(g["prompt_groups"] for g in batch) if batch else None,
                        **{key: average([g[key] for g in batch]) for key in
                           ("reward", "accuracy", "forced", "tokens", "mixed_groups")}})
    run_id = directory / "wandb_run_id.txt"
    repeated = [values for values in prompt_scores.values() if len(values) > 1]
    decisions = {d["rollout_id"]: d for d in read_events(directory / "optimizer-decisions.jsonl")
                 if d["rollout_id"] < updates}
    return {
        "model": plan.get("models", ["Qwen3-0.6B"])[0],
        "task": task, "variant": variant, "directory": str(directory), "updates": updates,
        "max_sampled_tokens": cap, "objective_passes": passes,
        "completed": (directory / "completed.json").exists(),
        "responses": sum(g["responses"] for g in groups),
        "training_integrity_checks": {"passed": True, "prompt_groups_checked": sum(g["prompt_groups"] for g in groups),
                                      "responses_checked": sum(g["responses"] for g in groups),
                                      "expected_prompts_per_iteration": prompts, "samples_per_prompt": 8},
        "optimizer_updates": sum(d["optimizer_update_applied"] for d in decisions.values()) if decisions else updates,
        "optimizer_updates_skipped": sum(not d["optimizer_update_applied"] for d in decisions.values()),
        "optimizer_steps_applied": sum(d.get("optimizer_steps", int(d["optimizer_update_applied"]))
                                       for d in decisions.values()),
        "importance_diagnostics": {
            "reduction": "sampled-token-mean" if variant.startswith("fixed") else "response-mean",
            "batches": len(importance),
            **{key: {"mean": average([row[key] for row in importance]),
                     "min": min((row[key] for row in importance), default=None),
                     "max": max((row[key] for row in importance), default=None)}
               for key in ("tis", "ois", "tis_clipfrac")},
        },
        "validation": panels, "training_windows": windows, "post_update": probe_summary,
        "zero_gradient_updates": sum(m.get("train/grad_norm", 0) == 0 for m in metrics),
        "largest_gradient_update": {"rollout_id": largest_gradient[0], "gradient_norm": largest_gradient[1]}
        if largest_gradient is not None else None,
        "repeated_training_problems": {
            "problems": len(repeated), "first_success": average([scores[0] for scores in repeated]),
            "last_success": average([scores[-1] for scores in repeated]),
            "paired_change": paired_interval([scores[-1] - scores[0] for scores in repeated]) if repeated else None,
        },
        "wandb_id": run_id.read_text().strip() if run_id.exists() else None,
    }


def report(experiment):
    plan = json.loads((experiment / "plan.json").read_text())
    model, = plan.get("models", ["Qwen3-0.6B"])
    runs = [inspect_run(task, variant, experiment / task / variant / f"{model}-{task}")
            for task in plan["tasks"] for variant in plan["variants"]]
    comparisons = []
    for task in plan["tasks"]:
        by_variant = {r["variant"]: r for r in runs if r["task"] == task}
        if "baseline" not in by_variant or not by_variant["baseline"]["validation"]:
            continue
        control = by_variant["baseline"]["validation"][-1]
        for variant, run in by_variant.items():
            if variant == "baseline" or not run["validation"]:
                continue
            candidate = run["validation"][-1]
            if candidate["updates"] != control["updates"] or not candidate["problems"]:
                continue
            comparisons.append({"task": task, "candidate": variant, "updates": candidate["updates"],
                                **paired_interval([candidate["problems"][k] - control["problems"][k]
                                                   for k in candidate["problems"]]),
                                "change_vs_baseline": paired_interval([
                                    (candidate["problems"][k] - run["validation"][0]["problems"][k])
                                    - (control["problems"][k] - by_variant["baseline"]["validation"][0]["problems"][k])
                                    for k in candidate["problems"]
                                ])})
    value = {"experiment": str(experiment), "runs": runs, "comparisons": comparisons}
    write_json(experiment / "analysis.json", value)
    for run in runs:
        print(json.dumps({k: run[k] for k in ("task", "variant", "updates", "completed", "post_update")}))
    for comparison in comparisons:
        print(json.dumps(comparison))
    return value


def plot(value):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(rows=2, cols=2, subplot_titles=(
        "Math: held-out pass@1", "Code: held-out pass@1",
        "Math: change from own initial policy", "Code: change from own initial policy",
    ))
    colors = {"baseline": "#707070", "fixed": "#0072B2", "fixed-lr-low": "#009E73", "fixed-kl": "#D55E00"}
    plan = json.loads((Path(value["experiment"]) / "plan.json").read_text())
    batch = plan.get("prompts_per_update", 1) * 8
    for run in value["runs"]:
        col = 1 if run["task"] == "math" else 2
        x = [e["updates"] * batch for e in run["validation"]]
        y = [100 * e["pass@1"] for e in run["validation"]]
        color = colors.get(run["variant"], "#CC79A7")
        common = {"name": run["variant"], "legendgroup": run["variant"],
                  "line": {"color": color}, "mode": "lines+markers"}
        figure.add_trace(go.Scatter(x=x, y=y, showlegend=col == 1, **common), row=1, col=col)
        changes = [e["vs_initial"] for e in run["validation"]]
        if changes and all(changes):
            means = [100 * e["difference"] for e in changes]
            figure.add_trace(go.Scatter(
                x=x, y=means, showlegend=False,
                error_y={"type": "data", "symmetric": False,
                         "array": [100 * e["paired_problem_95_interval"][1] - m for e, m in zip(changes, means, strict=True)],
                         "arrayminus": [m - 100 * e["paired_problem_95_interval"][0] for e, m in zip(changes, means, strict=True)]},
                **common,
            ), row=2, col=col)
    figure.update_xaxes(title_text="Training responses")
    figure.update_yaxes(title_text="Accuracy (%)", row=1)
    figure.update_yaxes(title_text="Change (percentage points)", row=2)
    figure.update_layout(template="plotly_white", height=850, width=1200,
                         title=f"Fresh {value['runs'][0]['model']} / 2k controlled trials; 32 fixed validation problems")
    figure.write_html(str(Path(value["experiment"]) / "ablation-curves.html"), include_plotlyjs=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    value = report(args.experiment)
    if args.plot:
        plot(value)
