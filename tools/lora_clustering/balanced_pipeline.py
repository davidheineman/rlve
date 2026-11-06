"""Balanced transfer search followed by disjoint merged-adapter confirmation."""

import csv
import os
import time
from pathlib import Path

import numpy as np

from async_branching.io import read_json, write_json
from lora_clustering.algorithm import members_label, random_partition


def confirmation_assignments(groups, seed):
    """Keep own-task training out of transfer controls and match adapter counts."""
    n = sum(map(len, groups))
    if (
        sorted(i for group in groups for i in group) != list(range(n))
        or len({len(g) for g in groups}) != 1
    ):
        raise ValueError("Expected a balanced partition of all tasks")
    if len(groups) < 2 or len(groups[0]) < 2:
        raise ValueError("Confirmation needs nontrivial clusters")
    random_groups = random_partition(groups, seed)
    rng = np.random.default_rng(seed + 1)
    selected = {i: group for group in groups for i in group}
    random = {i: group for group in random_groups for i in group}
    controls = {
        key: []
        for key in ("selected-loo", "random-loo", "outside-matched", "selected-full")
    }
    for i in range(n):
        group = selected[i]
        outside = [j for j in range(n) if j not in group]
        controls["selected-loo"].append([j for j in group if j != i])
        controls["random-loo"].append([j for j in random[i] if j != i])
        controls["outside-matched"].append(
            sorted(rng.choice(outside, len(group) - 1, replace=False).tolist())
        )
        controls["selected-full"].append(group)
    return {"random_groups": random_groups, "controls": controls, "seed": seed}


def bootstrap_mean(values, seed):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "environment_bootstrap_95ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }


async def run_balanced(experiment, plan, results, model_path, base_url):
    import torch
    from analyze_balanced_lora_clustering import analyze, plot, write_report

    from lora_clustering.evaluation import Evaluator
    from lora_clustering.pipeline import cluster_adapter

    torch.set_num_threads(8)
    evaluator = Evaluator(
        experiment, plan, model_path, base_url, os.environ["LORA_CHECKPOINT_ROOT"]
    )
    directory = experiment / "clustering"
    output = Path(plan["plot_output"])
    balanced = output / "balanced"
    base = await evaluator.vector("selection", "base")
    individual = []
    for i, result in enumerate(results):
        individual.append(
            await evaluator.vector("selection", f"individual-{i:02d}", bundle=result)
        )
    write_json(
        directory / "transfer.json",
        {
            "environments": plan["environments"],
            "base": base,
            "individual": individual,
            "transfer": (np.asarray(individual) - np.asarray(base)[None, :]).tolist(),
        },
    )
    selection_file = directory / "balanced-selected.json"
    if selection_file.exists():
        analysis = read_json(selection_file)
        if (
            analysis["evaluation_input_sha256"]
            != plan["clustering"]["selection_sha256"]
        ):
            raise ValueError("Balanced selection inputs changed")
    else:
        write_json(
            experiment / "pipeline-status.json",
            {"stage": "balanced_search", "time": time.time()},
        )
        analysis = analyze(
            experiment, balanced, plan["clustering"]["balanced_restarts"], plan["seed"]
        )
        write_json(selection_file, analysis)
    balanced.mkdir(parents=True, exist_ok=True)
    write_json(balanced / "results.json", analysis)
    plot(balanced)
    groups = analysis["selected"]["fit"]["groups"]
    specification = confirmation_assignments(groups, plan["seed"] + 20000)
    spec_path = directory / "confirmation-assignment-spec.json"
    if spec_path.exists() and read_json(spec_path) != specification:
        raise ValueError("Frozen confirmation assignments changed")
    write_json(spec_path, specification)
    write_json(
        experiment / "pipeline-status.json",
        {"stage": "merged_confirmation", "time": time.time()},
    )
    base = await evaluator.vector("confirmation", "base")
    measured = {"base": base, "specialist": []}
    for i, result in enumerate(results):
        label, environment = f"individual-{i:02d}", plan["environments"][i]
        cell = evaluator.cached("confirmation", label, environment)
        if cell is None:
            sampler = await evaluator.sampler(label, bundle=result)
            cell = await evaluator.cell("confirmation", label, environment, sampler)
        measured["specialist"].append(cell["accuracy"])
    for label, memberships in specification["controls"].items():
        measured[label] = []
        for i, members in enumerate(memberships):
            key = members_label(members)
            environment = plan["environments"][i]
            cell = evaluator.cached("confirmation", key, environment)
            if cell is None:
                adapter = cluster_adapter(experiment, plan, results, members)
                sampler = await evaluator.sampler(key, adapter=adapter)
                cell = await evaluator.cell("confirmation", key, environment, sampler)
            measured[label].append(cell["accuracy"])
        write_json(directory / "balanced-confirmation-progress.json", measured)
    summary = {
        key: bootstrap_mean(values, plan["seed"]) for key, values in measured.items()
    }
    comparisons = {}
    for label in ("random-loo", "outside-matched", "base"):
        comparisons[f"selected-loo-minus-{label}"] = bootstrap_mean(
            np.asarray(measured["selected-loo"]) - measured[label], plan["seed"]
        )
    confirmation = {
        "environments": plan["environments"],
        "measurements": measured,
        "summary": summary,
        "paired_comparisons": comparisons,
        "assignments": specification,
        "confirmation_inputs_sha256": plan["clustering"]["confirmation_sha256"],
        "interpretation": "LOO controls exclude the target task's own LoRA; selected-full and specialist include own-task training. Adapter composition is not joint training.",
    }
    write_json(directory / "balanced-confirmation.json", confirmation)
    write_json(output / "confirmation.json", confirmation)
    analysis["merged_adapter_confirmation"] = {
        "path": "../confirmation.json",
        "summary": summary,
        "paired_comparisons": comparisons,
    }
    write_json(balanced / "results.json", analysis)
    write_report(analysis, balanced)
    manifest = {
        "selected_k": len(groups),
        "source_experiment": str(experiment),
        "method": "balanced",
        "clusters": [
            {
                "environments": [plan["environments"][i] for i in group],
                "adapter": str(cluster_adapter(experiment, plan, results, group)),
                "specialist_teachers": {
                    plan["environments"][i]: results[i]["hf"] for i in group
                },
            }
            for group in groups
        ],
    }
    write_json(directory / "mopd-clusters.json", manifest)
    write_json(output / "mopd-clusters.json", manifest)
    write_json(
        output / "results.json",
        {"balanced_selection": analysis["selection"], "confirmation": confirmation},
    )
    plot_confirmation(experiment, plan, results, confirmation, output)
    lines = [
        f"# N{plan['size']} balanced LoRA clustering",
        "",
        f"Experiment: `{experiment}`",
        "",
        f"Selected k={len(groups)}, {len(groups[0])} tasks per cluster. [Balanced table and random controls](balanced/README.md).",
        "",
        "Fresh confirmation uses disjoint prompts after assignments are frozen. Leave-one-task-out (LOO) merges exclude the evaluated task's own adapter.",
        "",
        "| Confirmation policy | Macro pass@1 |",
        "|---|---|",
    ]
    lines += [
        f"| {key} | {100 * value['mean']:.2f}% |" for key, value in summary.items()
    ]
    lines += [
        "",
        "| Paired comparison | Pass@1 difference | 95% environment bootstrap interval |",
        "|---|---|---|",
    ]
    for key, value in comparisons.items():
        lo, hi = value["environment_bootstrap_95ci"]
        lines.append(
            f"| {key} | {100 * value['mean']:+.2f} pp | [{100 * lo:+.2f}, {100 * hi:+.2f}] pp |"
        )
    lines += [
        "",
        "These intervals resample environments and do not certify every individual group. Full merges include own-task training; compare LOO controls for transfer. Merged adapters are a proxy for joint training.",
        "",
        "[Cluster adapters for a subsequent MOPD run](mopd-clusters.json).",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    return {"selection": {"selected": {"k": len(groups)}}, "confirmation": confirmation}


def plot_confirmation(experiment, plan, results, confirmation, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = confirmation["summary"]
    labels = list(summary)
    values = np.array([summary[key]["mean"] for key in labels]) * 100
    intervals = (
        np.array([summary[key]["environment_bootstrap_95ci"] for key in labels]) * 100
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, values)
    ax.errorbar(
        labels,
        values,
        yerr=np.maximum(
            np.stack([values - intervals[:, 0], intervals[:, 1] - values]), 0
        ),
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.set_ylabel("Fresh confirmation macro pass@1 (%)")
    ax.set_title("LOO excludes own-task LoRA; full merges include it")
    for ext in ("png", "pdf"):
        fig.savefig(
            output / f"confirmation_performance.{ext}", dpi=180, bbox_inches="tight"
        )
    plt.close(fig)
    fig, axes = plt.subplots(
        int(np.ceil(len(results) / 8)),
        8,
        figsize=(22, 2.5 * np.ceil(len(results) / 8)),
        squeeze=False,
    )
    for ax, teacher in zip(axes.flat, plan["teachers"]):
        directory = experiment / "teacher" / teacher["source_label"]
        points = {}
        path = directory / "metrics.jsonl"
        if path.exists():
            import json

            for line in path.read_text().splitlines():
                row = json.loads(line)
                if "david/eval/val_envs/pass@1" in row:
                    points[row["david/rollouts"]] = row["david/eval/val_envs/pass@1"]
        x = sorted(points)
        ax.plot(x, [100 * points[t] for t in x])
        ax.axhline(98, color="gray", ls="--", lw=0.5)
        ax.set(xlim=(0, plan["teacher_rollouts"]), ylim=(0, 100))
        ax.set_title(teacher["environment"], fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"teacher_curves.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    with (output / "teacher_summary.csv").open("w") as stream:
        fields = [
            "environment",
            "training_rollouts",
            "accuracy",
            "stop_reason",
            "wandb_url",
        ]
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows([[row[key] for key in fields] for row in results])
