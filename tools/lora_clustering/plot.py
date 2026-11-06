"""Export real measurements, paired uncertainty estimates, plots and cluster inputs."""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np

from async_branching.io import read_json, write_json
from lora_clustering.algorithm import members_label


def bootstrap_interval(values, seed=1234):
    """Two-level bootstrap of tasks and their paired prompt measurements."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n, prompts = values.shape
    tasks = rng.integers(n, size=(2000, n))
    draws = rng.integers(prompts, size=(2000, n, prompts))
    means = values[tasks[:, :, None], draws].mean(axis=(1, 2))
    return np.quantile(means, [0.025, 0.975]).tolist()


def cell_values(directory, label, environment):
    value = read_json(directory / "confirmation" / label / f"{environment}.json")
    return np.asarray([r["correct"] / r["total"] for r in value["observations"]])


def uncertainty(experiment, final):
    directory = experiment / "clustering/evaluations"
    groups = final["selection"]["selected"]["groups"]
    within, between = [], []
    for index, name in enumerate(final["environments"]):
        base = cell_values(directory, "base", name)
        vectors = [cell_values(directory, members_label(g), name) for g in groups]
        own = next(i for i, g in enumerate(groups) if index in g)
        within.append(vectors[own] - base)
        if len(groups) > 1:
            between.append(
                np.mean([v for i, v in enumerate(vectors) if i != own], axis=0) - base
            )
    return {
        "method": "2000 paired prompt/task bootstrap replicates; 95% percentile intervals",
        "within_transfer_95ci": bootstrap_interval(within),
        "between_transfer_95ci": bootstrap_interval(between) if between else None,
    }


def plot_results(experiment, output):
    # Keep matplotlib's cache out of both the container's read-only home and shared user config.
    os.environ.setdefault("MPLCONFIGDIR", str(experiment / "matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    plan = read_json(experiment / "plan.json")
    final = read_json(experiment / "clustering/results.json")
    transfer = read_json(experiment / "clustering/transfer.json")
    names = final["environments"]
    selected = final["selection"]["selected"]
    groups = selected["groups"]
    order = [i for g in groups for i in g]
    ci = uncertainty(experiment, final)
    write_json(output / "uncertainty.json", ci)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    def heatmap(matrix, rows, columns, title, name):
        fig, ax = plt.subplots(figsize=(15, max(5, len(rows) * 0.35)))
        bound = max(1, float(np.abs(matrix).max()))
        image = ax.imshow(matrix, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
        ax.set_xticks(range(len(columns)), columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(rows)), rows, fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("Evaluation environment")
        ax.set_ylabel("Training adapter / cluster")
        fig.colorbar(image, ax=ax, label="Pass@1 change vs. base (percentage points)")
        save(fig, name)

    matrix = np.asarray(transfer["transfer"])
    heatmap(
        100 * matrix[np.ix_(order, order)],
        [names[i] for i in order],
        [names[i] for i in order],
        "Individual-adapter transfer (selection inputs)",
        "individual_transfer",
    )
    heatmap(
        100 * np.asarray(transfer["hierarchy"]["affinity"])[np.ix_(order, order)],
        [names[i] for i in order],
        [names[i] for i in order],
        "Signed affinity used for clustering (diagonal excluded)",
        "signed_affinity",
    )
    confirmation = final["confirmation"]
    best = confirmation["partitions"][0]
    delta = np.asarray(best["matrix"]) - np.asarray(confirmation["base"])[None, :]
    heatmap(
        100 * delta[:, order],
        [f"Cluster {i + 1} ({len(g)} tasks)" for i, g in enumerate(groups)],
        [names[i] for i in order],
        "Combined-adapter transfer on fresh confirmation inputs",
        "cluster_transfer",
    )

    labels = ["Base", "Own specialist"] + [
        p["label"] for p in confirmation["partitions"]
    ]
    scores = [np.mean(confirmation["base"]), np.mean(confirmation["specialist"])] + [
        p["assigned_accuracy"] for p in confirmation["partitions"]
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, np.asarray(scores) * 100)
    ax.set_ylabel("Macro pass@1 (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        "Fresh confirmation: base, specialists, clustered and global adapter means"
    )
    ax.tick_params(axis="x", rotation=20)
    save(fig, "confirmation_performance")

    values = [best["within_transfer"]]
    intervals = [ci["within_transfer_95ci"]]
    labels = ["Within cluster"]
    if best["between_transfer"] is not None:
        values.append(best["between_transfer"])
        intervals.append(ci["between_transfer_95ci"])
        labels.append("Between clusters")
    fig, ax = plt.subplots(figsize=(7, 4))
    errors = np.abs(np.asarray(intervals).T - np.asarray(values)[None, :]) * 100
    ax.bar(labels, np.asarray(values) * 100, yerr=errors, capsize=5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Pass@1 change vs. base (percentage points)")
    ax.set_title("Transfer signs on fresh prompts (95% bootstrap intervals)")
    save(fig, "within_between_transfer")

    fig, ax = plt.subplots(figsize=(8, 4))
    candidates = final["selection"]["candidates"]
    for key, label in [
        ("within_transfer", "Within-cluster transfer"),
        ("between_transfer", "Between-cluster transfer"),
        ("separation", "Selection objective"),
    ]:
        points = sorted((p["k"], p[key]) for p in candidates if p[key] is not None)
        if points:
            ax.plot(
                [p[0] for p in points],
                [p[1] * 100 for p in points],
                marker="o",
                label=label,
            )
    ax.axvline(len(groups), color="gray", ls="--", label="Selected k")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Pass@1 difference (percentage points)")
    ax.legend(fontsize=8)
    ax.set_title("Candidate cuts on selection prompts only")
    save(fig, "candidate_scores")

    fig, axes = plt.subplots(4, 8, figsize=(24, 11), sharex=True, sharey=True)
    teacher_rows = []
    for i, (teacher, ax) in enumerate(zip(plan["teachers"], axes.flat)):
        directory = experiment / "teacher" / teacher["source_label"]
        records = [
            json.loads(line)
            for line in (directory / "metrics.jsonl").read_text().splitlines()
        ]
        records = [r for r in records if "david/eval/val_envs/pass@1" in r]
        # Replayed uncommitted progress can have duplicate x coordinates; latest wins.
        by_rollout = {
            r["david/rollouts"]: r["david/eval/val_envs/pass@1"] for r in records
        }
        x = sorted(by_rollout)
        ax.plot(x, [100 * by_rollout[r] for r in x])
        ax.axhline(98, color="gray", ls="--", lw=0.5)
        ax.set_title(f"{i}: {teacher['environment']}", fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 8000)
        teacher_rows.append(read_json(directory / "completed.json"))
    fig.supxlabel("Training rollouts per teacher")
    fig.supylabel("Original teacher validation pass@1 (%)")
    save(fig, "teacher_curves")

    with (output / "individual_transfer.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["adapter_environment", *names])
        writer.writerows([[name, *row] for name, row in zip(names, matrix)])
    with (output / "teacher_summary.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "environment",
                "training_rollouts",
                "validation_pass_at_1",
                "stop_reason",
                "wandb_url",
            ]
        )
        writer.writerows(
            [
                [
                    r[k]
                    for k in [
                        "environment",
                        "training_rollouts",
                        "accuracy",
                        "stop_reason",
                        "wandb_url",
                    ]
                ]
                for r in teacher_rows
            ]
        )
    for name in (
        "results.json",
        "transfer.json",
        "candidate-results.json",
        "mopd-clusters.json",
    ):
        shutil.copy2(experiment / "clustering" / name, output / name)
    within_ci = ci["within_transfer_95ci"]
    between_ci = ci["between_transfer_95ci"]
    lines = [
        "# MOPD task clustering",
        "",
        f"Experiment: `{experiment}`",
        "",
        f"Selected **{len(groups)} clusters** on selection prompts; confirmation inputs were held out.",
        "",
        f"Fresh confirmation macro pass@1: base **{np.mean(confirmation['base']):.2%}**, "
        f"own specialists **{np.mean(confirmation['specialist']):.2%}**, assigned cluster adapters **{best['assigned_accuracy']:.2%}**.",
        "",
        f"Within-cluster transfer: {best['within_transfer'] * 100:+.2f} percentage points; "
        f"95% interval [{within_ci[0] * 100:+.2f}, {within_ci[1] * 100:+.2f}].",
    ]
    if between_ci:
        lines += [
            f"Between-cluster transfer: {best['between_transfer'] * 100:+.2f} percentage points; "
            f"95% interval [{between_ci[0] * 100:+.2f}, {between_ci[1] * 100:+.2f}]."
        ]
    lines += [
        "",
        "The method seeks positive transfer within clusters and negative transfer between them. "
        "The data need not exhibit both signs. Adapter averaging is a proxy for joint training, not proof of it.",
        "",
        "Clusters and adapter paths for a subsequent MOPD experiment: [mopd-clusters.json](mopd-clusters.json).",
        "",
    ]
    lines += [
        f"- Cluster {i + 1}: " + ", ".join(names[j] for j in g)
        for i, g in enumerate(groups)
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_results(args.experiment, args.output)
