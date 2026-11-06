"""Search equal-size clusters using saved LoRA evaluations, without GPU training.

Only original selection observations are used. Four prompt folds retain one
example from each probe difficulty in every test fold. Earlier confirmation
measurements do not influence assignments or k selection.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from lora_clustering.balanced import (
    balanced_metrics,
    candidate_ks,
    fit_balanced,
    random_balanced_baseline,
    select_k,
)


def read(path):
    return json.loads(Path(path).read_text())


def singleton_reference(transfer):
    matrix = np.asarray(transfer)
    n = len(matrix)
    return {
        "k": n,
        "size": 1,
        "cv_within_mean": None,
        "cv_between_mean": float(matrix[~np.eye(n, dtype=bool)].mean()),
        "cv_gap_mean": None,
        "self_gain": float(np.diag(matrix).mean()),
        "eligible_for_selection": False,
        "reason": "Singleton clusters contain no distinct within-cluster task pairs",
    }


def add_random_baselines(result, cube, *, draws=10000):
    held = [
        cube[:, :, fold["held_prompt_indices"]].mean(axis=-1)
        for fold in result["candidates"][0]["folds"]
    ]
    for row in result["candidates"] + result["reference_rows"]:
        row["random_baseline"] = random_balanced_baseline(
            held, row["k"], draws=draws, seed=result["seed"] + 10000 + row["k"]
        )


def load_observations(experiment):
    plan = read(experiment / "plan.json")
    names = plan["environments"]
    root = experiment / "clustering/evaluations/selection"
    base, individual = [], []
    expected_ids = {}
    hashes = {}

    def observations(path, environment, *, baseline=False):
        cell = read(path)
        if cell["eval_sha256"] != plan["clustering"]["selection_sha256"]:
            raise ValueError("Selection inputs do not match the frozen experiment")
        rows = cell["observations"]
        ids = [r["prompt_id"] for r in rows]
        if baseline:
            expected_ids[environment] = ids
        elif ids != expected_ids[environment]:
            raise ValueError("Prompt-level outcomes are not aligned")
        if len(rows) != 8 or any(r["total"] != 2 for r in rows):
            raise ValueError(
                "Expected eight original selection prompts and two responses each"
            )
        hashes[str(path.relative_to(experiment))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        return [r["correct"] / r["total"] for r in rows]

    for name in names:
        base.append(observations(root / "base" / f"{name}.json", name, baseline=True))
    for i in range(len(names)):
        individual.append(
            [
                observations(root / f"individual-{i:02d}" / f"{name}.json", name)
                for name in names
            ]
        )
    return plan, np.asarray(individual) - np.asarray(base)[None, :, :], hashes


def analyze(experiment, output, restarts, seed):
    plan, cube, hashes = load_observations(experiment)
    # Selection data were generated as four level-a then four level-b prompts.
    folds = [[i, i + 4] for i in range(4)]
    candidates = []
    for k in candidate_ks(len(cube)):
        evaluations = []
        for fold, held in enumerate(folds):
            training = [i for i in range(cube.shape[-1]) if i not in held]
            fit = fit_balanced(
                cube[:, :, training].mean(axis=-1),
                k,
                restarts=restarts,
                seed=seed + k * 100 + fold,
            )
            measured = balanced_metrics(cube[:, :, held].mean(axis=-1), fit["groups"])
            evaluations.append(
                {
                    "held_prompt_indices": held,
                    "groups": fit["groups"],
                    "train": fit,
                    "held_out": measured,
                }
            )
        qualities = [r["held_out"]["weakest_quality"] for r in evaluations]
        gaps = [r["held_out"]["mean_gap"] for r in evaluations]
        fit = fit_balanced(cube.mean(axis=-1), k, restarts=restarts * 2, seed=seed + k)
        candidate = {
            "k": k,
            "size": len(cube) // k,
            "fit": fit,
            "folds": evaluations,
            "cv_quality_mean": float(np.mean(qualities)),
            "cv_quality_se": float(np.std(qualities, ddof=1) / np.sqrt(len(qualities))),
            "cv_gap_mean": float(np.mean(gaps)),
            "cv_gap_se": float(np.std(gaps, ddof=1) / np.sqrt(len(gaps))),
            "cv_within_mean": float(
                np.mean([r["held_out"]["within"] for r in evaluations])
            ),
            "cv_between_mean": float(
                np.mean([r["held_out"]["between"] for r in evaluations])
            ),
        }
        candidates.append(candidate)
        print(
            json.dumps(
                {
                    key: candidate[key]
                    for key in [
                        "k",
                        "size",
                        "cv_quality_mean",
                        "cv_quality_se",
                        "cv_gap_mean",
                        "cv_within_mean",
                        "cv_between_mean",
                    ]
                }
            ),
            flush=True,
        )
    selection = select_k(candidates)
    chosen = next(r for r in candidates if r["k"] == selection["k"])
    result = {
        "method": "equal-size balanced pair swaps; optimize the weakest cluster before mean separation",
        "objective": "min over clusters of min(within, within - max(outgoing_between,incoming_between))",
        "between_transfer_must_be_negative": False,
        "self_transfer_excluded": True,
        "confirmation_data_used": False,
        "source_experiment": str(experiment),
        "seed": seed,
        "restarts_per_fold": restarts,
        "environments": plan["environments"],
        "candidates": candidates,
        "reference_rows": [singleton_reference(cube.mean(axis=-1))],
        "selection": selection,
        "selected": chosen,
        "evaluation_input_sha256": plan["clustering"]["selection_sha256"],
        "observation_sha256": hashes,
        "validation_scope": "prompt cross-validation of individual-adapter transfer; new merged-adapter confirmation is pending",
    }
    add_random_baselines(result, cube)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "assignments.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cluster", "environment"])
        for cluster, group in enumerate(chosen["fit"]["groups"], start=1):
            writer.writerows([[cluster, plan["environments"][i]] for i in group])
    (output / "transfer.json").write_text(
        json.dumps(cube.mean(axis=-1).tolist()) + "\n"
    )
    write_report(result, output)
    return result


def write_report(result, output):
    candidates, selection, chosen = (
        result["candidates"],
        result["selection"],
        result["selected"],
    )
    lines = [
        "# Equal-size LoRA task clusters",
        "",
        f"Selected k={selection['k']}, {chosen['size']} tasks per cluster.",
        "",
        "This development analysis uses only the original selection observations. Self-transfer is excluded. "
        "Exterior transfer need not be negative; it should be lower than interior transfer in both directions.",
        "",
        "Assignments maximize the weakest cluster's positive internal gain and internal/external gap, "
        "with mean separation as a tie-breaker. k is selected by mean internal/external gap on four "
        "stratified prompt folds; prefer fewer clusters within one fold standard error of the best score.",
        "",
        "| k | Tasks/cluster | Held-out within | Held-out between | Random within | Random between | Held-out gap | Weakest-cluster score |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in candidates:
        random = row["random_baseline"]
        lines.append(
            f"| {row['k']} | {row['size']} | {100 * row['cv_within_mean']:+.2f} pp | "
            f"{100 * row['cv_between_mean']:+.2f} pp | "
            f"{100 * random['within']['expected_mean']:+.2f} pp | "
            f"{100 * random['between']['expected_mean']:+.2f} pp | "
            f"{100 * row['cv_gap_mean']:+.2f} pp | "
            f"{100 * row['cv_quality_mean']:+.2f} ± {100 * row['cv_quality_se']:.2f} pp |"
        )
    reference = result["reference_rows"][0]
    lines.append(
        f"| {reference['k']} (reference) | 1 | — | "
        f"{100 * reference['cv_between_mean']:+.2f} pp | — | "
        f"{100 * reference['random_baseline']['between']['expected_mean']:+.2f} pp | — | — |"
    )
    lines += [
        "",
        "Random columns show exact expected means over uniform equal-size assignments, with self-transfer excluded. "
        "Both expectations equal the global off-diagonal mean: every distinct task pair has the same "
        "probability of falling within a random cluster. Expected random gap is zero. "
        "We also sampled 10,000 random assignments per k on the same held-out prompts, drawing a fresh "
        "partition for each fold and averaging its four fold scores. No random partition was optimized or selected.",
        "",
        "| k | Random within: 95% assignment range | Random between: 95% assignment range | Random gap: 95% assignment range |",
        "|---|---|---|---|",
    ]
    for row in candidates + result["reference_rows"]:
        intervals = []
        for key in ("within", "between", "gap"):
            metric = row["random_baseline"][key]
            intervals.append(
                "—"
                if metric is None
                else " to ".join(
                    f"{100 * value:+.2f} pp"
                    for value in metric["assignment_interval_95"]
                )
            )
        lines.append(f"| {row['k']} | " + " | ".join(intervals) + " |")
    lines += [
        "",
        "These ranges measure variation across random assignments of the fixed observations; "
        "they are not confidence intervals or fresh confirmation of the chosen clusters.",
        "",
        f"At k={reference['k']}, each cluster contains one task, so within-transfer and the gap are undefined. "
        "This reference is excluded from k selection. Its between-transfer averages all distinct task pairs "
        "over the same held-out prompts. "
        f"Own-task gain is {100 * reference['self_gain']:+.2f} pp; this is self-transfer, excluded from the within-transfer metric above.",
        "",
        "Errors describe variation across the four prompt folds, not an independent confirmation interval.",
        "",
        "A positive average gap does not certify every cluster. The weakest-cluster statistic is reported "
        "separately; with only two held-out prompts per task in each fold, individual-group reliability "
        "needs a larger fresh evaluation.",
        "",
        (
            "Fresh merged-adapter and leave-one-task-out confirmation is complete; see "
            "[confirmation results](../README.md). This pairwise table remains the original prompt cross-validation result."
            if result.get("merged_adapter_confirmation")
            else "This is a pairwise-transfer search result. Fresh combined-adapter or leave-one-task-out evaluation "
            "is still required before claiming that merged adapters reproduce the same grouping."
        ),
        "",
    ]
    for i, group in enumerate(chosen["fit"]["groups"], start=1):
        lines.append(
            f"- Cluster {i}: " + ", ".join(result["environments"][j] for j in group)
        )
    (output / "README.md").write_text("\n".join(lines) + "\n")


def plot(output):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/balanced-lora-clustering-mpl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = read(output / "results.json")
    candidates, selected = result["candidates"], result["selected"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ks = [r["k"] for r in candidates]
    for key, label in [
        ("cv_within_mean", "Within"),
        ("cv_between_mean", "Between"),
        ("cv_gap_mean", "Gap"),
    ]:
        axes[0].plot(ks, [100 * r[key] for r in candidates], marker="o", label=label)
    axes[0].set_title("Held-out transfer, excluding self-transfer")
    axes[0].set_ylabel("Pass@1 change / difference (percentage points)")
    axes[0].legend()
    axes[1].errorbar(
        ks,
        [100 * r["cv_quality_mean"] for r in candidates],
        yerr=[100 * r["cv_quality_se"] for r in candidates],
        marker="o",
        capsize=4,
    )
    axes[1].set_title("Weakest-cluster score across held-out prompts")
    for ax in axes:
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(result["selection"]["k"], color="gray", ls="--")
        ax.set_xticks(ks)
        ax.set_xlabel("Equal-sized clusters (k)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"k_selection.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    matrix = np.asarray(read(output / "transfer.json"))
    groups = selected["fit"]["groups"]
    order = [i for g in groups for i in g]
    matrix = matrix[np.ix_(order, order)] * 100
    np.fill_diagonal(matrix, np.nan)
    names = [result["environments"][i] for i in order]
    fig, ax = plt.subplots(figsize=(14, 12))
    bound = float(np.nanmax(np.abs(matrix)))
    im = ax.imshow(matrix, cmap="RdBu", vmin=-bound, vmax=bound)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    for boundary in np.cumsum([len(g) for g in groups])[:-1] - 0.5:
        ax.axvline(boundary, color="black", lw=1)
        ax.axhline(boundary, color="black", lw=1)
    ax.set_title(
        f"Equal-size k={len(groups)}; individual-adapter transfer; self-transfer excluded"
    )
    ax.set_xlabel("Evaluation task")
    ax.set_ylabel("Adapter training task")
    fig.colorbar(im, ax=ax, label="Pass@1 change vs. base (percentage points)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"balanced_transfer.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restarts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        analyze(
            args.experiment.resolve(), args.output.resolve(), args.restarts, args.seed
        )
    if not args.no_plots:
        plot(args.output.resolve())


if __name__ == "__main__":
    main()
