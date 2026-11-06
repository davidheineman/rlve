"""Find low-transfer balanced groups using the original saved prompt outcomes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_balanced_lora_clustering import load_observations
from lora_clustering.algorithm import members_label
from lora_clustering.low_transfer import evaluate_fold, fit_low_transfer


def add_cached_confirmation(result, experiment):
    """Evaluate frozen discovery-selected pairs only where prior controls exist."""
    plan = json.loads((experiment / "plan.json").read_text())
    names = result["environments"]
    if names != plan["environments"]:
        raise ValueError("Confirmation and selected pairs must use the same task order")
    root = experiment / "clustering/evaluations"
    indices = {name: i for i, name in enumerate(names)}
    records, hashes = [], {}
    for pair in result["full_fit_pairs"]:
        for source, target in [
            (pair["task_a"], pair["task_b"]),
            (pair["task_b"], pair["task_a"]),
        ]:
            path = (
                root
                / "confirmation"
                / members_label([indices[source]])
                / f"{target}.json"
            )
            if not path.exists():
                continue
            base_path = root / "confirmation/base" / f"{target}.json"
            cell, base = [json.loads(p.read_text()) for p in [path, base_path]]
            selected = json.loads(
                (root / "selection/base" / f"{target}.json").read_text()
            )
            ids = [o["prompt_id"] for o in cell["observations"]]
            if (
                not ids
                or len(set(ids)) != len(ids)
                or ids != [o["prompt_id"] for o in base["observations"]]
                or not set(ids).isdisjoint(
                    o["prompt_id"] for o in selected["observations"]
                )
                or any(
                    v["eval_sha256"] != plan["clustering"]["confirmation_sha256"]
                    for v in [cell, base]
                )
            ):
                raise ValueError(
                    "Confirmation prompts must be aligned, frozen, and disjoint from selection"
                )
            differences = np.asarray(
                [
                    a["correct"] / a["total"] - b["correct"] / b["total"]
                    for a, b in zip(cell["observations"], base["observations"])
                ]
            )
            if not np.isclose(differences.mean(), cell["accuracy"] - base["accuracy"]):
                raise ValueError("Cell mean and paired-prompt confirmation disagree")
            rng = np.random.default_rng(result["seed"])
            sampled = differences[rng.integers(len(ids), size=(10000, len(ids)))].mean(
                axis=1
            )
            records.append(
                {
                    "source": source,
                    "target": target,
                    "base_pass1": base["accuracy"],
                    "adapter_pass1": cell["accuracy"],
                    "gain": float(differences.mean()),
                    "prompt_bootstrap_95ci": np.quantile(
                        sampled, [0.025, 0.975]
                    ).tolist(),
                    "prompts": len(ids),
                    "completions_per_prompt": sorted(
                        {o["total"] for o in cell["observations"]}
                    ),
                }
            )
            for p in [path, base_path]:
                hashes[str(p.relative_to(experiment))] = hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
    result["cached_confirmation"] = {
        "records": records,
        "available_directed_cells": len(records),
        "possible_directed_cells": 2 * len(result["full_fit_pairs"]),
        "used_for_assignment_selection": False,
        "scope": "Opportunistic disjoint confirmation for previously cached controls; not a representative or complete evaluation of this partition.",
        "uncertainty": "Paired bootstrap over prompts, conditional on these trained adapters and the fixed pair selection.",
        "observation_sha256": hashes,
    }


def write_outputs(result, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    names = result["environments"]
    with (output / "assignments.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["k", "cluster", "environment"])
        for row in result["candidates"]:
            for cluster, group in enumerate(row["fit"]["groups"], 1):
                writer.writerows((row["k"], cluster, names[i]) for i in group)
    with (output / "pairs.csv").open("w") as stream:
        fields = [
            "task_a",
            "task_b",
            "a_to_b",
            "b_to_a",
            "mutual_mean",
            "both_negative",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["full_fit_pairs"])
    lines = [
        "# Low-transfer task groups",
        "",
        "Equal-sized groups covering the same 64 tasks exactly once. Minimize mean within-group transfer, excluding self-transfer. "
        "Pairs use minimum-weight perfect matching; larger groups use multi-start improving swaps. This objective does not require every group or direction to be negative.",
        "",
        "Assignments are refit on six selection prompts per task and evaluated on two held prompts in each of the same four difficulty-stratified folds as the positive clustering table. "
        "All numbers are pass@1 changes versus the base model, in percentage points. k is shown as a sweep; no k is selected using held scores.",
        "",
        "Each measurement applies one source-task LoRA to a different target task. Larger groups average these pairwise measurements; this analysis does not compose adapters. "
        "Plot error bars show ±1 standard error across the four prompt folds (descriptive). "
        "The axis label '% over base' denotes absolute percentage-point change, not relative percent increase.",
        "",
        "| k | Tasks/cluster | Positive-search within | Low-transfer within | Outside | Random within/outside | Within−outside | Negative groups/fold |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["candidates"]:
        lines.append(
            f"| {row['k']} | {row['size']} | {100 * row['positive_cv_within']:+.2f} | "
            f"{100 * row['cv_within_mean']:+.2f} | {100 * row['cv_between_mean']:+.2f} | "
            f"{100 * row['random_mean']:+.2f} | {100 * row['cv_gap_mean']:+.2f} | "
            f"{row['cv_negative_clusters_mean']:.1f}/{row['k']} |"
        )
    lines += [
        f"| 64 | 1 | — | — | {100 * result['global_offdiagonal_mean']:+.2f} | {100 * result['global_offdiagonal_mean']:+.2f} | — | — |",
        "",
        "Negative means applying a LoRA trained on another task reduces accuracy relative to the unchanged base. "
        "It does not by itself establish interference under joint training or a full adapter merge. A negative mutual mean can also hide a positive direction; pairs.csv retains both directions.",
        "",
        "Cross-validation scores describe the grouping method; fold assignments differ. Full-data assignments and pair scores are exploratory fits, not independently confirmed fixed groups. "
        "There are only eight original selection prompts with two completions each. Fold variability is not a formal confidence interval, and individual negative pairs need fresh confirmation.",
        "",
        "## Most negative pairs in the full-data matching (exploratory)",
        "",
        "| Task A | Task B | A→B | B→A | Mutual mean |",
        "|---|---|---:|---:|---:|",
    ]
    for pair in result["full_fit_pairs"][:10]:
        lines.append(
            f"| {pair['task_a']} | {pair['task_b']} | {100 * pair['a_to_b']:+.2f} | {100 * pair['b_to_a']:+.2f} | {100 * pair['mutual_mean']:+.2f} |"
        )
    cached = result.get("cached_confirmation")
    if cached:
        lines += [
            "",
            "## Available disjoint confirmation",
            "",
            f"{cached['available_directed_cells']} of {cached['possible_directed_cells']} directed cells happened to be cached by earlier controls. "
            "Assignments were frozen using selection observations before reading these cells. Their prompts are disjoint from selection. "
            "This incomplete subset cannot confirm the whole partition. Intervals resample paired prompts conditional on the trained adapters.",
            "",
            "| Adapter trained on | Evaluated task | Base pass@1 | Adapter pass@1 | Change | 95% prompt-bootstrap interval |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in cached["records"]:
            lo, hi = row["prompt_bootstrap_95ci"]
            lines.append(
                f"| {row['source']} | {row['target']} | {100 * row['base_pass1']:.2f}% | {100 * row['adapter_pass1']:.2f}% | {100 * row['gain']:+.2f} pp | [{100 * lo:+.2f}, {100 * hi:+.2f}] pp |"
            )
    (output / "README.md").write_text("\n".join(lines) + "\n")


def plot(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    rows = result["candidates"]
    ks = [row["k"] for row in rows]
    high_color, low_color, random_color = "#126FA4", "#C85A27", "#758291"
    fig, ax = plt.subplots(figsize=(8, 5))
    (high_line,) = ax.plot(
        ks,
        [100 * r["positive_cv_within"] for r in rows],
        marker="o",
        color=high_color,
        label="High transfer grouping",
    )
    low_line = ax.errorbar(
        ks,
        [100 * r["cv_within_mean"] for r in rows],
        yerr=[100 * r["cv_within_se"] for r in rows],
        marker="o",
        color=low_color,
        ecolor=to_rgba(low_color, 0.7),
        capsize=3,
        label="Low transfer grouping",
    )
    (random_line,) = ax.plot(
        ks,
        [100 * result["global_offdiagonal_mean"] for _ in rows],
        marker="o",
        color=random_color,
        label="Random grouping",
    )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(
        xlabel="Number of equal-sized task groups (k)",
        ylabel="pass@1 improvement (% over base)",
        title=f"Pairwise LoRA transfer ({len(result['environments'])} envs)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.grid(alpha=0.2)
    ax.legend(handles=[high_line, low_line, random_line], fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output / f"low_transfer.{ext}", dpi=180)
    plt.close(fig)


def analyze(experiment, positive_path, output, *, restarts=256, seed=1234):
    positive = json.loads(positive_path.read_text())
    plan, cube, hashes = load_observations(experiment)
    if (
        positive["environments"] != plan["environments"]
        or positive["observation_sha256"] != hashes
    ):
        raise ValueError(
            "Original positive and reverse searches must use identical observations"
        )
    candidates = []
    for original in positive["candidates"]:
        k = original["k"]
        evaluations = [
            evaluate_fold(
                cube,
                k,
                f["held_prompt_indices"],
                restarts=restarts,
                seed=seed + k * 100 + i,
            )
            for i, f in enumerate(original["folds"])
        ]
        within = [f["held_out"]["within"] for f in evaluations]
        row = {
            "k": k,
            "size": len(cube) // k,
            "folds": evaluations,
            "fit": fit_low_transfer(
                cube.mean(axis=-1), k, restarts=2 * restarts, seed=seed + k
            ),
            "cv_within_mean": float(np.mean(within)),
            "cv_within_se": float(np.std(within, ddof=1) / np.sqrt(len(within))),
            "cv_between_mean": float(
                np.mean([f["held_out"]["between"] for f in evaluations])
            ),
            "cv_gap_mean": float(
                np.mean([f["held_out"]["mean_gap"] for f in evaluations])
            ),
            "cv_negative_clusters_mean": float(
                np.mean([f["held_out"]["negative_clusters"] for f in evaluations])
            ),
            "cv_both_directions_negative_pair_fraction": float(
                np.mean(
                    [
                        f["held_out"]["both_directions_negative_pair_fraction"]
                        for f in evaluations
                    ]
                )
            ),
            "positive_cv_within": original["cv_within_mean"],
            "random_mean": original["random_baseline"]["within"]["expected_mean"],
        }
        candidates.append(row)
        print(
            json.dumps({k: v for k, v in row.items() if k not in ["folds", "fit"]}),
            flush=True,
        )
    transfer = cube.mean(axis=-1)
    pairing = next(row for row in candidates if row["size"] == 2)
    pairs = [
        {
            "task_a": plan["environments"][i],
            "task_b": plan["environments"][j],
            "a_to_b": float(transfer[i, j]),
            "b_to_a": float(transfer[j, i]),
            "mutual_mean": float((transfer[i, j] + transfer[j, i]) / 2),
            "both_negative": bool(transfer[i, j] < 0 and transfer[j, i] < 0),
        }
        for i, j in pairing["fit"]["groups"]
    ]
    pairs.sort(key=lambda p: p["mutual_mean"])
    result = {
        "source_experiment": str(experiment),
        "positive_results": str(positive_path),
        "positive_results_sha256": hashlib.sha256(
            positive_path.read_bytes()
        ).hexdigest(),
        "evaluation_input_sha256": plan["clustering"]["selection_sha256"],
        "observation_sha256": hashes,
        "seed": seed,
        "restarts_per_fold": restarts,
        "environments": plan["environments"],
        "objective": "minimize mean within-group off-diagonal transfer; equal size and full task coverage",
        "confirmation_data_used": False,
        "self_transfer_excluded": True,
        "global_offdiagonal_mean": float(
            transfer[~np.eye(len(cube), dtype=bool)].mean()
        ),
        "candidates": candidates,
        "full_fit_pairs": pairs,
        "validation_scope": "Four prompt folds on individual-adapter transfer. No new joint-training or adapter-composition confirmation.",
    }
    add_cached_confirmation(result, experiment)
    write_outputs(result, output)
    plot(result, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--positive-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restarts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cached-confirmation-only", action="store_true")
    args = parser.parse_args()
    if args.cached_confirmation_only:
        result = json.loads((args.output / "results.json").read_text())
        if Path(result["source_experiment"]).resolve() != args.experiment.resolve():
            raise ValueError(
                "Cached confirmation must use the original discovery experiment"
            )
        add_cached_confirmation(result, args.experiment)
        write_outputs(result, args.output)
        return
    analyze(
        args.experiment,
        args.positive_results,
        args.output,
        restarts=args.restarts,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
