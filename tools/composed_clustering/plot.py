"""Report direct composed-adapter ID improvement on disjoint member-task probes."""

import json
from pathlib import Path

import numpy as np


def summary(values, seed=1234):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "environment_bootstrap_95ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def render(directory, records, base_map, *, singleton=None, all_tasks=None):
    directory = Path(directory)
    if all_tasks is None and (directory / "held-k01.json").exists():
        all_tasks = json.loads((directory / "held-k01.json").read_text())
    plan = json.loads((directory / "plan.json").read_text())
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    base = np.asarray([base_map[name] for name in plan["environments"]])
    measured = []
    for row in records:
        random = np.asarray([r["per_environment"] for r in row["random"]]).mean(axis=0)
        entry = {"k": row["k"], "size": len(base) // row["k"]}
        for label, values in [
            ("best", row["best"]["per_environment"]),
            ("worst", row["worst"]["per_environment"]),
            ("random", random),
        ]:
            entry[label] = {
                "accuracy": float(np.mean(values)),
                **summary(np.asarray(values) - base),
            }
        entry["best_minus_random"] = summary(
            np.asarray(row["best"]["per_environment"]) - random
        )
        entry["worst_minus_random"] = summary(
            np.asarray(row["worst"]["per_environment"]) - random
        )
        measured.append(entry)
    payload = {
        "method": "direct exact-composition group search",
        "composition": "uniform mean of full weight updates, represented by concatenated LoRA factors; no SVD truncation",
        "evaluation": "macro ID on member tasks; includes each target task's own adapter",
        "selection_prompts_per_task": 8,
        "selection_samples_per_prompt": plan["evaluation"]["selection"]["samples"],
        "confirmation_prompts_per_task": 32,
        "confirmation_samples_per_prompt": plan["evaluation"]["confirmation"][
            "samples"
        ],
        "base_accuracy": float(base.mean()),
        "records": measured,
        "complete": singleton is not None,
        "k1_evaluated": all_tasks is not None,
    }
    if singleton:
        payload["singleton"] = {
            "k": len(base),
            "accuracy": singleton["accuracy"],
            **summary(np.asarray(singleton["per_environment"]) - base),
        }
    if all_tasks:
        payload["all_tasks"] = {
            "k": 1,
            "accuracy": all_tasks["accuracy"],
            **summary(np.asarray(all_tasks["per_environment"]) - base),
        }
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    fig, ax = plt.subplots(figsize=(8, 5))
    measured.sort(key=lambda r: r["k"])
    for label, color, title in [
        ("best", "#126FA4", "High transfer grouping"),
        ("worst", "#C85A27", "Low transfer grouping"),
        ("random", "#758291", "Random grouping"),
    ]:
        points = [(r["k"], r[label]) for r in measured]
        if all_tasks:
            points.insert(0, (1, payload["all_tasks"]))
        if singleton:
            points.append((len(base), payload["singleton"]))
        means = np.asarray([r["mean"] for _, r in points]) * 100
        if len(means):
            ax.plot(
                [k for k, _ in points],
                means,
                color=color,
                marker="o",
                label=title,
            )
        # Shared endpoints have one CI each, avoiding stacked opacity.
        if measured:
            middle = np.asarray([r[label]["mean"] for r in measured]) * 100
            intervals = (
                np.asarray([r[label]["environment_bootstrap_95ci"] for r in measured])
                * 100
            )
            ax.errorbar(
                [r["k"] for r in measured],
                middle,
                yerr=np.vstack([middle - intervals[:, 0], intervals[:, 1] - middle]),
                fmt="none",
                ecolor=to_rgba(color, 0.7),
                capsize=3,
            )
    if all_tasks:
        r = payload["all_tasks"]
        lo, hi = r["environment_bootstrap_95ci"]
        ax.errorbar(
            [1],
            [100 * r["mean"]],
            yerr=[[100 * (r["mean"] - lo)], [100 * (hi - r["mean"])]],
            fmt="o",
            color="#758291",
            ecolor=to_rgba("#758291", 0.7),
            capsize=3,
            zorder=5,
        )
    if singleton:
        r = payload["singleton"]
        lo, hi = r["environment_bootstrap_95ci"]
        ax.errorbar(
            [len(base)],
            [100 * r["mean"]],
            yerr=[[100 * (r["mean"] - lo)], [100 * (hi - r["mean"])]],
            fmt="o",
            color="black",
            ecolor=to_rgba("black", 0.7),
            capsize=3,
            label="Single task",
            zorder=5,
        )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xscale("log", base=2)
    ks = sorted(
        {r["k"] for r in measured}
        | ({len(base)} if singleton else set())
        | ({1} if all_tasks else set())
    )
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set(
        xlabel="Number of equal-sized task groups (k)",
        ylabel="pass@1 improvement\n(% over base)",
        title=f"LoRA adapter mixing ({len(base)} envs)",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output / f"composed_group_id_improvement.{ext}", dpi=180)
    plt.close(fig)
    lines = [
        "# Composed-adapter task grouping",
        "",
        "Assignments are optimized by measured ID accuracy of the full member-adapter composition. Pairwise transfer only seeds and proposes assignments; it does not determine acceptance.",
        "",
        f"Selection: 8 prompts/task × {plan['evaluation']['selection']['samples']} completions. Confirmation: 32 disjoint prompts/task × {plan['evaluation']['confirmation']['samples']} completions. All groups include the target task's own adapter. Composition is a uniform mean of updates, with exact concatenated rank 32×group size and no SVD compression.",
        "",
        "| k | Tasks/group | High-ID gain | Random gain | Low-ID gain |",
        "|---:|---:|---:|---:|---:|",
    ]
    if all_tasks:
        value = 100 * payload["all_tasks"]["mean"]
        lines.append(
            f"| 1 | {len(base)} | {value:+.2f} pp | {value:+.2f} pp | {value:+.2f} pp |"
        )
    for r in measured:
        lines.append(
            f"| {r['k']} | {r['size']} | {100 * r['best']['mean']:+.2f} pp | {100 * r['random']['mean']:+.2f} pp | {100 * r['worst']['mean']:+.2f} pp |"
        )
    lines += [
        "",
        "High/low labels describe a budgeted local search, not certified global extrema. Search uses 3 fixed random initializations and the earlier high-/low-transfer assignments, then up to 16 proposed swaps per direction; acceptance is based only on selection data. Confirmation outcomes are read after all k assignments are frozen. Random assignments are not selected by their confirmation scores.",
        "",
        "Raw ID improvement here differs from the earlier off-diagonal cross-task transfer metric: these compositions include own-task training. The mean-update normalization is fixed across group sizes; there is no fitted scaling coefficient. Bootstrap intervals condition on the trained adapters and selected assignments.",
        "",
        "The axis label '% over base' denotes the existing absolute percentage-point difference, 100 × (pass@1 − base pass@1), not a relative percent increase. Error bars are 95% bootstrap intervals over environments. All curves share k=64 (single-task adapters); k=1, when measured, is also shared because there is only one all-task grouping.",
        "",
        f"Run: `{directory}`",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    completed = json.loads((args.experiment / "completed.json").read_text())
    render(
        args.experiment,
        completed["candidates"],
        completed["base"],
        singleton=completed["singleton"],
    )
