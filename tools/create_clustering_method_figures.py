"""Render an explanatory schematic and measured evidence for LoRA task grouping."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lora_clustering.balanced import balanced_metrics
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

BLUE = "#126FA4"
ORANGE = "#C85A27"
INK = "#172C40"
MUTED = "#576779"
BORDER = "#D6DEE7"
ROOT = Path(__file__).resolve().parents[1]


def box(
    ax, x, y, w, h, *, face="white", edge=BORDER, radius=0.009, linewidth=1, **kwargs
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        **kwargs,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, value, *, size=10, color=INK, weight="normal", ha="left", **kwargs):
    return ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va="center",
        linespacing=1.5,
        **kwargs,
    )


def save(fig, output, name):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=190, facecolor="white")
    plt.close(fig)


def overview(output, n, high, low, random_mean):
    fig = plt.figure(figsize=(16, 8.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    text(
        ax,
        0.035,
        0.95,
        "Train once. Measure transfer. Search task groups.",
        size=25,
        weight="bold",
    )
    text(
        ax,
        0.035,
        0.902,
        "One shared base + task-specific LoRAs → many candidate groupings, without retraining.",
        size=12,
        color=MUTED,
    )
    xs, width = [0.035, 0.273, 0.511, 0.749], 0.216
    titles = [
        "Train adapters",
        "Measure transfer",
        "Search assignments",
        "Check generalization",
    ]
    for i, (x, title) in enumerate(zip(xs, titles), 1):
        box(ax, x, 0.265, width, 0.60, face="#FBFCFE")
        ax.add_patch(Circle((x + 0.025, 0.826), 0.014, facecolor=INK, edgecolor="none"))
        text(
            ax,
            x + 0.025,
            0.826,
            str(i),
            size=10,
            color="white",
            weight="bold",
            ha="center",
        )
        text(ax, x + 0.048, 0.826, title, size=12.5, weight="bold")
    for x in xs[:-1]:
        ax.add_patch(
            FancyArrowPatch(
                (x + width + 0.003, 0.585),
                (x + 0.238 - 0.004, 0.585),
                arrowstyle="->",
                mutation_scale=15,
                linewidth=1.3,
                color="#8493A4",
            )
        )

    x = xs[0]
    box(ax, x + 0.017, 0.689, 0.182, 0.077, face="#E9F0F7", edge="#B5C6D8")
    text(
        ax,
        x + width / 2,
        0.735,
        "Shared frozen base",
        size=11,
        weight="bold",
        ha="center",
    )
    text(ax, x + width / 2, 0.708, r"$W_0$", size=12, ha="center")
    ax.add_patch(
        FancyArrowPatch(
            (x + width / 2, 0.682),
            (x + width / 2, 0.642),
            arrowstyle="->",
            mutation_scale=13,
            color=MUTED,
        )
    )
    for i, letter in enumerate("ABCDEFGH"):
        left, bottom = x + 0.017 + (i % 4) * 0.047, 0.58 - (i // 4) * 0.071
        box(
            ax, left, bottom, 0.041, 0.052, face="#E8F3FA", edge="#B6D1E2", radius=0.005
        )
        text(
            ax,
            left + 0.0205,
            bottom + 0.026,
            rf"$\Delta_{{{letter}}}$",
            size=12,
            color=BLUE,
            ha="center",
        )
    text(ax, x + width / 2, 0.475, "One adapter per task", size=10, ha="center")
    text(
        ax,
        x + 0.017,
        0.395,
        "Train N adapters once.\nReuse them for every grouping.",
        size=10.5,
    )
    text(
        ax,
        x + 0.017,
        0.299,
        "No grouped-model retraining in search.",
        size=8.7,
        color=MUTED,
    )

    x = xs[1]
    text(
        ax,
        x + width / 2,
        0.782,
        "Schematic task-to-task matrix",
        size=9,
        color=MUTED,
        ha="center",
    )
    rng = np.random.default_rng(4)
    toy = rng.normal(-0.025, 0.035, (8, 8))
    toy[:4, :4] += 0.15
    toy[4:, 4:] += 0.15
    np.fill_diagonal(toy, np.nan)
    matrix_ax = fig.add_axes([x + 0.043, 0.521, 0.145, 0.229])
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#DDE3E9")
    matrix_ax.imshow(toy, cmap=cmap, vmin=-0.15, vmax=0.15)
    matrix_ax.set_xticks(range(8), list("ABCDEFGH"), fontsize=7)
    matrix_ax.set_yticks(range(8), list("ABCDEFGH"), fontsize=7)
    matrix_ax.set_xlabel("Evaluation task", fontsize=8)
    matrix_ax.set_ylabel("Adapter task", fontsize=8)
    matrix_ax.tick_params(length=0, pad=2)
    for spine in matrix_ax.spines.values():
        spine.set_visible(False)
    text(ax, x + 0.033, 0.449, "● Helps", size=9, color=BLUE)
    text(ax, x + 0.124, 0.449, "● Hurts", size=9, color=ORANGE)
    text(
        ax,
        x + width / 2,
        0.405,
        r"$T_{ij}=\mathrm{score}(W_0+\Delta_i,j)$",
        size=10.2,
        ha="center",
    )
    text(
        ax, x + width / 2, 0.373, r"$-\ \mathrm{score}(W_0,j)$", size=10.2, ha="center"
    )
    text(
        ax,
        x + 0.017,
        0.299,
        "Ignore self-transfer; keep direction.",
        size=8.7,
        color=MUTED,
    )

    x = xs[2]
    for y, color, title, groups in [
        (0.755, BLUE, "High-transfer search", ["A  B  C  D", "E  F  G  H"]),
        (0.548, ORANGE, "Low-transfer search", ["A  C  E  G", "B  D  F  H"]),
    ]:
        text(ax, x + 0.017, y, title, size=11, color=color, weight="bold")
        for j, label in enumerate(groups):
            left = x + 0.017 + j * 0.095
            box(ax, left, y - 0.086, 0.087, 0.05, edge=color, radius=0.005)
            text(ax, left + 0.0435, y - 0.061, label, size=9, color=color, ha="center")
    text(
        ax,
        x + 0.017,
        0.626,
        "Protect the weakest group's\npositive gain and separation.",
        size=9.5,
    )
    text(ax, x + 0.017, 0.421, "Minimize average\nwithin-group gain.", size=9.5)
    box(ax, x + 0.015, 0.331, 0.186, 0.039, face="#F0F3F7", edge="none", radius=0.005)
    text(
        ax, x + width / 2, 0.35, "Equal sizes · every task once", size=8.8, ha="center"
    )
    text(
        ax, x + 0.017, 0.299, "CPU search over task assignments.", size=8.7, color=MUTED
    )

    x = xs[3]
    text(
        ax, x + 0.017, 0.748, "Separate fitting from scoring", size=10.5, weight="bold"
    )
    for i in range(8):
        box(
            ax,
            x + 0.017 + i * 0.0236,
            0.686,
            0.0205,
            0.035,
            face="#D7DFE8" if i < 6 else "#B9DCCB",
            edge="none",
            radius=0.003,
        )
    text(
        ax,
        x + width / 2,
        0.654,
        "6 prompts fit · 2 prompts held out",
        size=8.8,
        color=MUTED,
        ha="center",
    )
    text(ax, x + 0.017, 0.59, "Repeat four stratified folds.", size=10)
    text(ax, x + 0.017, 0.527, "Compare with random groups.", size=10)
    text(ax, x + 0.017, 0.473, "Freeze task assignments.", size=10)
    box(ax, x + 0.017, 0.342, 0.182, 0.087, face="white", linestyle="--")
    text(
        ax,
        x + width / 2,
        0.4,
        "Confirm on fresh prompts",
        size=10,
        weight="bold",
        ha="center",
    )
    text(
        ax,
        x + width / 2,
        0.368,
        "Leave the target's own adapter out.",
        size=8.5,
        ha="center",
    )
    text(
        ax, x + 0.017, 0.299, "Report uncertainty and direction.", size=8.7, color=MUTED
    )

    box(ax, 0.035, 0.13, 0.93, 0.085, face="#F0F4F8", edge="none")
    text(
        ax,
        0.5,
        0.19,
        f"{n} tasks · k = {high['k']} · held-out within-group transfer",
        size=10,
        color=MUTED,
        ha="center",
    )
    for xx, label, value, color in [
        (0.25, "High", high["cv_within_mean"], BLUE),
        (0.5, "Random", random_mean, MUTED),
        (0.75, "Low", low["cv_within_mean"], ORANGE),
    ]:
        text(
            ax,
            xx,
            0.153,
            f"{label}  {100 * value:+.2f} pp",
            size=16,
            weight="bold",
            color=color,
            ha="center",
        )
    text(
        ax,
        0.035,
        0.085,
        "Task letters and the small matrix are schematic; the numeric summary uses the measured N64 experiment.",
        size=9,
        color=MUTED,
    )
    text(
        ax,
        0.035,
        0.051,
        "Transfer is the change in pass@1 under another task's adapter. Joint-training interference requires a separate test.",
        size=9,
        color=MUTED,
    )
    save(fig, output, "clustering_procedure")


def evidence(output, positive, low, high_row, low_row, transfer):
    fig = plt.figure(figsize=(16, 7.5), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    text(
        ax,
        0.035,
        0.952,
        "Same measurements, different task groups",
        size=24,
        weight="bold",
    )
    text(
        ax,
        0.035,
        0.905,
        "Discovery views show how tasks are arranged; held-out evaluations test whether the grouping generalizes.",
        size=11.5,
        color=MUTED,
    )
    k = high_row["k"]
    matrices = []
    for row in [high_row, low_row]:
        measured = balanced_metrics(transfer, row["fit"]["groups"])
        np.testing.assert_allclose(
            measured["block_matrix"], row["fit"]["block_matrix"], atol=1e-12
        )
        matrices.append(100 * np.asarray(measured["block_matrix"]))
    bound = float(np.ceil(max(np.abs(m).max() for m in matrices)))
    cmap, norm = plt.get_cmap("RdBu"), TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    for left, title, matrix, color, row in zip(
        [0.05, 0.365],
        ["High-transfer assignment", "Low-transfer assignment"],
        matrices,
        [BLUE, ORANGE],
        [high_row, low_row],
    ):
        h = fig.add_axes([left, 0.325, 0.265, 0.51])
        image = h.imshow(matrix, cmap=cmap, norm=norm)
        h.set_title(title, fontsize=13, color=color, fontweight="bold", pad=14)
        h.set_xticks(range(k), [str(i + 1) for i in range(k)], fontsize=9)
        h.set_yticks(range(k), [str(i + 1) for i in range(k)], fontsize=9)
        h.set_xlabel("Evaluation task group", fontsize=10)
        h.set_ylabel("Adapter training-task group", fontsize=10)
        h.tick_params(length=0)
        for i in range(k):
            for j in range(k):
                value = matrix[i, j]
                h.text(
                    j,
                    i,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    color="white" if abs(value) > 0.56 * bound else INK,
                    fontweight="bold" if i == j else "normal",
                )
            h.add_patch(
                Rectangle(
                    (i - 0.5, i - 0.5), 1, 1, fill=False, linewidth=1.8, edgecolor=color
                )
            )
        text(
            ax,
            left + 0.1325,
            0.265,
            f"Discovery within: {100 * row['fit']['within']:+.2f} pp",
            size=10,
            color=color,
            weight="bold",
            ha="center",
        )
    cax = fig.add_axes([0.13, 0.195, 0.40, 0.023])
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    colorbar.set_label(
        "Mean individual-adapter transfer vs. base (percentage points)", fontsize=9
    )
    colorbar.ax.tick_params(labelsize=8)
    text(
        ax,
        0.342,
        0.132,
        f"Discovery data · {len(positive['environments'])} tasks · k = {k} · {high_row['size']} tasks per group",
        size=9,
        color=MUTED,
        ha="center",
    )

    h = fig.add_axes([0.72, 0.315, 0.255, 0.52])
    ks = [row["k"] for row in positive["candidates"]]
    high_se = [
        np.std([fold["held_out"]["within"] for fold in row["folds"]], ddof=1) / 2
        for row in positive["candidates"]
    ]
    h.errorbar(
        ks,
        [100 * r["cv_within_mean"] for r in positive["candidates"]],
        yerr=100 * np.asarray(high_se),
        color=BLUE,
        marker="o",
        markersize=5,
        capsize=3,
        label="High-transfer groups",
    )
    h.errorbar(
        ks,
        [100 * r["cv_within_mean"] for r in low["candidates"]],
        yerr=[100 * r["cv_within_se"] for r in low["candidates"]],
        color=ORANGE,
        marker="o",
        markersize=5,
        capsize=3,
        label="Low-transfer groups",
    )
    h.axhline(
        100 * low["global_offdiagonal_mean"],
        color="#8592A0",
        linestyle="--",
        label="Random expectation",
    )
    h.axhline(0, color=INK, linewidth=0.7)
    h.set_xscale("log", base=2)
    h.set_xticks(ks, [str(k) for k in ks])
    h.set(
        xlabel="Number of balanced groups (k)",
        ylabel="Within-group transfer (pp)",
        title="Held-out transfer",
    )
    h.legend(fontsize=8.5, loc="upper left")
    h.grid(alpha=0.17)
    text(
        ax,
        0.72,
        0.225,
        "Low transfer generalizes.\nNegative averages remain uncertain.",
        size=10.5,
        weight="bold",
    )
    text(
        ax,
        0.72,
        0.151,
        "Error bars: ±1 standard error\nacross four prompt folds (descriptive).",
        size=9,
        color=MUTED,
    )
    text(
        ax,
        0.035,
        0.075,
        "Heatmap cells average individually trained adapters over source and target task groups; diagonal cells exclude self-transfer.",
        size=9,
        color=MUTED,
    )
    text(
        ax,
        0.035,
        0.041,
        "The assignment changes, not the adapter weights. These are transfer proxies, not measurements of jointly trained group models.",
        size=9,
        color=MUTED,
    )
    save(fig, output, "clustering_evidence")


def write_explanation(output, k):
    content = f"""# Communicating LoRA transfer clustering

Suggested one-sentence framing:

**Train one adapter per task, measure which other tasks it helps, then search for groups that keep helpful—or deliberately unhelpful—relationships together.**

## Figures

- [Procedure overview](clustering_procedure.png) · [PDF](clustering_procedure.pdf) · [editable SVG](clustering_procedure.svg)
- [Measured evidence](clustering_evidence.png) · [PDF](clustering_evidence.pdf) · [editable SVG](clustering_evidence.svg)

The procedure schematic is intended for a talk or the first methods figure. The evidence figure separates an appealing discovery-data arrangement from held-out evidence. Its heatmaps use k={k} for readability; the final panel sweeps all measured k values. Matrix rows group individually trained task adapters: these are not group-trained models.

## Suggested caption: procedure

**Learning task groupings from reusable LoRA transfer measurements.** We train one task-specific adapter per task on a shared frozen base, then evaluate every adapter on every task. Directed transfer is the change in pass@1 relative to the unchanged base on the target task; self-transfer is excluded from clustering. We search equal-sized partitions of all tasks using the cached transfer matrix, without retraining candidate groups. The high-transfer objective protects the weakest cluster's positive internal transfer and separation from its exterior. The low-transfer objective minimizes mean internal transfer. Assignments are fit on six evaluation prompts per task and scored on two held-out prompts in each of four difficulty-stratified folds. Fixed assignments can then be checked on disjoint prompts, withholding the target task's own adapter when measuring transfer. Task letters and the small matrix are illustrative; the numerical strip reports measured N64 results.

## Suggested caption: evidence

**Grouping the same tasks to encourage or suppress transfer.** Left and middle: group-averaged discovery transfer under high- and low-transfer assignments of the same 64 tasks. Each cell averages directed individual-adapter effects from source-group tasks to target-group tasks; diagonal cells omit a task's effect on itself. Right: mean within-group transfer on held-out prompt folds, with descriptive ±1 fold standard error. Uniform random equal-sized partitions have expected within-group transfer equal to the global off-diagonal mean. Low-transfer assignments reduce internal transfer, but their average negative transfer is not firmly established: at k=32, held-out within-transfer is −0.49 pp with a fold standard error of 1.68 pp.

## Short methods paragraph

Let T[i,j] be pass@1 on task j after applying the adapter trained on task i, minus base-model pass@1 on task j. For each k we partition all N tasks into k disjoint groups of N/k tasks. For cluster c, W[c] is mean directed transfer between distinct members; O[c] and I[c] are mean outgoing and incoming transfer across its boundary. The high-transfer search maximizes min_c min(W[c], W[c] − max(O[c], I[c])), with mean separation as a tie-breaker. The low-transfer search minimizes mean_c W[c]. Both preserve exact group sizes using multi-start improving swaps; the low-transfer search uses minimum-weight perfect matching for pairs. These are different objectives, not exact sign reversals. The larger-group searches are heuristics, not global optimum certificates.

## Interpretation and validation

- **The cheap part is reusing measurements.** Training N adapters and measuring the N×N transfer matrix is paid once. Candidate partition searches do not require training a model for every grouping.
- **Transfer is directional.** T[i,j] need not equal T[j,i]. A negative pair average does not prove harm in both directions.
- **Keep discovery and validation separate.** Four-fold prompt cross-validation refits assignments on each fold; it evaluates the grouping procedure, not one independently confirmed final partition.
- **Use “high-transfer” and “low-transfer” rather than universally “best” and “worst.”** The positive objective targets its weakest cluster, while the reverse objective minimizes an average and can sacrifice individual groups.
- **Confirmation has different coverage.** The selected positive pairs have a complete disjoint confirmation experiment, including leave-one-task-out and full adapter-composition controls. Only 7 of 64 directed cells of the low-transfer full-data pairing happened to be available in the earlier confirmation cache; the whole low-transfer partition has not received fresh confirmation.
- **Adapter composition is a separate check.** When used, we average adapter weight updates and compress to rank 32 using SVD; we do not separately average LoRA factors. Neither single-adapter transfer nor adapter composition proves interference under actual joint training.
- **Next stage: MOPD.** Use the discovered partitions to define teacher task groups, then evaluate student learning under the chosen resource budget. Clustering measurements alone do not optimize or establish final MOPD performance.

## Reproduce

```bash
apptainer exec --no-home --bind /juice5b/u/dhei/tatsu/rlve \\
  --env PYTHONPATH=/juice5b/u/dhei/tatsu/rlve/tools \\
  --env MPLCONFIGDIR=/tmp/clustering-method-mpl \\
  /juice5b/scr5b/dhei/images/miles-dev-202610231907.sif \\
  python3 tools/create_clustering_method_figures.py
```

The script reads the existing positive and low-transfer analyses. It launches no training or GPU evaluation jobs. Exact inputs, summary data and group assignments are recorded in figure_data.json.
"""
    (output / "README.md").write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "plot/mopd-clustering/n64"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "plot/mopd-clustering/method"
    )
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    paths = [
        args.data_root / "balanced/results.json",
        args.data_root / "low-transfer/results.json",
        args.data_root / "balanced/transfer.json",
    ]
    positive, low, raw = [json.loads(path.read_text()) for path in paths]
    if (
        positive["environments"] != low["environments"]
        or positive["observation_sha256"] != low["observation_sha256"]
    ):
        raise ValueError("Both searches must use identical tasks and observations")
    high_row = next(row for row in positive["candidates"] if row["k"] == args.k)
    low_row = next(row for row in low["candidates"] if row["k"] == args.k)
    transfer = np.asarray(raw)
    for row in [high_row, low_row]:
        if sorted(i for group in row["fit"]["groups"] for i in group) != list(
            range(len(transfer))
        ):
            raise ValueError("Every task must appear once")
        if len({len(group) for group in row["fit"]["groups"]}) != 1:
            raise ValueError("Groups must be equal-sized")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.labelcolor": INK,
            "text.color": INK,
        }
    )
    overview(
        args.output, len(transfer), high_row, low_row, low["global_offdiagonal_mean"]
    )
    evidence(args.output, positive, low, high_row, low_row, transfer)
    write_explanation(args.output, args.k)
    payload = {
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "k_for_discovery_heatmaps": args.k,
        "environments": positive["environments"],
        "high_groups": high_row["fit"]["groups"],
        "low_groups": low_row["fit"]["groups"],
        "high_discovery_block_matrix": high_row["fit"]["block_matrix"],
        "low_discovery_block_matrix": low_row["fit"]["block_matrix"],
        "high_cv_within": [
            {"k": r["k"], "mean": r["cv_within_mean"]} for r in positive["candidates"]
        ],
        "low_cv_within": [
            {"k": r["k"], "mean": r["cv_within_mean"], "fold_se": r["cv_within_se"]}
            for r in low["candidates"]
        ],
        "random_expectation": low["global_offdiagonal_mean"],
        "schematic_is_illustrative": True,
        "actual_data_panels_use_discovery_observations": True,
        "held_out_scores_separate_from_discovery": True,
    }
    (args.output / "figure_data.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
