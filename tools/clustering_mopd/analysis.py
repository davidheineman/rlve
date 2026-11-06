"""Compare measured student quality at common simulated deadlines."""

import argparse
import json
from pathlib import Path


def observations(experiment):
    path = experiment / "workers/student/metrics.jsonl"
    if not path.exists():
        return []
    by_clock = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "david/eval/val_envs/pass@1" in row:
            by_clock[row["async/simulated_clock"]] = row
    return sorted(by_clock.values(), key=lambda row: row["async/simulated_clock"])


def at_clock(rows, clock):
    eligible = [row for row in rows if row["async/simulated_clock"] <= clock]
    return eligible[-1] if eligible else None


def score(rows, horizon):
    if not rows or rows[0]["async/simulated_clock"] != 0:
        raise ValueError("A measured baseline is required")
    marks = sorted(
        {
            0,
            horizon,
            *[
                r["async/simulated_clock"]
                for r in rows
                if r["async/simulated_clock"] <= horizon
            ],
        }
    )
    area = sum(
        (b - a) * at_clock(rows, a)["david/eval/val_envs/pass@1"]
        for a, b in zip(marks, marks[1:])
    )
    final = at_clock(rows, horizon)
    return {
        "final_id_pass1": final["david/eval/val_envs/pass@1"],
        "final_unseen_pass1": final.get("david/eval/test_envs/pass@1"),
        "mean_measured_id_over_clock": area / horizon,
        "student_rollouts": final["david/rollouts"],
        "physical_elapsed_hours": final.get("physical/elapsed_seconds", 0) / 3600,
        "time_to_id_accuracy": {
            str(target): next(
                (
                    r["async/simulated_clock"]
                    for r in rows
                    if r["async/simulated_clock"] <= horizon
                    and r["david/eval/val_envs/pass@1"] >= target
                ),
                None,
            )
            for target in (0.25, 0.35, 0.45)
        },
    }


def plot_composed_comparison(manifest, curves, ks):
    """Match the n32 comparison's panel allocation, axes, and typography."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

    style = {
        "font.size": 8,
        "axes.titlesize": 9.6,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 6.8,
        "lines.linewidth": 1.2,
    }
    colors = {
        "high-transfer": "#126FA4",
        "random": "#758291",
        "low-transfer": "#C85A27",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            2,
            len(ks),
            figsize=(3.2 * 0.8 * len(ks), 2.8 * 0.8 * 2),
            sharex="row",
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        for item, rows in curves:
            column = ks.index(item["cluster_count"])
            assignment = item["assignment"]
            for index, field in enumerate(("david/rollouts", "async/simulated_clock")):
                axes[index, column].step(
                    [row[field] for row in rows],
                    [100 * row["david/eval/val_envs/pass@1"] for row in rows],
                    where="post",
                    color=colors[assignment],
                    marker="o",
                    markersize=2.4,
                    label=assignment.replace("-", " "),
                )
        for column, k in enumerate(ks):
            environments = manifest.get("size", 64) // k
            axes[0, column].set_title(f"{k} teacher groups × {environments} envs")
            axes[0, column].set_xlabel("student rollouts")
            axes[1, column].set_xlabel("simulated wall time (rollouts)")
            axes[1, column].set_xlim(0, manifest["simulated_clock_budget"])
            for ax in axes[:, column]:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda n, _: "0" if n == 0 else f"{n / 1000:g}K")
                )
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
                ax.grid(alpha=0.2)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(loc="lower right")
                else:
                    ax.text(
                        0.5, 0.5, "Awaiting student evaluations",
                        transform=ax.transAxes, ha="center", color="#576779",
                    )
        fig.suptitle(f"{manifest.get('size', 64)} env val acc", fontsize=12.8)
        fig.supylabel("pass@1", fontsize=9.6)
    return fig


def analyze(sweep, output=None):
    manifest = json.loads((sweep / "sweep.json").read_text())
    items = list(manifest["experiments"])
    for reference in manifest.get("comparison_sweeps", []):
        previous = json.loads((Path(reference) / "sweep.json").read_text())
        for field in (
            "size",
            "simulated_devices",
            "simulated_clock_budget",
            "assignment_method",
        ):
            if previous[field] != manifest[field]:
                raise ValueError(f"Comparison sweep differs in {field}")
        items.extend(previous["experiments"])
    keys = [(item["cluster_count"], item["assignment"]) for item in items]
    if len(set(keys)) != len(keys):
        raise ValueError("Comparison sweeps contain duplicate assignment panels")
    output = output or sweep / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    records, curves = [], []
    for item in items:
        experiment = Path(item["experiment"])
        rows = observations(experiment)
        if not rows:
            continue
        complete = (experiment / "completed.json").exists()
        horizon = (
            manifest["simulated_clock_budget"]
            if complete
            else rows[-1]["async/simulated_clock"]
        )
        if horizon == 0:
            continue
        record = {
            "label": item["label"],
            "k": item["cluster_count"],
            "assignment": item["assignment"],
            "complete": complete,
            "scored_until_clock": horizon,
            **score(rows, horizon),
        }
        records.append(record)
        curves.append((item, rows))
    payload = {
        "metric": f"Equal-weight ID pass@1 over all {manifest.get('size', 64)} environments",
        "clock": "Fixed-device training-rollout proxy; evaluation time excluded",
        "offline_assignment_cost_included": False,
        "auc": "Piecewise-constant measured curve, carrying the latest available evaluation forward",
        "records": records,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Clustered async MOPD",
        "",
        "Common simulated device budget and deadline; all runs use four physical GPUs.",
        "",
        "| Run | Complete | Student rollouts | ID pass@1 | Unseen pass@1 | Mean ID over clock | Physical hours |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        test = (
            f"{100 * r['final_unseen_pass1']:.2f}%"
            if r["final_unseen_pass1"] is not None
            else "—"
        )
        lines.append(
            f"| {r['label']} | {r['complete']} | {r['student_rollouts']} | {100 * r['final_id_pass1']:.2f}% | {test} | {100 * r['mean_measured_id_over_clock']:.2f}% | {r['physical_elapsed_hours']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Initial queue time is excluded from physical hours; subsequent preemption/queue delays are included.",
            "The simulated clock excludes evaluation and assumes equal per-rollout device speed. This tests cluster granularity under that resource model, not every allocation or hardware scaling policy.",
            "Optimized assignments reuse the earlier LoRA transfer study; that offline discovery cost is excluded here.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n")
    if curves:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter

        ks = sorted({item["cluster_count"] for item in items})
        columns = min(
            3
            if manifest.get("assignment_method") == "composed-search-assignments"
            else 2,
            len(ks),
        )
        panel_rows = (len(ks) + columns - 1) // columns
        colors = {"optimized": "#D55E00", "joint-repeat2": "#D55E00"}
        if manifest.get("assignment_method") == "composed-search-assignments":
            colors.update(
                {
                    "high-transfer": "#126FA4",
                    "random": "#758291",
                    "low-transfer": "#C85A27",
                }
            )

        for axis, field, scale, xlabel in (
            ("simulated", "async/simulated_clock", 1, "Simulated device rollout clock"),
            (
                "physical",
                "physical/elapsed_seconds",
                3600,
                "Physical elapsed hours on four GPUs",
            ),
        ):
            fig, axes = plt.subplots(
                panel_rows,
                columns,
                figsize=(5.5 * columns, 3.5 * panel_rows),
                sharex=True,
                sharey=True,
                squeeze=False,
                layout="constrained",
            )
            panels = dict(zip(ks, axes.flat))
            for item, rows in curves:
                ax = panels[item["cluster_count"]]
                assignment = item["assignment"]
                values = [r for r in rows if field in r]
                ax.step(
                    [r[field] / scale for r in values],
                    [100 * r["david/eval/val_envs/pass@1"] for r in values],
                    label=assignment.replace("-", " "),
                    color=colors.get(assignment, "#0072B2"),
                    linestyle="--" if assignment == "joint-repeat2" else "-",
                    where="post",
                    marker="o",
                    markersize=3,
                )
            for k, ax in panels.items():
                environments = manifest.get("size", 64) // k
                students = "student" if k == 1 else "students"
                envs = "env" if environments == 1 else "envs"
                ax.set_title(
                    f"{k} teacher groups × {environments} environments"
                    if manifest.get("assignment_method")
                    == "composed-search-assignments"
                    else f"{k} {students} at {environments} {envs} per student"
                )
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
                if axis == "simulated":
                    ax.set_xlim(0, manifest["simulated_clock_budget"])
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=9)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "Awaiting student evaluations",
                        transform=ax.transAxes,
                        ha="center",
                        color="#576779",
                    )
                ax.grid(alpha=0.2)
            for ax in list(axes.flat)[len(ks) :]:
                ax.set_visible(False)
            fig.suptitle(f"{manifest.get('size', 64)} env val acc", fontsize=16)
            fig.supxlabel(xlabel)
            fig.supylabel("pass@1")
            for extension in ("png", "pdf"):
                fig.savefig(output / f"student_quality_vs_{axis}_time.{extension}")
            if axis == "simulated" and manifest.get("publish_plot_prefix"):
                prefix = Path(manifest["publish_plot_prefix"])
                prefix.parent.mkdir(parents=True, exist_ok=True)
                comparison = (
                    plot_composed_comparison(manifest, curves, ks)
                    if manifest.get("assignment_method") == "composed-search-assignments"
                    else fig
                )
                for extension in manifest.get("publish_plot_formats", ("png", "pdf", "svg")):
                    comparison.savefig(prefix.with_suffix("." + extension), dpi=180)
                if comparison is not fig:
                    plt.close(comparison)
            plt.close(fig)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.sweep, args.output), indent=2))


if __name__ == "__main__":
    main()
