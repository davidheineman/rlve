"""Per-quartet learning curves and the independent final evaluation endpoint."""

import argparse
import json
import time
from pathlib import Path


def learning_line(item, rows, field, scale):
    """Interpolate evaluations, preserving the unchanged policy across teacher phases."""
    experiment = Path(item["experiment"])
    synchronous = item["algorithm"] == "sync-mopd"
    start_path = experiment / "physical-start.json"
    physical_start = (
        json.loads(start_path.read_text())["time"] if start_path.exists() else None
    )
    xs, ys = [], []
    previous = None

    def barrier(round_index, first_evaluation=None):
        task = experiment / "tasks" / f"round-{round_index:02d}-student" / "task.json"
        if field == "budget/total_training_rollouts":
            if task.exists():
                return json.loads(task.read_text())["start_work"] / scale
            if first_evaluation is not None and previous is not None:
                student_work = (
                    first_evaluation["david/rollouts"] - previous["david/rollouts"]
                )
                return (first_evaluation[field] - student_work) / scale
        elif physical_start is not None:
            completed_teachers = list(
                (experiment / "tasks").glob(
                    f"round-{round_index:02d}-teacher-*/result.json"
                )
            )
            if completed_teachers:
                return (
                    max(p.stat().st_mtime for p in completed_teachers) - physical_start
                ) / scale
        return None

    for row in rows:
        x, y = row[field] / scale, 100 * row["david/eval/val_envs/pass@1"]
        if (
            synchronous
            and previous is not None
            and row.get("sync/round", 0) > previous.get("sync/round", 0)
        ):
            idle_end = barrier(row["sync/round"], row)
            if idle_end is not None and xs[-1] < idle_end <= x:
                xs.append(idle_end)
                ys.append(ys[-1])
        xs.append(x)
        ys.append(y)
        previous = row

    if synchronous and previous is not None:
        state_path = experiment / "sync-state.json"
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "round": 1,
                "phase": "teachers",
                "total_training_rollouts": 0,
            }
        )
        if state["phase"] == "teachers" and state["round"] > previous.get(
            "sync/round", 0
        ):
            if field == "physical/elapsed_seconds":
                idle_end = (
                    (time.time() - physical_start) / scale
                    if physical_start is not None
                    else None
                )
            else:
                spent = 0
                for task in (experiment / "tasks").glob(
                    f"round-{state['round']:02d}-teacher-*"
                ):
                    result, latest = task / "result.json", task / "latest.json"
                    used = 0
                    if result.exists():
                        used = json.loads(result.read_text())["rollouts"]
                    else:
                        if latest.exists():
                            name = Path(
                                json.loads(latest.read_text())["resume"]
                            ).parent.name
                            if name.startswith("u") and name[1:].isdigit():
                                used = 8 * int(name[1:])
                        for heartbeat in (experiment / "workers").glob(
                            "teacher-*/heartbeat.json"
                        ):
                            record = json.loads(heartbeat.read_text())
                            if record.get("task_id") == task.name:
                                used = max(used, record.get("used", 0))
                    spent += used
                idle_end = (state["total_training_rollouts"] + spent) / scale
            if idle_end is not None and idle_end > xs[-1]:
                xs.append(idle_end)
                ys.append(ys[-1])
        elif state["phase"] == "student" and state["round"] > previous.get(
            "sync/round", 0
        ):
            idle_end = barrier(state["round"])
            if idle_end is not None and idle_end > xs[-1]:
                xs.append(idle_end)
                ys.append(ys[-1])
    return xs, ys


def analyze(sweep):
    manifest = json.loads((sweep / "sweep.json").read_text())
    output = Path(manifest["analysis_output"])
    output.mkdir(parents=True, exist_ok=True)
    records, curves = [], []
    for item in manifest["experiments"]:
        experiment = Path(item["experiment"])
        path = experiment / "workers/student/metrics.jsonl"
        rows = (
            [json.loads(line) for line in path.read_text().splitlines() if line]
            if path.exists()
            else []
        )
        monitoring = [r for r in rows if "david/eval/val_envs/pass@1" in r]
        final_path = experiment / "final-evaluation.json"
        final = json.loads(final_path.read_text()) if final_path.exists() else {}
        done = (
            json.loads((experiment / "completed.json").read_text())
            if (experiment / "completed.json").exists()
            else {}
        )
        records.append(
            {
                **item,
                "complete": bool(done),
                "total_training_rollouts": done.get("total_training_rollouts"),
                "student_rollouts": done.get("student_rollouts"),
                "final_id_pass1": final.get("david/eval/final_id_envs/pass@1"),
                "final_unseen_pass1": final.get("david/eval/final_test_envs/pass@1"),
                "final_per_task_pass1": {
                    key.split("/")[-2]: value
                    for key, value in final.items()
                    if key.startswith("david/eval/final_id_envs/")
                    and key.count("/") == 4
                },
                "physical_hours": final.get("physical/elapsed_seconds", 0) / 3600
                if final
                else None,
            }
        )
        curves.append((item, monitoring))
    paired = []
    for group in sorted({r["group"] for r in records}):
        members = {r["algorithm"]: r for r in records if r["group"] == group}
        a, b = members.get("joint-rl"), members.get("sync-mopd")
        if a and b and a["complete"] and b["complete"]:
            paired.append(
                {
                    "group": group,
                    "mopd_minus_joint_id_pp": 100
                    * (b["final_id_pass1"] - a["final_id_pass1"]),
                    "mopd_minus_joint_unseen_pp": 100
                    * (b["final_unseen_pass1"] - a["final_unseen_pass1"]),
                }
            )
    payload = {
        "records": records,
        "paired_final_differences": paired,
        "primary_endpoint": "Disjoint final 96-prompt macro ID pass@1, eight completions per prompt",
        "budget": "Common training-completion cap; actual teacher plus student work is reported, and early-stopped MOPD can spend less",
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Four-task transfer-regime comparison",
        "",
        "| Group | Method | Complete | Actual training completions | Final ID | Final unseen | Elapsed hours |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    def percent(value):
        return "—" if value is None else f"{100 * value:.2f}%"

    for record in records:
        hours = (
            "—"
            if record["physical_hours"] is None
            else f"{record['physical_hours']:.1f}"
        )
        lines.append(
            f"| {record['group']} | {record['algorithm']} | {record['complete']} | {record['total_training_rollouts'] or '—'} | {percent(record['final_id_pass1'])} | {percent(record['final_unseen_pass1'])} | {hours} |"
        )
    lines += [
        "",
        "Final ID is isolated from monitoring and teacher stopping. Near-zero controls are selected by net discovery transfer; opposing directional effects can cancel. Any random controls were sampled without scoring and need not have zero transfer. Discovery LoRA transfer is a proxy, and task difficulty differs across groups. Two groups per condition and a single training seed are exploratory.",
        "",
        "Physical elapsed hours exclude initial queue time and include later preemption/queue delays; each job requests four GPUs. Completion counts include teachers and students. Evaluation and token costs can differ even at equal training-completion counts.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    if paired:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=(9, 4))
        axis.bar(
            [r["group"] for r in paired], [r["mopd_minus_joint_id_pp"] for r in paired]
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(
            ylabel="MOPD − JointRL final ID pass@1 (pp)",
            title="Independent final evaluation",
        )
        axis.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        for extension in ["png", "pdf"]:
            fig.savefig(output / f"final-mopd-gain.{extension}")
        plt.close(fig)
    if any(rows for _, rows in curves):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        condition_order = {"best": 0, "neutral": 1, "random": 1, "worst": 2}
        titles = {
            "best-1": "Best",
            "best-2": "2nd Best",
            "neutral-1": "Near-zero (set 1)",
            "neutral-2": "Near-zero (set 2)",
            "random-1": "Random (set 1)",
            "random-2": "Random (set 2)",
            "worst-1": "Worst",
            "worst-2": "2nd Worst",
        }
        groups = sorted(
            {item["group"] for item, _ in curves},
            key=lambda group: (
                int(group.rsplit("-", 1)[1]),
                condition_order[group.rsplit("-", 1)[0]],
            ),
        )
        for field, scale, label, suffix in [
            (
                "budget/total_training_rollouts",
                1,
                "total rollouts",
                "work",
            ),
            (
                "physical/elapsed_seconds",
                3600,
                "Elapsed hours on four GPUs",
                "wall-time",
            ),
        ]:
            fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharey=True)
            for group, axis in zip(groups, axes.flat):
                for item, rows in curves:
                    if item["group"] == group and rows:
                        x, y = learning_line(item, rows, field, scale)
                        observed = {
                            (r[field] / scale, 100 * r["david/eval/val_envs/pass@1"])
                            for r in rows
                        }
                        axis.plot(
                            x,
                            y,
                            linewidth=1.0,
                            marker="o",
                            markersize=2.5,
                            markeredgewidth=0,
                            markevery=[
                                i
                                for i, point in enumerate(zip(x, y))
                                if point in observed
                            ],
                            label={"joint-rl": "GRPO", "sync-mopd": "MOPD (sync LoRA)"}[
                                item["algorithm"]
                            ],
                            color={"joint-rl": "#1f77b4", "sync-mopd": "#ff7f0e"}[
                                item["algorithm"]
                            ],
                        )
                axis.set(title=titles.get(group, group), xlabel=label, ylabel="pass@1")
                axis.grid(alpha=0.2)
                axis.set_ylim(0, 100)
                if suffix == "work":
                    plan = json.loads(
                        (
                            Path(manifest["experiments"][0]["experiment"]) / "plan.json"
                        ).read_text()
                    )
                    axis.set_xlim(0, plan["total_training_rollout_cap"])
                    axis.set_xticks(
                        range(0, plan["total_training_rollout_cap"] + 1, 5000)
                    )
                    axis.xaxis.set_major_formatter(
                        FuncFormatter(
                            lambda value, _: "0" if value == 0 else f"{value / 1000:g}K"
                        )
                    )
                if axis.lines:
                    axis.legend(fontsize=8)
            fig.suptitle("4 env RL vs. MOPD")
            fig.tight_layout()
            for extension in ["png", "pdf"]:
                fig.savefig(output / f"learning-vs-{suffix}.{extension}")
            plt.close(fig)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.sweep), indent=2))


if __name__ == "__main__":
    main()
