"""Publish completed controlled trials and their scored rollouts to one audit run."""

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

from modular_rl.prepare import write_json

from .ablation_report import plot, report
from .inspect_rollouts import annotate, read_events
from .verify_probe_alignment import verify


def publish(experiments, review, more_trials_pending=False):
    import wandb

    analyses = [report(experiment) for experiment in experiments]
    for experiment in experiments:
        verify(experiment)
    if any(not run["completed"] for value in analyses for run in value["runs"]):
        raise ValueError("Publish final results only after every requested trial completes")
    models = sorted({run["model"] for value in analyses for run in value["runs"]})
    overview, examples, series = [], [], {"math": [], "code": []}
    for value in analyses:
        plot(value)
        experiment = Path(value["experiment"])
        plan = json.loads((experiment / "plan.json").read_text())
        for info in value["runs"]:
            evaluations = info["validation"]
            label = (f"{info['model']}; {info['variant']}; p{plan.get('prompts_per_update', 1)}; "
                     f"{info['responses']} responses; {plan.get('training_pool', 'full')}")
            overview.append([
                experiment.name, info["task"], label, info["wandb_id"],
                "https://forge.coreweave.com/wandb/david-heineman/modular-mopd-rl-initial/runs/" + info["wandb_id"],
                info["updates"], info["optimizer_updates"], info["optimizer_updates_skipped"],
                info["responses"], evaluations[0]["pass@1"],
                max(e["pass@1"] for e in evaluations), evaluations[-1]["pass@1"],
                evaluations[-1]["pass@8"],
                evaluations[-1]["behavior"]["finished_correct_fraction"],
                evaluations[-1]["behavior"]["truncated_fraction"],
                json.dumps(evaluations[-1]["vs_initial"]), json.dumps(info["post_update"]),
            ])
            directory = Path(info["directory"])
            signals = read_events(directory / "signal.jsonl")
            mixed = [int(row["david/step"]) for row in signals if row["david/train/signal_groups"]]
            indices = sorted({0, info["updates"] - 1, *mixed[:2], *mixed[-2:]})
            if info.get("largest_gradient_update"):
                indices = sorted({*indices, info["largest_gradient_update"]["rollout_id"]})
            for index in indices:
                path = directory / "rollouts/train" / f"{index:04d}.jsonl"
                for raw in read_events(path):
                    row = annotate(raw, path, label)
                    examples.append([
                        experiment.name, info["task"], label, index + 1, row["sample_index"],
                        json.dumps(row["prompt"], ensure_ascii=False), row["response"],
                        row["final_answer"], row["accuracy"], row["reward"], row["forced"], row["sampled_tokens"],
                        row["status"],
                        json.dumps(row["target"], ensure_ascii=False),
                    ])
            series[info["task"]].append((label, [e["updates"] * plan.get("prompts_per_update", 1) * 8 for e in evaluations],
                                         [e["pass@1"] for e in evaluations], [e["pass@8"] for e in evaluations],
                                         [e["behavior"]["finished_correct_fraction"] for e in evaluations],
                                         [e["behavior"]["truncated_fraction"] for e in evaluations]))
    with tempfile.TemporaryDirectory(prefix="modular-ablation-review-") as local:
        previous = review / "initial-publication.json"
        resume_args = {"id": json.loads(previous.read_text())["run_id"], "resume": "must"} if previous.exists() else {}
        run = wandb.init(
            entity="david-heineman", project="modular-mopd-rl-initial",
            name="Qwen3 / 2k | trainer audit and controlled trials", job_type="ablation-review",
            group="modular-v2-rollout-audits", tags=["diagnostics", "controlled-ablation"],
            config={"inspection_only": True, "experiments": [str(p) for p in experiments]},
            settings=wandb.Settings(mode="online", init_timeout=120), dir=local,
            **resume_args,
        )
        run.config.update({"controlled_trials_in_progress": more_trials_pending}, allow_val_change=True)
        metrics = {
            "ablation/outcomes": wandb.Table(columns=[
                "experiment", "task", "variant", "source_run_id", "source_run_url", "rollout_iterations",
                "optimizer_updates", "skipped_optimizer_updates", "training_responses",
                "initial_pass@1", "peak_pass@1", "final_pass@1", "final_pass@8", "final_finished_correct_fraction",
                "final_truncated_fraction", "change_from_initial", "post_update_measurements",
            ], data=overview),
            "ablation/train_rollouts": wandb.Table(columns=[
                "experiment", "task", "variant", "update", "sample", "prompt", "full_response",
                "final_answer", "accuracy", "training_reward", "forced", "sampled_tokens", "status", "grader_target",
            ], data=examples),
        }
        for task, curves in series.items():
            if curves:
                metrics[f"ablation/{task}/pass@1"] = wandb.plot.line_series(
                    xs=[c[1] for c in curves], ys=[c[2] for c in curves], keys=[c[0] for c in curves],
                    title=f"{task}: 32-problem held-out pass@1", xname="training_responses",
                )
                metrics[f"ablation/{task}/pass@8"] = wandb.plot.line_series(
                    xs=[c[1] for c in curves], ys=[c[3] for c in curves], keys=[c[0] for c in curves],
                    title=f"{task}: 32-problem held-out pass@8", xname="training_responses",
                )
                for key, index in (("finished_correct_fraction", 4), ("truncated_fraction", 5)):
                    metrics[f"ablation/{task}/{key}"] = wandb.plot.line_series(
                        xs=[c[1] for c in curves], ys=[c[index] for c in curves], keys=[c[0] for c in curves],
                        title=f"{task}: {key}", xname="training_responses",
                    )
        run.log(metrics)
        artifact_name = "qwen06b-2k-controlled-trials" if models == ["Qwen3-0.6B"] else "qwen3-2k-controlled-trials"
        artifact = wandb.Artifact(artifact_name, type="rl-diagnostics", metadata={"models": models})
        for value in analyses:
            experiment = Path(value["experiment"])
            for name in ("analysis.json", "plan.json", "jobs.json", "ablation-curves.html", "probe-alignment.json"):
                artifact.add_file(str(experiment / name), name=f"{experiment.name}/{name}")
            if (experiment / "runtime-adjustments.jsonl").exists():
                artifact.add_file(str(experiment / "runtime-adjustments.jsonl"), name=f"{experiment.name}/runtime-adjustments.jsonl")
            if (experiment / "resume-audit").exists():
                artifact.add_dir(str(experiment / "resume-audit"), name=f"{experiment.name}/resume-audit")
            artifact.add_dir(str(experiment / "selection"), name=f"{experiment.name}/selection")
            archive_path = Path(local) / (experiment.name + "-responses.tar.gz")
            with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
                for info in value["runs"]:
                    directory = Path(info["directory"])
                    prefix = Path(info["task"]) / info["variant"]
                    archive.add(directory / "rollouts", arcname=str(prefix / "rollouts"))
                    for name in ("update-probes.jsonl", "progress.jsonl", "generation.jsonl", "signal.jsonl", "completed.json", "train.log"):
                        archive.add(directory / name, arcname=str(prefix / name))
                    if (directory / "optimizer-decisions.jsonl").exists():
                        archive.add(directory / "optimizer-decisions.jsonl", arcname=str(prefix / "optimizer-decisions.jsonl"))
            artifact.add_file(str(archive_path), name=archive_path.name)
        artifact.add_dir(str(review), name="original-run-audit")
        report_path = Path(__file__).resolve().parents[2] / "AGENTS.md"
        artifact.add_file(str(report_path), name=report_path.name)
        for directory in review.parent.glob("optimizer-gpu-check-*"):
            artifact.add_dir(str(directory), name=directory.name)
        logged = run.log_artifact(artifact)
        logged.wait()
        result = {"run_id": run.id, "run_url": run.url, "artifact_url": logged.url,
                  "outcome_rows": len(overview), "rollout_rows": len(examples)}
        run.finish()
    write_json(review / "controlled-trials-publication.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", type=Path, nargs="+")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--more-trials-pending", action="store_true")
    args = parser.parse_args()
    publish(args.experiments, args.review, args.more_trials_pending)
