"""Publish a clearly labeled W&B review of existing training rollouts."""

import argparse
import concurrent.futures
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

from modular_rl.prepare import write_json


def archive_run(review, info):
    path = review / "raw" / f"{info['run']}.tar.gz"
    path.parent.mkdir(exist_ok=True)
    source = Path(info["directory"]) / "rollouts/train"
    count = 0
    with tarfile.open(path, "w:gz", compresslevel=1) as archive:
        for batch in sorted(source.glob("*.jsonl")):
            if int(batch.stem) >= info["completed_updates"]:
                continue
            archive.add(batch, arcname=f"{info['run']}/{batch.name}", recursive=False)
            count += 1
    if count != info["saved_groups"]:
        raise RuntimeError("Raw batch archive disagrees with audit cutoff")
    return {
        "run": info["run"],
        "archive": str(path),
        "batches": count,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def attach_to_training_runs(review):
    import wandb

    summary = json.loads((review / "summary.json").read_text())
    batches = json.loads((review / "selected-batches.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        archives = list(
            pool.map(lambda info: archive_run(review, info), summary["runs"])
        )
    write_json(review / "raw-archives.json", archives)
    columns = [
        "update",
        "sample",
        "selection_reason",
        "prompt",
        "final_answer",
        "full_response",
        "grading_target",
        "saved_reward",
        "accuracy",
        "status",
        "forced",
        "sampled_tokens",
        "group_rewards",
        "reconstructed_advantage",
        "gradient_norm",
        "test_diagnostics",
    ]
    api = wandb.Api(timeout=60)
    statuses = {
        r["wandb_id"]: api.run(
            "david-heineman/modular-mopd-rl-initial/" + r["wandb_id"]
        ).state
        for r in summary["runs"]
    }
    artifact_name = "rollout-review-" + review.name
    published = []
    # The longer-budget code run owns the cross-run artifact; each run gets its table.
    ordered = sorted(summary["runs"], key=lambda info: info["run"] != "code-2048")
    artifact_url = None
    for info in ordered:
        rows = []
        for group in batches:
            if group["run"] != info["run"]:
                continue
            for sample in group["samples"]:
                rows.append(
                    [
                        group["rollout_id"] + 1,
                        sample["sample_index"],
                        "; ".join(group["selection_reasons"]),
                        json.dumps(sample["prompt"], ensure_ascii=False),
                        sample["final_answer"],
                        sample["response"],
                        json.dumps(sample["target"], ensure_ascii=False),
                        sample["reward"],
                        sample["accuracy"],
                        sample["status"],
                        sample["forced"],
                        sample["sampled_tokens"],
                        json.dumps(sample["group_rewards"]),
                        sample["reconstructed_group_advantage"],
                        sample["gradient_norm"],
                        json.dumps(sample.get("replay"), ensure_ascii=False),
                    ]
                )
        with tempfile.TemporaryDirectory(prefix="rl-rollout-review-") as local:
            run = wandb.init(
                entity="david-heineman",
                project="modular-mopd-rl-initial",
                id=info["wandb_id"],
                dir=local,
                settings=wandb.Settings(
                    mode="shared",
                    x_primary=False,
                    x_update_finish_state=False,
                    x_label="rollout-review",
                ),
            )
            run.log({"debug/train_rollouts": wandb.Table(columns=columns, data=rows)})
            if artifact_url is None:
                artifact = wandb.Artifact(
                    artifact_name,
                    type="rollout-review",
                    metadata={
                        "created_at": summary["created_at"],
                        "responses": sum(r["saved_responses"] for r in summary["runs"]),
                        "reward_replays": summary["replay"]["responses"],
                    },
                )
                artifact.add_dir(str(review))
                logged = run.log_artifact(artifact)
                logged.wait()
                artifact_url = logged.url
            run.finish()
        api.flush()
        actual = api.run("david-heineman/modular-mopd-rl-initial/" + info["wandb_id"])
        if statuses[info["wandb_id"]] == "running" and actual.state != "running":
            raise RuntimeError(
                "Secondary writer unexpectedly changed the training run's state"
            )
        record = {
            "run": info["run"],
            "wandb_id": info["wandb_id"],
            "table": "debug/train_rollouts",
            "rows": len(rows),
            "state": actual.state,
            "url": info["wandb_url"],
        }
        published.append(record)
        print(json.dumps(record), flush=True)
    result = {
        "artifact_url": artifact_url,
        "artifact_name": artifact_name,
        "runs": published,
        "archives": archives,
    }
    write_json(review / "published.json", result)
    print(json.dumps({"artifact_url": artifact_url}), flush=True)


def publish(review):
    """Use one primary writer so the audit has stable tables of its own."""
    import wandb

    summary = json.loads((review / "summary.json").read_text())
    groups = json.loads((review / "selected-batches.json").read_text())
    if not (review / "raw-archives.json").exists():
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            archives = list(
                pool.map(lambda info: archive_run(review, info), summary["runs"])
            )
        write_json(review / "raw-archives.json", archives)
    columns = [
        "source_run",
        "update",
        "sample",
        "prompt",
        "final_answer",
        "full_response",
        "grading_target",
        "reward",
        "accuracy",
        "status",
        "forced",
        "sampled_tokens",
        "group_rewards",
        "reconstructed_advantage",
        "gradient_norm",
        "test_diagnostics",
    ]
    examples, overview = [], []
    per_run = {}
    for info in summary["runs"]:
        evaluations = info["validation"]
        overview.append(
            [
                info["run"],
                info["completed_updates"],
                info["saved_responses"],
                evaluations[0]["pass@1"],
                max(e["pass@1"] for e in evaluations),
                evaluations[-1]["pass@1"],
                info["mixed_reward_groups"] / info["saved_groups"],
                info["train_truncated_fraction"],
            ]
        )
        rows = []
        for group in groups:
            if group["run"] != info["run"]:
                continue
            for sample in group["samples"]:
                rows.append(
                    [
                        info["run"],
                        group["rollout_id"] + 1,
                        sample["sample_index"],
                        json.dumps(sample["prompt"], ensure_ascii=False),
                        sample["final_answer"],
                        sample["response"],
                        json.dumps(sample["target"], ensure_ascii=False),
                        sample["reward"],
                        sample["accuracy"],
                        sample["status"],
                        sample["forced"],
                        sample["sampled_tokens"],
                        json.dumps(sample["group_rewards"]),
                        sample["reconstructed_group_advantage"],
                        sample["gradient_norm"],
                        json.dumps(sample.get("replay"), ensure_ascii=False),
                    ]
                )
        per_run["rollouts/" + info["run"]] = rows
        examples.extend(rows)
    with tempfile.TemporaryDirectory(prefix="rl-rollout-audit-") as local:
        run = wandb.init(
            entity="david-heineman",
            project="modular-mopd-rl-initial",
            name="ROLLOUT AUDIT | Qwen3-0.6B | 2026-10-27",
            job_type="rollout-audit",
            group="modular-v2-rollout-audits",
            tags=["diagnostics", "rollout-review"],
            config={
                "inspection_only": True,
                "snapshot_time": summary["created_at"],
                "source_run_ids": [r["wandb_id"] for r in summary["runs"]],
            },
            settings=wandb.Settings(mode="online", init_timeout=120),
            dir=local,
        )
        run.log(
            {
                "rollouts/summary": wandb.Table(
                    columns=[
                        "source_run",
                        "updates",
                        "saved_responses",
                        "initial_pass1",
                        "best_pass1",
                        "latest_pass1",
                        "mixed_reward_fraction",
                        "truncated_fraction",
                    ],
                    data=overview,
                ),
                "rollouts/all_examples": wandb.Table(columns=columns, data=examples),
                **{
                    key: wandb.Table(columns=columns, data=rows)
                    for key, rows in per_run.items()
                },
            }
        )
        artifact = wandb.Artifact(
            "rollout-review-" + review.name,
            type="rollout-review",
            metadata={
                "created_at": summary["created_at"],
                "responses": sum(r["saved_responses"] for r in summary["runs"]),
                "reward_replays": summary["replay"]["responses"],
            },
        )
        artifact.add_dir(str(review))
        logged = run.log_artifact(artifact)
        logged.wait()
        result = {
            "run_id": run.id,
            "run_url": run.url,
            "artifact_url": logged.url,
            "table_rows": {key: len(rows) for key, rows in per_run.items()},
            "all_example_rows": len(examples),
        }
        run.finish()
    write_json(review / "audit-publication.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--attach-to-training-runs", action="store_true")
    args = parser.parse_args()
    if args.attach_to_training_runs:
        attach_to_training_runs(args.review.resolve())
    else:
        publish(args.review.resolve())
