"""Freeze the tested source and launch resumable jobs for the selected tasks."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from modular_rl.prepare import write_json

from .driver import training_updates, validate_generation_budget


def submit(
    repo,
    experiment,
    data,
    selection,
    runtime,
    dry_run=False,
    rollouts=8000,
    afterok=(),
    tasks=("math", "code", "chat"),
    max_response_tokens=1024,
    answer_forcing_tokens=None,
    exclude=None,
):
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or set(tasks) - {"math", "code", "chat"}
    ):
        raise ValueError("Select distinct tasks from math, code, chat")
    if answer_forcing_tokens is None:
        answer_forcing_tokens = max_response_tokens // 8
    validate_generation_budget(max_response_tokens, answer_forcing_tokens)
    steps = training_updates(rollouts)
    if steps != 1000:
        raise ValueError(
            "The v2 launch budget is fixed at 8000 training responses per run"
        )
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY is required")
    if experiment.exists():
        raise ValueError("Use a fresh experiment directory")
    if any(not str(job).isdigit() or int(job) <= 0 for job in afterok):
        raise ValueError("Dependencies must be positive Slurm job IDs")
    if not (data / "ready.json").is_file() or not (runtime / "math_verify").is_dir():
        raise ValueError("Prepare v2 data and the pinned math runtime first")
    audit = json.loads((data.parent / "math-reference-audit.json").read_text())
    if not audit["passed"] or audit["failed"]:
        raise ValueError("MATH reference audit must pass before launch")
    for split, checksum in audit["sha256"].items():
        if (
            hashlib.sha256((data / "math" / f"{split}.jsonl").read_bytes()).hexdigest()
            != checksum
        ):
            raise ValueError("MATH data changed after reference audit")
    for task in tasks:
        descriptor = json.loads((selection / task / "selection.json").read_text())
        if len(descriptor["keys"]["val"]) != 32 or not descriptor["enable_thinking"]:
            raise ValueError(
                "v2 requires 32 eligible reasoning-mode validation prompts"
            )
        for split, checksum in descriptor["sha256"].items():
            if (
                hashlib.sha256(
                    (selection / task / f"{split}.jsonl").read_bytes()
                ).hexdigest()
                != checksum
            ):
                raise ValueError("Selected data changed")
    source = experiment / "source"
    source.mkdir(parents=True)
    ignore = shutil.ignore_patterns(
        "__pycache__", ".pytest_cache", ".ruff_cache", "*.pyc"
    )
    for name in ("slime", "slime_plugins", "Gym", "modular_rl", "scripts"):
        shutil.copytree(repo / name, source / name, ignore=ignore)
    (source / "tools").mkdir()
    for name in ("yaml_to_args.py", "convert_hf_to_torch_dist.py"):
        shutil.copy2(repo / "tools" / name, source / "tools" / name)
    shutil.copy2(repo / "train.py", source / "train.py")
    original_selection = selection
    selection = experiment / "selection"
    for task in tasks:
        shutil.copytree(original_selection / task, selection / task)
    runtime_snapshot = experiment / "runtime"
    shutil.copytree(runtime, runtime_snapshot, ignore=ignore)
    runtime = runtime_snapshot
    group = experiment.name
    jobs, commands = [], []
    plan = {
        "version": 2,
        "project": "modular-mopd-rl-initial",
        "group": group,
        "source": str(source),
        "data": str(data),
        "selection": str(selection),
        "original_selection": str(original_selection),
        "runtime": str(runtime),
        "math_reference_audit": audit,
        "afterok": list(afterok),
        "models": ["Qwen3-0.6B", "Qwen3-1.7B"],
        "tasks": list(tasks),
        "excluded_nodes": exclude,
        "steps_per_run": steps,
        "max_train_rollouts_per_run": rollouts,
        "prompts_per_update": 1,
        "responses_per_prompt": 8,
        "max_sampled_tokens": max_response_tokens,
        "answer_forcing_reserved_tokens": answer_forcing_tokens,
        "enable_thinking": True,
        "eval_interval": 50,
        "validation_prompts": 32,
        "validation_samples": 8,
        "early_stop_pass@1": 0.98,
        "max_eval_rollouts_per_run": (steps // 50 + 1) * 256,
        "host_memory_gb": {"math": 128, "code": 128, "chat": 160},
        "gpu_memory_features": ["40G", "48G", "80G", "141G"],
        "source_sha256": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob("*"))
            if p.is_file()
        },
    }
    write_json(experiment / "plan.json", plan)
    for task in tasks:
        output = experiment / task
        output.mkdir()
        command = [
            "sbatch",
            "--parsable",
            f"--job-name=modular-v2-{task}-gen{max_response_tokens}",
            "--gres=gpu:" + ("2" if task == "chat" else "1"),
            "--cpus-per-task=" + ("12" if task == "chat" else "8"),
            "--mem=" + ("160G" if task == "chat" else "128G"),
            f"--chdir={source}",
            f"--output={output}/slurm-%j.out",
            str(source / "modular_rl/v2/run.sbatch"),
        ]
        if afterok:
            command.insert(2, "--dependency=afterok:" + ":".join(map(str, afterok)))
        if exclude:
            command.insert(2, f"--exclude={exclude}")
        commands.append(command)
        environment = {
            **os.environ,
            "RLVE_REPO": str(source),
            "MODULAR_RL_PROFILE": "v2",
            "MODULAR_RL_TASKS": task,
            "MODULAR_RL_MODELS": "Qwen3-0.6B Qwen3-1.7B",
            "MODULAR_RL_SHARE_REWARD_GPU": "0",
            "MODULAR_RL_GROUP": group,
            "MODULAR_RL_DATA": str(data),
            "MODULAR_RL_SELECTION": str(selection),
            "MODULAR_RL_RUNTIME": str(runtime),
            "MODULAR_RL_OUTPUT": str(output),
            "MODULAR_RL_STEPS": str(steps),
            "MODULAR_RL_EVAL_INTERVAL": "50",
            "MODULAR_RL_MAX_RESPONSE_TOKENS": str(max_response_tokens),
            "MODULAR_RL_ANSWER_FORCING_TOKENS": str(answer_forcing_tokens),
        }
        if not dry_run:
            job = (
                subprocess.check_output(command, env=environment, text=True)
                .strip()
                .split(";")[0]
            )
            jobs.append({"task": task, "job_id": job, "output": str(output)})
            write_json(experiment / "jobs.json", jobs)
            print(json.dumps(jobs[-1]), flush=True)
    write_json(experiment / "submit-commands.json", commands)
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    base = Path("/nlp/scr") / os.environ["USER"] / "modular-mopd-rl/v2"
    parser.add_argument(
        "--experiment",
        type=Path,
        default=base
        / "runs"
        / ("modular-v2-" + datetime.now().strftime("%Y%m%d-%H%M%S")),
    )
    parser.add_argument("--data", type=Path, default=base / "data")
    parser.add_argument("--selection", type=Path, default=base / "selection")
    parser.add_argument("--runtime", type=Path, default=base / "runtime")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("math", "code", "chat"),
        default=["math", "code", "chat"],
    )
    parser.add_argument("--max-response-tokens", type=int, default=1024)
    parser.add_argument(
        "--answer-forcing-tokens",
        type=int,
        help="Reserved final-answer tokens inside the cap; defaults to one eighth",
    )
    parser.add_argument("--exclude", help="Comma-separated Slurm nodes to exclude")
    parser.add_argument(
        "--afterok",
        nargs="+",
        type=int,
        default=[],
        help="Wait for successful preflight jobs",
    )
    args = vars(parser.parse_args())
    for key in ("repo", "experiment", "data", "selection", "runtime"):
        args[key] = args[key].resolve()
    submit(**args)
