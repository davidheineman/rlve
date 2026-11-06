"""Freeze and submit fresh-model Qwen3/2k objective comparisons.

Unlike the full v2 launcher, this explicitly labels smaller budgets as ablations.
The same frozen source, data, seed and update diagnostics serve both objectives.
"""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from modular_rl.prepare import write_json

from .data import stratum
from .inspect_rollouts import read_events
from .memory import instrument_model


def freeze(repo, experiment, base, tasks, skip_zero=False):
    source = experiment / "source"
    source.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache")
    for name in ("slime", "slime_plugins", "Gym", "modular_rl", "scripts"):
        shutil.copytree(repo / name, source / name, ignore=ignore)
    (source / "tools").mkdir()
    for name in ("yaml_to_args.py", "convert_hf_to_torch_dist.py"):
        shutil.copy2(repo / "tools" / name, source / "tools" / name)
    shutil.copy2(repo / "train.py", source / "train.py")
    shutil.copytree(base / "runtime", experiment / "runtime", ignore=ignore)
    for task in tasks:
        shutil.copytree(base / "selection" / task, experiment / "selection" / task)
    # Instrument only this frozen snapshot. The working native actor is untouched.
    actor_path = source / "slime/backends/megatron_utils/actor.py"
    original = actor_path.read_text()
    marker = "            # Profiling.\n"
    if original.count(marker) != 1:
        raise ValueError("Native actor structure changed; inspect probe placement")
    call = (
        "            from modular_rl.v2.update_probe import probe\n"
        "            probe(self, rollout_id, rollout_data, data_iterator, num_microbatches)\n\n"
    )
    instrumented = original.replace(marker, call + marker)
    if skip_zero:
        train_marker = "                train(\n                    rollout_id,\n"
        if instrumented.count(train_marker) != 1:
            raise ValueError("Native actor training call changed; inspect skip placement")
        instrumented = instrumented.replace(train_marker,
            "                from modular_rl.v2.updates import maybe_train\n"
            "                maybe_train(\n                    self.args, rollout_data, train,\n"
            "                    rollout_id,\n")
    actor_path.write_text(instrumented)
    model_path = source / "slime/backends/megatron_utils/model.py"
    model_path.write_text(instrument_model(model_path.read_text()))
    if skip_zero:
        text = model_path.read_text()
        marker = '        for optimizer in optimizer.chained_optimizers:\n'
        if text.count(marker) != 1:
            raise ValueError('Inspect optimizer construction before adding lazy-state initialization')
        text = text.replace(marker, marker
            + '            from modular_rl.v2.optimizer_state import initialize_hybrid_state\n'
            + '            initialize_hybrid_state(optimizer.optimizer)\n')
        model_path.write_text(text)
    return source


def select_training_probe(experiment, base, task, count):
    """Repeat known mixed-outcome training problems; preserve the held-out panel."""
    references = {
        "math": "modular-v2-math-gen2048-20261027-154441",
        "code": "modular-v2-code-gen2048-20261027-140220",
    }
    source = base / "runs" / references[task] / task / f"Qwen3-0.6B-{task}" / "rollouts/train"
    eligible = {}
    for path in sorted(source.glob("*.jsonl"))[:200]:
        rows = read_events(path)
        if len({row["correct"] for row in rows}) > 1:
            metadata = json.loads(rows[0]["metadata_json"])
            eligible[metadata["key"]] = int(path.stem)
    selected = sorted(eligible)
    random.Random(42).shuffle(selected)
    selected = selected[:count]
    if len(selected) != count:
        raise ValueError("Too few known mixed-outcome training prompts")
    directory = experiment / "selection" / task
    path = directory / "train.jsonl"
    original = read_events(path)
    rows = [row for row in original if row["metadata"]["key"] in selected]
    if len(rows) != count:
        raise ValueError("Probe prompts are not all in the original training pool")
    val_keys = {row["metadata"]["key"] for row in read_events(directory / "val.jsonl")}
    if val_keys.intersection(selected):
        raise ValueError("Training probe must not overlap validation")
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    descriptor = json.loads((directory / "selection.json").read_text())
    descriptor["keys"]["train"] = [row["metadata"]["key"] for row in rows]
    descriptor["sha256"]["train"] = hashlib.sha256(path.read_bytes()).hexdigest()
    descriptor["strata"]["train"] = dict(Counter("/".join(stratum(row)) for row in rows))
    descriptor["training_probe"] = {
        "kind": "positive-control-known-mixed-training-problems",
        "prompts": count, "original_training_prompts": len(original),
        "reference_run": str(source.parent.parent),
        "reference_rollout_ids": {key: eligible[key] for key in selected},
        "validation_unchanged": True,
    }
    write_json(directory / "selection.json", descriptor)


def trial_settings(steps, prompts, model):
    if model not in ("Qwen3-0.6B", "Qwen3-1.7B"):
        raise ValueError("Select an explicit released Qwen3 reasoning checkpoint")
    if prompts not in (1, 4, 8) or steps <= 0:
        raise ValueError("Use 1, 4 or 8 prompts and positive steps")
    full = steps * prompts * 8 == 8000
    if not full and (steps * prompts) % 48:
        raise ValueError("Use 8,000 responses or a positive multiple of 384 responses")
    return {
        "kind": "full-budget-controlled-trial" if full else "short-controlled-ablation",
        "eval_interval": (50 + prompts - 1) // prompts if full else 48 // prompts,
        "memory": "128G" if model == "Qwen3-1.7B" else "64G",
        "time": "12:00:00" if full
        else ("03:00:00" if model == "Qwen3-1.7B" else "01:30:00"),
    }


def launch(repo, experiment, base, steps, tasks, variants, dry_run=False, prompts=1, probe_prompts=0, afterok=(), model="Qwen3-0.6B", node=None, conservative_memory=False):
    repo, experiment, base = repo.resolve(), experiment.resolve(), base.resolve()
    if experiment.exists():
        raise ValueError("Use a fresh ablation directory")
    settings = trial_settings(steps, prompts, model)
    if conservative_memory and model != "Qwen3-0.6B":
        raise ValueError("The conservative memory profile has been GPU-validated only for 0.6B")
    if node is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", node):
        raise ValueError("Select one explicit hostname")
    eval_interval = settings["eval_interval"]
    filtered_variants = {"fixed-kl-filtered", "fixed-kl-filtered-cosine"}
    allowed = {"baseline", "fixed", "fixed-lr-low", "fixed-kl", "fixed-partial", "fixed-skip-zero", "fixed-finished", "fixed-kl-lr4"} | filtered_variants
    if not set(variants) <= allowed or not set(tasks) <= {"math", "code"}:
        raise ValueError("Unknown task or variant")
    if "fixed-partial" in variants and tasks != ["code"]:
        raise ValueError("The partial-test reward trial must select only code")
    if "fixed-finished" in variants and tasks != ["math"]:
        raise ValueError("The finished-answer reward trial must select only math")
    if probe_prompts and model != "Qwen3-0.6B":
        raise ValueError("Known-mixed positive controls were selected for 0.6B only")
    if any(not str(job).isdigit() or int(job) <= 0 for job in afterok):
        raise ValueError("Dependencies must be positive Slurm job IDs")
    skip_variants = {"fixed-skip-zero"} | filtered_variants
    source = freeze(repo, experiment, base, tasks, skip_zero=bool(skip_variants.intersection(variants)))
    if conservative_memory:
        config = source / "modular_rl/v2/config.yaml"
        contents = config.read_text()
        if contents.count("use-dynamic-batch-size: true") != 1:
            raise ValueError("Inspect the frozen configuration before selecting static microbatches")
        config.write_text(contents.replace("use-dynamic-batch-size: true", "use-dynamic-batch-size: false"))
        actor = source / "slime/backends/megatron_utils/actor.py"
        contents = actor.read_text()
        marker = '            # Train\n            with timer("actor_train"):'
        if contents.count(marker) != 1:
            raise ValueError("Inspect the native actor before placing transient cache cleanup")
        actor.write_text(contents.replace(marker, "            clear_memory()\n" + marker))
    if probe_prompts:
        for task in tasks:
            select_training_probe(experiment, base, task, probe_prompts)
    plan = {
        "kind": settings["kind"], "models": [model],
        "checkpoint_family": "released-reasoning", "hf_checkpoint": f"Qwen/{model}",
        "tasks": tasks, "variants": variants, "updates": steps,
        "training_responses": steps * prompts * 8, "eval_interval": eval_interval,
        "prompts_per_update": prompts,
        "checkpoint_interval": eval_interval if prompts > 1 else 50,
        "gpu_profile": "A6000",
        "requested_node": node,
        "training_pool": "positive-control" if probe_prompts else "full",
        "training_probe_prompts": probe_prompts,
        "afterok": list(afterok),
        "validation_prompts": 32, "samples_per_prompt": 8,
        "max_sampled_tokens": 2048, "answer_reserve": 256,
        "conservative_memory_profile": conservative_memory,
        "sglang_mem_fraction_static": 0.15 if conservative_memory else 0.3,
        "use_dynamic_batch_size": not conservative_memory,
        "sglang_max_running_requests": 16 if conservative_memory else None,
        "sglang_cuda_graph_max_bs": 16 if conservative_memory else None,
        "post_update_probe_interval": 10,
        "cuda_cache_clear_microbatch_interval": 4 if prompts > 1 or conservative_memory else 0,
        "uniform_reference_batch_filter_variants": [v for v in variants if v in filtered_variants],
        "optimizer_lr_schedules": {v: {"style": "cosine", "initial_lr": 1e-6, "min_lr": 1e-7,
                                       "decay_optimizer_steps": 500, "clock": "applied_optimizer_updates"}
                                   for v in variants if v == "fixed-kl-filtered-cosine"},
        "source": str(source), "selection": str(experiment / "selection"),
        "source_sha256": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob("*")) if p.is_file()
        },
    }
    write_json(experiment / "plan.json", plan)
    jobs = []
    for task in tasks:
        for variant in variants:
            output = experiment / task / variant
            output.mkdir(parents=True)
            extra = ("--sglang-mem-fraction-static 0.15 --sglang-max-running-requests 16 "
                     "--sglang-cuda-graph-max-bs 16") if conservative_memory else "--sglang-mem-fraction-static 0.3"
            if variant != "baseline":
                extra += (" --loss-type custom_loss --custom-loss-function-path "
                         "modular_rl.v2.losses.fixed_token_policy_loss")
            if variant == "fixed-lr-low":
                extra += " --lr 5e-7"
            if variant in ("fixed-kl", "fixed-kl-lr4"):
                extra += " --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type k3"
            if variant == "fixed-kl-lr4":
                extra += " --lr 4e-6"
            if variant in filtered_variants:
                extra += " --lr 1e-6 --use-kl-loss --kl-loss-coef 0.01 --kl-loss-type k3"
            if variant == "fixed-kl-filtered-cosine":
                extra += " --lr-decay-style cosine --lr-decay-iters 500 --min-lr 1e-7"
            if variant == "fixed-partial":
                extra += " --custom-rm-path modular_rl.v2.code_reward.partial_test_reward"
            if variant == "fixed-finished":
                extra += " --custom-rm-path modular_rl.v2.math_reward.correct_and_finished_reward"
            if prompts != 1:
                extra += (f" --rollout-batch-size {prompts} --over-sampling-batch-size {prompts}"
                          f" --global-batch-size {prompts * 8}")
            environment = {
                **os.environ, "RLVE_REPO": str(source),
                "MODULAR_RL_TASKS": task, "MODULAR_RL_MODELS": model,
                "MODULAR_RL_GROUP": experiment.name,
                "MODULAR_RL_VARIANT": f"{variant} p{prompts}" + (f" probe{probe_prompts}" if probe_prompts else " full-pool"),
                "MODULAR_RL_DATA": str(base / "data"),
                "MODULAR_RL_SELECTION": str(experiment / "selection"),
                "MODULAR_RL_RUNTIME": str(experiment / "runtime"),
                "MODULAR_RL_OUTPUT": str(output), "MODULAR_RL_STEPS": str(steps),
                "MODULAR_RL_EVAL_INTERVAL": str(eval_interval), "MODULAR_RL_EXTRA_ARGS": extra,
                "MODULAR_RL_CHECKPOINT_INTERVAL": str(eval_interval if prompts > 1 else 50),
                "MODULAR_RL_PROMPTS_PER_UPDATE": str(prompts),
                "MODULAR_RL_MAX_RESPONSE_TOKENS": "2048",
                "MODULAR_RL_ANSWER_FORCING_TOKENS": "256",
                "MODULAR_RL_UPDATE_PROBE_INTERVAL": "10",
                "MODULAR_RL_SKIP_ZERO_ADVANTAGE": "1" if variant in skip_variants else "0",
                "MODULAR_RL_FILTER_UNIFORM_REFERENCE_BATCHES": "1" if variant in filtered_variants else "0",
                "MODULAR_RL_CUDA_CACHE_CLEAR_INTERVAL": "4" if prompts > 1 or conservative_memory else "0",
            }
            command = [
                "sbatch", "--parsable", f"--job-name=modular-ablate-{task}-{variant}-{model}-p{prompts}" + ("-probe" if probe_prompts else ""),
                "--gres=gpu:a6000:1",
                "--cpus-per-task=8", f"--mem={settings['memory']}", f"--time={settings['time']}",
                "--constraint=48G&ampere",
                "--exclude=viscam10,jagupard33,jagupard34",
                f"--chdir={source}", f"--output={output}/slurm-%j.out",
                str(source / "modular_rl/v2/run.sbatch"),
            ]
            if afterok:
                command.insert(2, "--dependency=afterok:" + ":".join(map(str, afterok)))
            if node:
                command.insert(2, "--nodelist=" + node)
            job_id = None if dry_run else subprocess.check_output(
                command, env=environment, text=True
            ).strip().split(";")[0]
            jobs.append({"task": task, "variant": variant, "job_id": job_id,
                         "output": str(output), "command": command, "extra_args": extra})
            write_json(experiment / "jobs.json", jobs)
            print(json.dumps({key: jobs[-1][key] for key in (
                "task", "variant", "job_id", "output"
            )}), flush=True)
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", type=Path, default=Path("/nlp/scr/dhei/modular-mopd-rl/v2"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--tasks", nargs="+", default=["math", "code"])
    parser.add_argument("--variants", nargs="+", default=["baseline", "fixed"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompts-per-update", type=int, default=1)
    parser.add_argument("--probe-prompts", type=int, default=0)
    parser.add_argument("--afterok", nargs="*", default=[])
    parser.add_argument("--model", choices=["Qwen3-0.6B", "Qwen3-1.7B"], default="Qwen3-0.6B")
    parser.add_argument("--node")
    parser.add_argument("--conservative-memory", action="store_true",
                        help="Validated 0.6B static-microbatch and bounded inference-memory profile")
    args = parser.parse_args()
    launch(args.repo, args.experiment, args.base, args.steps, args.tasks,
           args.variants, args.dry_run, args.prompts_per_update, args.probe_prompts, args.afterok, args.model,
           args.node, args.conservative_memory)
