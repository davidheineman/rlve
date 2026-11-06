"""One isolated environment, random stream, LoRA slot, optimizer and W&B run."""

import argparse
import asyncio
import json
import os
import random
import shutil
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tinker
import torch
from safetensors import safe_open
from tinker import types
from transformers import AutoTokenizer

import wandb
from lora_teachers.objective import grpo_loss
from lora_teachers.plan import sha256
from lora_teachers.sampling import prompt_tokens, sample_group, training_datums
from slime.ray.rollout_data_source import RLVEManager


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def log_metrics(run, directory, metrics):
    # W&B's internal history counter advances per record. The explicit train
    # step is the chart axis, so evaluation and training can share an update.
    run.log(metrics)
    with (directory / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(metrics) + "\n")
    print(json.dumps(metrics), flush=True)


def snapshot_directory(root, sampler_path):
    model_id, kind, version = sampler_path.removeprefix("tinker://").split("/")
    if kind != "sampler_weights":
        raise ValueError("Expected immutable sampler weights")
    directory = (root / model_id / kind / version).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError("Sampler path escaped this experiment")
    return directory


def validate_final_adapter(directory, plan):
    config = json.loads((directory / "adapter_config.json").read_text())
    if config["r"] != plan["lora_rank"] or config["lora_alpha"] != plan["lora_alpha"]:
        raise ValueError("Final adapter has the wrong rank or scaling")
    with safe_open(directory / "adapter_model.safetensors", framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
        if not keys or not any("lora_A" in key for key in keys) or not any("lora_B" in key for key in keys):
            raise ValueError("Final export is missing adapter factors")
        for key in keys:
            if not torch.isfinite(weights.get_tensor(key)).all():
                raise ValueError(f"Nonfinite final adapter tensor: {key}")


def persist_final_adapter(source, destination, plan):
    validate_final_adapter(source, plan)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copytree(source, temporary)
    validate_final_adapter(temporary, plan)
    temporary.replace(destination)


def persist_progress_adapter(source, directory, plan, step):
    """Retain one validated adapter on shared scratch, without optimizer state."""
    root = directory.resolve() / "adapter-checkpoints"
    root.mkdir(exist_ok=True)
    latest = directory / "latest-adapter.json"
    previous = None
    if latest.exists():
        previous = Path(json.loads(latest.read_text())["adapter_path"]).resolve()
        if not previous.is_relative_to(root):
            raise ValueError("Previous progress adapter escaped its checkpoint directory")
    destination = root / f"update-{step:04d}"
    persist_final_adapter(source, destination, plan)
    atomic_json(
        latest,
        {
            "updates": step,
            "training_rollouts": step * plan["teacher_config"]["n-samples-per-prompt"],
            "adapter_path": str(destination),
            "base_model": plan["base_model"],
            "optimizer_state_saved": False,
        },
    )
    # Publish the new, fully validated copy before deleting the older one.
    if previous is not None and previous != destination:
        shutil.rmtree(previous)


async def publish(training, service, checkpoint_root, step, previous_path):
    saved = await (await training.save_weights_for_sampler_async(name=f"update-{step:04d}"))
    directory = snapshot_directory(checkpoint_root, saved.path)
    if not (directory / "adapter_model.safetensors").is_file() or not (directory / "META.json").is_file():
        raise ValueError(f"Incomplete sampler export: {directory}")
    sampler = await service.create_sampling_client_async(model_path=saved.path)
    # All sampling of the previous version has completed before publication.
    # Reclaim only this client's disposable snapshots; retain its final version.
    if previous_path is not None:
        shutil.rmtree(snapshot_directory(checkpoint_root, previous_path))
    return sampler, saved.path


async def evaluate(sampler, tokenizer, rows, settings, *, concurrency=2):
    if concurrency < 1:
        raise ValueError("Evaluation concurrency must be positive")
    semaphore = asyncio.Semaphore(concurrency)

    async def score(row):
        async with semaphore:
            meta = row["metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            return await sample_group(
                sampler,
                tokenizer,
                prompt_tokens(tokenizer, row["user_prompt"]),
                meta["environment"],
                meta["config"],
                settings,
                evaluation=True,
            )

    groups = await asyncio.gather(*(score(row) for row in rows))
    completions = [completion for group in groups for completion in group]
    return {
        "david/eval/val_envs/pass@1": sum(c.accuracy for c in completions) / len(completions),
        "eval/val_envs/answer_forcing_fraction": sum(c.forced for c in completions) / len(completions),
        "eval/val_envs/truncated_fraction": sum(c.truncated for c in completions) / len(completions),
    }


async def train_updates(
    plan, teacher, *, training, service, tokenizer, sampler, sampler_path, run, directory, experiment, checkpoint_root
):
    settings = plan["teacher_config"]
    source = RLVEManager(
        SimpleNamespace(
            environment_list=[teacher["environment"]],
            initial_difficulty=settings["initial-difficulty"],
            rlve_balanced_difficulty=settings["rlve-balanced-difficulty"],
            rlve_prompt_pool=None,
            difficulty_sliding_window_size=settings["difficulty-sliding-window-size"],
        ),
        tokenizer,
    )
    eval_path = experiment / "source/rlve" / teacher["eval"]
    if sha256(eval_path) != teacher["eval_sha256"]:
        raise ValueError("Frozen teacher validation changed")
    rows = json.loads(eval_path.read_text())
    if len(rows) != 32:
        raise ValueError("Expected the original 32 validation prompts")
    adam = types.AdamParams(
        learning_rate=settings["lr"],
        beta1=settings["adam-beta1"],
        beta2=settings["adam-beta2"],
        eps=plan["optimizer_eps"],
        weight_decay=settings["weight-decay"],
        grad_clip_norm=plan["grad_clip_norm"],
    )
    cap = settings["num-rollout"]
    for step in range(cap + 1):
        metrics = {"train/step": step, "david/rollouts": step * settings["n-samples-per-prompt"]}
        stopping = step == cap
        if step % settings["eval-interval"] == 0 or stopping:
            metrics.update(await evaluate(sampler, tokenizer, rows, settings))
            accuracy = metrics["david/eval/val_envs/pass@1"]
            # Match the original driver: initial evaluation never stops an untrained run.
            stopping |= step > 0 and accuracy >= settings["early-stop-val-accuracy"]
            log_metrics(run, directory, metrics)
        if stopping:
            adapter = directory / "adapter-final"
            persist_final_adapter(snapshot_directory(checkpoint_root, sampler_path), adapter, plan)
            atomic_json(
                directory / "completed.json",
                {
                    "environment": teacher["environment"],
                    "updates": step,
                    "training_rollouts": metrics["david/rollouts"],
                    "sampler_path": sampler_path,
                    "adapter_path": str(adapter),
                    "base_model": plan["base_model"],
                    "wandb_url": run.url,
                    "metrics": metrics,
                    "stop_reason": "budget" if step == cap else "validation_accuracy",
                },
            )
            print(json.dumps({"completed": teacher["environment"], **metrics}), flush=True)
            return
        for _ in range(1000):
            environment, difficulty, problem = source.generate_problem()
            if problem is not None:
                break
        else:
            raise RuntimeError("Failed to generate a scoreable RLVE training prompt")
        prompt = prompt_tokens(tokenizer, problem.prompt_generator())
        completions = await sample_group(sampler, tokenizer, prompt, environment, problem.get_config(), settings)
        datums, masks, logprobs = training_datums(prompt, completions)
        objective = partial(
            grpo_loss,
            rollout_logprobs=logprobs,
            masks=masks,
            rewards=[c.reward for c in completions],
            eps_low=settings["eps-clip"],
            eps_high=settings["eps-clip-high"],
            tis_low=plan["tis_clip_low"],
            tis_high=plan["tis_clip_high"],
        )
        result = await (
            await training.forward_backward_custom_async(datums, lambda _data, lp, objective=objective: objective(lp))
        )
        optimizer_result = await (await training.optim_step_async(adam))
        if optimizer_result.metrics.get("skipped_nonfinite", 0):
            raise RuntimeError("Optimizer skipped a nonfinite update")
        metrics.update(
            {
                "train/step": step + 1,
                "train/lr": adam.learning_rate,
                "david/rollouts": (step + 1) * settings["n-samples-per-prompt"],
                "train/reward": sum(c.reward for c in completions) / len(completions),
                "train/accuracy": sum(c.accuracy for c in completions) / len(completions),
                "train/difficulty": difficulty,
                "train/pg_loss": result.metrics["pg_loss"],
                "train/tis_mean": result.metrics["tis_mean"],
                **optimizer_result.metrics,
            }
        )
        # Evaluation above describes the preceding policy, not this update.
        metrics = {key: value for key, value in metrics.items() if not key.startswith(("eval/", "david/eval/"))}
        log_metrics(run, directory, metrics)
        sampler, sampler_path = await publish(training, service, checkpoint_root, step + 1, sampler_path)
        interval = plan.get("progress_adapter_interval", 0)
        if interval > 0 and (step + 1) % interval == 0:
            persist_progress_adapter(snapshot_directory(checkpoint_root, sampler_path), directory, plan, step + 1)


async def main(args):
    experiment = Path(args.experiment).resolve()
    plan = json.loads((experiment / "plan.json").read_text())
    teacher = plan["teachers"][args.index]
    checkpoint_root = Path(os.environ["LORA_CHECKPOINT_ROOT"])
    directory = experiment / "teacher" / teacher["source_label"]
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    random.seed(plan["seed"])
    np.random.seed(plan["seed"])
    torch.manual_seed(plan["seed"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    service = tinker.ServiceClient(base_url=args.base_url, api_key=f"tml-{experiment.name}-{teacher['environment']}")
    training = await service.create_lora_training_client_async(
        base_model=plan["base_model"],
        rank=plan["lora_rank"],
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
    )
    sampler, sampler_path = await publish(training, service, checkpoint_root, 0, None)
    run = wandb.init(
        entity=plan["entity"],
        project=plan["project"],
        group=experiment.name,
        name=f"{plan['project']}-{teacher['source_label']}",
        dir=str(directory),
        config={
            **plan,
            "environment": teacher["environment"],
            "teacher": teacher,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )
    run.define_metric("train/step")
    run.define_metric("*", step_metric="train/step")
    atomic_json(
        directory / "identity.json", {"wandb_url": run.url, "wandb_id": run.id, "environment": teacher["environment"]}
    )
    # Every adapter exists before any teacher starts RL or initial evaluation.
    (directory / "ready").touch()
    while not (experiment / "clients.start").is_file():
        await asyncio.sleep(1)
    exit_code = 1
    try:
        await train_updates(
            plan,
            teacher,
            training=training,
            service=service,
            tokenizer=tokenizer,
            sampler=sampler,
            sampler_path=sampler_path,
            run=run,
            directory=directory,
            experiment=experiment,
            checkpoint_root=checkpoint_root,
        )
        exit_code = 0
    finally:
        run.finish(exit_code=exit_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-url", required=True)
    asyncio.run(main(parser.parse_args()))
