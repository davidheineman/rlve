"""Slime adapters: Qwen non-thinking prompts and independent task rewards."""

import asyncio
import json
import math
import os
from pathlib import Path

from .graders import grade_code, grade_math


async def generate(args, sample, sampling_params):
    from slime.rollout.sglang_rollout import GenerateState
    from slime.rollout.sglang_rollout import generate as sglang_generate

    if not sample.tokens:
        # Exactly the HF Qwen3 enable_thinking=False template, in train and eval.
        tokenizer = GenerateState(args).tokenizer
        sample.prompt = tokenizer.apply_chat_template(
            sample.metadata["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return await sglang_generate(args, sample, sampling_params)


async def reward(args, sample, **kwargs):
    metadata = sample.metadata
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    task, target = metadata["task"], metadata["target"]
    if task == "math":
        return grade_math(sample.response, target["answer"])
    if task == "code":
        return await asyncio.to_thread(grade_code, sample.response, target)
    if task != "chat":
        raise ValueError(f"Unknown RL task: {task}")
    import aiohttp

    endpoint = json.loads(Path(os.environ["MODULAR_RL_REWARD_ENDPOINT"]).read_text())
    messages = metadata["messages"]
    response = sample.response.rsplit("</think>", 1)[-1].strip()
    conversations = [
        messages + [{"role": "assistant", "content": text}]
        for text in (response, target["rejected_reference"])
    ]
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session,
        session.post(
            endpoint["url"] + "/score",
            headers={"Authorization": "Bearer " + endpoint["token"]},
            json={"conversations": conversations},
        ) as result,
    ):
        result.raise_for_status()
        score, baseline = (await result.json())["scores"]
    if not all(math.isfinite(x) for x in (score, baseline)):
        raise ValueError("Nonfinite reward service output")
    return {
        "reward": score,
        "accuracy": float(score > baseline),
        "rejected_score": baseline,
        "margin": score - baseline,
    }


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    from slime.ray.buffer import wandb_step
    from slime.rollout.sglang_rollout import generate_rollout as original

    data = original(args, rollout_id, data_buffer, evaluation=evaluation)
    if not evaluation:
        groups = data
        samples = [sample for group in groups for sample in group]
        signal = sum(
            max(s.reward["reward"] for s in group)
            > min(s.reward["reward"] for s in group)
            for group in groups
        )
        metrics = {
            "david/step": wandb_step(rollout_id, args),
            "david/train/mean_reward": sum(s.reward["reward"] for s in samples)
            / len(samples),
            "david/train/accuracy": sum(s.reward["accuracy"] for s in samples)
            / len(samples),
            "david/train/signal_groups": signal,
        }
        print(f"modular_rl {rollout_id}: {metrics}", flush=True)
        if args.use_wandb:
            import wandb

            wandb.log(metrics)
        directory = Path(os.environ["MODULAR_RL_RUN_DIR"])
        with (directory / "signal.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
    return data
