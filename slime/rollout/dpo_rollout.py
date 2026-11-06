"""Matched prompt pairs from the current student and a frozen base policy."""

import asyncio

import ray
import wandb

from slime.rollout.rm_hub import async_rm
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm
from slime.utils.async_utils import run
from slime.utils.types import Sample


async def _generate_batch(args, groups, base_generator):
    base_future = base_generator.generate.remote([group[1].to_dict() for group in groups])
    policy_task = asyncio.gather(*[
        generate_and_rm(args, group[0], GenerateState(args).sampling_params.copy())
        for group in groups
    ])
    policy_samples, base_rows = await asyncio.gather(
        policy_task, asyncio.to_thread(ray.get, base_future)
    )
    base_samples = [Sample.from_dict(row) for row in base_rows]
    base_rewards = await asyncio.gather(*[async_rm(args, sample) for sample in base_samples])
    for sample, reward in zip(base_samples, base_rewards):
        sample.reward = reward
    return list(zip(policy_samples, base_samples))


async def _rollout(args, rollout_id, data_source):
    base_generator = getattr(data_source, "dpo_base_generator", None)
    if base_generator is None:
        raise RuntimeError("Online DPO base generator is not initialized")
    accepted = []
    attempts = 0
    policy_correct = base_correct = agreements = 0
    while len(accepted) < args.rollout_batch_size:
        count = args.rollout_batch_size - len(accepted)
        groups = data_source.get_samples(count)
        generated = await _generate_batch(args, groups, base_generator)
        attempts += count
        for policy, base in generated:
            if (policy.prompt != base.prompt or
                    policy.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) or
                    base.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) or
                    policy.response_length < 1 or base.response_length < 1 or
                    len(policy.tokens) > args.seq_length or len(base.tokens) > args.seq_length):
                continue
            # The preference is by source: the current policy is chosen and
            # the frozen base is rejected. Rewards are logged as diagnostics.
            policy.rollout_log_probs = None
            accepted.append([policy, base])
            p_correct = policy.reward["accuracy"] >= 1.0 - 1e-6
            b_correct = base.reward["accuracy"] >= 1.0 - 1e-6
            policy_correct += p_correct
            base_correct += b_correct
            agreements += p_correct == b_correct
        if attempts >= 8 * args.rollout_batch_size and len(accepted) < args.rollout_batch_size:
            raise RuntimeError(
                f"Only {len(accepted)}/{args.rollout_batch_size} complete online DPO pairs "
                f"after {attempts} prompts"
            )
    samples = [sample for pair in accepted for sample in pair]
    data_source.rlve_manager.update(samples)
    count = len(accepted)
    print(
        f"dpo {rollout_id}: {count} policy/base pairs from {attempts} prompts; "
        f"policy accuracy={policy_correct/count:.3f}, base accuracy={base_correct/count:.3f}",
        flush=True,
    )
    if args.use_wandb:
        wandb.log({
            "david/step": rollout_id,
            "david/dpo/policy_accuracy": policy_correct / count,
            "david/dpo/base_accuracy": base_correct / count,
            "david/dpo/reward_agreement": agreements / count,
            "david/dpo/attempted_prompts": attempts,
        })
    return accepted


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        raise ValueError("Use the standard SGLang evaluation rollout for online DPO")
    return run(_rollout(args, rollout_id, data_source))
