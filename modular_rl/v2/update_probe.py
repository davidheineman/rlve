"""Opt-in measurements on sampled tokens before/after an update.

The ablation launcher inserts a call into its frozen actor snapshot after train()
and before CPU weight synchronization. This module never changes policy weights.
The selected-action KL estimate is a diagnostic, not a full-vocabulary KL.
"""

import json
import os
from pathlib import Path


def probe(actor, rollout_id, rollout_data, data_iterator, num_microbatches):
    import torch

    import wandb
    from slime.backends.megatron_utils.model import forward_only

    interval = int(os.environ.get("MODULAR_RL_UPDATE_PROBE_INTERVAL", "10"))
    if rollout_id % interval:
        return
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank() if world > 1 else 0
    if world not in (1, 2, 4) or (world > 1 and (
            actor.args.tensor_model_parallel_size != 1
            or actor.args.pipeline_model_parallel_size != 1
            or actor.args.context_parallel_size != 1 or actor.args.balance_data)):
        raise ValueError('Update probe requires one, two, or four strided data-parallel ranks')
    after = forward_only(
        actor.args, actor.model, data_iterator, num_microbatches
    )["log_probs"]
    rows = []
    rewards = rollout_data.get("raw_reward", rollout_data["rewards"])
    positions = list(range(rank, len(rewards), world))
    rewards = [rewards[i] for i in positions]
    count = len(rewards)
    epochs = getattr(actor.args, "train_reuse_epochs", 1)
    if len(after) != count * epochs:
        raise ValueError("Probe output count differs from the original responses and declared objective passes")
    for i, (old, new, mask, advantage) in enumerate(zip(
        rollout_data["log_probs"][:count], after[:count], rollout_data["loss_masks"][:count],
        rollout_data["advantages"][:count], strict=True,
    )):
        old, new = old.detach().float().cpu(), new.detach().float().cpu()
        mask = torch.as_tensor(mask, dtype=torch.bool).cpu()
        delta = (new - old)[mask]
        adv = advantage.detach().float().cpu()[mask]
        if delta.numel() == 0:
            raise ValueError("Probe response has no sampled tokens")
        ratio = delta.exp()
        rows.append({
            "sample": positions[i], "reward": float(rewards[i]),
            "tokens": delta.numel(), "advantage": float(adv.mean()),
            "mean_delta_logp": float(delta.mean()),
            "mean_abs_delta_logp": float(delta.abs().mean()),
            "max_abs_delta_logp": float(delta.abs().max()),
            "sampled_k3_old_to_new": float((delta.exp() - 1 - delta).mean()),
            "advantage_delta_logp": float((adv * delta).mean()),
            "fixed_token_advantage_delta_logp": float((adv * delta).sum() / actor.args.rollout_max_response_len),
            "post_update_pg_clipfrac": float((
                ((adv > 0) & (ratio > 1 + actor.args.eps_clip_high))
                | ((adv < 0) & (ratio < 1 - actor.args.eps_clip))
            ).float().mean()),
        })
    if world > 1:
        from megatron.core import mpu

        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, rows,
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=True))
        rows = sorted([row for shard in gathered for row in shard], key=lambda row: row['sample'])
        if [row['sample'] for row in rows] != list(range(len(rollout_data['raw_reward']))):
            raise ValueError('Distributed probe duplicated or omitted rollout samples')
    if rank != 0:
        return
    mean = lambda key: sum(row[key] for row in rows) / len(rows)
    metrics = {f"david/update_probe/{key}": mean(key) for key in (
        "mean_delta_logp", "mean_abs_delta_logp", "sampled_k3_old_to_new",
        "advantage_delta_logp",
        "fixed_token_advantage_delta_logp", "post_update_pg_clipfrac",
    )}
    metrics["david/update_probe/max_abs_delta_logp"] = max(
        row["max_abs_delta_logp"] for row in rows
    )
    metrics["david/step"] = rollout_id
    directory = Path(os.environ["MODULAR_RL_RUN_DIR"])
    with (directory / "update-probes.jsonl").open("a") as stream:
        stream.write(json.dumps({"rollout_id": rollout_id, "samples": rows,
                                 "metrics": metrics}) + "\n")
    if actor.args.use_wandb:
        wandb.log(metrics)
