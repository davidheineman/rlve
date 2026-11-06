"""Opt-in comparison that skips Adam entirely on zero-advantage batches."""

import json
import math
import os
from pathlib import Path


def maybe_train(args, rollout_data, train_function, *train_args):
    if os.environ.get("MODULAR_RL_SKIP_ZERO_ADVANTAGE") != "1":
        return train_function(*train_args)
    # A separate, explicit recipe may discard a uniform batch altogether,
    # including its reference penalty. Preserve the original diagnostic's guard
    # against silently suppressing another objective.
    filter_with_reference = os.environ.get("MODULAR_RL_FILTER_UNIFORM_REFERENCE_BATCHES") == "1"
    allow_reuse = os.environ.get("MODULAR_RL_ALLOW_OBJECTIVE_REUSE") == "1"
    if (
        args.advantage_estimator != "grpo"
        or (args.use_kl_loss and not filter_with_reference)
        or args.kl_coef != 0 or args.entropy_coef != 0
        or args.weight_decay != 0
        or args.actor_num_nodes * args.actor_num_gpus_per_node not in (1, 2, 4)
        or getattr(args, 'tensor_model_parallel_size', 1) != 1
        or getattr(args, 'pipeline_model_parallel_size', 1) != 1
        or getattr(args, 'context_parallel_size', 1) != 1
        or args.num_steps_per_rollout != 1
        or (args.train_reuse_epochs != 1 and not allow_reuse)
        or args.train_reuse_epochs not in (1, 2, 4)
    ):
        raise ValueError("Skip-zero trial requires one/two/four data-parallel GPUs, single-step GRPO without unrequested losses or decay")
    if filter_with_reference and (not args.use_kl_loss or args.kl_loss_type != "k3" or args.kl_loss_coef <= 0):
        raise ValueError("Reference-batch filtering requires an explicit positive K3 loss")
    import torch

    world = args.actor_num_nodes * args.actor_num_gpus_per_node
    distributed = torch.distributed.is_initialized()
    if world > 1 and (not distributed or torch.distributed.get_world_size() != world):
        raise ValueError('Distributed skip requires an initialized matching process group')
    rank = torch.distributed.get_rank() if distributed else 0
    advantages = rollout_data['advantages']
    valid = bool(advantages) and all(a.numel() and torch.isfinite(a).all() for a in advantages)
    active = valid and any(torch.count_nonzero(a) for a in advantages)
    if 'raw_reward' in rollout_data:
        raw = rollout_data['raw_reward']
        group_size = getattr(args, 'n_samples_per_prompt', 8)
        raw_valid = bool(raw) and len(raw) % group_size == 0 and all(math.isfinite(float(r)) for r in raw)
        valid = valid and raw_valid
        # Independent defense against a numerical centering residual: a tie in
        # every complete prompt group cannot justify a policy update.
        if raw_valid and not any(max(raw[i:i + group_size]) > min(raw[i:i + group_size])
                                 for i in range(0, len(raw), group_size)):
            active = False
    if world > 1:
        # Every rank must enter backward if any shard has a signal. A local-only
        # decision can deadlock DDP when one shard has all-zero advantages.
        fallback_device = 'cuda' if torch.distributed.get_backend() == 'nccl' else 'cpu'
        flags = torch.tensor([not valid, bool(active)], dtype=torch.int,
                             device=advantages[0].device if advantages else fallback_device)
        torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX)
        valid, active = not bool(flags[0]), bool(flags[1])
    if not valid:
        raise ValueError("Skip-zero trial needs finite, nonempty advantages")
    skipped = not active
    rollout_id = train_args[0]
    result = None
    if not skipped:
        result = train_function(*train_args)
    elif rank == 0:
        # Avoid calling either optimizer.step or the LR scheduler. In particular,
        # do not decay the Adam moments or increment Adam's bias-correction clock.
        metrics = {"train/grad_norm": 0.0, "train/step": rollout_id,
                   "train/optimizer_update_skipped": 1.0}
        print(f"step {rollout_id}: {metrics}", flush=True)
        if args.use_wandb:
            import wandb

            wandb.log(metrics)
    record = {"rollout_id": rollout_id, "optimizer_update_applied": not skipped}
    if allow_reuse:
        record["optimizer_steps"] = 0 if skipped else args.train_reuse_epochs
    directory = Path(os.environ["MODULAR_RL_RUN_DIR"])
    if world > 1:
        record['policy_world_size'] = world
    if rank == 0:
        with (directory / "optimizer-decisions.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
    if args.use_wandb and rank == 0:
        import wandb

        wandb.log({"david/step": rollout_id,
                   "david/optimizer/update_applied": int(not skipped),
                   "david/optimizer/steps_applied": 0 if skipped else args.train_reuse_epochs})
    return result
