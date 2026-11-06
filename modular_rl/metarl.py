"""MetaRL: harmonically weighted pass@1..K policy-gradient reweighting.

Implements the supplied PassAtKReweight code: w(p)=sum((1-p)**j)/H_K.
The native scalar-reward hook then subtracts the *weighted* prompt mean.
No per-prompt standard-deviation normalization or advantage whitening follows.
A fixed common dataset reward scale of one preserves the objective's gradient;
estimating a dataset std from our single-prompt batches would cancel w(p).
"""

import json
import os
from pathlib import Path

import torch


def pass_at_k_weights(rewards, max_k=4):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if isinstance(max_k, bool) or not isinstance(max_k, int) or max_k < 1:
        raise ValueError('optimize_pass_k must be a positive integer')
    if rewards.ndim != 2 or rewards.shape[1] == 0 or not torch.isfinite(rewards).all():
        raise ValueError('Expected finite per-prompt reward groups')
    probability = rewards.mean(dim=1, keepdim=True).clamp(0., 1.)
    q = 1 - probability
    harmonic = sum(1 / k for k in range(1, max_k + 1))
    # The finite sum avoids pass@K/p's removable singularity at p=0.
    return sum(q ** j for j in range(max_k)) / harmonic


def reweight(rewards, max_k=4, per_token_rewards=None):
    """Scale both scalar and optional token rewards before any centering."""
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    weights = pass_at_k_weights(rewards, max_k)
    tokens = None
    if per_token_rewards is not None:
        tokens = torch.as_tensor(per_token_rewards, dtype=torch.float32)
        if tokens.ndim != 3 or tokens.shape[:2] != rewards.shape or not torch.isfinite(tokens).all():
            raise ValueError('Token rewards must have shape [prompts, responses, tokens]')
        tokens = tokens * weights.unsqueeze(-1)
    return rewards * weights, tokens


def normalize(rewards, max_k=4):
    """Scalar outcome advantages after PassAtKReweight and MinusMean."""
    weighted, _ = reweight(rewards, max_k)
    return weighted - weighted.mean(dim=1, keepdim=True)


def post_process(args, samples):
    if args.advantage_estimator != 'grpo' or getattr(args, 'normalize_advantages', False):
        raise ValueError('MetaRL requires GRPO broadcasting without subsequent advantage whitening')
    group_size = args.n_samples_per_prompt
    if group_size < 2 or len(samples) != group_size * args.rollout_batch_size:
        raise ValueError('MetaRL requires complete multi-sample prompt groups')
    max_k = args.optimize_pass_k
    for start in range(0, len(samples), group_size):
        keys = [getattr(s, 'metadata', {}).get('key') for s in samples[start:start + group_size]]
        if None in keys or len(set(keys)) != 1:
            raise ValueError('MetaRL group contains missing or different prompt keys')
    if any(getattr(s, 'per_token_rewards', None) is not None for s in samples):
        raise ValueError('Native math adapter uses terminal rewards; reweight() supports token-reward backends')
    raw = [s.get_reward_value(args) for s in samples]
    rewards = torch.tensor(raw, dtype=torch.float32).reshape(-1, group_size)
    if not torch.isfinite(rewards).all() or not ((rewards == 0) | (rewards == 1)).all():
        raise ValueError('This MetaRL math run requires binary correctness rewards')
    weights = pass_at_k_weights(rewards, max_k)
    if directory := os.environ.get('MODULAR_RL_RUN_DIR'):
        with (Path(directory) / 'metarl-reweight.jsonl').open('a') as stream:
            stream.write(json.dumps({'optimize_pass_k': max_k, 'pass_rates': rewards.mean(1).tolist(),
                                     'weights': weights.flatten().tolist(), 'dataset_reward_scale': 1.,
                                     'postprocessing': 'reweight_then_prompt_mean; no group std'}) + '\n')
    # Returning here bypasses Slime's group/std reward-normalization path.
    return raw, normalize(rewards, max_k).flatten().tolist()
