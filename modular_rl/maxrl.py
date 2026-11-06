"""Binary-outcome MaxRL, matching tajwarfahim/maxrl's centered estimator.

Use Slime's custom reward post-processing hook with the GRPO broadcast
estimator. Further advantage whitening would undo this estimator's weighting.
Reference: verl/trainer/ppo/core_algos.py, compute_maxrl_outcome_advantage.
"""

import torch


def normalize(rewards):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError('MaxRL needs complete groups of at least two outcomes')
    if not torch.isfinite(rewards).all() or not ((rewards == 0) | (rewards == 1)).all():
        raise ValueError('This MaxRL experiment requires binary rewards')
    mean = rewards.mean(dim=1, keepdim=True)
    return (rewards - mean) / (mean + 1e-6)


def post_process(args, samples):
    if args.advantage_estimator != 'grpo' or getattr(args, 'normalize_advantages', False):
        raise ValueError('MaxRL requires GRPO broadcasting without advantage whitening')
    n = args.n_samples_per_prompt
    if len(samples) != n * args.rollout_batch_size:
        raise ValueError('MaxRL received incomplete rollout groups')
    for start in range(0, len(samples), n):
        group = samples[start:start + n]
        # Slime stores original RLVE metadata directly on each sample.
        keys = [getattr(s, 'metadata', {}).get('key') for s in group]
        if len(set(keys)) != 1:
            raise ValueError('MaxRL group contains different prompts')
    raw = [s.get_reward_value(args) for s in samples]
    return raw, normalize(torch.tensor(raw).reshape(-1, n)).flatten().tolist()
