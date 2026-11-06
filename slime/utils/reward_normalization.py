"""Numerically stable GRPO centering for constant reward groups."""


def normalize_group_rewards(rewards, standardize):
    """Preserve native nonuniform reductions and force exact zeros for ties.

    Repeated float32 fractions can have a rounded mean different from every
    element. Dividing that residual by epsilon creates a false policy signal.
    """
    if rewards.ndim != 2 or rewards.shape[1] < 1:
        raise ValueError('Reward groups must be a nonempty two-dimensional tensor')
    uniform = rewards.isfinite().all(dim=-1, keepdim=True) & (rewards == rewards[:, :1]).all(dim=-1, keepdim=True)
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    if standardize and rewards.shape[1] > 1:
        centered = centered / (centered.std(dim=-1, keepdim=True) + 1e-6)
    return centered.masked_fill(uniform, 0)
