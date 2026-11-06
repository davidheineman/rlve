"""Original single-update GRPO/PPO objective, evaluated through Tinker custom loss."""

import torch


def grpo_advantages(rewards):
    values = torch.tensor(rewards, dtype=torch.float32)
    centered = values - values.mean()
    return centered / (centered.std() + 1e-6)


def grpo_loss(logprobs, *, rollout_logprobs, masks, rewards, eps_low, eps_high, tis_low, tis_high):
    """Mean of response means; injected tokens and prompt positions have zero loss.

    Each adapter takes exactly one optimizer step per sampled group. The detached
    forward values are therefore its pre-update policy, matching slime's actor
    logprob pass. Other tenants cannot update this adapter between the two calls.
    """
    advantages = grpo_advantages(rewards)
    losses, corrections = [], []
    for current, sampled, mask, advantage in zip(logprobs, rollout_logprobs, masks, advantages, strict=True):
        active = torch.tensor(mask, dtype=current.dtype, device=current.device)
        sampled = torch.tensor(sampled, dtype=current.dtype, device=current.device)
        old = current.detach()
        ratio = torch.exp(current - old)
        correction = torch.exp(old - sampled).clamp(tis_low, tis_high)
        unclipped = ratio * advantage
        clipped = ratio.clamp(1 - eps_low, 1 + eps_high) * advantage
        denominator = active.sum().clamp_min(1)
        losses.append((torch.maximum(-unclipped, -clipped) * correction * active).sum() / denominator)
        corrections.append((correction * active).sum() / denominator)
    loss = torch.stack(losses).mean()
    return loss, {"pg_loss": loss.detach().item(), "tis_mean": torch.stack(corrections).mean().item()}
