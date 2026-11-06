"""Matched-prompt DPO objective for interleaved chosen/rejected trajectories."""

import torch
import torch.nn.functional as F


def pair_loss(policy_scores, reference_scores, beta=0.1):
    """Return one loss per pair; inputs alternate chosen, rejected."""
    if len(policy_scores) != len(reference_scores) or len(policy_scores) % 2:
        raise ValueError("DPO requires equally sized, interleaved chosen/rejected scores")
    if not policy_scores or beta <= 0:
        raise ValueError("DPO requires at least one pair and a positive beta")
    policy = torch.stack(policy_scores)
    reference = torch.stack(reference_scores).detach()
    margin = beta * ((policy[0::2] - policy[1::2]) - (reference[0::2] - reference[1::2]))
    losses = -F.logsigmoid(margin)
    if not torch.isfinite(losses).all():
        raise RuntimeError("Non-finite DPO loss")
    return losses, margin
