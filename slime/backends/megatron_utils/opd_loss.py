"""Megatron loss for on-policy distillation from teacher-scored student tokens."""

import torch

from slime.utils.opd import sampled_reverse_kl_loss

from .loss import get_log_probs_and_entropy


def opd_loss_function(args, batch, logits, sum_of_sample_mean):
    student_log_probs = torch.cat(
        get_log_probs_and_entropy(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
        )["log_probs"]
    )
    rollout_log_probs = torch.cat(batch["rollout_log_probs"])
    teacher_log_probs = torch.cat(batch["teacher_log_probs"])
    # The shared data iterator includes requested optional keys with value None.
    # Fresh-only MOPD has no replay mask even though that key is present.
    replay_mask = batch.get("opd_replay_mask")
    loss, metrics = sampled_reverse_kl_loss(
        student_log_probs, rollout_log_probs, teacher_log_probs, sum_of_sample_mean,
        replay_mask=(torch.cat(replay_mask) if replay_mask is not None else None),
        replay_clip_epsilon=getattr(args, "opd_replay_clip_epsilon", 0.2),
    )
    if student_log_probs.numel() == 0:
        loss += 0 * logits.sum()
    return loss, {"loss": loss.detach(), **{key: value.detach() for key, value in metrics.items()}}
