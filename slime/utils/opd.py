"""Token-level on-policy distillation helpers."""

import torch


@torch.inference_mode()
def score_response_log_probs(model, tokens, response_length, device, *, stop_token_permutation=None):
    if response_length < 1 or response_length >= len(tokens):
        raise ValueError("A scored trajectory needs a prompt and at least one response token")
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    prompt_length = len(tokens) - response_length
    logits = model(input_ids=input_ids, use_cache=False).logits[0, prompt_length - 1 : -1].float()
    targets = input_ids[0, prompt_length:]
    if stop_token_permutation is not None:
        student_eos, teacher_eos = stop_token_permutation
        if not (0 <= student_eos < logits.shape[-1] and 0 <= teacher_eos < logits.shape[-1]):
            raise ValueError("Stop-token permutation must use valid vocabulary IDs")
        if ((targets[:-1] == student_eos) | (targets[:-1] == teacher_eos)).any():
            raise ValueError("Stop-token alignment requires generation to stop at either EOS alias")
        # Transport teacher termination labels into the student's convention.
        # A swap is a bijection: probability mass remains normalized. Prompt
        # delimiters and all model input/context tokens retain their original IDs.
        targets = torch.where(targets == student_eos, teacher_eos,
                              torch.where(targets == teacher_eos, student_eos, targets))
    scores = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if scores.numel() != response_length or not torch.isfinite(scores).all():
        raise RuntimeError("Teacher response log probabilities are invalid")
    return scores.cpu().tolist()


def sampled_reverse_kl_loss(student_log_probs, rollout_log_probs, teacher_log_probs, reduce,
                            replay_mask=None, replay_clip_epsilon=0.2):
    if student_log_probs.shape != rollout_log_probs.shape or student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("Student, rollout, and teacher log probabilities must align tokenwise")
    advantage = (teacher_log_probs - rollout_log_probs).detach()
    ratio = (student_log_probs - rollout_log_probs).exp()
    objective = ratio * advantage
    if replay_mask is not None:
        if replay_mask.shape != ratio.shape or not 0 < replay_clip_epsilon < 1:
            raise ValueError("Replay masks must align with token scores and clipping must be in (0, 1)")
        # Cached advantages stay signed: replay can discourage a sampled token.
        # Only stale replay uses the PPO clipped surrogate. Fresh OPD is unchanged.
        clipped = ratio.clamp(1 - replay_clip_epsilon, 1 + replay_clip_epsilon) * advantage
        objective = torch.where(replay_mask.bool(), torch.minimum(objective, clipped), objective)
    loss = -reduce(objective)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite OPD loss")
    return loss, {
        "opd_reverse_kl": -reduce(advantage),
        "opd_importance_ratio": reduce(ratio.detach()),
    }
