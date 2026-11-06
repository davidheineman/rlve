"""Opt-in fixed-token normalization for controlled RL objective comparisons.

Enable with ``loss-type: custom_loss`` and
``custom-loss-function-path: modular_rl.v2.losses.fixed_token_policy_loss``.
This does not change the default v2 objective or any frozen experiment.
"""


def fixed_token_policy_loss(args, batch, logits, sum_of_sample_mean):
    """Replace per-response averaging with a common, fixed token denominator.

    The native outer loss still divides by global batch size and handles gradient
    accumulation. Every unmasked token gets the same normalization constant;
    one-token responses no longer receive a response-length amplification.
    Prefix masks, PPO/TIS, optional KL, and context parallel reduction remain native.
    """
    import torch

    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
    from slime.backends.megatron_utils.loss import policy_loss_function

    if args.calculate_per_token_loss:
        raise ValueError(
            "Fixed-token policy loss requires calculate_per_token_loss=False; "
            "the custom reduction already supplies its normalization"
        )
    if args.rollout_max_response_len <= 0:
        raise ValueError("A positive sampled-token cap is required")
    del sum_of_sample_mean
    token_sum = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        calculate_per_token_loss=True,
    )

    def reduction(values):
        return token_sum(values) / args.rollout_max_response_len

    loss, reported = policy_loss_function(args, batch, logits, reduction)
    # Native diagnostics use the same reducer as the objective. Record its mass:
    # e.g. reported tis / normalization_mass is a valid-token mean ratio, whereas
    # reported tis alone includes the mean sampled length / fixed-cap factor.
    ones = torch.ones_like(torch.cat(batch["advantages"]))
    reported["normalization_mass"] = reduction(ones).detach()
    return loss, reported
