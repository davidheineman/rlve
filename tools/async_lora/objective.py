"""Encode the existing sampled reverse KL as native Tinker importance sampling."""

import math


def opd_advantages(sampled, teacher, mask, batch_size):
    if not (len(sampled) == len(teacher) == len(mask)) or batch_size <= 0:
        raise ValueError("Teacher, sampled log probabilities and masks must align")
    denominator = batch_size * max(sum(mask), 1)
    values = [(t - s) * m / denominator for s, t, m in zip(sampled, teacher, mask, strict=True)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite distillation advantage")
    return values
