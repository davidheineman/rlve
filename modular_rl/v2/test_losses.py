"""Exercise native loss algebra on CPU with synthetic selected-token logprobs.

Megatron model execution is stubbed; the reducer, PPO loss, and outer scaling are
loaded from their actual source. This checks loss gradients, not GPU kernels.
"""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional

import pytest
import torch

from modular_rl.v2.losses import fixed_token_policy_loss

REPO = Path(__file__).resolve().parents[2]


def load_function(path, name, namespace):
    tree = ast.parse((REPO / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    # Execute only the selected function from this repository's trusted source.
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(REPO / path), "exec"), namespace)  # noqa: S102
    return namespace[name]


@pytest.fixture
def native(monkeypatch):
    namespace = {
        "torch": torch,
        "Optional": Optional,
        "mpu": SimpleNamespace(
            get_context_parallel_world_size=lambda: 1,
            get_data_parallel_world_size=lambda **kwargs: 1,
        ),
    }
    reducer = load_function("slime/backends/megatron_utils/cp_utils.py", "get_sum_of_sample_mean", namespace)
    load_function("slime/utils/ppo_utils.py", "compute_policy_loss", namespace)
    load_function("slime/utils/ppo_utils.py", "compute_approx_kl", namespace)

    def selected_logprobs(logits, *, response_lengths, **kwargs):
        return {
            "log_probs": list(logits.split(response_lengths)),
            "entropy": list((logits * 0).split(response_lengths)),
        }

    namespace["get_log_probs_and_entropy"] = selected_logprobs
    policy = load_function("slime/backends/megatron_utils/loss.py", "policy_loss_function", namespace)
    cp_module = ModuleType("slime.backends.megatron_utils.cp_utils")
    cp_module.get_sum_of_sample_mean = reducer
    loss_module = ModuleType("slime.backends.megatron_utils.loss")
    loss_module.policy_loss_function = policy
    monkeypatch.setitem(sys.modules, cp_module.__name__, cp_module)
    monkeypatch.setitem(sys.modules, loss_module.__name__, loss_module)
    namespace["load_function"] = lambda path: fixed_token_policy_loss
    outer = load_function("slime/backends/megatron_utils/loss.py", "loss_function", namespace)
    return SimpleNamespace(policy=policy, outer=outer)


def run_loss(native, *, fixed, chunks=1, uniform=False, prefix=0, kl_coef=0, ref_delta=0, kl_type="k3"):
    lengths = [1821 + prefix, 1, 1859 + prefix] + [1821 + prefix] * 5
    rewards = torch.tensor([1, 0, 1, 1, 1, 1, 1, 1], dtype=torch.float64)
    if uniform:
        rewards[:] = 1
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    current = torch.full((sum(lengths),), -0.5, dtype=torch.float64, requires_grad=True)
    old = list(current.detach().clone().split(lengths))
    masks = [torch.ones(n, dtype=torch.float64) for n in lengths]
    for index, mask in enumerate(masks):
        if index != 1:
            mask[:prefix] = 0
    args = SimpleNamespace(
        calculate_per_token_loss=False, rollout_max_response_len=2048,
        loss_type="custom_loss" if fixed else "policy_loss",
        custom_loss_function_path="modular_rl.v2.losses.fixed_token_policy_loss",
        global_batch_size=8, advantage_estimator="grpo", use_tis=True,
        use_kl_loss=bool(kl_coef), kl_loss_coef=kl_coef, kl_loss_type=kl_type,
        eps_clip=0.2, eps_clip_high=0.28, entropy_coef=0.0,
        tis_clip_low=0.0, tis_clip=2.0,
    )
    indices = list(range(8))
    chunks_of_indices = [indices[i:i + 8 // chunks] for i in range(0, 8, 8 // chunks)]
    offset = [0]
    for n in lengths:
        offset.append(offset[-1] + n)
    logs = []
    for indexes in chunks_of_indices:
        selected = torch.cat([current[offset[i]:offset[i + 1]] for i in indexes])
        batch = {
            "response_lengths": [lengths[i] for i in indexes],
            "total_lengths": [lengths[i] + 10 for i in indexes],
            "loss_masks": [masks[i] for i in indexes],
            "unconcat_tokens": [],
            "advantages": [advantages[i].expand(lengths[i]) for i in indexes],
            "log_probs": [old[i] for i in indexes],
            "ref_log_probs": [old[i] + ref_delta for i in indexes],
            "rollout_log_probs": [old[i] for i in indexes],
        }
        loss, _, log = native.outer(args, batch, chunks, selected)
        # Megatron averages microbatch losses in the non-per-token mode.
        (loss / chunks).backward()
        logs.append(dict(zip(log["keys"], log["values"][1:].tolist(), strict=True)))
    return current.grad, lengths, masks, logs


@pytest.mark.parametrize("fixed", [False, True])
def test_policy_gradient_sign_and_masked_prefixes(native, fixed):
    gradient, lengths, masks, _ = run_loss(native, fixed=fixed, prefix=6)
    offset = 0
    for i, (length, mask) in enumerate(zip(lengths, masks, strict=True)):
        actual = gradient[offset:offset + length]
        assert torch.all(actual[mask == 0] == 0)
        assert torch.all(actual[mask == 1] > 0) if i == 1 else torch.all(actual[mask == 1] < 0)
        offset += length


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("chunks", [2, 8])
def test_gradient_accumulation_is_invariant_to_packing(native, fixed, chunks):
    expected, *_ = run_loss(native, fixed=fixed, chunks=1)
    actual, *_ = run_loss(native, fixed=fixed, chunks=chunks)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("fixed", [False, True])
def test_uniform_rewards_have_zero_policy_gradient(native, fixed):
    gradient, *_ = run_loss(native, fixed=fixed, uniform=True)
    assert torch.all(gradient == 0)


def test_fixed_normalization_removes_response_length_amplification(native):
    old, lengths, _, old_logs = run_loss(native, fixed=False)
    new, *_ = run_loss(native, fixed=True)
    # The failed response is a single EOS token with advantage -7x a winner's.
    eos = lengths[0]
    assert float(abs(old[eos] / old[eos + 1])) == pytest.approx(7 * lengths[2])
    assert float(abs(new[eos] / new[eos + 1])) == pytest.approx(7)
    assert all(log["pg_clipfrac"] == 0 and log["ppo_kl"] == 0 for log in old_logs)


def test_rejects_a_second_token_normalization(native):
    args = SimpleNamespace(calculate_per_token_loss=True)
    with pytest.raises(ValueError, match="already supplies"):
        fixed_token_policy_loss(args, {}, None, None)


def test_rejects_invalid_generation_cap(native):
    args = SimpleNamespace(calculate_per_token_loss=False, rollout_max_response_len=0)
    with pytest.raises(ValueError, match="positive"):
        fixed_token_policy_loss(args, {}, None, None)


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("ref_delta", [-0.2, 0.2])
def test_explicit_k3_regularizer_pulls_toward_reference(native, fixed, ref_delta):
    grad, _, masks, _ = run_loss(native, fixed=fixed, uniform=True, prefix=6,
                                kl_coef=0.001, ref_delta=ref_delta)
    mask = torch.cat(masks).bool()
    assert torch.all(grad[~mask] == 0)
    assert torch.all(grad[mask] * ref_delta < 0)


@pytest.mark.parametrize("fixed", [False, True])
def test_k3_regularizer_has_zero_gradient_at_reference(native, fixed):
    grad, *_ = run_loss(native, fixed=fixed, uniform=True, kl_coef=0.001)
    assert torch.all(grad == 0)


@pytest.mark.parametrize("fixed", [False, True])
def test_linear_direct_kl_loss_does_not_anchor_to_reference(native, fixed):
    low, *_ = run_loss(native, fixed=fixed, uniform=True, kl_coef=0.001,
                       ref_delta=-0.2, kl_type="kl")
    high, *_ = run_loss(native, fixed=fixed, uniform=True, kl_coef=0.001,
                        ref_delta=0.2, kl_type="kl")
    torch.testing.assert_close(low, high)
    assert torch.all(low > 0)


def test_native_ppo_matches_standard_clipped_surrogate_on_both_advantage_signs(native):
    # Include ratios outside both clip boundaries; the negative-advantage branch
    # is particularly easy to reverse when implementing the clipped surrogate.
    function = native.policy.__globals__["compute_policy_loss"]
    ratio = torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5] * 2, dtype=torch.float64)
    advantage = torch.tensor([1.0] * 5 + [-1.0] * 5, dtype=torch.float64)
    log_ratio = ratio.log().requires_grad_()
    actual, _ = function(-log_ratio, advantage, 0.2, 0.28)
    expected = -torch.minimum(log_ratio.exp() * advantage,
                              log_ratio.exp().clamp(0.8, 1.28) * advantage)
    torch.testing.assert_close(actual, expected)
    actual_grad, = torch.autograd.grad(actual.sum(), log_ratio, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected.sum(), log_ratio)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grad[4] == 0  # rewarded action above the upper ratio bound
    assert actual_grad[5] == 0  # failed action below the lower ratio bound
    assert actual_grad[0] < 0 and actual_grad[9] > 0


def test_native_grpo_returns_ignore_kl_values(native):
    # Guard the documented configuration trap: GRPO's reward path currently
    # uses these tensors for response shape, not for a reference penalty.
    function = load_function("slime/utils/ppo_utils.py", "get_grpo_returns", {"torch": torch})
    rewards = torch.tensor([0.5, -0.5])
    zero = function(rewards, [torch.zeros(3), torch.zeros(2)])
    large = function(rewards, [torch.full((3,), 10.0), torch.full((2,), 20.0)])
    for before, after in zip(zero, large, strict=True):
        torch.testing.assert_close(before, after)


@pytest.mark.parametrize("chunks", [1, 2, 8])
def test_fixed_objective_logs_importance_diagnostic_normalization_mass(native, chunks):
    _, _, masks, logs = run_loss(native, fixed=True, chunks=chunks, prefix=6)
    mass = sum(log["normalization_mass"] for log in logs)
    assert mass == pytest.approx(sum(float(mask.sum()) for mask in masks) / 2048)
    assert sum(log["tis"] for log in logs) / mass == pytest.approx(1)
    assert sum(log["ois"] for log in logs) / mass == pytest.approx(1)
