"""Regress the real repeated-1/3 IF phantom-update failure."""

import json

import pytest
import torch

from slime.utils.reward_normalization import normalize_group_rewards

from .test_updates import options
from .updates import maybe_train


@pytest.mark.parametrize('value', [1/3, 2/3, .1, -.1, 0., 1.])
@pytest.mark.parametrize('standardize', [False, True])
def test_every_uniform_group_is_exactly_zero(value, standardize):
    rewards = torch.full((3, 8), value, dtype=torch.float32)
    assert torch.count_nonzero(normalize_group_rewards(rewards, standardize)) == 0


def test_actual_fractional_failure_and_nonuniform_native_parity():
    rewards = torch.tensor([[1/3]*8, [0., 1/3, 1., 0., 2/3, 1., 0., 0.]], dtype=torch.float32)
    native = rewards - rewards.mean(dim=-1, keepdim=True)
    native = native / (native.std(dim=-1, keepdim=True) + 1e-6)
    assert torch.count_nonzero(native[0]) == 8
    fixed = normalize_group_rewards(rewards, True)
    assert torch.count_nonzero(fixed[0]) == 0
    torch.testing.assert_close(fixed[1], native[1], rtol=0, atol=0)


def test_skip_guard_rejects_spurious_fractional_advantages_without_stepping(monkeypatch, tmp_path):
    monkeypatch.setenv('MODULAR_RL_SKIP_ZERO_ADVANTAGE', '1')
    monkeypatch.setenv('MODULAR_RL_RUN_DIR', str(tmp_path))
    calls = []
    maybe_train(options(), {'raw_reward': [1/3]*8, 'advantages': [torch.full((3,), .029802322)]*8},
                calls.append, 0)
    assert calls == []
    assert not json.loads((tmp_path/'optimizer-decisions.jsonl').read_text())['optimizer_update_applied']


def test_single_response_group_cannot_create_nan():
    assert normalize_group_rewards(torch.tensor([[1/3]]), True).item() == 0


@pytest.mark.parametrize('value', [float('inf'), float('nan')])
def test_nonfinite_rewards_are_not_hidden_as_zero(value):
    assert not normalize_group_rewards(torch.full((1, 8), value), True).isfinite().all()
