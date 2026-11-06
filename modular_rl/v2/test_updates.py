"""Exercise actual Adam state preservation when skipping a signal-free batch."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from .updates import maybe_train


def options(**overrides):
    return SimpleNamespace(**{
        "advantage_estimator": "grpo", "use_kl_loss": False, "kl_coef": 0,
        "entropy_coef": 0, "weight_decay": 0, "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 1, "num_steps_per_rollout": 1,
        "train_reuse_epochs": 1, "use_wandb": False, **overrides,
    })


def test_skip_preserves_parameters_moments_and_adam_clock(tmp_path, monkeypatch):
    monkeypatch.setenv("MODULAR_RL_SKIP_ZERO_ADVANTAGE", "1")
    monkeypatch.setenv("MODULAR_RL_RUN_DIR", str(tmp_path))
    weight = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([weight], lr=0.01, betas=(0.9, 0.98))
    weight.grad = torch.ones_like(weight)
    optimizer.step()
    optimizer.zero_grad(set_to_none=False)
    before = weight.detach().clone()
    state = copy.deepcopy(optimizer.state[weight])

    def ordinary_step(rollout_id):
        optimizer.step()

    maybe_train(options(), {"advantages": [torch.zeros(2)]}, ordinary_step, 1)
    torch.testing.assert_close(weight, before, rtol=0, atol=0)
    for key, value in state.items():
        torch.testing.assert_close(optimizer.state[weight][key], value, rtol=0, atol=0)
    assert json.loads((tmp_path / "optimizer-decisions.jsonl").read_text()) == {
        "rollout_id": 1, "optimizer_update_applied": False,
    }
    # The control really does move on a zero-gradient step through Adam momentum.
    ordinary_step(1)
    assert not torch.equal(weight, before)


def test_active_advantages_call_native_trainer(tmp_path, monkeypatch):
    monkeypatch.setenv("MODULAR_RL_SKIP_ZERO_ADVANTAGE", "1")
    monkeypatch.setenv("MODULAR_RL_RUN_DIR", str(tmp_path))
    calls = []
    maybe_train(options(), {"advantages": [torch.ones(2), -torch.ones(2)]}, calls.append, 5)
    assert calls == [5]
    assert json.loads((tmp_path / "optimizer-decisions.jsonl").read_text())["optimizer_update_applied"]


@pytest.mark.parametrize("overrides", [
    {"use_kl_loss": True}, {"entropy_coef": 0.01}, {"weight_decay": 0.01},
    {"actor_num_gpus_per_node": 2}, {"train_reuse_epochs": 2},
])
def test_skip_cannot_suppress_another_objective_or_parallel_step(monkeypatch, overrides):
    monkeypatch.setenv("MODULAR_RL_SKIP_ZERO_ADVANTAGE", "1")
    with pytest.raises(ValueError, match="requires"):
        maybe_train(options(**overrides), {"advantages": [torch.zeros(2)]}, None, 0)


def test_default_does_not_change_native_training(monkeypatch):
    monkeypatch.delenv("MODULAR_RL_SKIP_ZERO_ADVANTAGE", raising=False)
    calls = []
    maybe_train(None, None, calls.append, 7)
    assert calls == [7]


@pytest.mark.parametrize("epochs", [2, 4])
def test_declared_objective_reuse_skips_all_passes_or_keeps_exact_optimizer_count(tmp_path, monkeypatch, epochs):
    monkeypatch.setenv("MODULAR_RL_SKIP_ZERO_ADVANTAGE", "1")
    monkeypatch.setenv("MODULAR_RL_ALLOW_OBJECTIVE_REUSE", "1")
    monkeypatch.setenv("MODULAR_RL_RUN_DIR", str(tmp_path))
    calls = []
    args = options(train_reuse_epochs=epochs)
    maybe_train(args, {"advantages": [torch.zeros(2)] * epochs}, calls.append, 0)
    maybe_train(args, {"advantages": [torch.tensor([1., -1.])] * epochs}, calls.append, 1)
    assert calls == [1]
    rows = [json.loads(x) for x in (tmp_path / "optimizer-decisions.jsonl").read_text().splitlines()]
    assert [r["optimizer_steps"] for r in rows] == [0, epochs]


def test_explicit_reference_batch_filter_skips_uniform_but_keeps_active_objective(tmp_path, monkeypatch):
    monkeypatch.setenv("MODULAR_RL_SKIP_ZERO_ADVANTAGE", "1")
    monkeypatch.setenv("MODULAR_RL_FILTER_UNIFORM_REFERENCE_BATCHES", "1")
    monkeypatch.setenv("MODULAR_RL_RUN_DIR", str(tmp_path))
    args = options(use_kl_loss=True, kl_loss_type="k3", kl_loss_coef=.01)
    calls = []
    maybe_train(args, {"advantages": [torch.zeros(2)]}, calls.append, 0)
    assert calls == []
    maybe_train(args, {"advantages": [torch.ones(2), -torch.ones(2)]}, calls.append, 1)
    assert calls == [1]
    decisions = [json.loads(line) for line in (tmp_path / "optimizer-decisions.jsonl").read_text().splitlines()]
    assert [d["optimizer_update_applied"] for d in decisions] == [False, True]


@pytest.mark.parametrize("overrides", [
    {"use_kl_loss": False}, {"kl_loss_type": "kl"}, {"kl_loss_coef": 0},
])
def test_reference_filter_rejects_unanchored_or_wrong_penalty(monkeypatch, overrides):
    monkeypatch.setenv("MODULAR_RL_SKIP_ZERO_ADVANTAGE", "1")
    monkeypatch.setenv("MODULAR_RL_FILTER_UNIFORM_REFERENCE_BATCHES", "1")
    args = options(use_kl_loss=True, kl_loss_type="k3", kl_loss_coef=.01)
    for name, value in overrides.items():
        setattr(args, name, value)
    with pytest.raises(ValueError, match="positive K3"):
        maybe_train(args, {"advantages": [torch.zeros(2)]}, None, 0)
