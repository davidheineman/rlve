"""Check that update diagnostics exclude injected tokens and preserve signs."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from .driver import validate_run_settings
from .update_probe import probe


@pytest.mark.parametrize("zero_advantages", [False, True])
@pytest.mark.parametrize("epochs", [1, 2, 4])
def test_probe_measures_actual_change_and_excludes_forced_prefix(tmp_path, monkeypatch, zero_advantages, epochs):
    monkeypatch.setenv("MODULAR_RL_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("MODULAR_RL_UPDATE_PROBE_INTERVAL", "1")
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    old = [torch.tensor([-0.5, -0.5, -0.5]), torch.tensor([-0.5, -0.5])]
    after = [old[0] + torch.tensor([100.0, 0.1, 0.1]), old[1] - 0.1]
    model = ModuleType("slime.backends.megatron_utils.model")
    model.forward_only = lambda *args: {"log_probs": after * epochs}
    monkeypatch.setitem(sys.modules, model.__name__, model)
    data = {
        "log_probs": old, "loss_masks": [[0, 1, 1], [1, 1]],
        "advantages": [torch.full((3,), 0.0 if zero_advantages else 1.0),
                       torch.full((2,), 0.0 if zero_advantages else -1.0)],
        "raw_reward": [1, 0], "rewards": [1, -1],
    }
    for key in ("log_probs", "loss_masks", "advantages"):
        data[key] *= epochs
    actor = SimpleNamespace(args=SimpleNamespace(
        use_wandb=False, rollout_max_response_len=2048, eps_clip=0.2, eps_clip_high=0.28,
        train_reuse_epochs=epochs,
    ), model=None)
    probe(actor, 0, data, None, None)
    result = json.loads((tmp_path / "update-probes.jsonl").read_text())
    samples, metrics = result["samples"], result["metrics"]
    assert [s["tokens"] for s in samples] == [2, 2]
    assert [s["reward"] for s in samples] == [1, 0]
    assert samples[0]["mean_delta_logp"] == pytest.approx(0.1)
    assert samples[1]["mean_delta_logp"] == pytest.approx(-0.1)
    assert metrics["david/update_probe/max_abs_delta_logp"] == pytest.approx(0.1)
    assert metrics["david/update_probe/mean_abs_delta_logp"] == pytest.approx(0.1)
    assert metrics["david/update_probe/advantage_delta_logp"] == pytest.approx(
        0.0 if zero_advantages else 0.1
    )
    assert metrics["david/update_probe/sampled_k3_old_to_new"] > 0
    assert metrics["david/update_probe/post_update_pg_clipfrac"] == 0
    assert metrics["david/update_probe/fixed_token_advantage_delta_logp"] == pytest.approx(
        0 if zero_advantages else 0.2 / 2048
    )


def test_four_prompt_batch_requires_explicit_declaration(monkeypatch):
    monkeypatch.delenv("MODULAR_RL_PROMPTS_PER_UPDATE", raising=False)
    args = SimpleNamespace(
        rollout_batch_size=4, n_samples_per_prompt=8, n_samples_per_eval_prompt=8,
        rollout_max_response_len=2048, eval_max_response_len=2048,
        answer_forcing_extra_tokens=256,
    )
    with pytest.raises(ValueError, match="declared train prompt count"):
        validate_run_settings(args)
    monkeypatch.setenv("MODULAR_RL_PROMPTS_PER_UPDATE", "4")
    validate_run_settings(args)


@pytest.mark.parametrize('group', [16, 32])
def test_larger_training_group_requires_declaration_and_keeps_eval_at_eight(monkeypatch, group):
    monkeypatch.delenv('MODULAR_RL_SAMPLES_PER_PROMPT', raising=False)
    args = SimpleNamespace(rollout_batch_size=1, n_samples_per_prompt=group,
        n_samples_per_eval_prompt=8, rollout_max_response_len=2048,
        eval_max_response_len=2048, answer_forcing_extra_tokens=256)
    with pytest.raises(ValueError, match='group size'):
        validate_run_settings(args)
    monkeypatch.setenv('MODULAR_RL_SAMPLES_PER_PROMPT', str(group))
    validate_run_settings(args)
    args.n_samples_per_eval_prompt = group
    with pytest.raises(ValueError, match='8 validation samples'):
        validate_run_settings(args)


@pytest.mark.parametrize("settings,error", [
    ({"kl_coef": 0.001}, "ignores kl_coef"),
    ({"use_kl_loss": True, "kl_loss_type": "kl"}, "reference-independent"),
    ({"use_kl_loss": True, "kl_loss_type": "k3"}, None),
    ({"use_kl_loss": False, "kl_loss_type": "kl"}, None),
])
def test_v2_rejects_ineffective_reference_penalty_configuration(monkeypatch, settings, error):
    monkeypatch.delenv("MODULAR_RL_PROMPTS_PER_UPDATE", raising=False)
    args = SimpleNamespace(
        rollout_batch_size=1, n_samples_per_prompt=8, n_samples_per_eval_prompt=8,
        rollout_max_response_len=2048, eval_max_response_len=2048,
        answer_forcing_extra_tokens=256, **settings,
    )
    if error:
        with pytest.raises(ValueError, match=error):
            validate_run_settings(args)
    else:
        validate_run_settings(args)
