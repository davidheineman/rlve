from types import SimpleNamespace

import pytest
import torch

from modular_rl.metarl import normalize, pass_at_k_weights, post_process, reweight


@pytest.mark.parametrize('max_k', [1, 4, 8])
def test_weights_are_derivative_of_harmonic_pass_at_k_objective(max_k):
    p = torch.tensor([[.02], [.125], [.5], [.875], [.99]], requires_grad=True)
    harmonic = sum(1 / k for k in range(1, max_k + 1))
    objective = sum((1 - (1 - p)**k) / k for k in range(1, max_k + 1)) / harmonic
    objective.sum().backward()
    torch.testing.assert_close(pass_at_k_weights(p.detach(), max_k), p.grad)


def test_k_four_endpoint_limits_and_clamped_probability_are_finite():
    weights = pass_at_k_weights([[0,0], [1,1], [-2,0], [2,4]], 4)
    torch.testing.assert_close(weights.flatten(), torch.tensor([1.92, .48, 1.92, .48]))
    assert torch.isfinite(normalize([[0]*8,[1]*8], 4)).all()
    assert torch.count_nonzero(normalize([[0]*8,[1]*8], 4)) == 0


def test_weight_survives_centering_and_remains_different_across_prompts():
    rewards = torch.tensor([[1.] + [0.]*7, [1.]*7 + [0.]])
    centered = rewards - rewards.mean(1, keepdim=True)
    advantages = normalize(rewards, 4)
    weights = pass_at_k_weights(rewards, 4)
    torch.testing.assert_close(advantages, centered * weights)
    torch.testing.assert_close(advantages.sum(1), torch.zeros(2), atol=1e-6, rtol=0)
    assert weights[0] > weights[1]
    # K=1 is centered pass@1 PG, not group-standardized GRPO.
    torch.testing.assert_close(normalize(rewards, 1), centered)
    torch.testing.assert_close(advantages / centered, weights.expand_as(centered))


def test_token_rewards_receive_same_prompt_weight_as_scalar_rewards():
    rewards = torch.tensor([[1.,0.], [1.,1.]])
    tokens = torch.arange(12.).reshape(2,2,3)
    scalar, per_token = reweight(rewards, 4, tokens)
    weights = pass_at_k_weights(rewards,4)
    torch.testing.assert_close(scalar,rewards*weights)
    torch.testing.assert_close(per_token,tokens*weights.unsqueeze(-1))


def test_native_hook_bypasses_std_and_preserves_raw_correctness():
    args=SimpleNamespace(advantage_estimator='grpo',normalize_advantages=False,
                         n_samples_per_prompt=8,rollout_batch_size=2,optimize_pass_k=4)
    rewards=[1.]+[0.]*7+[1.]*7+[0.]
    samples=[SimpleNamespace(metadata={'key':str(i//8)},get_reward_value=lambda args,r=r:r)
             for i,r in enumerate(rewards)]
    raw,advantages=post_process(args,samples)
    assert raw==rewards
    torch.testing.assert_close(torch.tensor(advantages).reshape(2,8),normalize(torch.tensor(rewards).reshape(2,8),4))
    args.normalize_advantages=True
    with pytest.raises(ValueError,match='whitening'):post_process(args,samples)
    args.normalize_advantages=False
    samples[1].metadata['key']='different'
    with pytest.raises(ValueError,match='prompt keys'):post_process(args,samples)


@pytest.mark.parametrize('max_k', [0,-1,True,1.5])
def test_invalid_k_is_rejected(max_k):
    with pytest.raises(ValueError):pass_at_k_weights([[0,1]],max_k)


def test_resume_rejects_changing_pass_k(tmp_path,monkeypatch):
    import json
    from dataclasses import asdict
    from modular_rl.adaptive_entropy import Controller,initialize
    monkeypatch.setenv('MODULAR_RL_EXPLORATION_METHOD','metarl')
    directory=tmp_path/'iter_0000049';directory.mkdir()
    state={'method':'metarl','rollout_id':49,'controller':asdict(Controller(steps=80)),
           'method_config':{'optimize_pass_k':4}}
    (directory/'exploration-state.json').write_text(json.dumps(state))
    args=SimpleNamespace(actor_num_nodes=1,actor_num_gpus_per_node=4,tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,context_parallel_size=1,calculate_per_token_loss=False,
        use_kl_loss=False,kl_coef=0,entropy_coef=0,train_reuse_epochs=4,
        load=str(tmp_path),save=str(tmp_path),optimize_pass_k=4)
    initialize(args,49)
    args.optimize_pass_k=1
    with pytest.raises(ValueError,match='objective settings changed'):initialize(args,49)
