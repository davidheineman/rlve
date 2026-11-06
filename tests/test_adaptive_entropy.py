import json
import math
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from modular_rl.adaptive_entropy import Controller, clipped_surrogate, weighted_surprisal, initialize


def test_integral_controller_direction_bounds_and_equality():
    state = Controller(maximum=.5)
    state.update(.1)
    assert state.k == .25
    state.update(.1)
    state.update(.1)
    assert state.k == .5
    state.update(.3)
    assert state.k == .5
    for _ in range(4): state.update(.6)
    assert state.k == 0 and state.steps == 8
    with pytest.raises(ValueError): state.update(float('nan'))


def test_two_level_clipping_preserves_correct_gradient_branches():
    log_ratio = torch.tensor([math.log(2), math.log(.2), math.log(2), math.log(.2)], requires_grad=True)
    advantage = torch.tensor([1., 1., -1., -1.])
    value = clipped_surrogate(log_ratio, advantage, .2, 1.25, outer=50)
    torch.testing.assert_close(value, torch.tensor([1.25, .2, -2., -.8]))
    value.sum().backward()
    torch.testing.assert_close(log_ratio.grad, torch.tensor([0., .2, -2., 0.]))


def test_outer_bound_has_finite_zero_gradient_for_extreme_mismatches():
    log_ratio = torch.tensor([1000., 1000.], requires_grad=True)
    value = clipped_surrogate(log_ratio, torch.tensor([1., -1.]), .2, 1.25, outer=50)
    torch.testing.assert_close(value, torch.tensor([1.25, -50.]))
    value.sum().backward()
    torch.testing.assert_close(log_ratio.grad, torch.zeros(2))


def test_widening_upper_bound_changes_gradient_after_policy_moves():
    def gradient(log_ratio, upper):
        ratio = torch.tensor([log_ratio], requires_grad=True)
        clipped_surrogate(ratio, torch.ones(1), .2, upper, outer=50).sum().backward()
        return ratio.grad
    # Explains why a single on-policy optimization pass cannot test this method.
    torch.testing.assert_close(gradient(0., 1.25), gradient(0., 2.))
    torch.testing.assert_close(gradient(math.log(1.5), 1.25), torch.zeros(1))
    torch.testing.assert_close(gradient(math.log(1.5), 1.75), torch.tensor([1.5]))


def test_entropy_estimator_uses_importance_weights_and_global_token_counts():
    logp = torch.tensor([-.1, -.8, -1., -2., -3.], requires_grad=True)
    sampling = torch.tensor([-.2, -.5, -.9, -2., -2.])
    mask = torch.tensor([1, 0, 1, 1, 1])
    complete = weighted_surprisal(logp, sampling, mask)
    sharded = weighted_surprisal(logp[:1], sampling[:1], mask[:1]) + weighted_surprisal(logp[1:], sampling[1:], mask[1:])
    torch.testing.assert_close(complete, sharded)
    expected = sum(-logp[i].item() * math.exp(logp[i].item()-sampling[i].item()) for i in (0,2,3,4)) / 4
    assert complete[0] / complete[1] == pytest.approx(expected)
    assert not complete.requires_grad
    assert complete[0] / complete[1] != pytest.approx(float((-logp.detach()*mask).sum()/mask.sum()))


def test_global_token_loss_gradient_is_invariant_to_uneven_dp_partition():
    # Four ranks, two microbatches/rank, unequal response lengths and forced masks.
    values = [torch.tensor(x) for x in ([1., 2.], [3.], [4.,5.,6.], [7.], [8.,9.], [10.], [11.], [12.,13.,14.])]
    total = sum(v.numel() for v in values)
    direct = torch.tensor(2., requires_grad=True)
    (sum((direct*v).sum() for v in values)/total).backward()
    grads = []
    for rank in range(4):
        parameter = torch.tensor(2.,requires_grad=True)
        for v in values[rank::4]:
            custom = (parameter*v).sum()*8/total
            native = custom*2/8*4
            (native/2).backward()  # Megatron microbatch scaling.
        grads.append(parameter.grad)
    torch.testing.assert_close(torch.stack(grads).mean(), direct.grad)


def test_resume_restores_controller_from_exact_model_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv('MODULAR_RL_EXPLORATION_METHOD','mai')
    folder = tmp_path/'iter_0000049'; folder.mkdir()
    controller = Controller(k=.75,steps=120)
    state = {'method':'mai','rollout_id':49,'controller':asdict(controller)}
    (folder/'exploration-state.json').write_text(json.dumps(state))
    args = SimpleNamespace(actor_num_nodes=1,actor_num_gpus_per_node=4,tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,context_parallel_size=1,calculate_per_token_loss=False,
        use_kl_loss=False,kl_coef=0,entropy_coef=0,train_reuse_epochs=4,load=str(tmp_path),save=str(tmp_path))
    initialize(args,49)
    from modular_rl import adaptive_entropy
    assert adaptive_entropy._runtime['controller'] == controller
    monkeypatch.setenv('MODULAR_RL_ENTROPY_TARGET','.4')
    with pytest.raises(ValueError,match='settings changed'): initialize(args,49)
    monkeypatch.delenv('MODULAR_RL_ENTROPY_TARGET')
    with pytest.raises(FileNotFoundError): initialize(args,48)


@pytest.mark.parametrize('kwargs',[{'target':0},{'delta':-1},{'maximum':-1},{'k':3},{'steps':-1}])
def test_invalid_controller_settings_are_rejected(kwargs):
    with pytest.raises(ValueError): Controller(**kwargs)


def test_eight_prompt_weighting_changes_gradient_direction_after_norm_clipping():
    from modular_rl.maxrl import normalize as maxrl
    from modular_rl.metarl import normalize as metarl
    from slime.utils.reward_normalization import normalize_group_rewards

    rewards = torch.tensor([[1.]*n + [0.]*(8-n) for n in (1,7,2,6,3,5,4,4)])

    def gradient(advantages):
        # Each prompt supplies an independent model-feature direction. The
        # global clip can rescale their sum but cannot undo relative weights.
        parameter = torch.nn.Parameter(torch.zeros(8))
        (-(advantages*rewards*parameter[:,None]).sum()).backward()
        torch.nn.utils.clip_grad_norm_([parameter], .1)
        return parameter.grad

    reference = gradient(normalize_group_rewards(rewards, True))
    for advantage in (maxrl(rewards), metarl(rewards, 4)):
        weighted = gradient(advantage)
        torch.testing.assert_close(weighted.norm(), torch.tensor(.1))
        assert torch.nn.functional.cosine_similarity(reference,weighted,dim=0) < .99


def test_single_pass_grpo_and_batch_recipe_are_checked_on_resume(tmp_path, monkeypatch):
    from modular_rl import adaptive_entropy
    monkeypatch.setenv('MODULAR_RL_EXPLORATION_METHOD','grpo')
    args = SimpleNamespace(actor_num_nodes=1,actor_num_gpus_per_node=4,tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,context_parallel_size=1,calculate_per_token_loss=False,
        use_kl_loss=False,kl_coef=0,entropy_coef=0,train_reuse_epochs=1,
        rollout_batch_size=8,n_samples_per_prompt=8,load=str(tmp_path),save=str(tmp_path))
    initialize(args,-1)
    runtime = adaptive_entropy._runtime
    assert runtime['recipe_config']['objective_passes'] == 1
    directory=tmp_path/'iter_0000005';directory.mkdir()
    (directory/'exploration-state.json').write_text(json.dumps({
        'method':'grpo','rollout_id':5,'controller':asdict(runtime['controller']),
        'method_config':{},'recipe_config':runtime['recipe_config']}))
    initialize(args,5)
    args.rollout_batch_size=1
    with pytest.raises(ValueError,match='batch or objective passes changed'):
        initialize(args,5)
