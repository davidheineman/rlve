from types import SimpleNamespace

import pytest
import torch

from modular_rl.maxrl import normalize, post_process


def test_binary_estimator_matches_centered_official_maxrl():
    for successes in range(9):
        r = torch.tensor([[1.] * successes + [0.] * (8 - successes)])
        a = normalize(r)
        assert torch.isfinite(a).all()
        assert abs(a.sum().item()) < 1e-5
        expected = (r - successes / 8) / (successes / 8 + 1e-6)
        torch.testing.assert_close(a, expected)
        if successes in (0, 8):
            assert torch.count_nonzero(a) == 0
    # Several prompts remain separate rather than normalizing the whole batch.
    a = normalize([[1, 0, 0, 0], [1, 1, 0, 0]])
    torch.testing.assert_close(a[0, 0], torch.tensor(3.), atol=3e-5, rtol=0)
    torch.testing.assert_close(a[1, 0], torch.tensor(1.), atol=3e-5, rtol=0)


@pytest.mark.parametrize('rewards', [[[.5, 1]], [[float('nan'), 0]], [[-1, 1]], [[1]]])
def test_rejects_invalid_rewards(rewards):
    with pytest.raises(ValueError):
        normalize(rewards)


def test_hook_preserves_raw_rewards_and_rejects_whitening_or_mixed_groups():
    args = SimpleNamespace(advantage_estimator='grpo', normalize_advantages=False,
                           n_samples_per_prompt=2, rollout_batch_size=1)
    samples = [SimpleNamespace(metadata={'key': 'one'}, get_reward_value=lambda args: 1),
               SimpleNamespace(metadata={'key': 'one'}, get_reward_value=lambda args: 0)]
    raw, advantages = post_process(args, samples)
    assert raw == [1, 0]
    assert advantages == pytest.approx([1, -1], abs=1e-5)
    args.normalize_advantages = True
    with pytest.raises(ValueError): post_process(args, samples)
    args.normalize_advantages = False
    samples[1].metadata['key'] = 'two'
    with pytest.raises(ValueError): post_process(args, samples)
