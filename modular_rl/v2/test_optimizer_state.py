"""A skipped first batch must still have a resumable zero-update Adam state."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from .optimizer_state import initialize_hybrid_state


@pytest.mark.parametrize('fused', [False, True])
def test_initialize_and_roundtrip_without_changing_weights_or_clock(fused):
    p = torch.nn.Parameter(torch.tensor([1., -2.]))
    optimizer = torch.optim.AdamW([p], lr=.01, weight_decay=0, betas=(.9, .98), fused=fused)
    master = p.detach().clone()
    state = defaultdict(dict)

    def sync():
        state[p] = {**optimizer.state[p], 'master_param': master}

    initialize_hybrid_state(SimpleNamespace(sub_optimizers=[optimizer],
                                           _sync_sub_optimizers_state_to_hdo=sync))
    assert p.grad is None
    torch.testing.assert_close(p, master, rtol=0, atol=0)
    assert optimizer.state[p]['step'] == 0
    assert torch.count_nonzero(optimizer.state[p]['exp_avg']) == 0
    assert torch.count_nonzero(optimizer.state[p]['exp_avg_sq']) == 0
    assert state[p]['master_param'] is master
    # First real update must exactly match an ordinary lazily initialized AdamW.
    q = torch.nn.Parameter(master.clone())
    control = torch.optim.AdamW([q], lr=.01, weight_decay=0, betas=(.9, .98), fused=fused)
    optimizer.load_state_dict(optimizer.state_dict())
    p.grad = q.grad = torch.tensor([.2, -.1])
    optimizer.step()
    control.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    for key, value in optimizer.state[p].items():
        torch.testing.assert_close(value, control.state[q][key], rtol=0, atol=0)
