"""Exercise the native loss with the data iterator's optional-None convention."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('replay', ['missing', None, 'mask'])
def test_fresh_opd_loss_accepts_missing_or_none_replay_mask(monkeypatch, replay):
    student = torch.tensor([-.8, -.9], requires_grad=True)
    package = ModuleType('opd_loss_fixture')
    package.__path__ = []
    helper = ModuleType('opd_loss_fixture.loss')
    helper.get_log_probs_and_entropy = lambda *a, **kw: {'log_probs': [student]}
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    path = Path(__file__).parents[1] / 'backends/megatron_utils/opd_loss.py'
    spec = importlib.util.spec_from_file_location('opd_loss_fixture.opd_loss', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    batch = {'unconcat_tokens': [], 'total_lengths': [], 'response_lengths': [],
             'rollout_log_probs': [torch.tensor([-.8, -.9])],
             'teacher_log_probs': [torch.tensor([-.5, -1.0])]}
    if replay is None:
        batch['opd_replay_mask'] = None
    elif replay == 'mask':
        batch['opd_replay_mask'] = [torch.tensor([0., 1.])]
    loss, metrics = module.opd_loss_function(SimpleNamespace(), batch, torch.zeros(1), lambda t: t.mean())
    torch.testing.assert_close(loss, torch.tensor(-.1))
    loss.backward()
    torch.testing.assert_close(student.grad, torch.tensor([-.15, .05]))
    assert torch.isfinite(metrics['opd_reverse_kl'])
