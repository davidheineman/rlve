"""The preferred response should receive the positive DPO gradient."""

import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location("dpo", Path(__file__).resolve().parents[1] / "utils" / "dpo.py")
dpo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dpo)


def test_dpo_pair_gradient_and_fixed_reference():
    chosen = torch.tensor(-3.0, requires_grad=True)
    rejected = torch.tensor(-4.0, requires_grad=True)
    losses, margin = dpo.pair_loss([chosen, rejected], [torch.tensor(-3.0), torch.tensor(-4.0)])
    assert margin.item() == pytest.approx(0)
    assert losses.item() == pytest.approx(0.693147, abs=1e-5)
    losses.sum().backward()
    assert chosen.grad.item() < 0
    assert rejected.grad.item() > 0


def test_dpo_rejects_unpaired_scores():
    with pytest.raises(ValueError, match="interleaved"):
        dpo.pair_loss([torch.tensor(0.0)], [torch.tensor(0.0)])
