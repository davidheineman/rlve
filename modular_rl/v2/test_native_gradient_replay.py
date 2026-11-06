"""Check diagnostic inverse storage mappings with nonuniform grouped heads."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from .native_gradient_replay import pack_native_weight


@pytest.mark.parametrize("suffix,shape", [
    ("self_attention.linear_qkv.weight", (32, 8)),
    ("mlp.linear_fc1.weight", (24, 8)),
    ("self_attention.q_layernorm.weight", (4,)),
])
def test_native_replay_roundtrips_every_grouped_qkv_head_and_gate_row(suffix, shape):
    root = Path(__file__).resolve().parents[2]
    converter = runpy.run_path(str(root / "slime/backends/megatron_utils/megatron_to_hf/qwen2.py"))[
        "convert_qwen2_to_hf"
    ]
    args = SimpleNamespace(num_attention_heads=4, num_query_groups=2, kv_channels=4, hidden_size=8)
    original = torch.arange(torch.tensor(shape).prod(), dtype=torch.float32).reshape(shape)
    name = "module.module.decoder.layers.0." + suffix
    converted = converter(args, name, original)
    weights = {key: value.clone() for key, value in converted}
    actual = pack_native_weight(name, converted, weights, args)
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
