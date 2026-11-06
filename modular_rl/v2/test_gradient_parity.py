"""Check parity capture placement and that inspection preserves native gradients."""

import ast
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from .gradient_parity import capture, instrument, reference_precision

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_capture_only_adds_observation_before_native_optimizer_step():
    original = (REPO / "slime/backends/megatron_utils/model.py").read_text()
    modified = instrument(original)
    before, after = ast.parse(original), ast.parse(modified)
    function = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == "train_one_step")
    observed = [n for n in function.body if (
        isinstance(n, ast.ImportFrom) and n.module == "modular_rl.v2.gradient_parity"
    ) or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
          and isinstance(n.value.func, ast.Name) and n.value.func.id == "capture")]
    assert len(observed) == 2
    function.body = [n for n in function.body if n not in observed]
    assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)
    assert modified.index("losses_reduced = forward_backward_func(") < modified.index("    capture(args,")
    assert modified.index("    capture(args,") < modified.index("    valid_step = True")


def test_capture_preserves_live_parameters_and_fp32_gradient_buffers(monkeypatch, tmp_path):
    converter = runpy.run_path(str(REPO / "slime/backends/megatron_utils/megatron_to_hf/qwen2.py"))[
        "convert_qwen2_to_hf"
    ]
    param = torch.nn.Parameter(torch.arange(64, dtype=torch.bfloat16).reshape(16, 4))
    param.main_grad = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 100
    original_weight, original_grad = param.detach().clone(), param.main_grad.clone()
    module = ModuleType("slime.backends.megatron_utils.update_weight_utils")
    module.named_parameters = lambda args, model: iter([(
        "module.module.decoder.layers.0.self_attention.linear_qkv.weight", param,
    )])
    module.convert_to_hf = lambda args, model_name, name, value: converter(args, name, value)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    path = tmp_path / "native.pt"
    monkeypatch.setenv("MODULAR_RL_GRADIENT_PARITY_CAPTURE", str(path))
    args = SimpleNamespace(
        entropy_coef=0, use_kl_loss=False, advantage_estimator="grpo", global_batch_size=32,
        rollout_max_response_len=2048, tis_clip_low=0, tis_clip=2, use_tis=True,
        loss_type="custom_loss", custom_loss_function_path="modular_rl.v2.losses.fixed_token_policy_loss",
        num_attention_heads=4, num_query_groups=2, hidden_size=4, kv_channels=2,
    )
    capture(args, 0, 0, [object()], SimpleNamespace(get_loss_scale=lambda: torch.tensor(1)))
    value = torch.load(path, weights_only=False)
    assert set(value["weights"]) == {
        f"model.layers.0.self_attn.{kind}_proj.weight" for kind in ("q", "k", "v")
    }
    assert all(g.dtype == torch.float32 for g in value["gradients"].values())
    assert sum(g.numel() for g in value["gradients"].values()) == param.numel()
    assert value["metadata"]["native_unclipped_gradient_norm"] > 0
    torch.testing.assert_close(param, original_weight)
    torch.testing.assert_close(param.main_grad, original_grad)


def test_later_update_cannot_overwrite_initial_capture(monkeypatch, tmp_path):
    module = ModuleType("slime.backends.megatron_utils.update_weight_utils")
    module.named_parameters = module.convert_to_hf = None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    path = tmp_path / "initial.pt"
    path.write_bytes(b"initial capture")
    monkeypatch.setenv("MODULAR_RL_GRADIENT_PARITY_CAPTURE", str(path))
    capture(None, 1, 0, None, None)
    assert path.read_bytes() == b"initial capture"


@pytest.mark.parametrize("mode", ["norm-rope", "fused"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_precision_audit_preserves_dtypes_and_restores_official_methods(one_cpu_thread, mode, dtype):
    from transformers import AutoModelForCausalLM, Qwen3Config
    from transformers.models.qwen3 import modeling_qwen3 as qwen

    model = AutoModelForCausalLM.from_config(
        Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                    num_attention_heads=2, num_key_value_heads=1, head_dim=8),
        torch_dtype=dtype, attn_implementation="sdpa",
    ).eval()
    original_apply = qwen.apply_rotary_pos_emb
    norm, rotary = model.model.norm, model.model.rotary_emb
    original_norm, original_rotary = norm.forward, rotary.forward
    mlp, original_mlp = model.model.layers[0].mlp, model.model.layers[0].mlp.forward
    tokens = torch.tensor([[1, 2, 3, 4]])
    with reference_precision(model, mode):
        cos, sin = rotary(torch.zeros(1, 4, 16, dtype=dtype), torch.arange(4).reshape(1, -1))
        assert cos.dtype == sin.dtype == torch.float32
        output = model(tokens).logits
        assert output.dtype == dtype
        output.float().sum().backward()
        assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters())
    assert norm.forward == original_norm and rotary.forward == original_rotary
    assert qwen.apply_rotary_pos_emb is original_apply
    assert mlp.forward == original_mlp
