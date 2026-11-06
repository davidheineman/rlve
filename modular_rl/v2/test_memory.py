import ast
import sys
from pathlib import Path
from types import SimpleNamespace

from .memory import instrument_model, release_transient_cuda_cache


def test_cache_release_completes_streams_before_releasing_unused_blocks(monkeypatch):
    calls = []
    cuda = SimpleNamespace(synchronize=lambda: calls.append("synchronize"),
                           empty_cache=lambda: calls.append("empty_cache"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setenv("MODULAR_RL_CUDA_CACHE_CLEAR_INTERVAL", "8")
    for index in range(17):
        release_transient_cuda_cache(index)
    assert calls == ["synchronize", "empty_cache"] * 3
    monkeypatch.setenv("MODULAR_RL_CUDA_CACHE_CLEAR_INTERVAL", "0")
    release_transient_cuda_cache(0)
    assert len(calls) == 6


def test_cache_guard_only_changes_forward_boundaries_and_preserves_no_grad():
    path = Path(__file__).resolve().parents[2] / "slime/backends/megatron_utils/model.py"
    source = path.read_text()
    result = instrument_model(source)
    original_nodes = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    new_nodes = {n.name: n for n in ast.parse(result).body if isinstance(n, ast.FunctionDef)}
    for name, node in original_nodes.items():
        if name not in ("forward_only", "train_one_step"):
            assert ast.dump(new_nodes[name]) == ast.dump(node)
        else:
            assert [ast.dump(n) for n in node.decorator_list] == [
                ast.dump(n) for n in new_nodes[name].decorator_list]
    assert result.count("release_transient_cuda_cache(cache_guard_microbatch,") == 2
