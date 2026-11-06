"""Keep model identity, response budgets and objective reuse distinct."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import base_sweep
from .base_models import verify_hf
from .base_sweep import Cell, arguments, load_cells, matrix, validate_cells
from .rollout import forcing_prefix, validation_prompt_count


def test_sweep_covers_every_requested_axis_without_reasoning_weights():
    cells = matrix()
    assert len(cells) == len({c.name for c in cells}) == 24
    assert {c.model for c in cells} == {"Qwen3-0.6B-Base", "Qwen3-1.7B-Base"}
    for model in {c.model for c in cells}:
        for cap in (2048, 4096):
            group = [c for c in cells if c.model == model and c.cap == cap]
            assert len(group) == 6
            assert {c.responses for c in group} == {8000, 16000}
            assert {c.passes for c in group} == {1, 2, 4}
            assert {c.lr for c in group} == {2e-6, 4e-6, 1e-5}
            assert all(c.responses // 8 in (1000, 2000) for c in group)
            assert all("--max-position-embeddings 32768" in arguments(c) for c in group)


@pytest.mark.parametrize("name", ["Qwen3-0.6B", "Qwen3-1.7B", "trained-Base-copy"])
def test_reasoning_or_renamed_checkpoint_is_rejected_before_file_loading(tmp_path, name):
    with pytest.raises(ValueError, match="exactly"):
        verify_hf(tmp_path / name)
    with pytest.raises(ValueError, match="exact"):
        arguments(Cell(name, 2048, "control"))


def test_only_declared_base_preflight_can_reduce_validation_size(tmp_path, monkeypatch):
    (tmp_path / "math").mkdir()
    info = tmp_path / "math/selection.json"
    info.write_text(json.dumps({"preflight_only": True, "validation_prompts": 4}))
    monkeypatch.setenv("MODULAR_RL_SELECTION", str(tmp_path))
    monkeypatch.delenv("MODULAR_RL_REQUIRE_PRETRAINED_BASE", raising=False)
    assert validation_prompt_count() == 32
    monkeypatch.setenv("MODULAR_RL_REQUIRE_PRETRAINED_BASE", "1")
    assert validation_prompt_count() == 4
    info.write_text(json.dumps({"validation_prompts": 4}))
    assert validation_prompt_count() == 32


def test_base_forcing_does_not_insert_an_unopened_thinking_tag():
    assert forcing_prefix("math", "Some calculation", base_model=True) == "\n\n\\boxed{"
    assert "</think>" in forcing_prefix("math", "<think>Some calculation", base_model=True)


def test_stability_conditions_use_an_explicit_reference_loss():
    cells = load_cells(Path(__file__).with_name("base_stability_cells.json"))
    assert len(cells) == 4
    assert {c.reference_kl for c in cells} == {0, .01, .03}
    for cell in cells:
        args = arguments(cell)
        assert ("--use-kl-loss" in args) == bool(cell.reference_kl)
        if cell.reference_kl:
            assert "--kl-loss-type k3" in args
            assert f"--kl-loss-coef {cell.reference_kl}" in args
        assert "--kl-coef " not in args  # GRPO does not use this as an anchor.


@pytest.mark.parametrize("changes", [
    {"lr": float("nan")}, {"reference_kl": -.01}, {"reference_kl": float("inf")},
    {"responses": 7}, {"passes": 3}, {"variant": "../other-run"},
    {"seed": -1}, {"rollout_seed": True}, {"seed": 2**31},
])
def test_invalid_explicit_conditions_cannot_reach_submission(changes):
    values = {"model": "Qwen3-0.6B-Base", "cap": 2048, "variant": "control", **changes}
    with pytest.raises(ValueError):
        validate_cells([Cell(**values)])


def test_distinct_conditions_cannot_overwrite_the_same_directory():
    with pytest.raises(ValueError, match="unique"):
        validate_cells([Cell("Qwen3-0.6B-Base", 2048, "control"),
                        Cell("Qwen3-0.6B-Base", 2048, "control", lr=4e-6)])


def test_explicit_replication_seeds_control_training_and_prompt_order():
    ordinary = arguments(Cell("Qwen3-1.7B-Base", 2048, "control"))
    assert "--seed " not in ordinary and "--rollout-seed " not in ordinary
    seeded = arguments(Cell("Qwen3-1.7B-Base", 2048, "seed2027", seed=2027, rollout_seed=2027))
    assert "--seed 2027" in seeded and "--rollout-seed 2027" in seeded


def test_code_cells_keep_partial_training_reward_and_answer_reserve_explicit():
    cell = Cell("Qwen3-1.7B-Base", 4096, "partial", task="code", reward_variant="partial", answer_reserve=1024)
    assert "-code-" in cell.name
    assert "--custom-rm-path modular_rl.v2.code_reward.partial_test_reward" in arguments(cell)
    with pytest.raises(ValueError, match="only supported for code"):
        arguments(Cell("Qwen3-1.7B-Base", 4096, "bad", reward_variant="partial"))
    with pytest.raises(ValueError, match="inside"):
        arguments(Cell("Qwen3-1.7B-Base", 2048, "bad", task="code", answer_reserve=2048))


@pytest.mark.parametrize("task", ["math", "code"])
def test_custom_kl_launch_is_independent_and_keeps_filter_settings(tmp_path, monkeypatch, task):
    checked, calls = [], []
    tokenizer = SimpleNamespace(eos_token_id=151643, chat_template="official Base template",
                                apply_chat_template=lambda *a, **kw: "assistant prefix")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)))

    def verify(hf, native):
        checked.append((hf.name, native.name))
        return {"repo_id": "Qwen/" + hf.name}

    def freeze(repo, experiment, base, tasks, skip_zero=False):
        assert tasks == [task] and skip_zero
        source = experiment / "source"
        (source / "modular_rl/v2").mkdir(parents=True)
        (source / "modular_rl/v2/config.yaml").write_text("use-dynamic-batch-size: true\n")
        selection = experiment / "selection" / task
        selection.mkdir(parents=True)
        (selection / "train.jsonl").write_text(json.dumps({"metadata": {"messages": [
            {"role": "user", "content": "Compute1+1"}]}}) + "\n")
        return source

    def submit(command, *, env, text):
        calls.append((command, env))
        return "12345\n"

    monkeypatch.setattr(base_sweep, "verify_initialization", verify)
    monkeypatch.setattr(base_sweep, "freeze", freeze)
    monkeypatch.setattr(base_sweep.subprocess, "check_output", submit)
    jobs = base_sweep.launch(tmp_path / "repo", tmp_path / "base/v2", tmp_path / "run",
                             cells=[Cell("Qwen3-0.6B-Base", 2048, "kl003", reference_kl=.03,
                                         task=task, answer_reserve=512)],
                             preflights=False)
    assert len(jobs) == len(calls) == 1
    assert checked == [("Qwen3-0.6B-Base", "Qwen3-0.6B-Base_torch_dist")]
    command, env = calls[0]
    assert not any(x.startswith("--dependency") for x in command)
    assert env["MODULAR_RL_FILTER_UNIFORM_REFERENCE_BATCHES"] == "1"
    assert env["MODULAR_RL_REQUIRE_PRETRAINED_BASE"] == "1"
    assert env["MODULAR_RL_TASKS"] == task
    assert env["MODULAR_RL_ANSWER_FORCING_TOKENS"] == "512"
    assert jobs[0]["run"].endswith("Qwen3-0.6B-Base-" + task)
    assert "--use-kl-loss --kl-loss-type k3 --kl-loss-coef 0.03" in env["MODULAR_RL_EXTRA_ARGS"]
    assert jobs[0]["reference_kl"] == .03
