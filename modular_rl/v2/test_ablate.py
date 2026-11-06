import json
from pathlib import Path

import pytest

from . import ablate


def test_full_minibatch_job_keeps_response_budget_and_recovery_cadence(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    captured = {}

    def freeze(repo, experiment, base, tasks, skip_zero=False):
        source = experiment / "source"
        source.mkdir(parents=True)
        (source / "placeholder").write_text("frozen")
        repository = Path(ablate.__file__).resolve().parents[2]
        for name in ("modular_rl/v2/config.yaml", "slime/backends/megatron_utils/actor.py"):
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text((repository / name).read_text())
        return source

    def submit(command, *, env, text):
        captured.update(command=command, env=env)
        return "12345\n"

    monkeypatch.setattr(ablate, "freeze", freeze)
    monkeypatch.setattr(ablate.subprocess, "check_output", submit)
    experiment = tmp_path / "trial"
    ablate.launch(tmp_path, experiment, tmp_path / "base", 125, ["math"], ["fixed-kl-lr4"],
                  prompts=8, afterok=[123], node="jagupard35", conservative_memory=True)
    plan = json.loads((experiment / "plan.json").read_text())
    env, command = captured["env"], captured["command"]
    assert plan["training_responses"] == 8000
    assert env["MODULAR_RL_STEPS"] == "125"
    assert env["MODULAR_RL_PROMPTS_PER_UPDATE"] == "8"
    assert env["MODULAR_RL_EVAL_INTERVAL"] == env["MODULAR_RL_CHECKPOINT_INTERVAL"] == "7"
    assert env["MODULAR_RL_CUDA_CACHE_CLEAR_INTERVAL"] == "4"
    assert plan["sglang_mem_fraction_static"] == .15 and not plan["use_dynamic_batch_size"]
    assert "use-dynamic-batch-size: false" in (experiment / "source/modular_rl/v2/config.yaml").read_text()
    for option in ("--sglang-mem-fraction-static 0.15", "--sglang-max-running-requests 16",
                   "--sglang-cuda-graph-max-bs 16"):
        assert option in env["MODULAR_RL_EXTRA_ARGS"]
    for option in ("--global-batch-size 64", "--rollout-batch-size 8", "--lr 4e-6", "--kl-loss-type k3"):
        assert option in env["MODULAR_RL_EXTRA_ARGS"]
    assert "--mem=64G" in command and "--constraint=48G&ampere" in command
    assert "--gres=gpu:a6000:1" in command
    assert "--time=12:00:00" in command and "--dependency=afterok:123" in command
    assert "--nodelist=jagupard35" in command


def test_full_larger_model_requests_host_memory_for_model_and_checkpoint():
    settings = ablate.trial_settings(1000, 1, "Qwen3-1.7B")
    assert settings["memory"] == "128G"
    assert settings["time"] == "12:00:00"


@pytest.mark.parametrize("variant", ["fixed-kl-filtered", "fixed-kl-filtered-cosine"])
def test_filtered_reference_recipe_explicitly_keeps_rollout_budget_and_labels_skips(tmp_path, monkeypatch, variant):
    captured = {}

    def freeze(repo, experiment, base, tasks, skip_zero=False):
        captured["skip_zero"] = skip_zero
        source = experiment / "source"
        source.mkdir(parents=True)
        (source / "placeholder").write_text("frozen")
        return source

    def submit(command, *, env, text):
        captured.update(command=command, env=env)
        return "12345\n"

    monkeypatch.setattr(ablate, "freeze", freeze)
    monkeypatch.setattr(ablate.subprocess, "check_output", submit)
    experiment = tmp_path / "filtered"
    ablate.launch(tmp_path, experiment, tmp_path / "base", 1000, ["math"], [variant], afterok=[123])
    env = captured["env"]
    plan = json.loads((experiment / "plan.json").read_text())
    assert captured["skip_zero"]
    assert plan["training_responses"] == 8000 and plan["prompts_per_update"] == 1
    assert plan["uniform_reference_batch_filter_variants"] == [variant]
    assert env["MODULAR_RL_SKIP_ZERO_ADVANTAGE"] == "1"
    assert env["MODULAR_RL_FILTER_UNIFORM_REFERENCE_BATCHES"] == "1"
    assert env["MODULAR_RL_EVAL_INTERVAL"] == env["MODULAR_RL_CHECKPOINT_INTERVAL"] == "50"
    assert "--lr 1e-6" in env["MODULAR_RL_EXTRA_ARGS"]
    assert "--kl-loss-coef 0.01 --kl-loss-type k3" in env["MODULAR_RL_EXTRA_ARGS"]
    if variant == "fixed-kl-filtered-cosine":
        assert "--lr-decay-style cosine --lr-decay-iters 500 --min-lr 1e-7" in env["MODULAR_RL_EXTRA_ARGS"]
        assert plan["optimizer_lr_schedules"][variant]["clock"] == "applied_optimizer_updates"
        assert plan["optimizer_lr_schedules"][variant]["decay_optimizer_steps"] == 500
    else:
        assert not plan["optimizer_lr_schedules"]
        assert "--lr-decay" not in env["MODULAR_RL_EXTRA_ARGS"]
