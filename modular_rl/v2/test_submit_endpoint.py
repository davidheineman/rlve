import json

import pytest

from . import submit_endpoint


@pytest.mark.parametrize("model", ["Qwen3-0.6B-Base", "Qwen3-1.7B-Base"])
@pytest.mark.parametrize("cap", [2048, 4096])
@pytest.mark.parametrize("task", ["math", "code", "if"])
def test_base_endpoints_preserve_identity_and_cap_without_default_dependencies(tmp_path, monkeypatch, model, cap, task):
    base = tmp_path / "scratch/modular-mopd-rl/v2"
    baseline = base.parents[1] / "models" / (model + "_torch_dist")
    baseline.mkdir(parents=True)
    (baseline / "latest_checkpointed_iteration.txt").write_text("release")
    trained = tmp_path / "trained/ckpt"
    trained.mkdir(parents=True)
    (trained / "latest_checkpointed_iteration.txt").write_text("49")
    (trained / "rlve_checkpoint_state.json").write_text(json.dumps({"completed": True}))
    identity = trained.parent / "initial-model-identity.json"
    identity.write_text(json.dumps({"repo_id": "Qwen/" + model}))
    verified, calls = [], []

    def verify(hf, native):
        verified.append((hf, native))
        return {"repo_id": "Qwen/" + model}

    def freeze(repo, experiment, base, tasks):
        source = experiment / "source"
        source.mkdir(parents=True)
        return source

    def submit(command, env, text):
        calls.append((command, env))
        return str(100 + len(calls))

    monkeypatch.setattr(submit_endpoint, "verify_initialization", verify)
    monkeypatch.setattr(submit_endpoint, "freeze", freeze)
    monkeypatch.setattr(submit_endpoint, "prepare_panel", lambda *args: None)
    monkeypatch.setattr(submit_endpoint, "prepare_if_panel", lambda *args: None)
    monkeypatch.setattr(submit_endpoint.subprocess, "check_output", submit)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: object())
    policies = [{"task": task, "name": name, "checkpoint": str(path)}
                for name, path in (("base", baseline), ("candidate", trained))]
    experiment = tmp_path / "endpoint"
    reserve = 1024 if task == "code" else 256
    submit_endpoint.submit(tmp_path, base, experiment, policies, model=model,
                           max_sampled_tokens=cap, answer_reserve=reserve,
                           code_format="prefix" if task == "code" else "chat")
    assert verified == [(baseline.parent / model, baseline)]
    assert len(calls) == 2
    for command, environment in calls:
        assert not any(arg.startswith("--dependency=") for arg in command)
        assert environment["MODULAR_RL_MODELS"] == model
        assert environment["MODULAR_RL_MAX_RESPONSE_TOKENS"] == str(cap)
        assert environment["MODULAR_RL_ANSWER_FORCING_TOKENS"] == str(reserve)
        assert environment["MODULAR_RL_TASKS"] == task
        if task == "code":
            assert "--custom-generate-function-path modular_rl.v2.code_generation.generate" in environment["MODULAR_RL_EXTRA_ARGS"]
            assert "--custom-rm-path modular_rl.v2.code_generation.reward" in environment["MODULAR_RL_EXTRA_ARGS"]
        if task == "if":
            assert "--custom-generate-function-path modular_rl.v2.if_rollout.generate" in environment["MODULAR_RL_EXTRA_ARGS"]
            assert "--custom-rm-path modular_rl.if_grader.reward" in environment["MODULAR_RL_EXTRA_ARGS"]
        assert f"--seq-length {4096 if cap == 2048 else 8192}" in environment["MODULAR_RL_EXTRA_ARGS"]
        assert "--no-load-optim --no-load-rng" in environment["MODULAR_RL_EXTRA_ARGS"]
    assert json.loads((experiment / "plan.json").read_text())["max_sampled_tokens"] == cap
    identity.write_text(json.dumps({"repo_id": "Qwen/Qwen3-0.6B"}))
    with pytest.raises(ValueError, match="different Base model"):
        submit_endpoint.submit(tmp_path, base, tmp_path / "wrong-family", policies, model=model)
    assert len(calls) == 2
