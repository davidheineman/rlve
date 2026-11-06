import asyncio
import hashlib
import json
import sys
from collections import Counter
from enum import Enum
from types import SimpleNamespace

import pytest

from modular_rl.v2 import math_reward, rollout, submit
from modular_rl.v2.data import boxed_answer, normalize_math, stratified_sample, stratum
from modular_rl.v2.driver import (
    commit_validation,
    training_updates,
    validate_run_settings,
)
from modular_rl.v2.inspect_rollouts import read_events
from modular_rl.v2.mbpp_spec import with_public_example
from modular_rl.v2.report import report
from slime.utils.checkpoint_state import read_checkpoint_state, write_checkpoint_state
from slime.utils.eval_metrics import early_stop_accuracy, pass_at_k


def test_response_budget_and_pass1_stopping_boundary():
    assert training_updates(8000) == 1000
    for budget in (0, -8, 8001):
        with pytest.raises(ValueError):
            training_updates(budget)
    assert (
        early_stop_accuracy(
            {"val_envs": {"rewards": [1] * 250 + [0] * 6}}, ["val_envs"]
        )
        < 0.98
    )
    assert (
        early_stop_accuracy(
            {"val_envs": {"rewards": [1] * 251 + [0] * 5}}, ["val_envs"]
        )
        >= 0.98
    )
    assert pass_at_k([4] * 32, 8)[8] == 1
    assert (
        early_stop_accuracy(
            {"val_envs": {"rewards": [1] * 128 + [0] * 128}}, ["val_envs"]
        )
        == 0.5
    )


@pytest.mark.parametrize("correct,stops", [(250, False), (251, True)])
def test_recovered_validation_commits_early_stop_before_training(
    tmp_path, correct, stops
):
    write_checkpoint_state(
        tmp_path,
        {"rollout_id": 49, "evaluated_rollout_id": -1, "completed": False},
    )
    commit_validation(tmp_path, 49, correct / 256, 0.98)
    persisted = read_checkpoint_state(tmp_path)
    assert persisted["evaluated_rollout_id"] == 49
    assert persisted.get("early_stopped", False) is stops


def test_math_targets_and_proportional_validation_do_not_expose_solutions():
    rows = [
        normalize_math(
            {
                "problem": f"Problem {index}",
                "solution": r"Secret reasoning. \boxed{\frac{1}{\sqrt{2}}}",
                "level": "Level 1",
            },
            "algebra" if index < 80 else "geometry",
            index,
        )
        for index in range(100)
    ]
    sample = stratified_sample(rows, 32, seed=42)
    assert len({r["metadata"]["key"] for r in sample}) == 32
    assert Counter(stratum(r)[0] for r in sample) == {"algebra": 26, "geometry": 6}
    assert sample == stratified_sample(rows, 32, seed=42)
    assert all("Secret reasoning" not in r["user_prompt"][0]["content"] for r in rows)
    assert (
        boxed_answer(r"old \boxed{3}; final \fbox{\frac{1}{\sqrt{2}}}")
        == r"\frac{1}{\sqrt{2}}"
    )


@pytest.mark.parametrize("solution", ["Unfinished", r"\boxed{}", r"\boxed{  }"])
def test_math_preparation_rejects_missing_or_empty_targets(solution):
    with pytest.raises(ValueError, match="nonempty complete boxed answer"):
        normalize_math(
            {"problem": "P", "solution": solution, "level": "Level 1"},
            "number_theory",
            0,
        )


@pytest.mark.parametrize(
    "prediction,gold,correct",
    [
        (r"\boxed{\frac{\sqrt{2}}{2}}", r"\frac{1}{\sqrt{2}}", 1),
        (r"\boxed{\frac{2}{4}}", r"\frac12", 1),
        (r"\boxed{(3,-1)}", "(3,-1)", 1),
        (r"\boxed{(-1,3)}", "(3,-1)", 0),
        (r"\boxed{[0,1]}", "(0,1)", 0),
        (r"\boxed{\{-1,2\}}", r"\{2,-1\}", 1),
        ("Still thinking about 3", "3", 0),
    ],
)
def test_symbolic_math_grader_controls(prediction, gold, correct):
    assert math_reward.grade(prediction, gold)["accuracy"] == correct


def test_symbolic_grading_in_spawned_cpu_worker():
    try:
        result = asyncio.run(math_reward.reward(r"\boxed{\sqrt{4}}", "2"))
        assert result["accuracy"] == 1
    finally:
        if math_reward._POOL:
            math_reward._POOL.shutdown()
            math_reward._POOL = None


class Status(Enum):
    PENDING = "pending"
    ABORTED = "aborted"
    COMPLETED = "completed"
    TRUNCATED = "truncated"


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is True
        return "P"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(text.encode())}


def fake_backend(monkeypatch, replies):
    calls = []
    tokenizer = Tokenizer()

    async def generate(args, sample, params):
        assert args.answer_forcing is False
        remaining = params["max_new_tokens"] - sample.response_length
        text, status = replies[len(calls)]
        assert len(text.encode()) <= remaining
        calls.append(remaining)
        if not sample.tokens:
            sample.tokens = [80]
        ids = list(text.encode())
        sample.tokens.extend(ids)
        sample.response += text
        sample.response_length += len(ids)
        sample.rollout_log_probs.extend([-0.5] * len(ids))
        if sample.loss_mask is not None:
            sample.loss_mask.extend([1] * len(ids))
        sample.status = status
        return sample

    backend = SimpleNamespace(
        GenerateState=lambda args: SimpleNamespace(tokenizer=tokenizer),
        generate=generate,
    )
    monkeypatch.setitem(sys.modules, "slime.rollout.sglang_rollout", backend)
    monkeypatch.setitem(
        sys.modules,
        "slime.utils.types",
        SimpleNamespace(Sample=SimpleNamespace(Status=Status)),
    )
    rollout._COUNTS.clear()
    rollout._SAMPLES.clear()
    sample = SimpleNamespace(
        prompt="",
        metadata={"task": "math", "messages": [{"role": "user", "content": "Q"}]},
        tokens=[],
        response="",
        response_length=0,
        loss_mask=None,
        rollout_log_probs=[],
        status=Status.PENDING,
    )
    args = SimpleNamespace(answer_forcing=True, answer_forcing_extra_tokens=128)
    return calls, sample, args


@pytest.mark.parametrize("task", ["math", "code", "chat"])
@pytest.mark.parametrize("total,reserve", [(1024, 128), (2048, 256)])
def test_answer_forcing_shares_token_budget_and_masks_prefix(
    monkeypatch, task, total, reserve
):
    initial = total - reserve
    calls, sample, args = fake_backend(
        monkeypatch,
        [("r" * initial, Status.TRUNCATED), ("a" * reserve, Status.COMPLETED)],
    )
    args.answer_forcing_extra_tokens = reserve
    sample.metadata["task"] = task
    result = asyncio.run(rollout.generate(args, sample, {"max_new_tokens": total}))
    assert calls == [initial, reserve]
    assert rollout.sampled_tokens(result) == total
    prefix = result.metadata[rollout.FORCING_KEY]["prefix_tokens"]
    assert result.loss_mask == [1] * initial + [0] * prefix + [1] * reserve
    assert (
        len(result.loss_mask) == len(result.rollout_log_probs) == result.response_length
    )
    assert result.response.count("</think>") == 1


def test_forcing_resumes_without_duplicate_prefix_or_exceeding_budget(monkeypatch):
    calls, sample, args = fake_backend(
        monkeypatch,
        [
            ("r" * 896, Status.TRUNCATED),
            ("a" * 40, Status.ABORTED),
            ("b" * 88, Status.COMPLETED),
        ],
    )

    async def run():
        sample1 = await rollout.generate(args, sample, {"max_new_tokens": 1024})
        assert sample1.status == Status.ABORTED
        return await rollout.generate(args, sample1, {"max_new_tokens": 1024})

    result = asyncio.run(run())
    assert calls == [896, 128, 88]
    assert rollout.sampled_tokens(result) == 1024
    assert result.response.count(r"\boxed{") == 1


def test_completed_response_is_never_forced(monkeypatch):
    calls, sample, args = fake_backend(monkeypatch, [(r"\boxed{3}", Status.COMPLETED)])
    result = asyncio.run(rollout.generate(args, sample, {"max_new_tokens": 1024}))
    assert len(calls) == 1 and result.loss_mask is None
    assert rollout.FORCING_KEY not in result.metadata


def test_driver_rejects_mismatched_train_eval_or_launch_budgets(monkeypatch):
    monkeypatch.setenv("MODULAR_RL_MAX_RESPONSE_TOKENS", "2048")
    monkeypatch.setenv("MODULAR_RL_ANSWER_FORCING_TOKENS", "256")
    args = SimpleNamespace(
        rollout_batch_size=1,
        n_samples_per_prompt=8,
        n_samples_per_eval_prompt=8,
        rollout_max_response_len=2048,
        eval_max_response_len=2048,
        answer_forcing_extra_tokens=256,
    )
    validate_run_settings(args)
    args.eval_max_response_len = 1024
    with pytest.raises(ValueError, match="equal train/eval"):
        validate_run_settings(args)
    args.rollout_max_response_len = 1024
    with pytest.raises(ValueError, match="MODULAR_RL_MAX_RESPONSE_TOKENS"):
        validate_run_settings(args)
    args.rollout_max_response_len = args.eval_max_response_len = 2048
    args.answer_forcing_extra_tokens = 2048
    with pytest.raises(ValueError, match="reserve"):
        validate_run_settings(args)


@pytest.mark.parametrize("cap,reserve", [(1024, 128), (2048, 256)])
def test_code_only_launcher_freezes_data_and_exports_generation_settings(
    tmp_path, monkeypatch, cap, reserve
):
    repo, data = tmp_path / "repo", tmp_path / "data"
    for name in ("slime", "slime_plugins", "Gym", "modular_rl", "scripts", "tools"):
        (repo / name).mkdir(parents=True)
    for name in (
        "train.py",
        "tools/yaml_to_args.py",
        "tools/convert_hf_to_torch_dist.py",
    ):
        (repo / name).write_text("# frozen source\n")
    data.mkdir()
    (data / "ready.json").write_text("{}")
    (tmp_path / "math-reference-audit.json").write_text(
        json.dumps({"passed": True, "failed": 0, "sha256": {}})
    )
    selection = tmp_path / "selection" / "code"
    selection.mkdir(parents=True)
    checksums = {}
    for split in ("train", "val"):
        content = "{}\n"
        (selection / f"{split}.jsonl").write_text(content)
        checksums[split] = hashlib.sha256(content.encode()).hexdigest()
    (selection / "selection.json").write_text(
        json.dumps(
            {
                "keys": {"val": list(range(32))},
                "enable_thinking": True,
                "sha256": checksums,
            }
        )
    )
    runtime = tmp_path / "runtime"
    (runtime / "math_verify").mkdir(parents=True)
    monkeypatch.setenv("WANDB_API_KEY", "unit-test-key")
    calls = []

    def sbatch(command, env, text):
        calls.append((command, env))
        return "12345\n"

    monkeypatch.setattr(submit.subprocess, "check_output", sbatch)
    experiment = tmp_path / "experiment"
    jobs = submit.submit(
        repo,
        experiment,
        data,
        selection.parent,
        runtime,
        tasks=["code"],
        max_response_tokens=cap,
        exclude="viscam10",
    )
    assert len(jobs) == len(calls) == 1
    command, environment = calls[0]
    assert "--gres=gpu:1" in command
    assert "--exclude=viscam10" in command
    assert environment["MODULAR_RL_TASKS"] == "code"
    assert environment["MODULAR_RL_MAX_RESPONSE_TOKENS"] == str(cap)
    assert environment["MODULAR_RL_ANSWER_FORCING_TOKENS"] == str(reserve)
    plan = json.loads((experiment / "plan.json").read_text())
    assert plan["tasks"] == ["code"]
    assert plan["max_sampled_tokens"] == cap
    assert plan["answer_forcing_reserved_tokens"] == reserve
    assert sorted(p.name for p in (experiment / "selection").iterdir()) == ["code"]


@pytest.mark.parametrize("prompts", [1, 4])
@pytest.mark.parametrize("preflight", [False, True])
@pytest.mark.parametrize("samples", [8, 16, 32])
def test_report_enforces_declared_2048_token_cap(tmp_path, prompts, preflight, samples, monkeypatch):
    monkeypatch.delenv("MODULAR_RL_REQUIRE_PRETRAINED_BASE", raising=False)
    if preflight:
        selection = tmp_path / "selection/math"
        selection.mkdir(parents=True)
        (selection / "selection.json").write_text(json.dumps({"preflight_only": True, "validation_prompts": 4}))
        monkeypatch.setenv("MODULAR_RL_SELECTION", str(selection.parent))
        monkeypatch.setenv("MODULAR_RL_REQUIRE_PRETRAINED_BASE", "1")
    directory = tmp_path / "Qwen3-0.6B-code"
    directory.mkdir()
    (directory / "completed.json").write_text(
        json.dumps(
            {
                "completed": True,
                "updates": 1,
                "train_rollouts": samples * prompts,
                "max_train_rollouts": samples * prompts,
                "prompts_per_update": prompts,
                "samples_per_prompt": samples,
                "reason": "rollout_budget",
                "sampled_token_budget": 2048,
            }
        )
    )
    (directory / "wandb_run_id.txt").write_text("test")
    (directory / "progress.jsonl").write_text(
        json.dumps(
            {
                "phase": "eval",
                "completed_updates": 1,
                "accuracy": 0.25,
            }
        )
        + "\n"
    )
    generation = {
        "phase": "train",
        "rollout_id": 0,
        "responses": samples * prompts,
        "sampled_token_budget": 2048,
        "max_sampled_tokens": 2048,
    }
    path = directory / "generation.jsonl"
    evaluation = {**generation, "phase": "eval", "responses": 32 if preflight else 256}
    path.write_text(json.dumps(generation) + "\n" + json.dumps(evaluation) + "\n")
    assert report(tmp_path)["passed"]
    path.write_text(json.dumps(generation) + "\n" + json.dumps({**evaluation, "responses": 8}) + "\n")
    with pytest.raises(RuntimeError, match="batch size"):
        report(tmp_path)
    for change in ({"max_sampled_tokens": 2049}, {"sampled_token_budget": 1024}):
        path.write_text(json.dumps({**generation, **change}) + "\n")
        with pytest.raises(RuntimeError, match="cap exceeded"):
            report(tmp_path)


def test_full_eval_export_preserves_prompt_order_targets_and_unicode(tmp_path):
    samples = [
        SimpleNamespace(
            index=index,
            metadata={"environment": "code", "target": {"tests": ["assert f()==1"]}},
            reward={"reward": 1.0, "accuracy": 1.0},
            prompt="P",
            response="text\u2028inside",
            response_length=3,
            status=Status.COMPLETED,
            label=None,
        )
        for index in (1, 0)
    ]
    samples[0].metadata["key"] = "second"
    samples[1].metadata["key"] = "first"
    path = rollout.export_evaluation_samples(tmp_path, 49, samples, "run", "group")
    rows = read_events(path)
    assert len(rows) == 2
    assert [json.loads(r["metadata_json"])["key"] for r in rows] == ["first", "second"]
    assert rows[0]["response"] == "text\u2028inside"
    samples[0].index = 0
    with pytest.raises(RuntimeError, match="complete and unique"):
        rollout.export_evaluation_samples(tmp_path, 49, samples, "run", "group")


def test_public_mbpp_example_keeps_other_tests_hidden_and_grading_unchanged():
    prompt = "Write f.\nRequired API:\ndef f(x): ...\nReturn a self-contained Python solution in a ```python code block."
    row = {
        "user_prompt": [{"role": "user", "content": prompt}],
        "metadata": {
            "task": "code",
            "key": "old",
            "messages": [{"role": "user", "content": prompt}],
            "target": {"tests": ["assert f(1)==2", "assert f(3)==4"]},
        },
    }
    revised = with_public_example(row)
    assert revised["metadata"]["target"] == row["metadata"]["target"]
    assert "assert f(1)==2" in revised["user_prompt"][0]["content"]
    assert "assert f(3)==4" not in revised["user_prompt"][0]["content"]
    assert revised["metadata"]["key"] != row["metadata"]["key"]
    assert row["user_prompt"][0]["content"] == prompt
    with pytest.raises(ValueError, match="already"):
        with_public_example(revised)
