import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import patch

from slime.utils.types import Sample
from spade_games import rollout
from spade_games.protocol import observation_suffix_ids


class FakeTokenizer:
    bos_token_id = None
    eos_token_id = 3
    eos_token = "<|im_end|>"

    def __call__(self, text, add_special_tokens=False):
        if text.startswith("\n<|im_start|>user\n"):
            assert "next observation" in text
            return {"input_ids": [11, 12]}
        return {"input_ids": [7, 8]}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return (
            messages[1]["content"] + self.eos_token +
            f"\n<|im_start|>user\n{messages[2]['content']}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


class FakeGame:
    def __init__(self):
        self.actions = []

    def reset(self, seed):
        assert seed == 4
        return "initial observation", {}

    def step(self, action):
        self.actions.append(action)
        if len(self.actions) == 1:
            return "next observation", 0.2, False, False, {}
        return "done", 1.0, True, False, {}

    def close(self):
        pass


def test_deepseek_turn_suffix_uses_model_template():
    class DeepSeekTokenizer(FakeTokenizer):
        eos_token = "<｜end▁of▁sentence｜>"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return (
                messages[1]["content"] + self.eos_token +
                f"<｜User｜>{messages[2]['content']}<｜Assistant｜><think>\n"
            )

        def __call__(self, text, add_special_tokens=False):
            assert text == "<｜User｜>next observation<｜Assistant｜><think>\n"
            return {"input_ids": [41, 42]}

    assert observation_suffix_ids(DeepSeekTokenizer(), "next observation") == [41, 42]


def test_multiturn_rollout_masks_observations_and_preserves_logprobs(monkeypatch):
    game = FakeGame()
    replies = iter([
        {"text": "look", "meta_info": {
            "output_token_logprobs": [(-0.1, 21), (-0.2, 3)],
            "finish_reason": {"type": "stop"},
        }},
        {"text": "finish", "meta_info": {
            "output_token_logprobs": [(-0.3, 22), (-0.4, 3)],
            "finish_reason": {"type": "stop"},
        }},
    ])

    async def fake_post(url, payload, use_http2):
        assert payload["input_ids"] in ([7, 8], [7, 8, 21, 3, 11, 12])
        return next(replies)

    monkeypatch.setattr(rollout, "post", fake_post)
    monkeypatch.setattr(rollout, "make_synthetic_env", lambda *args, **kwargs: game)
    fake_generation_module = SimpleNamespace(GenerateState=lambda args: SimpleNamespace(tokenizer=FakeTokenizer(), aborted=False))
    sample = Sample(
        prompt="rendered initial prompt",
        metadata={"game_file": "/tmp/game.py", "seed": 4,
                  "initial_observation": "initial observation"},
    )
    args = SimpleNamespace(sglang_router_ip="localhost", sglang_router_port=1234, use_http2=False)
    with patch.dict(sys.modules, {"slime.rollout.sglang_rollout": fake_generation_module}):
        result = asyncio.run(rollout.generate(args, sample, {"max_new_tokens": 16}))

    assert result.status == Sample.Status.COMPLETED
    assert result.tokens == [7, 8, 21, 3, 11, 12, 22, 3]
    assert result.response_length == 6
    assert result.loss_mask == [1, 1, 0, 0, 1, 1]
    assert result.rollout_log_probs == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4]
    assert game.actions == ["look", "finish"]
    assert result.metadata["spade_reward"] == {"reward": 1.0, "accuracy": 1.0, "positive_reward": 1.0}


def test_truncated_game_has_zero_episode_reward(monkeypatch):
    class TruncatedGame(FakeGame):
        def step(self, action):
            return "time up", 1.0, False, True, {}

    async def fake_post(url, payload, use_http2):
        return {"text": "guess", "meta_info": {
            "output_token_logprobs": [(-0.1, 21), (-0.2, 3)],
            "finish_reason": {"type": "stop"},
        }}

    monkeypatch.setattr(rollout, "post", fake_post)
    monkeypatch.setattr(rollout, "make_synthetic_env", lambda *args, **kwargs: TruncatedGame())
    fake_generation_module = SimpleNamespace(GenerateState=lambda args: SimpleNamespace(tokenizer=FakeTokenizer(), aborted=False))
    sample = Sample(
        prompt="rendered initial prompt",
        metadata={"game_file": "/tmp/game.py", "seed": 4,
                  "initial_observation": "initial observation"},
    )
    args = SimpleNamespace(sglang_router_ip="localhost", sglang_router_port=1234, use_http2=False)
    with patch.dict(sys.modules, {"slime.rollout.sglang_rollout": fake_generation_module}):
        result = asyncio.run(rollout.generate(args, sample, {"max_new_tokens": 16}))

    assert result.status == Sample.Status.TRUNCATED
    assert result.metadata["spade_reward"] == {"reward": 0.0, "accuracy": 0.0, "positive_reward": 0.0}
