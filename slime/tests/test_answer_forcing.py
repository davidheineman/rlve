import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch

from slime.rollout import sglang_rollout
from slime.utils.types import Sample


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text.encode())}


def reply(text, finish_reason):
    return {
        "text": text,
        "meta_info": {
            "output_token_logprobs": [(-0.5, token) for token in text.encode()],
            "finish_reason": {"type": finish_reason},
        },
    }


class AnswerForcingTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            sglang_router_ip="localhost",
            sglang_router_port=30000,
            use_http2=False,
            answer_forcing=True,
            answer_forcing_extra_tokens=128,
        )
        self.state = SimpleNamespace(tokenizer=FakeTokenizer())
        self.params = {"max_new_tokens": 8, "temperature": 1.0}

    async def test_forces_only_after_cap_and_masks_prefix(self):
        calls = []

        async def fake_post(url, payload, use_http2):
            calls.append(payload)
            return [reply("thinking", "length"), reply("correct\"}", "stop")][len(calls) - 1]

        with patch.object(sglang_rollout, "GenerateState", return_value=self.state), patch.object(
            sglang_rollout, "post", side_effect=fake_post
        ):
            sample = await sglang_rollout.generate(self.args, Sample(prompt="P"), self.params)

        prefix = sglang_rollout.ANSWER_FORCING_PREFIX
        assert sample.status == Sample.Status.COMPLETED
        assert sample.response == f'thinking{prefix}correct"}}'
        assert sample.response_length == len(sample.tokens) - 1
        assert len(sample.loss_mask) == len(sample.rollout_log_probs) == sample.response_length
        assert sample.loss_mask == [1] * 8 + [0] * len(prefix.encode()) + [1] * len('correct"}')
        assert calls[0]["sampling_params"]["max_new_tokens"] == 8
        assert calls[1]["sampling_params"]["max_new_tokens"] == 128
        assert calls[1]["input_ids"] == list(("Pthinking" + prefix).encode())
        assert self.params["max_new_tokens"] == 8

    async def test_completed_response_is_not_forced(self):
        async def fake_post(url, payload, use_http2):
            return reply('{"output": "ok"}', "stop")

        with patch.object(sglang_rollout, "GenerateState", return_value=self.state), patch.object(
            sglang_rollout, "post", side_effect=fake_post
        ):
            sample = await sglang_rollout.generate(self.args, Sample(prompt="P"), self.params)
        assert sample.status == Sample.Status.COMPLETED
        assert sample.loss_mask is None
        assert sglang_rollout.ANSWER_FORCING_KEY not in sample.metadata

    async def test_rlve_uses_its_answer_markers(self):
        self.args.rlve = True
        for marker, answer_prefix, answer in (
            (r"\boxed{}", r"\boxed{", "42}"),
            ("<answer></answer>", "<answer>", "42</answer>"),
        ):
            for reasoning in ("thinking", "</think>"):
                with self.subTest(marker=marker, reasoning=reasoning):
                    self.args.answer_marker_type = marker
                    calls = []

                    async def fake_post(url, payload, use_http2):
                        calls.append(payload)
                        return [reply(reasoning, "length"), reply(answer, "stop")][len(calls) - 1]

                    with patch.object(sglang_rollout, "GenerateState", return_value=self.state), patch.object(
                        sglang_rollout, "post", side_effect=fake_post
                    ):
                        sample = await sglang_rollout.generate(
                            self.args, Sample(prompt="P", metadata=json.dumps({"environment": "Multiplication"})), self.params
                        )
                    assert sample.response.endswith(answer_prefix + answer)
                    assert sample.response.count("</think>") == 1
                    assert sample.metadata["environment"] == "Multiplication"
                    assert sample.status == Sample.Status.COMPLETED
                    prefix_tokens = sample.metadata[sglang_rollout.ANSWER_FORCING_KEY]["prefix_tokens"]
                    assert sample.loss_mask == [1] * len(reasoning) + [0] * prefix_tokens + [1] * len(answer)

    async def test_aborted_continuation_resumes_without_second_prefix(self):
        calls = []

        async def fake_post(url, payload, use_http2):
            calls.append(payload)
            return [reply("thinking", "length"), reply("ans", "abort"), reply('wer"}', "stop")][len(calls) - 1]

        with patch.object(sglang_rollout, "GenerateState", return_value=self.state), patch.object(
            sglang_rollout, "post", side_effect=fake_post
        ):
            sample = await sglang_rollout.generate(self.args, Sample(prompt="P"), self.params)
            assert sample.status == Sample.Status.ABORTED
            sample = await sglang_rollout.generate(self.args, sample, self.params)
        assert sample.status == Sample.Status.COMPLETED
        assert sample.response.count('{"output": "') == 1
        assert calls[2]["sampling_params"]["max_new_tokens"] == 125
        assert len(sample.loss_mask) == sample.response_length


if __name__ == "__main__":
    unittest.main()
