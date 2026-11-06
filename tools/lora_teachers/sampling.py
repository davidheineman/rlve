"""Sample and verify RLVE responses with the original answer-forcing behavior."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from tinker import types

from Gym.environments import identifier2environment
from slime.utils.data import custom_prompt_preprocessor


@dataclass(frozen=True)
class Completion:
    tokens: list[int]
    logprobs: list[float]
    mask: list[int]
    reward: float
    accuracy: int
    forced: bool
    truncated: bool


def prompt_tokens(tokenizer, user_prompt):
    args = SimpleNamespace(custom_prompt_preprocessor="ChatTemplate_BoxedAnswer", answer_marker_type=r"\boxed{}")
    messages = custom_prompt_preprocessor(args, user_prompt, apply_chat_template=True)
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)


def forcing_prefix(response):
    return ("\n\n" if "</think>" in response else "\n</think>\n\n") + r"\boxed{"


async def sample_group(sampler, tokenizer, prompt, environment, config, settings, *, evaluation=False):
    count = settings["n-samples-per-eval-prompt" if evaluation else "n-samples-per-prompt"]
    limit = settings["eval-max-response-len" if evaluation else "rollout-max-response-len"]
    params = types.SamplingParams(
        max_tokens=limit,
        temperature=settings["eval-temperature" if evaluation else "rollout-temperature"],
        top_p=settings["eval-top-p"] if evaluation else 1.0,
    )
    result = await sampler.sample_async(
        prompt=types.ModelInput.from_ints(prompt), num_samples=count, sampling_params=params
    )

    async def finish(sequence):
        tokens = list(sequence.tokens)
        if sequence.logprobs is None or len(sequence.logprobs) != len(tokens):
            raise ValueError("Sampling must return a log probability for every generated token")
        logprobs = list(sequence.logprobs)
        mask = [1] * len(tokens)
        forced = sequence.stop_reason == "length" and settings["answer-forcing"]
        truncated = sequence.stop_reason == "length"
        response = tokenizer.decode(tokens, skip_special_tokens=True)
        if forced:
            injected = tokenizer.encode(forcing_prefix(response), add_special_tokens=False)
            continuation = await sampler.sample_async(
                prompt=types.ModelInput.from_ints(prompt + tokens + injected),
                num_samples=1,
                sampling_params=params.model_copy(update={"max_tokens": settings["answer-forcing-extra-tokens"]}),
            )
            tail = continuation.sequences[0]
            if tail.logprobs is None or len(tail.logprobs) != len(tail.tokens):
                raise ValueError("Answer-forcing continuation has missing log probabilities")
            tokens += injected + list(tail.tokens)
            logprobs += [0.0] * len(injected) + list(tail.logprobs)
            mask += [0] * len(injected) + [1] * len(tail.tokens)
            response += forcing_prefix(response) + tokenizer.decode(tail.tokens, skip_special_tokens=True)
            truncated = tail.stop_reason == "length"
        verifier = identifier2environment[environment](answer_markers=(r"\boxed{", "}"))
        verifier.set_config(config)
        score = verifier.verifier(response)
        return Completion(tokens, logprobs, mask, float(score["reward"]), int(score["accuracy"]), forced, truncated)

    return await asyncio.gather(*(finish(sequence) for sequence in result.sequences))


def training_datums(prompt, completions):
    datums, masks, logprobs = [], [], []
    for completion in completions:
        tokens = prompt + completion.tokens
        datums.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens[:-1]), loss_fn_inputs={"target_tokens": tokens[1:]}
            )
        )
        masks.append([0] * (len(prompt) - 1) + completion.mask)
        logprobs.append([0.0] * (len(prompt) - 1) + completion.logprobs)
    return datums, masks, logprobs
