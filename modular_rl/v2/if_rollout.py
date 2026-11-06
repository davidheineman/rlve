"""Full-budget IF generation without adding text that can violate constraints."""

import copy

from . import rollout


async def generate(args, sample, sampling_params):
    from slime.rollout.sglang_rollout import GenerateState
    from slime.rollout.sglang_rollout import generate as original
    from slime.utils.types import Sample

    if sample.metadata["task"] != "if":
        raise ValueError("IF generation requires IF metadata")
    native_args = copy.copy(args)
    native_args.answer_forcing = False
    if not sample.tokens:
        sample.prompt = GenerateState(args).tokenizer.apply_chat_template(
            sample.metadata["messages"], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
    sample = await original(native_args, sample, sampling_params)
    if sample.status == Sample.Status.ABORTED:
        return sample
    if not 0 <= sample.response_length <= sampling_params["max_new_tokens"]:
        raise RuntimeError("IF generation exceeded its sampled-token cap")
    rollout._COUNTS.append({"sampled_tokens": sample.response_length, "forced": 0,
                            "truncated": int(sample.status == Sample.Status.TRUNCATED)})
    rollout._SAMPLES.append(sample)
    return sample
