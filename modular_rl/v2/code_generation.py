"""Opt-in Python-prefix completion for raw Base models, with masked scaffolding."""
import copy

from .rollout import FORCING_KEY

PREFIX = '```python\n'


def initialize_prefix(sample, tokenizer, total):
    if sample.metadata['task'] != 'code' or sample.tokens:
        raise ValueError('Only initialize a fresh code sample')
    sample.prompt = tokenizer.apply_chat_template(sample.metadata['messages'], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=True)
    prompt_ids = tokenizer(sample.prompt, add_special_tokens=False)['input_ids']
    prefix_ids = tokenizer(PREFIX, add_special_tokens=False)['input_ids']
    sample.tokens = [*prompt_ids, *prefix_ids]
    sample.response = PREFIX
    sample.response_length = len(prefix_ids)
    sample.loss_mask = [0] * len(prefix_ids)
    sample.rollout_log_probs = [0.] * len(prefix_ids)
    sample.metadata[FORCING_KEY] = {'initial_tokens': 0, 'prefix_tokens': len(prefix_ids),
                                  'answer_char_start': 0, 'sampled_token_budget': total,
                                  'mode': 'python-prefix'}


def submission_text(response):
    """A supplied opening fence may end naturally at EOS without a closing fence."""
    response = response.rstrip()
    while True:
        before = response
        for marker in ('<|endoftext|>', '<|im_end|>'):
            response = response.removesuffix(marker).rstrip()
        if response == before:
            break
    if response.startswith(PREFIX) and response.count('```') == 1:
        response = response[len(PREFIX):]
    return response


async def generate(args, sample, sampling_params):
    from slime.rollout.sglang_rollout import GenerateState

    from .rollout import generate as original

    if not sample.tokens:
        initialize_prefix(sample, GenerateState(args).tokenizer, sampling_params['max_new_tokens'])
    return await original(args, sample, sampling_params)


async def reward(args, sample, **kwargs):
    from .rollout import reward as original

    if sample.metadata.get(FORCING_KEY, {}).get('mode') != 'python-prefix':
        raise ValueError('Code-prefix reward requires its matching generation mode')
    graded = copy.copy(sample)
    graded.response = submission_text(sample.response)
    # The prefix mode always grades from character0, so no stale offset is used.
    return await original(args, graded, **kwargs)
