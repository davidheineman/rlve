from types import SimpleNamespace

from modular_rl.graders import grade_code

from .code_generation import PREFIX, initialize_prefix, submission_text
from .rollout import FORCING_KEY, sampled_tokens


def test_python_prefix_is_prompt_scaffolding_with_zero_loss_and_no_sampled_tokens():
    tokenizer = type('Tokenizer', (), {'apply_chat_template': lambda *a, **k: 'question',
                                     '__call__': lambda self, s, **k: {'input_ids': [1, 2] if s == PREFIX else [3, 4, 5]}})()
    sample = SimpleNamespace(tokens=[], metadata={'task': 'code', 'messages': []})
    initialize_prefix(sample, tokenizer, 2048)
    assert sample.tokens == [3, 4, 5, 1, 2] and sample.response == PREFIX
    assert sample.loss_mask == [0, 0] and sample.rollout_log_probs == [0., 0.]
    assert sampled_tokens(sample) == 0 and sample.metadata[FORCING_KEY]['sampled_token_budget'] == 2048


def test_supplied_opening_fence_and_terminal_control_token_are_not_python_source():
    response = PREFIX + 'def add_one(x): return x+1\n<|endoftext|>'
    code = submission_text(response)
    assert grade_code(code, {'setup': '', 'tests': ['assert add_one(5) == 6']})['accuracy'] == 1
    assert '<|endoftext|>' in submission_text(PREFIX + 'def f(): return "<|endoftext|>"')


def test_closed_fence_and_invalid_implementation_keep_original_semantics():
    response = PREFIX + 'def add_one(x): return 0\n```<|endoftext|>'
    assert grade_code(submission_text(response), {'setup': '', 'tests': ['assert add_one(5) == 6']})['accuracy'] == 0
