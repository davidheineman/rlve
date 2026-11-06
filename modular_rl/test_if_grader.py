import random

import pytest

from .if_controls import IF_CASES
from .if_grader import grade_if, if_checkers, if_target, registry


@pytest.mark.parametrize('name,kwargs,positive,negative', IF_CASES)
def test_registered_constraint_known_answers(name, kwargs, positive, negative):
    target = {'instruction_id': [name], 'kwargs': [kwargs]}
    assert grade_if(positive, target)['accuracy'] == 1
    assert grade_if(negative, target)['accuracy'] == 0


def test_controls_cover_registry_and_grading_preserves_training_rng():
    assert {c[0] for c in IF_CASES} == set(registry())
    before = random.getstate()
    if_checkers({'instruction_id': ['length_constraints:number_words'], 'kwargs': [{}]})
    assert random.getstate() == before


def test_partial_reward_is_separate_from_strict_accuracy_and_empty_fails():
    target = {'instruction_id': ['keywords:existence', 'punctuation:no_comma'],
              'kwargs': [{'keywords': ['apple']}, {}]}
    result = grade_if('apple, pear', target)
    assert result['reward'] == .5 and result['accuracy'] == 0
    assert result['constraint_passes'] == [True, False]
    assert grade_if('', target)['reward'] == 0
    assert grade_if('<think>ignore, this</think>apple<|endoftext|>', target)['accuracy'] == 1


def test_copy_span_zero_is_inclusive_and_target_errors_fail_fast():
    target = {'instruction_id': ['new:copy_span_idx'],
              'kwargs': [{'prompt_to_repeat': 'abcdef', 'n_start': 0, 'n_end': 2}]}
    assert grade_if('abc', target)['accuracy'] == 1
    assert grade_if('ab', target)['accuracy'] == 0
    assert if_target([str([target])]) == target
    for invalid in ({}, {'instruction_id': ['x'], 'kwargs': []}):
        with pytest.raises(ValueError):
            if_target(invalid)
    with pytest.raises(ValueError, match='Unsupported IF'):
        if_checkers({'instruction_id': ['typo:unknown'], 'kwargs': [{}]})
