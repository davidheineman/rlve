import json

import pytest

from .response_quality import panel_quality, passes_guard


def save(tmp_path, response):
    path = tmp_path / 'panel.jsonl'
    rows = [{'metadata_json': json.dumps({'key': 'problem', 'task': 'code',
                                        'messages': [{'role': 'user', 'content': 'def solve(x): ...'}]}),
             'response': response, 'response_tokens': 100} for _ in range(8)]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


def test_code_quality_uses_real_python_api_and_short_responses_do_not_repeat(tmp_path):
    quality = panel_quality(save(tmp_path, 'def solve(x): return x+1'), prompts=1)
    assert quality['mean_repeated_6word_gram_fraction'] == 0
    assert quality['parseable_implementation_fraction'] == 1
    assert quality['required_api_present_fraction'] == 1


def test_repetitive_but_parseable_code_is_not_eligible(tmp_path):
    response = 'def solve(x):\n' + ('    x = x + 1 # repeated line\n' * 80) + '    return x\n'
    quality = panel_quality(save(tmp_path, response), prompts=1)
    assert not passes_guard(quality, {'mean_repeated_6word_fraction_at_most': .3,
                                     'response_fraction_repetition_above_half_at_most': .25})


def test_quality_refuses_missing_declared_problems(tmp_path):
    with pytest.raises(ValueError, match='every declared'):
        panel_quality(save(tmp_path, 'def solve(x): return x'), prompts=2)
