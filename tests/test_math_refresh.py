import json
from types import SimpleNamespace

import numpy as np
import pytest

from modular_rl.dolci import math_refresh as core
from modular_rl.dolci.math_epoch import assignments, OneEpochBlocks
from modular_rl.dolci.math_refresh_build import near_equal, simhash, verify_group


def test_question_normalization_preserves_mathematical_changes():
    assert core.question_key(r'Find $\dfrac{x+2}{3}$.') == core.question_key(r'Find \(\frac{x+2}{3}\).')
    assert core.question_key('Find x+2.') != core.question_key('Find x+3.')
    assert core.question_key(r'Find $\Gamma(x)$.') != core.question_key(r'Find $\gamma(x)$.')
    assert core.question_key('Find x+2.' + core.PROMPT_SUFFIX) == core.question_key('Find x+2.')


def nemo_row(**overrides):
    row = {'problem': 'Find the number of even integers from 4 through 130, inclusive.',
           'uuid': 'one', 'expected_answer': '64', 'subset': 'cot', 'source': 'AoPS',
           'license': 'cc-by-4.0', 'tools': [], 'messages': [
               {'role': 'user', 'content': 'question'},
               {'role': 'assistant', 'content': 'Since successive terms differ by 2, there are (130-4)/2 = 63 gaps. '
                'Including both endpoints gives 63+1 = 64 terms. Therefore the number of even integers is \\boxed{64}.',
                'reasoning_content': 'long hidden deliberation ' * 10000}]}
    row.update(overrides); return row


def test_preserves_short_final_working_and_rejects_tools_and_incomplete_answers():
    value, reason = core.candidate('nemotron', nemo_row())
    assert reason is None and 'long hidden' not in value['response']
    assert value['prediction'] == '64'
    assert core.candidate('nemotron', nemo_row(subset='tir'))[1] == 'tool_assisted_subset'
    assert core.candidate('nemotron', nemo_row(source='Math StackExchange'))[1] == 'outside_selected_competition_source'
    row = nemo_row(); row['messages'][1]['tool_calls'] = [{'id': 'call'}]
    assert core.candidate('nemotron', row)[1] == 'tool_dependent'
    row = nemo_row(); row['messages'][1]['content'] += '\n\\boxed{'
    assert core.candidate('nemotron', row)[1] == 'unfinished_box'
    assert core.short_solution('<think>unfinished') is None
    assert core.short_solution('<think>long</think> complete solution') == 'complete solution'


def test_conservative_near_dedup_does_not_merge_changed_numbers_or_questions():
    text = 'Find the least positive integer n such that n+123=456.'
    assert near_equal(text, text)
    assert not near_equal(text, text.replace('123', '124'))
    assert not near_equal(text, text.replace('least', 'greatest'))
    long = ('Consider the following collection of positive integers and its properties. ' * 20
            + 'Find the sum of x and y.')
    assert not near_equal(long, long.replace('sum', 'product'))
    assert not near_equal(long, long.replace('x and y', 'y and x'))
    h, _ = simhash(text)
    assert isinstance(h, int) and 0 <= h < 2**64


@pytest.fixture(scope='module')
def tokenizer():
    core.initialize_worker('/nlp/scr/dhei/dolci-mix-20261029/model',
                           '/nlp/scr/dhei/dolci-mix-20261029/rl/base/runtime')
    return core.TOKENIZER


def test_real_encoding_verifies_gold_and_keeps_all_assistant_targets(tokenizer):
    value, _ = core.candidate('nemotron', nemo_row())
    value['reference_answers'] = ['64']
    verified, reason = core.verify_candidate(value)
    assert reason is None and verified['response_tokens'] <= 1536 and verified['tokens'] <= 2048
    ids = np.frombuffer(verified['encoded_ids'], dtype='<u4')
    mask = np.frombuffer(verified['encoded_mask'], dtype='u1')
    assert len(ids) == len(mask) == verified['tokens']
    assert not mask[0] and mask[-1] and mask.sum() == verified['assistant_tokens']
    assert 'boxed' in tokenizer.decode(ids[mask.astype(bool)])
    value['reference_answers'] = ['62']
    assert core.verify_candidate(value)[1] == 'reference_answer_mismatch'
    value['reference_answers'] = []
    assert core.verify_candidate(value)[1] == 'reference_answer_mismatch'
    value['reference_answers'] = ['64', '62']
    assert core.verify_candidate(value)[1] == 'conflicting_reference_answers'


def test_symbolic_answer_check_and_length_budget(tokenizer):
    assert core.equivalent(r'\frac{1}{2}', r'\(0.5\)')
    assert not core.equivalent(r'-\frac{3}{2}', r'\frac{3}{2}')
    value, _ = core.candidate('nemotron', nemo_row())
    value['response'] = ('This is a long mathematical argument. ' * 500) + r'\boxed{64}'
    value['reference_answers'] = ['64']
    assert core.verify_candidate(value)[1] == 'response_over_1536'


def test_quality_gate_catches_observed_failures_and_numeric_errors(tokenizer):
    assert core.quality_reason('Find f(4).', 'This contradicts the given conditions; nevertheless the answer is 24.')
    assert core.quality_reason('Find the area.', 'Based on the clean answer, the intended sides are AB and CD.')
    assert core.quality_reason('Find CDE.', 'The problem most likely intends the similar triangle CDF.')
    assert core.quality_reason('Find the minimal polynomial over K.', 'Use the Galois group.')
    assert core.quality_reason('Maximize the volume.', 'Optimization leads to a maximum volume ratio M=8/25. Thus the answer is 704.')
    assert core.quality_reason(r'Evaluate $\int x^2 dx$.', r'The answer is $x^3/3$.')
    assert core.numeric_consistency(r'We calculate \(6\times7=42\), so the answer is \boxed{42}.')
    assert not core.numeric_consistency(r'We calculate \(6\times7=41\), so the answer is \boxed{41}.')
    assert core.numeric_consistency(r'Substitute into \(x+2=7\) to solve for x.')


def test_synthetic_reference_alone_does_not_qualify_a_question():
    import sqlite3
    from modular_rl.dolci.math_refresh_build import references
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE refs (key TEXT, answer TEXT, source TEXT, rank INTEGER)')
    db.execute("INSERT INTO refs VALUES ('one','24','openmath',2)")
    assert references(db, 'one') == ([], [])
    db.execute("INSERT INTO refs VALUES ('two','64','nemotron',1)")
    assert references(db, 'two') == (['64'], ['nemotron'])
    references.cache_clear(); db.close()


def test_reference_formatting_does_not_hide_correct_numeric_or_interval_answers(tokenizer):
    assert core.equivalent(r'2+\frac{\pi}{2}-\sqrt{2\pi}',
                           '\\,1+\\Bigl(1-\\sqrt{\\dfrac{\\pi}{2}}\\Bigr)^{2}\n'
                           '\\;=\\;\\dfrac{\\pi}{2}+2-2\\sqrt{\\dfrac{\\pi}{2}}\\,')
    assert core.equivalent(r'(1,2)', '\\;\n1<x<2\\;')
    assert not core.equivalent('41', r'\displaystyle 6\times7=41')


def test_non_thinking_rollout_prefix_matches_sft_context(tokenizer, monkeypatch):
    from modular_rl.v2.rollout import format_prompt
    from modular_rl.dolci.prepare import encode
    messages = [{'role': 'user', 'content': 'Find x when x+7=12.'},
                {'role': 'assistant', 'content': r'Subtracting 7 gives x=5, so \boxed{5}.'}]
    _, _, _, rendered = encode(messages)
    target_start = rendered.index(messages[1]['content'])
    monkeypatch.setenv('MODULAR_RL_ENABLE_THINKING', '0')
    assert format_prompt(tokenizer, messages[:1]) == rendered[:target_start]
    monkeypatch.delenv('MODULAR_RL_ENABLE_THINKING')
    assert format_prompt(tokenizer, messages[:1]) == tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=True)


def test_packing_is_a_partition_and_never_splits_documents():
    docs = [(0, 1300), (1, 700), (2, 1000), (3, 1000), (4, 48)]
    bins = assignments(docs)
    assert sorted(i for block in bins for i in block) == list(range(5))
    lengths = dict(docs)
    assert all(sum(lengths[i] for i in block) <= 2048 for block in bins)
    assert len(bins) == 2
    with pytest.raises(ValueError): assignments([(0, 2049)])


def test_one_epoch_loader_uses_each_real_block_once_and_masks_padding(tmp_path):
    source = tmp_path / 'sources'; source.mkdir()
    ids = np.array([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [0, 0, 0, 0, 0]], dtype='<u4')
    mask = np.array([[0, 0, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]], dtype='u1')
    ids.tofile(source / 'train.bin'); mask.tofile(source / 'train.mask')
    np.save(source / 'train.counts.npy', np.array([2, 3, 0], dtype=np.uint32))
    (tmp_path / 'manifest.json').write_text(json.dumps({'one_epoch': True, 'whole_conversations': True,
        'seq_len': 4, 'seed': 1, 'splits': {'train': {'blocks': 3, 'real_blocks': 2}}}))
    stream = OneEpochBlocks(tmp_path, 'math')
    assert set(stream.indices[:2]) == {0, 1} and stream.indices[-1] == 2
    assert stream.loss_tokens(0, 3) == 5
    x, y = stream.batch(0, 3, 'cpu')
    assert y[-1].eq(-100).all() and y.ne(-100).sum().item() == 5
    with pytest.raises(ValueError): stream.batch(3, 1, 'cpu')


def test_explicit_repetition_shuffles_each_pass_and_resumes_deterministically(tmp_path):
    from modular_rl.dolci.data import MixedBlocks
    source = tmp_path / 'sources'; source.mkdir()
    ids = np.array([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24],
                    [30, 31, 32, 33, 34]], dtype='<u4')
    masks = np.array([[0, 0, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 0, 1, 0]], dtype='u1')
    ids.tofile(source / 'curated.bin'); masks.tofile(source / 'curated.mask')
    np.save(source / 'curated.counts.npy', masks[:, 1:].sum(1, dtype=np.uint32))
    manifest = {'seq_len': 4, 'seed': 20261027, 'objective': 'assistant',
                'require_unique_blocks': False, 'sources': {'curated': {'packed_tokens': 15}},
                'quotas': {'math': {'curated': 8}}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    first, resumed = MixedBlocks(tmp_path, 'math'), MixedBlocks(tmp_path, 'math')
    assert np.array_equal(first.indices, resumed.indices)
    assert set(first.indices[:3]) == set(first.indices[3:6]) == {0, 1, 2}
    assert len(set(first.indices[6:])) == 2
    x, y = first.batch(4, 4, 'cpu')
    resumed_x, resumed_y = resumed.batch(4, 4, 'cpu')
    assert x.equal(resumed_x) and y.equal(resumed_y)
    assert first.loss_tokens(4, 4) == y.ne(-100).sum().item()
