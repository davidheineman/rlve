import pytest

from modular_rl.dolci import math_large as core
from modular_rl.dolci.math_large_pack import close_question


def test_incomplete_frozen_rl_launch_dependencies_fail_before_gpu_submission(tmp_path):
    from modular_rl.dolci.math_large_experiment import validate_orchestrator_source
    with pytest.raises(ValueError, match='slime/utils/checkpoint_state.py'):
        validate_orchestrator_source(tmp_path)


def test_eight_gpu_midtraining_preserves_global_batch_and_optimizer_steps(tmp_path):
    import json
    from modular_rl.dolci.math_large_experiment import sft_resources
    four = sft_resources(tmp_path)
    (tmp_path / 'sft-resources.json').write_text(json.dumps({'gpus': 8}))
    eight = sft_resources(tmp_path)
    assert four['gpus'] == 4 and eight['gpus'] == 8
    assert four['grad_accum'] == 16 and eight['grad_accum'] == 8
    for profile in (four, eight):
        assert profile['gpus'] * profile['micro_batch'] * profile['grad_accum'] == 128
        assert profile['tokens_per_step'] == 262144
        assert 1_000_000_000 // profile['tokens_per_step'] == 3814
    assert sft_resources(tmp_path, pilot=True)['global_sequences'] == 32
    (tmp_path / 'sft-resources.json').write_text('{"gpus":3}')
    with pytest.raises(ValueError):
        sft_resources(tmp_path)


def test_additional_gpu_names_remain_explicitly_identified(tmp_path):
    from modular_rl.dolci.math_large_experiment import extend_gpu_types
    path = tmp_path / 'gpu_policy.py'
    path.write_text('import re\ndef gpu_type(name):\n    words = name.lower()\n'
                    '    if "a100" in words:\n        return "a100"\n    return None\n')
    extend_gpu_types(path)
    namespace = {}
    exec(compile(path.read_text(), str(path), 'exec'), namespace)
    assert namespace['gpu_type']('NVIDIA RTX A5000') == 'a5000'
    assert namespace['gpu_type']('NVIDIA H100 80GB') == 'h100'
    assert namespace['gpu_type']('NVIDIA A100') == 'a100'
    assert namespace['gpu_type']('NVIDIA TITAN RTX') is None
    previous = path.read_text()
    extend_gpu_types(path)
    assert path.read_text() == previous


@pytest.mark.parametrize('world,objective,steps,expected', [(8, 'assistant', 10, True),
    (8, 'assistant', 4, False), (4, 'assistant', 10, 'reject'), (8, 'smoke', 10, 'reject')])
def test_full_startup_proof_requires_real_eight_gpu_training(tmp_path, world, objective, steps, expected):
    import json
    from modular_rl.dolci.math_large_experiment import training_startup
    (tmp_path / 'sft-resources.json').write_text('{"gpus":8}')
    folder = tmp_path / 'sft/math'
    folder.mkdir(parents=True)
    config = {'world_size': world, 'tokens_per_step': 262144, 'micro_batch': 2,
              'grad_accum': 8, 'model': {'repo': 'Qwen/Qwen3-0.6B-Base'}, 'objective': objective}
    timing = {'train_tokens': steps * 262144, 'tokens_per_s': 30000., 'remaining_training_hours': 8.}
    (folder / 'config.json').write_text(json.dumps(config))
    (folder / 'timing.json').write_text(json.dumps(timing))
    (tmp_path / 'sft-job.json').write_text('{"job_id":"123"}')
    (tmp_path / 'sft-123.log').write_text('step=10/3814 tokens=2621440 ce=1.20\n')
    if expected == 'reject':
        with pytest.raises(AssertionError):
            training_startup(tmp_path)
    else:
        result = training_startup(tmp_path)
        assert (result is not None) == expected
        if result:
            assert result['passed'] and result['real_curated_data'] and not result['synthetic']
            assert result['actual_training_steps_observed'] == steps


@pytest.mark.parametrize('alter_response', [False, True])
def test_available_screen_keeps_unknown_examples_and_requires_exact_verdict_binding(tmp_path, monkeypatch, alter_response):
    import hashlib
    import json
    import sqlite3
    from modular_rl.dolci.math_available_screen import apply_available
    monkeypatch.setattr(core, 'capacity', lambda *args: None)
    (tmp_path / 'quality-policy.json').write_text('{"mode":"available-checks","user_authorized":true}')
    (tmp_path / 'manual-rejections.json').write_text('[]')
    (tmp_path / 'merge-complete.json').write_text('{}')
    folder = tmp_path / 'judge-results'
    folder.mkdir()
    rows = [{'key': str(i), 'id': str(i), 'source': 'openmath', 'tokens': 64,
             'response': f'Complete solution {i}.'} for i in range(3)]
    verdicts = [{'key': v['key'], 'accepted': i == 1,
                 'solution_sha256': hashlib.sha256(v['response'].encode()).hexdigest()}
                for i, v in enumerate(rows[:2])]
    (folder / '00.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in verdicts) + 'incomplete trailing uncommitted data')
    (folder / '00.progress.json').write_text('{"reviewed":2}')
    db = sqlite3.connect(tmp_path / 'merged.sqlite')
    db.execute('CREATE TABLE accepted(key TEXT PRIMARY KEY, bucket TEXT, source TEXT, tokens INTEGER, payload TEXT)')
    if alter_response:
        rows[1]['response'] = 'A different solution than the reviewed one.'
    for row in rows:
        db.execute('INSERT INTO accepted VALUES(?,?,?,?,?)', (row['key'], 'math-labeled', 'openmath', 64, json.dumps(row)))
    db.execute('INSERT INTO accepted VALUES(?,?,?,?,?)', ('general', 'general', 'general', 64, '{}'))
    db.commit()
    db.close()
    if alter_response:
        with pytest.raises(AssertionError):
            apply_available(tmp_path)
        assert sqlite3.connect(tmp_path / 'merged.sqlite').execute('SELECT count(*) FROM accepted').fetchone()[0] == 4
    else:
        audit = apply_available(tmp_path)
        assert not audit['all_solutions_model_reviewed']
        assert audit['review_coverage']['reviewed_questions'] == 2
        assert audit['review_coverage']['unreviewed_questions'] == 1
        assert {row[0] for row in sqlite3.connect(tmp_path / 'merged.sqlite').execute('SELECT key FROM accepted')} == {'1', '2', 'general'}
        assert apply_available(tmp_path) == audit


def test_capacity_requires_enough_unique_tokens_in_both_buckets():
    with pytest.raises(ValueError, match='two|2-epoch'):
        core.capacity(6_206_000, 1_554_000)
    with pytest.raises(ValueError):
        core.capacity(900_000_000, 10_000_000)
    assert core.capacity(600_000_000, 150_000_000)['meets_preferred']
    allowed = core.capacity(450_000_000, 150_000_000)
    assert not allowed['meets_preferred'] and max(allowed['epochs'].values()) <= 2


def test_unicode_superscripts_and_changed_math_are_not_collapsed():
    assert core.key('Find x².') != core.key('Find x2.')
    assert core.key('Find x².') == core.key('Find x^{2}.')
    assert core.key('Find $\\dfrac{x+2}{3}$.') == core.key(r'Find \(\frac{x+2}{3}\).')
    assert core.presentation_key('Please find the sum of x and y.') == core.presentation_key('Find sum of x and y.')
    assert not close_question('Find the sum of x and y.', 'Find the product of x and y.')


def test_embedded_holdouts_are_excluded_inside_persona_prompts():
    question = 'A rectangular field has width 17 meters and length 29 meters. Find its area in square meters.'
    excludes = core.Exclusions([{'problem': question}])
    assert excludes.contains(question)
    assert excludes.contains('As a farmer, help me solve this: ' + question)
    assert not excludes.contains('What is the probability of two heads when tossing a fair coin twice?')


def test_short_math_scope_does_not_confuse_integral_values_with_calculus():
    assert not core.outside_scope('Find all integral values of n satisfying n squared less than 100.')
    assert not core.outside_scope('Find the maximum area of a rectangle with perimeter 24.')
    assert core.outside_scope(r'Evaluate $\int x^2 dx$.')
    assert core.outside_scope('Calculate the rate of change of P(t) at t = 5.')
    assert core.outside_scope('Determine whether the minimum average latency path can be computed in polynomial time.')


def test_reference_agreement_does_not_excuse_changing_the_question():
    assert core.unreliable('However, considering a possible typo, the additive equation gives x.')
    assert core.unreliable('Despite earlier contradictions, the likely answer is 2.')
    assert not core.unreliable('Assume a solution exists. Subtracting the two equations gives x = 3.')
    assert core.invented_integer_roots('A polynomial has constant term -18. Maximize the product of two roots.',
                                      'Considering integer coefficients and roots, the maximum is 18.')
    assert not core.invented_integer_roots('A monic polynomial has three distinct integer roots.',
                                          'The integer roots divide its constant term.')


def test_natural_format_dolci_math_is_retained_without_a_fabricated_answer():
    row = {'messages': [{'role': 'user', 'content': 'Find the area of a rectangle with sides 4 and 7.'},
                        {'role': 'assistant', 'content': 'The area is length times width. Thus 4 times 7 gives 28 square units.'}],
           'domain': 'Math', 'source_dataset': 'Tulu 3 Persona MATH', 'id': 'one'}
    value, reason = core.standardize('dolci', row)
    assert reason is None and value['answer_format'] == 'natural'
    assert value['prediction'] is None and value['expected'] is None
    row['messages'][1]['content'] = r'The area is $4\times7=28$, hence $\boxed{28}$.'
    value, reason = core.standardize('dolci', row)
    assert reason is None and value['prediction'] == '28'


def test_openmath_majority_labels_do_not_require_membership_in_a_small_external_index():
    row = {'problem_source': 'augmented_math', 'problem': 'What is seven times eight?',
           'generated_solution': r'Seven groups of eight give $7\times8=56$, so $\boxed{56}$.',
           'expected_answer': '56'}
    value, reason = core.standardize('openmath', row)
    assert reason is None and value['expected'] == '56'
    row['problem_source'] = 'augmented_gsm8k'
    assert core.standardize('openmath', row)[1] is None


def test_full_training_refuses_a_129_pass_pool_even_if_other_checks_pass(tmp_path):
    import json
    from modular_rl.dolci.config import fingerprint
    from modular_rl.dolci.math_large_experiment import trainer
    manifest = {'repo': 'math-large-20261101', 'maximum_epochs': 2}
    manifest['signature'] = fingerprint(manifest)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'curation-audit.json').write_text(json.dumps({'passed': True,
        'source_epochs': {'math': 129, 'general': 129}, 'input_token_slots': 999817216,
        'maximum_block_exposures': {'math': 129, 'general': 129}}))
    (tmp_path / 'preflight').mkdir()
    (tmp_path / 'preflight/validation.json').write_text('{"passed": true}')
    (tmp_path / 'REVIEWED.json').write_text('{"passed": true}')
    with pytest.raises(AssertionError):
        trainer(tmp_path)
    assert not (tmp_path / 'sft-source').exists()


def test_quality_weighting_respects_preferred_exposure_when_feasible():
    from modular_rl.dolci.math_large_pack import sampling_quotas
    quotas, limit = sampling_quotas(300, 300, 300, 1000 * 2048)
    assert limit == 1.5
    assert quotas == {'math-labeled': 450, 'math-unlabeled': 350, 'general': 200}
    quotas, limit = sampling_quotas(300, 200, 300, 1000 * 2048)
    assert limit == pytest.approx(1.6)
    assert quotas['math-labeled'] == 480 and quotas['math-unlabeled'] == 320
    assert sum(quotas.values()) == 1000 and quotas['general'] == 200
    with pytest.raises(ValueError):
        sampling_quotas(100, 100, 100, 1000 * 2048)


@pytest.mark.parametrize('preserve_existing', [False, True])
def test_appending_a_source_preserves_global_question_deduplication(tmp_path, preserve_existing):
    import json
    import sqlite3
    from modular_rl.dolci.math_large_pack import merge
    from modular_rl.dolci.math_refresh_build import simhash

    db = sqlite3.connect(tmp_path / 'merged.sqlite')
    db.execute('CREATE TABLE accepted (key TEXT PRIMARY KEY, presentation TEXT UNIQUE, bucket TEXT, '
               'source TEXT, tokens INTEGER, quality INTEGER, problem TEXT, payload TEXT, ids BLOB, '
               'mask BLOB, hash TEXT,b0 INTEGER,b1 INTEGER,b2 INTEGER,b3 INTEGER)')
    problem = 'What is the sum of 2 and 2?'
    old = {'id': 'old', 'source': 'openmath', 'problem': problem, 'response': 'The sum is four.',
           'verification': 'provider_reference'}
    h, _ = simhash(problem)
    db.execute('INSERT INTO accepted VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
               (core.key(problem), core.presentation_key(problem), 'math-labeled', 'openmath', 64,
                20000, problem, json.dumps(old), b'', b'', hex(h), *[(h >> (16*i)) & 65535 for i in range(4)]))
    db.commit()
    db.close()
    (tmp_path / 'merge-complete.json').write_text(json.dumps({'dedup': {}}))
    (tmp_path / 'manual-rejections.json').write_text('[]')
    candidate = tmp_path / 'reasoning-0.sqlite'
    db = sqlite3.connect(candidate)
    db.execute('CREATE TABLE units (name TEXT, stats TEXT)')
    db.execute('CREATE TABLE accepted (key TEXT, presentation TEXT, source TEXT, tokens INTEGER, '
               'quality INTEGER,payload TEXT,ids BLOB,mask BLOB)')
    for i, p in enumerate((problem, 'What is the sum of 3 and 3?')):
        value = {**old, 'id': str(i), 'source': 'reasoning', 'problem': p}
        db.execute('INSERT INTO accepted VALUES (?,?,?,?,?,?,?,?)',
                   (core.key(p), core.presentation_key(p), 'reasoning', 64, 30000,
                    json.dumps(value), b'', b''))
    db.commit()
    db.close()
    candidate.with_suffix('.complete.json').write_text('{}')
    merge(tmp_path, [candidate], preserve_existing_solutions=preserve_existing)
    db = sqlite3.connect(tmp_path / 'merged.sqlite')
    assert db.execute('SELECT count(*) FROM accepted').fetchone()[0] == 2
    assert db.execute('SELECT count(DISTINCT key) FROM accepted').fetchone()[0] == 2
    kept = json.loads(db.execute('SELECT payload FROM accepted WHERE key=?', (core.key(problem),)).fetchone()[0])
    assert kept['id'] == ('old' if preserve_existing else '0')
    assert json.loads((tmp_path / 'merge-complete.json').read_text())['counts']['math']['questions'] == 2


def test_independent_review_is_bound_to_the_exact_solution(tmp_path):
    import hashlib
    import json
    import sqlite3
    from modular_rl.dolci.math_large_judge import apply_screen

    (tmp_path / 'judge-input').mkdir()
    (tmp_path / 'judge-results').mkdir()
    (tmp_path / 'judge-input/manifest.json').write_text('{"shards": 1}')
    incoming = tmp_path / 'judge-input/00.jsonl'
    verdict = {'key': 'one', 'accepted': True, 'solution_sha256': hashlib.sha256(b'old solution').hexdigest()}
    incoming.write_text(json.dumps({'key': 'one', 'id': 'one', 'response': 'old solution',
                                    'solution_sha256': verdict['solution_sha256']}) + '\n')
    output = tmp_path / 'judge-results/00.jsonl'
    output.write_text(json.dumps(verdict) + '\n')
    output.with_suffix('.complete.json').write_text(json.dumps({'questions': 1,
        'input_sha256': hashlib.sha256(incoming.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))
    (tmp_path / 'manual-rejections.json').write_text('[]')
    db = sqlite3.connect(tmp_path / 'merged.sqlite')
    db.execute('CREATE TABLE accepted (key TEXT, bucket TEXT, payload TEXT)')
    db.execute('INSERT INTO accepted VALUES (?,?,?)', ('one', 'math-labeled',
               json.dumps({'id': 'one', 'response': 'different solution'})))
    db.commit()
    db.close()
    with pytest.raises(AssertionError):
        apply_screen(tmp_path)
    assert sqlite3.connect(tmp_path / 'merged.sqlite').execute('SELECT count(*) FROM accepted').fetchone()[0] == 1


def test_calibration_reports_static_filter_and_standalone_model_separately():
    from modular_rl.dolci.math_large_judge import calibration_report
    examples = [{'id': 'bad', 'problem': 'Find the rate of change of P(t).',
                 'response': 'Its positive derivative means the function is decreasing.'},
                {'id': 'good', 'problem': 'Find 2+2.', 'response': 'Adding two and two gives four.'}]
    rows = [{'id': 'bad', 'accepted': True, 'expected_accept': False},
            {'id': 'good', 'accepted': True, 'expected_accept': True}]
    report = calibration_report(rows, examples, 'reviewer', 'pinned')
    assert report['passed']
    assert report['standalone_model_score']['bad_case_recall'] == 0
    assert report['score']['bad_case_recall'] == 1
    assert report['cases'][0]['prefilter_rejected']


def test_numina_requires_valid_answered_questions_and_preserves_natural_solutions():
    row = {'problem': 'Problem 1. What is the sum of 2 and 2?',
           'solution': 'Adding two to two gives a total of four.', 'answer': '4',
           'problem_is_valid': 'Yes', 'solution_is_valid': 'Yes',
           'question_type': 'math-word-problem', 'problem_type': 'Algebra', 'source': 'cn_k12'}
    value, reason = core.standardize('numina', row)
    assert reason is None and value['problem'] == 'What is the sum of 2 and 2?'
    assert value['answer_format'] == 'natural' and value['expected'] == '4'
    row['solution'] += ' Use ![diagram](https://example.org/diagram.png).'
    assert core.standardize('numina', row)[1] == 'image'
    row['question_type'] = 'proof'
    assert core.standardize('numina', row)[0] is None


def test_reasoned_reviewer_requires_a_final_complete_verdict():
    from modular_rl.dolci import math_large_judge as judge
    judge.configure_reviewer('Qwen/Qwen3-30B-A3B-Instruct-2507')
    assert not judge.accepted('ACCEPT. I have not checked the calculation yet.')
    assert not judge.accepted('The derivation appears plausible but the output stopped here.')
    assert judge.accepted('The products give 14+18+20=52.\nVERDICT: ACCEPT')
    assert not judge.accepted('A counterexample violates the constraint.\nVERDICT: REJECT')
    judge.configure_reviewer('Qwen/Qwen3-32B')


def test_mixture_weights_count_real_tokens_instead_of_differential_padding():
    from modular_rl.dolci.math_large_pack import sampling_quotas
    means = {'math-labeled': 2040, 'math-unlabeled': 1700, 'general': 2048}
    q, _ = sampling_quotas(300, 300, 300, 1000 * 2048, average_lengths=means)
    actual = sum(q[k] * means[k] for k in ('math-labeled', 'math-unlabeled')) / sum(q[k] * means[k] for k in q)
    assert abs(actual - .8) < .001
    assert q['math-labeled'] + q['math-unlabeled'] > 800
    assert sum(q.values()) == 1000 and max(q.values()) <= 600


@pytest.mark.parametrize('final,expected', [(' ACCEPT', True), (' REJECT', False)])
def test_capped_review_requests_a_real_verdict_without_default_acceptance(monkeypatch, final, expected):
    from modular_rl.dolci import math_large_judge as judge
    class Engine:
        calls = 0
        def generate(self, prompts, parameters):
            self.calls += 1
            if self.calls == 1:
                assert parameters['max_new_tokens'] == 384
                return [{'text': 'The calculation needs a final decision.'}]
            assert parameters['max_new_tokens'] == 8 and prompts[0].endswith('VERDICT:')
            return [{'text': final}]
    judge.configure_reviewer('Qwen/Qwen3-30B-A3B-Instruct-2507')
    monkeypatch.setattr(judge, 'prompt', lambda tokenizer, value: 'Math problem and solution.\n')
    output, = judge.generate_reviews(Engine(), None, [{}])
    assert judge.accepted(output['text']) == expected
    judge.configure_reviewer('Qwen/Qwen3-32B')


def test_completed_curation_text_archival_is_lossless_and_resumable(tmp_path):
    import gzip
    import hashlib
    from modular_rl.dolci.math_large_storage import archive_audited_jsonl
    (tmp_path / 'curation-audit.json').write_text('{"passed": true}')
    (tmp_path / 'independent-screen-packed.json').write_text('{"passed": true}')
    (tmp_path / 'judge-input').mkdir()
    (tmp_path / 'judge-input/manifest.json').write_text('{"shards": 12}')
    source = tmp_path / 'math-labeled.jsonl'
    data = b'{"problem":"2+2","solution":"4"}\n' * 1000
    source.write_bytes(data)
    result = archive_audited_jsonl(tmp_path, force=True)
    assert not source.exists()
    assert gzip.decompress(source.with_suffix('.jsonl.gz').read_bytes()) == data
    assert result['files'][0]['original_sha256'] == hashlib.sha256(data).hexdigest()
    assert archive_audited_jsonl(tmp_path, force=True) == result


def test_prism_keeps_only_complete_final_solutions_without_fabricating_a_reference():
    row = {'id': 'prism-1', 'problem': 'Find the sum of 2 and 2.',
           'solution': '<think>Long separate reasoning.</think>Adding gives \\boxed{4}.'}
    value, reason = core.standardize('prism', row)
    assert reason is None and value['response'] == r'Adding gives \boxed{4}.'
    assert value['expected'] is None and value['license'] == 'cc-by-4.0'
    row['solution'] = '<think>Reasoning that never completed.'
    assert core.standardize('prism', row)[1] == 'incomplete_thinking'


def test_supplement_export_does_not_invalidate_existing_review_inputs(tmp_path):
    import hashlib
    import json
    import sqlite3
    from modular_rl.dolci.math_large_supplement import append_and_export

    (tmp_path / 'supplement-plan.json').write_text(json.dumps(
        {'source': 'prism', 'shard_start': 1, 'shard_stop': 3}))
    (tmp_path / 'supplement-appended.json').write_text('{"passed": true}')
    folder = tmp_path / 'judge-input'
    folder.mkdir()
    original = b'{"immutable":"previous response with a committed verdict"}\n'
    (folder / '00.jsonl').write_bytes(original)
    (folder / 'manifest.json').write_text('{"shards":1,"questions":{"0":1}}')
    db = sqlite3.connect(tmp_path / 'merged.sqlite')
    db.execute('CREATE TABLE accepted(key TEXT, source TEXT, payload TEXT)')
    for i, source in enumerate(('openmath', 'prism', 'prism')):
        value = {'key': f'{i:064x}', 'response': f'Complete solution {i}.'}
        db.execute('INSERT INTO accepted VALUES(?,?,?)', (value['key'], source, json.dumps(value)))
    db.commit()
    db.close()
    append_and_export(tmp_path)
    assert (folder / '00.jsonl').read_bytes() == original
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['shards'] == 3 and sum(manifest['questions'].values()) == 3
    added = [json.loads(line) for p in (folder / '01.jsonl', folder / '02.jsonl') for line in p.open()]
    assert len(added) == 2
    assert all(v['solution_sha256'] == hashlib.sha256(v['response'].encode()).hexdigest() for v in added)
    before = (folder / 'manifest.json').read_bytes()
    append_and_export(tmp_path)
    assert (folder / 'manifest.json').read_bytes() == before


def test_pilot_needs_accuracy_to_survive_rl_as_well_as_improved_sft_loss():
    from modular_rl.dolci.math_large_experiment import pilot_checks
    completed = {'completed': True, 'train_rollouts': 128, 'optimizer_updates': 8}
    loss = {'nll_change_sft_minus_base': -.2}
    assert all(pilot_checks(completed, [{'pass@1': .25}, {'pass@1': .26}], loss).values())
    bad = pilot_checks(completed, [{'pass@1': .25}, {'pass@1': .02}], loss)
    assert not bad['no_generation_collapse'] and not bad['no_large_short_rl_regression']
    assert not pilot_checks({**completed, 'optimizer_updates': 0}, [{'pass@1': .25}], loss)['real_rl_updates']


def test_pilot_resume_reuses_only_a_completed_base_model_sft_export(tmp_path):
    import json
    from modular_rl.dolci.math_large_experiment import pilot_sft_completed, PILOT_TOKENS
    assert not pilot_sft_completed(tmp_path)
    output = tmp_path / 'sft/math'
    (output / 'model').mkdir(parents=True)
    result = {'train_tokens': PILOT_TOKENS,
              'config': {'objective': 'assistant', 'model': {'repo': 'Qwen/Qwen3-0.6B-Base'}}}
    (output / 'results.json').write_text(json.dumps(result))
    assert not pilot_sft_completed(tmp_path)
    (output / 'model/model.safetensors').write_bytes(b'export checked by the conversion stage')
    assert pilot_sft_completed(tmp_path)
    result['config']['model']['repo'] = 'Qwen/Qwen3-0.6B'
    (output / 'results.json').write_text(json.dumps(result))
    with pytest.raises(AssertionError):
        pilot_sft_completed(tmp_path)


def test_screen_culling_can_resume_after_a_rejection_was_already_committed(tmp_path, monkeypatch):
    import hashlib
    import json
    import sqlite3
    from modular_rl.dolci.math_large_judge import apply_screen

    monkeypatch.setattr(core, 'capacity', lambda *args: None)
    (tmp_path / 'judge-input').mkdir()
    (tmp_path / 'judge-results').mkdir()
    (tmp_path / 'judge-input/manifest.json').write_text('{"shards":1}')
    (tmp_path / 'manual-rejections.json').write_text('[]')
    (tmp_path / 'merge-complete.json').write_text('{}')
    values = [{'key': str(i), 'id': str(i), 'source': 'openmath', 'tokens': 64,
               'problem': f'Find the sum of {i} and {i}.', 'response': f'The sum is {2*i}.'} for i in range(2)]
    for value in values:
        value['solution_sha256'] = hashlib.sha256(value['response'].encode()).hexdigest()
    incoming = tmp_path / 'judge-input/00.jsonl'
    incoming.write_text(''.join(json.dumps(v) + '\n' for v in values))
    verdicts = [{**v, 'accepted': i == 1} for i, v in enumerate(values)]
    output = tmp_path / 'judge-results/00.jsonl'
    output.write_text(''.join(json.dumps(v) + '\n' for v in verdicts))
    output.with_suffix('.complete.json').write_text(json.dumps({'questions': 2, 'revision': 'pinned',
        'input_sha256': hashlib.sha256(incoming.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))
    db = sqlite3.connect(tmp_path / 'merged.sqlite')
    db.execute('CREATE TABLE accepted(key TEXT, bucket TEXT, source TEXT, tokens INTEGER, payload TEXT)')
    # The rejected first row was removed by a prior, committed delete batch.
    db.execute('INSERT INTO accepted VALUES(?,?,?,?,?)', ('1', 'math-labeled', 'openmath', 64, json.dumps(values[1])))
    db.execute('INSERT INTO accepted VALUES(?,?,?,?,?)', ('general', 'general', 'general', 64, '{}'))
    db.commit()
    db.close()
    apply_screen(tmp_path)
    report = json.loads((tmp_path / 'independent-screen-audit.json').read_text())
    assert report['remaining_counts']['math']['questions'] == 1
    assert report['retained_math_tokens'] == {'openmath': 64}
    apply_screen(tmp_path)
