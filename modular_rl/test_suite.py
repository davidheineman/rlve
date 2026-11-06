import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from .initializers import validate_training, verify
from .suite import prepare_selection, reuse_frozen_source
from .v2.base_models import sha256
from .v2.driver import configured_run_name
from .v2.rollout import validation_prompt_count


def initializer(tmp_path):
    hf, native = tmp_path / 'hf', tmp_path / 'native'
    hf.mkdir()
    native.mkdir()
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'):
        (hf / name).write_text('{}')
    weights = hf / 'model.safetensors'
    weights.write_bytes(b'fixture weights')
    spec = {'id': 'code', 'kind': 'domain-midtrained-base', 'hf': str(hf), 'native': str(native),
            'config_sha256': sha256(hf / 'config.json'), 'weights_sha256': sha256(weights),
            'weights_bytes': weights.stat().st_size, 'weights_mtime_ns': weights.stat().st_mtime_ns,
            'tokenizer_assets_sha256': {n: sha256(hf / n) for n in (
                'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')}}
    (native / 'latest_checkpointed_iteration.txt').write_text('release')
    (native / 'initializer_identity.json').write_text(json.dumps({'initializer': spec, 'rl_optimizer_updates': 0}))
    return spec


def test_initializer_provenance_and_source_changes_are_enforced(tmp_path):
    spec = initializer(tmp_path)
    assert verify(spec, full_weights=True) == spec
    p = Path(spec['native']) / 'initializer_identity.json'
    p.write_text(json.dumps({'initializer': {**spec, 'id': 'math'}, 'rl_optimizer_updates': 0}))
    with pytest.raises(ValueError, match='different initializer'):
        verify(spec)
    p.write_text(json.dumps({'initializer': spec, 'rl_optimizer_updates': 0}))
    (Path(spec['hf']) / 'model.safetensors').write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='weights changed'):
        verify(spec)


def test_cross_initializer_resume_and_wrong_reference_are_rejected(tmp_path):
    spec = initializer(tmp_path)
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({name: spec for name in ('base', 'code', 'math', 'chat', 'web')}))
    run = tmp_path / 'run'
    args = SimpleNamespace(hf_checkpoint=spec['hf'], ref_load=spec['native'], load=spec['native'])
    validate_training(args, registry, 'code', run, sha256(registry))
    (run / 'ckpt').mkdir()
    (run / 'ckpt/latest_checkpointed_iteration.txt').write_text('49')
    args.load = str(run / 'ckpt')
    validate_training(args, registry, 'code', run, sha256(registry))
    (run / 'initial-model-identity.json').write_text(json.dumps({'initializer': {**spec, 'id': 'math'}}))
    with pytest.raises(ValueError, match='another initializer'):
        validate_training(args, registry, 'code', run, sha256(registry))
    args.ref_load = str(tmp_path / 'wrong')
    with pytest.raises(ValueError, match='paths differ'):
        validate_training(args, registry, 'code', run, sha256(registry))


@pytest.mark.parametrize('name', ['base math', 'BASE-MATH', 'base|math'])
def test_simple_names_and_declared_validation_count(monkeypatch, tmp_path, name):
    monkeypatch.setenv('MODULAR_RL_RUN_NAME', name)
    with pytest.raises(ValueError, match='lowercase'):
        configured_run_name('old')
    monkeypatch.setenv('MODULAR_RL_RUN_NAME', 'mid-chat-if')
    assert configured_run_name('old') == 'mid-chat-if'
    selection = tmp_path / 'math'
    selection.mkdir()
    info = selection / 'selection.json'
    info.write_text(json.dumps({'validation_prompts': 64}))
    monkeypatch.setenv('MODULAR_RL_SELECTION', str(tmp_path))
    monkeypatch.setenv('MODULAR_RL_VALIDATION_PROMPTS', '64')
    assert validation_prompt_count('math') == 64
    info.write_text(json.dumps({'validation_prompts': 32}))
    with pytest.raises(ValueError, match='frozen selection'):
        validation_prompt_count('math')


def test_expanded_validation_retains_old_prompts_and_excludes_training_groups(tmp_path):
    base = tmp_path / 'base'
    for task in ('math', 'code', 'if', 'chat'):
        rows = []
        for i in range(100):
            meta = {'task': task, 'environment': task, 'key': str(i), 'split_group': str(i),
                    'messages': [{'role': 'user', 'content': str(i)}],
                    'target': {'rejected_reference': 'reference'}}
            rows.append({'metadata': meta})
        for part in ('selection', 'data'):
            d = base / part / task
            d.mkdir(parents=True)
            (d / 'train.jsonl').write_text(json.dumps(rows[0]) + '\n')
            val = rows[1:33] if part == 'selection' else rows
            (d / 'val.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in val))
    tokenizer = type('Tokenizer', (), {'apply_chat_template': lambda *a, **k: 'prompt',
                                     '__call__': lambda *a, **k: {'input_ids': [1, 2]}})()
    result = prepare_selection(base, tmp_path / 'result', base / 'selection', [tokenizer] * 5, tokenizer)
    for task, info in result.items():
        assert info['counts'] == {'train': 1, 'val': 64}
        rows = [json.loads(line) for line in (tmp_path / 'result' / task / 'val.jsonl').open()]
        keys = {r['metadata']['key'] for r in rows}
        assert keys >= {str(i) for i in range(1, 33)} and '0' not in keys


def test_frozen_control_reuse_preserves_single_instrumentation_and_arguments(tmp_path):
    import shlex
    parent, repo = tmp_path / 'controls', tmp_path / 'repo'
    original = parent / 'source'
    for name, value in {'slime/actor.py': 'probe()\nmaybe_train()\n',
                        'modular_rl/__init__.py': '', 'modular_rl/v2/config.yaml':
                        'balance-data: false\nactor-num-gpus-per-node: 2\nnum-gpus-per-node: 2\ntensor-model-parallel-size: 1\n'}.items():
        path = original / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    helpers = repo / 'modular_rl'
    helpers.mkdir(parents=True)
    for name in ('initializers.py', 'initializer_convert.py', 'suite.py', 'suite_conversions.py'):
        (helpers / name).write_text('# current provenance helper\n')
    for directory in ('runtime', 'selection'):
        (parent / directory).mkdir()
        (parent / directory / 'fixture.json').write_text('{}')
    runs, expected = [], {}
    for task in ('math', 'code', 'if', 'chat'):
        extra = '--task ' + task + ' --lr 2e-6 --quoted "literal value"'
        script = parent / (task + '.sh')
        script.write_text('export MODULAR_RL_EXTRA_ARGS=' + shlex.quote(extra) + '\n')
        runs.append({'name': task, 'task': task, 'command': [str(script)], 'script_sha256': sha256(script)})
        expected[task] = extra
    plan = {'policy_gpus_per_run': 2, 'reward_gpus_per_chat_run': 1, 'preflight_only': False,
            'source': str(original), 'source_sha256': {str(p.relative_to(original)): sha256(p)
                                                     for p in original.rglob('*') if p.is_file()}, 'runs': runs}
    (parent / 'launch-plan.json').write_text(json.dumps(plan))
    child, extras = reuse_frozen_source(parent, tmp_path / 'student', repo, 2, 1)
    assert (child / 'slime/actor.py').read_text() == 'probe()\nmaybe_train()\n'
    assert extras == expected
    assert (child / 'modular_rl/initializers.py').read_text() == '# current provenance helper\n'
    with pytest.raises(ValueError, match='same production GPU arrangement'):
        reuse_frozen_source(parent, tmp_path / 'unrequested-resize', repo, 4, 1)
    resized, resized_extras = reuse_frozen_source(parent, tmp_path / 'dp4-student', repo, 4, 1, resize_policy=True)
    assert resized_extras == expected
    assert (resized / 'slime/actor.py').read_bytes() == (original / 'slime/actor.py').read_bytes()
    config = (resized / 'modular_rl/v2/config.yaml').read_text()
    assert 'actor-num-gpus-per-node: 4\n' in config and '\nnum-gpus-per-node: 4\n' in config
    assert 'tensor-model-parallel-size: 1' in config
    assert json.loads((resized.parent / 'policy-gpu-resize.json').read_text())['global_response_batch'] == 8
    (original / 'slime/actor.py').write_text('unexpected trainer edit\n')
    with pytest.raises(ValueError, match='Frozen control source changed'):
        reuse_frozen_source(parent, tmp_path / 'other-student', repo, 2, 1)


def test_code_reward_fix_installs_into_frozen_source_without_other_grader_edits(tmp_path):
    from modular_rl.dolci.code_reward import install
    implementation = Path(__file__).with_name('graders.py')
    current = implementation.read_text()
    start, end = 'def extract_code(response):\n', '    fenced = re.findall('
    fixed_prefix = start + current.split(start, 1)[1].split(end, 1)[0]
    old_prefix = start + '    response = response.rsplit("</think>", 1)[-1].strip()\n'
    old = current.replace(fixed_prefix, old_prefix)
    source = tmp_path / 'source'
    (source / 'modular_rl').mkdir(parents=True)
    path = source / 'modular_rl/graders.py'
    path.write_text(old)
    proof = install(source, implementation)
    assert path.read_text() == current and proof['before_sha256'] != proof['after_sha256']
    again = install(source, implementation)
    assert again['before_sha256'] == again['after_sha256']
