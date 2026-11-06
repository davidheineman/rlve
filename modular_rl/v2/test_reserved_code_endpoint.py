import hashlib
import json
from types import SimpleNamespace

import pytest

from .endpoint import validate_endpoint_identity
from .submit_endpoint import prepare_reserved_panel


def row(key, group=None):
    return {'metadata': {'task': 'code', 'environment': 'code', 'key': key,
                         'split_group': group or key,
                         'messages': [{'role': 'user', 'content': 'Implementf'}]}}


def setup_panel(tmp_path, fresh):
    directory = tmp_path / 'experiment/selection/code'
    directory.mkdir(parents=True)
    for name, rows in [('train', [row('training')]), ('val', [row('monitor')])]:
        (directory / (name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (directory / 'selection.json').write_text(json.dumps({'keys': {}, 'sha256': {}}))
    panel = tmp_path / 'reserved.jsonl'
    panel.write_text(''.join(json.dumps(r) + '\n' for r in fresh))
    tokenizer = type('Tokenizer', (), {'apply_chat_template': lambda *a, **k: 'prompt',
                                     '__call__': lambda *a, **k: {'input_ids': [1, 2, 3]}})()
    return directory, panel, tokenizer


def test_reserved_code_panel_preserves_exact_bytes_and_declared_count(tmp_path):
    directory, panel, tokenizer = setup_panel(tmp_path, [row('fresh1'), row('fresh2')])
    sha = hashlib.sha256(panel.read_bytes()).hexdigest()
    prepare_reserved_panel(directory.parents[1], 'code', tokenizer, panel, 2, sha)
    assert (directory / 'val.jsonl').read_bytes() == panel.read_bytes()
    info = json.loads((directory / 'selection.json').read_text())
    assert info['validation_prompts'] == 2 and info['sha256']['val'] == sha


@pytest.mark.parametrize('fresh', [[row('training')], [row('different-api', 'monitor')]])
def test_reserved_code_cannot_overlap_development_by_key_or_problem_group(tmp_path, fresh):
    directory, panel, tokenizer = setup_panel(tmp_path, fresh)
    with pytest.raises(ValueError, match='overlaps'):
        prepare_reserved_panel(directory.parents[1], 'code', tokenizer, panel, 1,
                               hashlib.sha256(panel.read_bytes()).hexdigest())


def test_reserved_code_sha_is_checked_before_overwriting_monitoring(tmp_path):
    directory, panel, tokenizer = setup_panel(tmp_path, [row('fresh')])
    before = (directory / 'val.jsonl').read_bytes()
    with pytest.raises(ValueError, match='SHA'):
        prepare_reserved_panel(directory.parents[1], 'code', tokenizer, panel, 1, '0' * 64)
    assert (directory / 'val.jsonl').read_bytes() == before


def test_endpoint_rechecks_base_initializer_at_runtime(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'candidate/ckpt'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'latest_checkpointed_iteration.txt').write_text('999')
    identity = checkpoint.parent / 'initial-model-identity.json'
    identity.write_text(json.dumps({'repo_id': 'Qwen/Qwen3-1.7B-Base'}))
    monkeypatch.setattr('modular_rl.v2.base_models.verify_initialization',
                        lambda *args: {'repo_id': 'Qwen/Qwen3-1.7B-Base'})
    args = SimpleNamespace(hf_checkpoint=tmp_path / 'Qwen3-1.7B-Base',
                           ref_load=tmp_path / 'zero', load=checkpoint)
    assert validate_endpoint_identity(args)['actor_load_for_this_attempt'] == str(checkpoint)
    identity.write_text(json.dumps({'repo_id': 'Qwen/Qwen3-1.7B'}))
    with pytest.raises(ValueError, match='different Base'):
        validate_endpoint_identity(args)
