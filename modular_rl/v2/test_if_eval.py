import asyncio
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from . import if_rollout, rollout
from .submit_endpoint import prepare_if_panel


def test_if_generation_uses_full_cap_without_adding_an_answer_prefix(monkeypatch):
    status = SimpleNamespace(ABORTED='aborted', TRUNCATED='truncated', COMPLETED='completed')
    calls = []
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: 'rendered prompt')

    async def generate(args, sample, parameters):
        assert not args.answer_forcing
        calls.append(dict(parameters))
        sample.response = 'apple'
        sample.response_length = 2048
        sample.status = status.COMPLETED
        return sample

    monkeypatch.setitem(sys.modules, 'slime.rollout.sglang_rollout', SimpleNamespace(
        GenerateState=lambda args: SimpleNamespace(tokenizer=tokenizer), generate=generate))
    monkeypatch.setitem(sys.modules, 'slime.utils.types', SimpleNamespace(Sample=SimpleNamespace(Status=status)))
    monkeypatch.setattr(rollout, '_COUNTS', [])
    monkeypatch.setattr(rollout, '_SAMPLES', [])
    args = SimpleNamespace(answer_forcing=True)
    sample = SimpleNamespace(tokens=[], metadata={'task': 'if', 'messages': []})
    asyncio.run(if_rollout.generate(args, sample, {'max_new_tokens': 2048}))
    assert args.answer_forcing and calls == [{'max_new_tokens': 2048}]
    assert sample.response == 'apple' and '_modular_v2_forcing' not in sample.metadata
    assert rollout._COUNTS == [{'sampled_tokens': 2048, 'forced': 0, 'truncated': 0}]


def test_fixed_if_panel_keeps_bytes_and_rejects_leakage_or_changed_data(tmp_path):
    d = tmp_path / 'selection/if'
    d.mkdir(parents=True)
    def row(key, group):
        return {'metadata': {'task': 'if', 'environment': 'if', 'key': key,
                             'split_group': group, 'messages': []}}
    (d / 'train.jsonl').write_text(json.dumps(row('train', 'train-group')) + '\n')
    panel = d / 'val.jsonl'
    panel.write_text(json.dumps(row('val', 'val-group')) + '\n')
    original = panel.read_bytes()
    (d / 'selection.json').write_text(json.dumps({'sha256': {
        'val': hashlib.sha256(original).hexdigest()}}))
    tokenizer = type('Tokenizer', (), {'apply_chat_template': lambda *a, **k: 'prompt',
                                     '__call__': lambda *a, **k: {'input_ids': [1, 2]}})()
    prepare_if_panel(tmp_path, tokenizer, 1)
    assert panel.read_bytes() == original
    (d / 'train.jsonl').write_text(json.dumps(row('another-key', 'val-group')) + '\n')
    with pytest.raises(ValueError, match='disjoint'):
        prepare_if_panel(tmp_path, tokenizer, 1)
    panel.write_text(panel.read_text() + '\n')
    with pytest.raises(ValueError, match='declared'):
        prepare_if_panel(tmp_path, tokenizer, 1)
