"""Checks for task routing and balanced, disjoint modular distillation data."""

import asyncio
import json
from collections import Counter
from types import SimpleNamespace

from modular_rl import mopd_runtime
from modular_rl.mopd_suite import TASKS, mixed_data


def test_mixed_data_balances_every_step_and_preserves_validation(tmp_path):
    for task in TASKS:
        directory = tmp_path / task
        directory.mkdir()
        for split, count in [('train', 3), ('val', 2)]:
            (directory / f'{split}.jsonl').write_text(''.join(json.dumps({'metadata': {
                'task': task, 'key': f'{task}/{split}/{i}'}}) + '\n' for i in range(count)))
    mixed_data(tmp_path, 7)
    train = [json.loads(line) for line in (tmp_path / 'mixed-train.jsonl').open()]
    val = [json.loads(line) for line in (tmp_path / 'mixed-val.jsonl').open()]
    assert len(train) == 56 and len(val) == 8
    for i in range(0, len(train), 8):
        assert Counter(row['metadata']['task'] for row in train[i:i+8]) == Counter(dict.fromkeys(TASKS, 2))
    assert not {r['metadata']['key'] for r in train} & {r['metadata']['key'] for r in val}
    before = (tmp_path / 'mixed-train.jsonl').read_bytes()
    mixed_data(tmp_path, 7)
    assert (tmp_path / 'mixed-train.jsonl').read_bytes() == before


def test_mixed_generate_uses_task_specific_caps_without_mutating_shared_args(monkeypatch):
    from modular_rl.v2 import if_rollout

    called = []
    async def ordinary(args, sample, params):
        called.append((sample.metadata['task'], args.answer_forcing_extra_tokens, params['max_new_tokens']))
        return sample
    async def instruction(args, sample, params):
        called.append(('if', 0, params['max_new_tokens']))
        return sample
    monkeypatch.setattr(mopd_runtime.rollout, 'generate', ordinary)
    monkeypatch.setattr(if_rollout, 'generate', instruction)
    args = SimpleNamespace(answer_forcing_extra_tokens=128)
    for task in TASKS:
        sample = SimpleNamespace(metadata={'task': task})
        assert asyncio.run(mopd_runtime.generate(args, sample, {'max_new_tokens': 2048})) is sample
    assert called == [('math', 256, 2048), ('code', 512, 2048), ('if', 0, 2048), ('chat', 128, 2048)]
    assert args.answer_forcing_extra_tokens == 128


def test_overlong_prompts_are_filtered_before_balanced_interleaving(tmp_path):
    class Tokenizer:
        def apply_chat_template(self, messages, *args, **kwargs):
            return messages[0]['content']
        def __call__(self, prompt):
            return {'input_ids': list(range(len(prompt)))}
    for task in TASKS:
        directory = tmp_path / task
        directory.mkdir()
        rows = [{'user_prompt': text, 'metadata': {'task': task, 'key': key}}
                for text, key in [('short', 'keep'), ('x' * 3000, 'drop')]]
        for split in ('train', 'val'):
            (directory / f'{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    mixed_data(tmp_path, 3, Tokenizer())
    info = json.loads((tmp_path / 'mixed-selection.json').read_text())
    assert info['original_pool_counts'] == dict.fromkeys(TASKS, 2)
    assert info['eligible_pool_counts'] == dict.fromkeys(TASKS, 1)
    rows = [json.loads(line) for line in (tmp_path / 'mixed-train.jsonl').open()]
    assert len(rows) == 24 and all(row['metadata']['key'] == 'keep' for row in rows)
