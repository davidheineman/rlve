"""Prepare pinned Dolci IF pools and a proportional, fixed validation panel."""

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from modular_rl.if_controls import IF_CASES
from modular_rl.if_grader import grade_if, if_target, registry
from modular_rl.prepare import digest, write_json

from .data import stratified_sample
from .inspect_rollouts import read_events

DATASET = 'allenai/Dolci-RL-Zero-IF-7B'
REVISION = 'c96e4e424c0a5d416f725c488ec7c61b7f758d85'


def normalize(row, index):
    messages = [{'role': 'user', 'content': re.sub(r'^user:\s*', '', row['prompt'], count=1)}]
    return {'id': str(row.get('id', row.get('key', index))), 'key': digest(messages),
            'messages': messages, 'target': if_target(row['ground_truth'])}


def source_rows(source):
    if source is not None:
        manifest = json.loads((source / 'manifest.json').read_text())
        if manifest['dataset'] != DATASET or manifest['revision'] != REVISION:
            raise ValueError('Prepared source is not the pinned Dolci IF dataset')
        for split in ('train', 'val'):
            for row in read_events(source / (split + '.jsonl')):
                yield split, row
    else:
        from datasets import load_dataset

        for i, raw in enumerate(load_dataset(DATASET, revision=REVISION, split='train')):
            row = normalize(raw, i)
            bucket = int(hashlib.sha256(f"42:{row['key']}".encode()).hexdigest()[:8], 16)
            yield 'val' if bucket % 10 == 0 else 'train', row


def prepare(base, source=None, prompts=32, seed=20261029):
    from transformers import AutoTokenizer

    os.environ['MODULAR_RL_RUNTIME'] = str(base / 'runtime')
    available = registry()
    if {case[0] for case in IF_CASES} != set(available):
        raise ValueError('IF positive/negative controls must cover every registered checker')
    for name, kwargs, positive, negative in IF_CASES:
        target = {'instruction_id': [name], 'kwargs': [kwargs]}
        assert grade_if(positive, target)['accuracy'] == 1, name
        assert grade_if(negative, target)['accuracy'] == 0, name
    pools = {'train': [], 'val': []}
    seen, ids = set(), Counter()
    for split, original in source_rows(source):
        key = digest(original['messages'])
        if original['key'] != key or key in seen:
            raise ValueError('Duplicate or noncanonical Dolci prompt')
        seen.add(key)
        target = if_target(original['target'])
        # Exercise every actual argument bundle; malformed data/dependencies fail fast.
        grade_if('This is a test response. Another test follows.', target)
        ids.update(target['instruction_id'])
        metadata = {'task': 'if', 'environment': 'if', 'id': original['id'], 'key': key,
                    'split_group': key, 'messages': original['messages'], 'target': target,
                    'subject': f"constraints-{len(target['instruction_id'])}", 'level': 'all'}
        pools[split].append({'user_prompt': original['messages'], 'metadata': metadata})
    if {k: len(v) for k, v in pools.items()} != {'train': 11875, 'val': 1304}:
        raise ValueError('Pinned Dolci source counts changed')
    directory, selection = base / 'data/if', base / 'selection/if'
    if directory.exists() or selection.exists():
        raise ValueError('Use fresh IF data and selection directories')
    tokenizers = [AutoTokenizer.from_pretrained(base.parents[1] / 'models' / model, local_files_only=True)
                  for model in ('Qwen3-0.6B-Base', 'Qwen3-1.7B-Base')]
    eligible, excluded = [], []
    for row in pools['val']:
        lengths = [len(t(t.apply_chat_template(row['metadata']['messages'], tokenize=False,
                    add_generation_prompt=True, enable_thinking=True), add_special_tokens=False)['input_ids'])
                   for t in tokenizers]
        (eligible if max(lengths) <= 2048 else excluded).append(row)
    panel = stratified_sample(eligible, prompts, seed)
    directory.mkdir(parents=True)
    selection.mkdir(parents=True)
    def save(path, rows):
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    data_sha = {s: save(directory / (s + '.jsonl'), rows) for s, rows in pools.items()}
    selection_sha = {'train': save(selection / 'train.jsonl', pools['train']),
                     'val': save(selection / 'val.jsonl', panel)}
    provenance = {'dataset': DATASET, 'revision': REVISION, 'split_seed': 42,
                  'counts': {s: len(v) for s, v in pools.items()}, 'sha256': data_sha,
                  'instruction_ids': dict(ids), 'registered_checkers': len(available),
                  'known_answer_controls_passed': 2 * len(IF_CASES), 'all_targets_exercised': len(seen),
                  'prepared_source': str(source) if source else None,
                  'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in source.glob('*.json*')} if source else {},
                  'max_input_tokens': 2048, 'eligible_val': len(eligible),
                  'oversized_val_keys': [r['metadata']['key'] for r in excluded]}
    write_json(directory / 'manifest.json', provenance)
    ready_path = base / 'data/ready.json'
    ready = json.loads(ready_path.read_text()) if ready_path.exists() else {'version': 2}
    ready['if'] = provenance
    write_json(ready_path, ready)
    write_json(selection / 'selection.json', {
        'task': 'if', 'seed': seed, 'validation_prompts': prompts, 'sha256': selection_sha,
        'keys': {'train': [r['metadata']['key'] for r in pools['train']],
                 'val': [r['metadata']['key'] for r in panel]},
        'strata': {'val': dict(Counter(r['metadata']['subject'] for r in panel))},
        'answer_forcing': False, 'sampling': 'Hamilton allocation by number of constraints',
        'dataset_manifest': provenance})
    print(json.dumps(provenance), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--prompts', type=int, default=32)
    args = parser.parse_args()
    prepare(args.base.resolve(), args.source.resolve() if args.source else None, args.prompts)
