"""Validate prepared token/mask files and deterministic global SFT sampling."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import ACTUAL_TOKENS, atomic_json, family
from .data import MixedBlocks


def validate(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['repo'] == 'allenai/Dolci-Instruct-SFT'
    assert sum(s['original_rows'] for s in manifest['sources'].values()) == 2152112
    source_audit = []
    for name, info in manifest['sources'].items():
        path = root / 'sources' / (name + '.bin')
        mask = path.with_suffix('.mask')
        assert path.stat().st_size == info['packed_tokens'] * 4
        assert mask.stat().st_size == info['packed_tokens']
        tokens = np.memmap(path, dtype='<u4', mode='r')
        masks = np.memmap(mask, dtype='u1', mode='r')
        assert int(tokens.max()) < 151936 and int(masks.max()) == 1
        assert int(masks.sum()) > 0
        counts = masks.reshape(-1, manifest['seq_len'] + 1)[:, 1:].sum(axis=1, dtype=np.uint32)
        count_path = path.with_suffix('.counts.npy')
        temporary = count_path.with_suffix('.npy.partial')
        with temporary.open('wb') as output:
            np.save(output, counts)
        temporary.replace(count_path)
        source_audit.append({'source_id': name, 'tokens': info['tokens'],
                             'packed_tokens': info['packed_tokens'], 'mask_targets': int(masks.sum())})
    mix_audit = []
    for mix, weights in manifest['recipe']['mixes'].items():
        assert abs(sum(weights.values()) - 1) < 1e-10
        if mix != 'baseline':
            target = 'web' if mix == 'web-knowledge' else mix
            definition = manifest.get('natural_definitions', {}).get(mix, manifest['natural_definition'])
            mass = sum(w for n, w in weights.items() if family(manifest['sources'][n]['domain'], definition) == target)
            assert abs(mass - .8) < 1e-10
        data = MixedBlocks(root, mix)
        assert len(data.labels) * data.seq_len == ACTUAL_TOKENS
        for start in (0, 128, len(data.labels) - 128):
            x, y = data.batch(start, 128, 'cpu')
            assert x.shape == y.shape == (128, 2048)
            assert data.loss_tokens(start, 128) == int(y.ne(-100).sum()) > 0
        counts = np.empty(len(data.labels), dtype=np.uint32)
        for i in data.target_counts:
            selected = data.labels == i
            counts[selected] = data.target_counts[i][data.indices[selected]]
        global_counts = counts.reshape(-1, 128).sum(axis=1)
        assert int(global_counts.min()) > 0
        mix_audit.append({'mix': mix, 'prediction_tokens': len(data.labels) * data.seq_len,
                          'first_batch_loss_tokens': data.loss_tokens(0, 128),
                          'realized_supervised_tokens': data.loss_tokens(0, len(data.labels)),
                          'minimum_global_batch_supervised_tokens': int(global_counts.min()),
                          'maximum_global_batch_supervised_tokens': int(global_counts.max()),
                          'realized_source_supervised_tokens': {
                              name: int(data.target_counts[i][data.indices[data.labels == i]].sum())
                              for i, name in enumerate(data.names) if i in data.target_counts}})
    plan = json.loads((root / 'sft-plan.json').read_text())
    for name, expected in plan['source_sha256'].items():
        assert hashlib.sha256((Path(plan['source']) / name).read_bytes()).hexdigest() == expected
    result = {'passed': True, 'source_audit': source_audit, 'mix_audit': mix_audit,
              'manifest_signature': manifest['signature'],
              'objective': manifest['objective'], 'natural_definition': manifest['natural_definition']}
    atomic_json(root / 'validation.json', result)
    print('VALIDATED_DOLCI', json.dumps(mix_audit), flush=True)
    return result
