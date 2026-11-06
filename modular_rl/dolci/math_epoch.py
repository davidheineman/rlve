"""Whole-conversation packing and an exactly one-pass SFT block loader."""

import bisect
import hashlib
import json
import math
import sqlite3
from collections import Counter, deque
from pathlib import Path

import numpy as np

from .config import SEED, atomic_json, fingerprint

SEQ_LEN = 2048
GLOBAL_SEQUENCES = 128
PAD = 151643


def assignments(documents, capacity=SEQ_LEN):
    """Best-fit decreasing bin packing; IDs are never duplicated or split."""
    buckets = [deque() for _ in range(capacity + 1)]
    available, bins = [], []
    for identity, length in sorted(documents, key=lambda v: (-v[1], v[0])):
        if not 1 < length <= capacity: raise ValueError('Conversation does not fit a complete context')
        i = bisect.bisect_left(available, length)
        if i == len(available):
            block, remaining = len(bins), capacity
            bins.append([])
        else:
            remaining = available[i]
            block = buckets[remaining].popleft()
            if not buckets[remaining]: available.pop(i)
        bins[block].append(identity)
        remaining -= length
        if remaining:
            if not buckets[remaining]: bisect.insort(available, remaining)
            buckets[remaining].append(block)
    return bins


def pack(root):
    root = Path(root)
    if (root / 'manifest.json').exists(): return
    prepared = json.loads((root / 'prepared.json').read_text())
    if not prepared['passed']: raise ValueError('Prepare and audit the question pool first')
    db = sqlite3.connect(root / 'corpus.sqlite')
    db.execute('PRAGMA cache_size=-524288')
    val = set(json.loads((root / 'sft-validation-keys.json').read_text()))
    # Reserve math examples here. General validation uses the existing frozen
    # continuation-eval suite; taking another 128 general docs out of this small
    # one-pass pool would distort the requested 80/20 training-token mixture.
    documents = {'train': [], 'val': []}
    for identity, key, length in db.execute('SELECT rowid,key,tokens FROM accepted'):
        documents['val' if key in val else 'train'].append((identity, length))
    output = root / 'sources'; output.mkdir(exist_ok=True)
    reports = {}
    for split, docs in documents.items():
        bins = assignments(docs)
        count = len(bins)
        padded = math.ceil(count / GLOBAL_SEQUENCES) * GLOBAL_SEQUENCES if split == 'train' else count
        targets, counts, lengths, response_lengths, source_counts = [], Counter(), [], [], Counter()
        all_keys, emitted = set(), 0
        with (output / (split + '.bin')).open('wb') as tokens, (output / (split + '.mask')).open('wb') as masks, \
                (root / (split + '.jsonl')).open('w') as examples, (root / (split + '-blocks.jsonl')).open('w') as mappings:
            for i in range(padded):
                block_ids = np.full(SEQ_LEN + 1, PAD, dtype='<u4')
                block_mask = np.zeros(SEQ_LEN + 1, dtype='u1')
                offset, keys = 0, []
                for identity in bins[i] if i < count else []:
                    key, payload, ids, mask = db.execute('SELECT key,payload,ids,mask FROM accepted WHERE rowid=?', (identity,)).fetchone()
                    if key in all_keys: raise ValueError('A question would be trained twice')
                    all_keys.add(key); value = json.loads(payload)
                    ids = np.frombuffer(ids, dtype='<u4'); mask = np.frombuffer(mask, dtype='u1')
                    if mask[0] or len(ids) != value['tokens'] or len(ids) != len(mask):
                        raise ValueError('Incorrect encoded conversation or assistant mask')
                    if offset + len(ids) > SEQ_LEN: raise ValueError('Packing split a conversation')
                    block_ids[offset:offset + len(ids)] = ids
                    block_mask[offset:offset + len(ids)] = mask
                    offset += len(ids); keys.append(key); emitted += 1
                    source_counts[value['source']] += 1
                    counts[value['source']] += len(ids)
                    lengths.append(len(ids)); response_lengths.append(value['response_tokens'])
                    metadata = {k: v for k, v in value.items() if k not in ('messages', 'problem', 'response')}
                    examples.write(json.dumps({'messages': value['messages'], 'metadata': metadata}, ensure_ascii=False) + '\n')
                block_ids.tofile(tokens); block_mask.tofile(masks)
                targets.append(int(block_mask[1:].sum()))
                mappings.write(json.dumps({'block': i, 'questions': keys, 'real_tokens': offset,
                                           'padding_tokens': SEQ_LEN - offset}) + '\n')
        if emitted != len(docs): raise ValueError('Lost or repeated a conversation during packing')
        np.save(output / (split + '.counts.npy'), np.asarray(targets, dtype=np.uint32))
        hashes = {}
        for suffix in ('.bin', '.mask'):
            digest = hashlib.sha256()
            with (output / (split + suffix)).open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b''): digest.update(chunk)
            hashes[suffix] = digest.hexdigest()
        total = sum(counts.values()); math_tokens = total - counts['general']
        reports[split] = {'documents': emitted, 'real_blocks': count, 'blocks': padded,
                          'input_token_slots': padded * SEQ_LEN, 'nonpadding_input_tokens': total,
                          'supervised_tokens': sum(targets), 'source_documents': dict(source_counts),
                          'source_input_tokens': dict(counts), 'math_input_share': math_tokens / total,
                          'padding_fraction': 1 - total / (padded * SEQ_LEN),
                          'response_token_max': max(response_lengths), 'response_token_median': float(np.median(response_lengths)),
                          'conversation_token_max': max(lengths), 'sha256': hashes}
        print('PACKED', split, reports[split], flush=True)
    db.close()
    train = reports['train']
    if not .79 <= train['math_input_share'] <= .81: raise ValueError('The retained mix is not 80/20')
    manifest = {'repo': 'math-refresh-20261101', 'revision': prepared['sources'], 'seq_len': SEQ_LEN,
                'seed': SEED, 'objective': 'assistant', 'natural_definition': '80% short math / 20% short tool-free general',
                'recipe': {'mixes': {'math': {'math': train['math_input_share'], 'general': 1 - train['math_input_share']}}},
                'pool_prediction_tokens': train['input_token_slots'], 'splits': reports,
                'one_epoch': True, 'math_epochs': 1, 'require_unique_blocks': True,
                'whole_conversations': True, 'response_limit': 1536, 'no_truncation': True,
                'realized_epochs': 1.0, 'global_sequences': GLOBAL_SEQUENCES,
                'train_steps': train['blocks'] // GLOBAL_SEQUENCES,
                'training_tokens': train['input_token_slots'],
                'independent_math_references_required': True}
    for name in ('final_quality_policy', 'nemotron_solution_subset', 'manual_invalid_questions'):
        if name in prepared: manifest[name] = prepared[name]
    manifest['signature'] = fingerprint(manifest)
    atomic_json(root / 'manifest.json', manifest)


class OneEpochBlocks:
    def __init__(self, root, mix, seed=None):
        if mix != 'math': raise ValueError('This corpus is a math specialization')
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if not self.manifest['one_epoch'] or not self.manifest['whole_conversations']:
            raise ValueError('Expected whole-conversation single-pass data')
        self.seq_len = self.manifest['seq_len']
        info = self.manifest['splits']['train']
        n, real = info['blocks'], info['real_blocks']
        self.ids = np.memmap(self.root / 'sources/train.bin', dtype='<u4', mode='r').reshape(n, self.seq_len + 1)
        self.masks = np.memmap(self.root / 'sources/train.mask', dtype='u1', mode='r').reshape(self.ids.shape)
        self.target_counts = np.load(self.root / 'sources/train.counts.npy')
        if self.target_counts.shape != (n,) or self.target_counts[real:].any():
            raise ValueError('Incorrect real/padding block counts')
        rng = np.random.default_rng(self.manifest['seed'] if seed is None else seed)
        self.indices = np.concatenate([rng.permutation(real), np.arange(real, n)])
        self.labels = np.zeros(n, dtype=np.uint8)  # Existing trainer's dataset-length interface.

    def batch(self, start, size, device):
        import torch
        if start < 0 or start + size > len(self.indices): raise ValueError('One-epoch corpus exhausted')
        indices = self.indices[start:start + size]
        ids = np.asarray(self.ids[indices], dtype=np.int64)
        mask = np.asarray(self.masks[indices, 1:], dtype=bool)
        targets = ids[:, 1:].copy(); targets[~mask] = -100
        return torch.from_numpy(ids[:, :-1].copy()).to(device), torch.from_numpy(targets).to(device)

    def loss_tokens(self, start, size):
        count = int(self.target_counts[self.indices[start:start + size]].sum())
        if count <= 0: raise ValueError('A global optimizer batch has no assistant targets')
        return count
