"""Globally shuffled packed sequences; small strata repeat with fresh permutations."""

import json
from pathlib import Path

import numpy as np
import torch

from modular_rl.dolci.config import fingerprint


class MixedBlocks:
    def __init__(self, root, mix, seed=None):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.seq_len = self.manifest['seq_len']
        self.names = sorted(self.manifest['quotas'][mix])
        seed = self.manifest['seed'] if seed is None else seed
        rng = np.random.default_rng(seed)
        quotas = self.manifest['quotas'][mix]
        self.labels = np.concatenate([np.full(quotas[n], i, dtype=np.uint8)
                                      for i, n in enumerate(self.names)])
        rng.shuffle(self.labels)
        self.indices = np.empty(len(self.labels), dtype=np.int64)
        self.streams, self.masks, self.target_counts = {}, {}, {}
        for i, name in enumerate(self.names):
            if not quotas[name]:
                continue
            info = self.manifest['sources'][name]
            path = self.root / 'sources' / f'{name}.bin'
            mask_path = path.with_suffix('.mask')
            if (path.stat().st_size != info['packed_tokens'] * 4
                    or mask_path.stat().st_size != info['packed_tokens']):
                raise ValueError(f'Truncated token/mask source: {name}')
            stream = np.memmap(path, dtype='<u4', mode='r').reshape(-1, self.seq_len + 1)
            masks = np.memmap(mask_path, dtype='u1', mode='r').reshape(stream.shape)
            if not len(stream):
                raise ValueError(f'Empty packed source {name}')
            if self.manifest.get('require_unique_blocks') and quotas[name] > len(stream):
                raise ValueError(f'Fresh corpus would repeat blocks in {name}')
            source_rng = np.random.default_rng(seed + int(fingerprint(name)[:8], 16))
            chunks, remaining = [], quotas[name]
            while remaining:
                take = min(len(stream), remaining)
                chunks.append(source_rng.permutation(len(stream))[:take])
                remaining -= take
            self.indices[self.labels == i] = np.concatenate(chunks)
            self.streams[i], self.masks[i] = stream, masks
            counts_path = path.with_suffix('.counts.npy')
            counts = (np.load(counts_path) if counts_path.exists()
                      else np.asarray(masks[:, 1:].sum(axis=1, dtype=np.uint32)))
            if counts.shape != (len(stream),):
                raise ValueError('Wrong per-block target counts: ' + name)
            self.target_counts[i] = counts

    def batch(self, start, size, device):
        if start < 0 or start + size > len(self.labels):
            raise RuntimeError('Training pool exhausted')
        labels, indices = self.labels[start:start + size], self.indices[start:start + size]
        result = np.empty((size, self.seq_len + 1), dtype=np.int64)
        mask = np.empty(result.shape, dtype=bool)
        for label in np.unique(labels):
            selected = labels == label
            result[selected] = self.streams[int(label)][indices[selected]]
            mask[selected] = self.masks[int(label)][indices[selected]]
        targets = result[:, 1:].copy()
        if self.manifest['objective'] == 'assistant':
            targets[~mask[:, 1:]] = -100
        packed = torch.from_numpy(result[:, :-1].copy()).to(device)
        return packed, torch.from_numpy(targets).to(device)

    def loss_tokens(self, start, size):
        if self.manifest['objective'] == 'all':
            return size * self.seq_len
        labels, indices = self.labels[start:start + size], self.indices[start:start + size]
        total = 0
        for label in np.unique(labels):
            total += int(self.target_counts[int(label)][indices[labels == label]].sum())
        if total <= 0:
            raise ValueError('Global SFT batch contains no assistant targets')
        return total
