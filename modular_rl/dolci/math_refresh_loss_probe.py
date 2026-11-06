"""Compare raw Base and pilot SFT on identical held-out assistant targets."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from .config import atomic_json


def probe(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    info = manifest['splits']['val']; width = manifest['seq_len'] + 1
    ids = np.memmap(root / 'sources/val.bin', mode='r', dtype='<u4').reshape(info['blocks'], width)
    masks = np.memmap(root / 'sources/val.mask', mode='r', dtype='u1').reshape(ids.shape)
    selected = np.random.default_rng(20261101).permutation(info['blocks'])[:32]
    results = {}
    for label, path in (('base', root / 'model'), ('pilot_sft', root / 'pilot/sft/math/model')):
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                    local_files_only=True, attn_implementation='sdpa').to('cuda:0').eval()
        total, count = 0., 0
        tick = time.monotonic()
        with torch.inference_mode():
            for block in selected:
                sequence = torch.tensor(np.asarray(ids[block], dtype=np.int64), device='cuda:0')
                keep = torch.tensor(np.asarray(masks[block, 1:], dtype=bool), device='cuda:0')
                hidden = model.model(input_ids=sequence[None, :-1], use_cache=False).last_hidden_state[0, keep]
                targets = sequence[1:][keep]
                for start in range(0, len(targets), 256):
                    logits = F.linear(hidden[start:start + 256], model.lm_head.weight).float()
                    total += F.cross_entropy(logits, targets[start:start + 256], reduction='sum').item()
                count += len(targets)
        if not count or not math.isfinite(total): raise ValueError('Invalid held-out loss probe')
        results[label] = {'assistant_nll': total / count, 'assistant_tokens': count,
                          'blocks': len(selected), 'seconds': time.monotonic() - tick}
        del model
        torch.cuda.empty_cache()
    if results['base']['assistant_tokens'] != results['pilot_sft']['assistant_tokens']:
        raise ValueError('Models were evaluated on different targets')
    report = {'results': results, 'nll_change_sft_minus_base': results['pilot_sft']['assistant_nll'] - results['base']['assistant_nll'],
              'blocks': [int(v) for v in selected], 'validation_hashes': info['sha256'],
              'scope': 'Paired held-out short-math teacher-forced loss; not an accuracy improvement claim.'}
    atomic_json(root / 'pilot-heldout-loss.json', report)
    print('HELDOUT_MATH_LOSS', json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    probe(parser.parse_args().root)
