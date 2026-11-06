"""Retire finished SFT optimizer states only after exact final-export verification."""

import hashlib
import json
from pathlib import Path

from .config import ACTUAL_TOKENS, STEPS, atomic_json

EXPECTED_PARAMETERS = 596049920


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(16 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def retire(directory, *, expected_tokens=ACTUAL_TOKENS):
    import torch
    from safetensors import safe_open

    directory = Path(directory)
    state_path = directory / 'train_state.pt'
    if not state_path.exists():
        return None
    result = json.loads((directory / 'results.json').read_text())
    if result['train_tokens'] != expected_tokens or result['config']['objective'] != 'assistant':
        raise ValueError('Only completed full-budget SFT states may retire')
    state = torch.load(state_path, map_location='cpu', weights_only=False, mmap=True)
    steps = expected_tokens // result['config']['tokens_per_step']
    if steps * result['config']['tokens_per_step'] != expected_tokens or state['next_step'] != steps:
        raise ValueError('SFT checkpoint has not completed its token budget')
    weight = directory / 'model/model.safetensors'
    elements, count = 0, 0
    with safe_open(weight, framework='pt', device='cpu') as hf:
        for name in hf.keys():
            tensor = hf.get_tensor(name)
            if not torch.isfinite(tensor).all() or not torch.equal(tensor, state['model'][name]):
                raise ValueError('Export differs from completed SFT checkpoint: ' + name)
            elements += tensor.numel()
            count += 1
    if elements != EXPECTED_PARAMETERS:
        raise ValueError('Incomplete SFT weight verification')
    record = {'retired_file': str(state_path), 'retired_bytes': state_path.stat().st_size,
              'reason': 'scratch capacity; training fully completed and every exported parameter exactly verified',
              'preserved_model': str(weight), 'preserved_weights_sha256': sha256(weight),
              'parameter_elements_compared': elements, 'tensor_count': count,
              'training_steps': steps, 'input_tokens': expected_tokens,
              'sft_optimizer_resume_available': False, 'all_final_model_weights_preserved': True}
    atomic_json(directory / 'optimizer-state-retired.json', record)
    state_path.unlink()
    print('RETIRED_COMPLETED_SFT_OPTIMIZER', directory.name, record['retired_bytes'], flush=True)
    return record
