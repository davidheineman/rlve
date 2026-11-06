"""Publish generation defaults matching the turn endings supervised during SFT."""

import json
from pathlib import Path

from .config import ACTUAL_TOKENS, atomic_json

ENDINGS = [151643, 151645]  # endoftext and im_end in the pinned Qwen vocabulary.


def finalize_generation(directory, *, expected_tokens=ACTUAL_TOKENS):
    directory = Path(directory)
    result = json.loads((directory / 'results.json').read_text())
    if result['train_tokens'] != expected_tokens or result['config']['objective'] != 'assistant':
        raise ValueError('Finalize only completed full-budget assistant-loss SFT exports')
    model = directory / 'model'
    config = json.loads((model / 'config.json').read_text())
    tokenizer = json.loads((model / 'tokenizer_config.json').read_text())
    expected = {151643: '<|endoftext|>', 151645: '<|im_end|>'}
    for token, text in expected.items():
        if tokenizer['added_tokens_decoder'][str(token)]['content'] != text:
            raise ValueError('Unexpected SFT turn-ending vocabulary')
    if config['eos_token_id'] != 151643:
        raise ValueError('The model architecture must remain the pinned Base configuration')
    path = model / 'generation_config.json'
    generation = json.loads(path.read_text())
    original = generation.get('eos_token_id')
    if original not in (151643, [151643], ENDINGS):
        raise ValueError('Unexpected inherited generation stop IDs')
    marker = directory / 'generation-stop-tokens.json'
    if original == ENDINGS and marker.exists():
        return json.loads(marker.read_text())
    generation['eos_token_id'] = ENDINGS
    atomic_json(path, generation)
    proof = {'before_eos_token_id': original, 'eos_token_id': ENDINGS,
             'tokens': expected, 'model_config_eos_unchanged': 151643,
             'reason': 'Dolci assistant targets end with im_end; SGLang unions generation/model EOS IDs',
             'weights_and_tokenizer_unchanged': True}
    atomic_json(marker, proof)
    return proof
