"""Pinned raw Base and explicitly supplied domain-midtraining initializers."""

import argparse
import json
from pathlib import Path

from .v2.base_models import BASE_MODELS, sha256, verify_hf, verify_initialization


def load_registry(path, expected_sha256=None):
    path = Path(path)
    if expected_sha256 and sha256(path) != expected_sha256:
        raise ValueError('Initializer registry changed after preparation')
    value = json.loads(path.read_text())
    if 'base' not in value or set(value) - {'base', 'baseline', 'code', 'math', 'chat', 'web', 'if20', 'if50'}:
        raise ValueError('Expected raw Base and explicitly registered mixture initializers')
    return value


def inspect_hf(alias, hf, native, *, training_mix=None, expected_training_tokens=999817216):
    hf, native = Path(hf).resolve(), Path(native).resolve()
    config = json.loads((hf / 'config.json').read_text())
    required = {'model_type': 'qwen3', 'hidden_size': 1024, 'num_hidden_layers': 28,
                'num_attention_heads': 16, 'num_key_value_heads': 8, 'vocab_size': 151936,
                'eos_token_id': 151643, 'tie_word_embeddings': True}
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError(f'{alias}: not the expected 0.6B Base architecture')
    if alias == 'base':
        ancestor = verify_hf(hf, full_weights=True)
        verify_initialization(hf, native)
        kind = 'pretrained-base'
    else:
        training = json.loads((hf.parent / 'config.json').read_text())
        ancestor = training['model']
        if (ancestor['repo'] != 'Qwen/Qwen3-0.6B-Base'
                or ancestor['revision'] != BASE_MODELS['Qwen3-0.6B-Base']['revision']
                or training['mix'] != (training_mix or alias)
                or training['actual_tokens'] != expected_training_tokens):
            raise ValueError('Supplied midtraining ancestry or domain does not match')
        kind = ('dolci-sft-base' if training.get('dataset_repo') == 'allenai/Dolci-Instruct-SFT'
                else 'domain-midtrained-base')
    weight = hf / 'model.safetensors'
    return {'id': alias, 'kind': kind, 'hf': str(hf), 'native': str(native),
            **({'training_mix': training_mix} if training_mix is not None else {}),
            **({'expected_training_tokens': expected_training_tokens} if alias != 'base' and expected_training_tokens != 999817216 else {}),
            'architecture': 'Qwen3-0.6B-Base', 'ancestor': ancestor,
            'config_sha256': sha256(hf / 'config.json'), 'weights_sha256': sha256(weight),
            'weights_bytes': weight.stat().st_size, 'weights_mtime_ns': weight.stat().st_mtime_ns,
            'tokenizer_assets_sha256': {name: sha256(hf / name) if (hf / name).exists() else None
                for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                             'added_tokens.json', 'vocab.json', 'merges.txt', 'generation_config.json',
                             'chat_template.jinja', 'chat_template.json')}}


def verify(spec, *, require_native=True, full_weights=False):
    hf, native = Path(spec['hf']), Path(spec['native'])
    if sha256(hf / 'config.json') != spec['config_sha256']:
        raise ValueError('Initializer configuration changed')
    weight = hf / 'model.safetensors'
    if (weight.stat().st_size != spec['weights_bytes']
            or weight.stat().st_mtime_ns != spec['weights_mtime_ns']
            or (full_weights and sha256(weight) != spec['weights_sha256'])):
        raise ValueError('Initializer weights changed')
    if any((sha256(hf / name) if (hf / name).exists() else None) != expected
           for name, expected in spec['tokenizer_assets_sha256'].items()):
        raise ValueError('Initializer tokenizer changed')
    if require_native:
        if (native / 'latest_checkpointed_iteration.txt').read_text().strip() != 'release':
            raise ValueError('RL initializer must be a zero-RL-update release conversion')
        if spec['kind'] == 'pretrained-base':
            verify_initialization(hf, native)
        else:
            converted = json.loads((native / 'initializer_identity.json').read_text())
            if converted != {'initializer': spec, 'rl_optimizer_updates': 0}:
                raise ValueError('Converted weights belong to a different initializer')
    return spec


def validate_training(args, registry_path, alias, directory, expected_sha256):
    spec = verify(load_registry(registry_path, expected_sha256)[alias])
    directory = Path(directory).resolve()
    if (Path(args.hf_checkpoint).resolve() != Path(spec['hf'])
            or Path(args.ref_load).resolve() != Path(spec['native'])):
        raise ValueError('Parsed actor/reference paths differ from the requested initializer')
    previous = directory / 'initial-model-identity.json'
    load = Path(args.load).resolve()
    if load != Path(spec['native']):
        if load != directory / 'ckpt':
            raise ValueError('Only the requested initializer or this run may load')
        if (load / 'latest_checkpointed_iteration.txt').exists():
            old = json.loads(previous.read_text())
            if old['initializer'] != spec:
                raise ValueError('Resume checkpoint belongs to another initializer')
    directory.mkdir(parents=True, exist_ok=True)
    previous.write_text(json.dumps({'initializer': spec, 'actor_load_for_this_attempt': args.load,
                                    'reference_load': args.ref_load}, indent=2) + '\n')
    return spec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registry', type=Path)
    parser.add_argument('alias', choices=['base', 'baseline', 'code', 'math', 'chat', 'web', 'if20', 'if50'])
    parser.add_argument('--expected-sha256')
    parser.add_argument('--paths', action='store_true')
    parser.add_argument('--converted', type=Path)
    args = parser.parse_args()
    spec = load_registry(args.registry, args.expected_sha256)[args.alias]
    if args.converted:
        verify(spec, require_native=False)
        assert (args.converted / 'latest_checkpointed_iteration.txt').read_text().strip() == 'release'
        (args.converted / 'initializer_identity.json').write_text(json.dumps(
            {'initializer': spec, 'rl_optimizer_updates': 0}, indent=2) + '\n')
    else:
        verify(spec)
        if args.paths:
            print(spec['hf'])
            print(spec['native'])
        else:
            print(json.dumps(spec))
