"""Convert a supplied initializer and compare native tensors with the HF source."""

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from .initializers import load_registry, verify


def verify_converted_weights(spec, output):
    """Compare every native Qwen parameter, including packed QKV/MLP matrices."""
    import re
    import torch
    from safetensors import safe_open
    from torch.distributed.checkpoint import FileSystemReader, load_state_dict
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader = FileSystemReader(Path(output) / 'release')
    metadata = reader.read_metadata().state_dict_metadata
    tensors = {key: torch.empty(value.size, dtype=value.properties.dtype)
               for key, value in metadata.items() if isinstance(value, TensorStorageMetadata)}
    load_state_dict(tensors, storage_reader=reader, no_dist=True)
    args = torch.load(Path(output) / 'release/common.pt', map_location='cpu', weights_only=False)['args']
    path = Path(__file__).resolve().parents[1] / 'slime/backends/megatron_utils/megatron_to_hf/qwen2.py'
    module_spec = importlib.util.spec_from_file_location('initializer_qwen_converter', path)
    converter = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(converter)
    names, elements = set(), 0
    with safe_open(Path(spec['hf']) / 'model.safetensors', framework='pt', device='cpu') as hf:
        expected = set(hf.keys())
        for name, tensor in tensors.items():
            if '.layers.' in name and not re.search(r'\.layers\.\d+\.', name):
                if tensor.shape[0] != args.num_layers:
                    raise ValueError('Unexpected stacked layer count: ' + name)
                layers = [(name.replace('.layers.', f'.layers.{i}.'), tensor[i]) for i in range(args.num_layers)]
            else:
                layers = [(name, tensor)]
            for native_name, value in layers:
                for hf_name, converted in converter.convert_qwen2_to_hf(args, 'module.module.' + native_name, value):
                    if hf_name == 'lm_head.weight' and hf_name not in expected:
                        original = hf.get_tensor('model.embed_tokens.weight')
                    else:
                        original = hf.get_tensor(hf_name)
                    if not torch.isfinite(original).all() or not torch.equal(converted, original.to(converted.dtype)):
                        raise ValueError('Native conversion differs from supplied HF tensor: ' + hf_name)
                    if hf_name in expected:
                        if hf_name in names:
                            raise ValueError('Duplicate converted tensor: ' + hf_name)
                        names.add(hf_name)
                        elements += converted.numel()
        if names != expected or elements != 596049920:
            raise ValueError(f'Incomplete native weight comparison: {len(names)}/{len(expected)}, {elements} elements')
    return {'all_hf_tensors_finite': True, 'hf_tensors': len(names), 'parameter_elements_compared': elements,
            'all_native_parameters_exactly_match_hf_after_dtype_cast': True, 'rl_optimizer_updates': 0}


def convert(registry, alias, output, architecture_args):
    spec = verify(load_registry(registry, os.environ['MODULAR_RL_INITIALIZERS_SHA256'])[alias],
                  require_native=False, full_weights=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = {**os.environ, 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port), 'WORLD_SIZE': '1', 'RANK': '0'}
    subprocess.run([sys.executable, 'tools/convert_hf_to_torch_dist.py', *architecture_args,
                    '--hf-checkpoint', spec['hf'], '--save', str(output)], env=env, check=True)
    audit = verify_converted_weights(spec, output)
    identity = {'initializer': spec, 'rl_optimizer_updates': 0}
    (output / 'initializer_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    (output / 'conversion-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('VERIFIED_INITIALIZER_CONVERSION', alias, spec['weights_sha256'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registry', type=Path)
    parser.add_argument('alias')
    parser.add_argument('output', type=Path)
    parser.add_argument('--architecture', nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    convert(args.registry, args.alias, args.output, args.architecture)
