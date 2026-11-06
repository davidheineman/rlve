"""Preserve diagnostic weights exactly while reclaiming derived scratch caches."""

import importlib.util
import gzip
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from .config import atomic_json


def archive_audited_jsonl(root, *, force=False):
    """Losslessly archive finished text evidence; training uses packed arrays."""
    root = Path(root)
    assert json.loads((root / 'curation-audit.json').read_text())['passed']
    full_screen = root / 'independent-screen-packed.json'
    available = root / 'available-screen-audit.json'
    assert ((full_screen.exists() and json.loads(full_screen.read_text())['passed']) or
            (available.exists() and json.loads(available.read_text())['passed']))
    marker = root / 'curation-text-compression.json'
    if marker.exists():
        return json.loads(marker.read_text())
    free = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
    if not force and free >= 40 * 1024**3:
        return None
    paths = [root / (name + '.jsonl') for name in ('math-labeled', 'math-unlabeled', 'general')]
    screening_shards = json.loads((root / 'judge-input/manifest.json').read_text())['shards']
    paths += [root / folder / f'{i:02d}.jsonl' for folder in ('judge-input', 'judge-results') for i in range(screening_shards)]
    paths += [root / 'independent-screen-rejections.jsonl']
    receipts = root / 'compression-audits'
    receipts.mkdir(exist_ok=True)

    def compress(path):
        receipt = receipts / (hashlib.sha256(str(path.relative_to(root)).encode()).hexdigest() + '.json')
        output = path.with_suffix(path.suffix + '.gz')
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            assert output.exists()
            if path.exists():
                assert hashlib.sha256(path.read_bytes()).hexdigest() == saved['original_sha256']
                path.unlink()
            return saved
        if not path.exists():
            return None
        temporary = output.with_suffix('.gz.partial')
        digest, size = hashlib.sha256(), 0
        with path.open('rb') as source, gzip.open(temporary, 'wb', compresslevel=1) as target:
            for data in iter(lambda: source.read(8 * 1024**2), b''):
                digest.update(data)
                size += len(data)
                target.write(data)
        actual, decoded = hashlib.sha256(), 0
        with gzip.open(temporary, 'rb') as source:
            for data in iter(lambda: source.read(8 * 1024**2), b''):
                actual.update(data)
                decoded += len(data)
        assert decoded == size == path.stat().st_size and actual.digest() == digest.digest()
        temporary.replace(output)
        value = {'original': str(path), 'archive': str(output), 'original_bytes': size,
                 'archive_bytes': output.stat().st_size, 'original_sha256': digest.hexdigest(),
                 'roundtrip_exactly_verified': True}
        # A durable per-file receipt precedes unlink, so interrupted archival
        # can safely resume even after some original files have been retired.
        atomic_json(receipt, value)
        path.unlink()
        print('ARCHIVED_AUDITED_TEXT', path.name, size, value['archive_bytes'], flush=True)
        return value

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = [v for v in pool.map(compress, paths) if v is not None]
    result = {'all_text_losslessly_preserved': True, 'files': results,
              'bytes_reclaimed': sum(v['original_bytes'] - v['archive_bytes'] for v in results),
              'packed_training_arrays_unchanged': True}
    atomic_json(marker, result)
    return result


def retire_merged_cache(root):
    root = Path(root)
    proof = json.loads((root / 'curation-audit.json').read_text())
    assert proof['passed'] and proof['all_assistant_targets_preserved']
    path = root / 'merged.sqlite'
    if path.exists():
        atomic_json(root / 'merged-cache-retired.json', {
            'path': str(path), 'bytes': path.stat().st_size,
            'reason': 'Derived merge cache; complete conversations/provenance, packed IDs/masks, hashes, and dedup audit are preserved.',
            'preserved': ['math-labeled.jsonl', 'math-unlabeled.jsonl', 'general.jsonl', 'sources/', 'curation-audit.json', 'cross-source-dedup.jsonl']})
        path.unlink()
    if (root / 'independent-screen-packed.json').exists() or (root / 'available-screen-audit.json').exists():
        archive_audited_jsonl(root)


def compact_pilot(root):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig

    root = Path(root).resolve()
    proof = json.loads((root / 'preflight/validation.json').read_text())
    assert proof['passed']
    plan = json.loads((root / 'preflight/rl/math/launch-plan.json').read_text())
    job, = plan['runs']
    policy = Path(job['output']) / 'policy'
    assert policy.is_relative_to(root / 'preflight')
    marker = policy / 'model-only-preservation.json'
    if marker.exists():
        return
    complete = json.loads((policy / 'completed.json').read_text())
    assert complete['completed'] and complete['train_rollouts'] <= 128
    checkpoint = policy / 'ckpt'
    state = json.loads((checkpoint / 'rlve_checkpoint_state.json').read_text())
    assert state['completed']
    directory = checkpoint / f"iter_{state['rollout_id']:07d}"
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('diagnostic_converter', repo / 'tools/convert_torch_dist_to_hf.py')
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    tensors = {}
    converter.dist_cp.state_dict_loader._load_state_dict(tensors,
        storage_reader=converter.WrappedStorageReader(str(directory)),
        planner=converter.EmptyStateDictLoadPlanner(), no_dist=True)
    source = root / 'preflight/sft/math/model'
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    args = SimpleNamespace(num_layers=config.num_hidden_layers, num_query_groups=config.num_key_value_heads,
                           num_attention_heads=config.num_attention_heads, hidden_size=config.hidden_size,
                           kv_channels=getattr(config, 'head_dim', None))
    export = policy / 'model-only-export'
    export.mkdir(exist_ok=True)
    converter.save_tensors(args, 'qwen3', tensors, str(export), 5 * 1024**3, 151936)
    converter.copy_assets(str(source), str(export))
    index = json.loads((export / 'model.safetensors.index.json').read_text())['weight_map']
    mapper = converter.model_converter('qwen3')
    seen, count = set(), 0
    for name, tensor in converter.get_named_params(args, tensors):
        for key, value in mapper(args, name, tensor):
            assert key not in seen
            with safe_open(export / index[key], framework='pt', device='cpu') as weights:
                actual = weights.get_tensor(key)
                assert torch.equal(actual, value) and torch.isfinite(actual).all()
            seen.add(key)
            count += actual.numel()
    assert seen == set(index) and count == 596049920
    payloads = [{'path': str(p), 'bytes': p.stat().st_size} for p in directory.glob('__*.distcp')]
    atomic_json(marker, {'all_model_parameters_exactly_preserved': True,
                         'parameter_elements_compared': count, 'tensors_compared': len(seen),
                         'export': str(export), 'retired_optimizer_payloads': payloads,
                         'native_optimizer_resume_available': False,
                         'reason': 'Completed diagnostic; primary RL uses the final fresh 1B SFT model.'})
    for row in payloads:
        Path(row['path']).unlink()
