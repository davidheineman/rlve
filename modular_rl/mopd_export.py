"""Export completed modular RL teachers without deleting resumable checkpoints."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

BANKS = {
    'base': dict(zip(('math', 'code', 'if', 'chat'), ('base-math', 'base-code', 'base-if', 'base-chat'))),
    'mid': dict(zip(('math', 'code', 'if', 'chat'), ('mid-math', 'mid-code', 'mid-chat-if', 'mid-web-chat'))),
    'web': dict(zip(('math', 'code', 'if', 'chat'), ('mid-web-math', 'mid-web-code', 'mid-web-if', 'mid-web-chat'))),
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def audit(directory, source):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer

    index = json.loads((directory / 'model.safetensors.index.json').read_text())['weight_map']
    count = 0
    for filename in sorted(set(index.values())):
        with safe_open(directory / filename, framework='pt', device='cpu') as weights:
            for name in weights.keys():
                tensor = weights.get_tensor(name)
                if index.get(name) != filename or not torch.isfinite(tensor).all():
                    raise ValueError('Invalid/nonfinite exported teacher tensor: ' + name)
                count += tensor.numel()
    if count != 596049920:
        raise ValueError(f'Unexpected teacher parameter count: {count}')
    if AutoConfig.from_pretrained(directory).to_dict()['vocab_size'] != 151936:
        raise ValueError('Teacher vocabulary changed')
    a, b = (AutoTokenizer.from_pretrained(p, local_files_only=True) for p in (directory, source))
    if a.get_vocab() != b.get_vocab() or a.chat_template != b.chat_template:
        raise ValueError('Teacher tokenizer changed')
    files = {name: {'bytes': (directory / name).stat().st_size, 'sha256': digest(directory / name)}
             for name in sorted(set(index.values()))}
    return {'tensor_count': len(index), 'parameter_count': count, 'weight_files': files,
            'weight_bytes': sum(x['bytes'] for x in files.values()), 'all_tensors_finite': True}


def build_banks(root):
    for bank, mapping in BANKS.items():
        directory = root / 'teachers/banks' / bank
        directory.mkdir(parents=True, exist_ok=True)
        bank_path = directory / 'bank.json'
        if bank_path.exists():
            if json.loads(bank_path.read_text()) != mapping:
                raise ValueError('Published teacher mapping changed: ' + bank)
            continue
        if not all((root / 'teachers/exports' / name / 'weights_only_complete.json').exists()
                   for name in mapping.values()):
            continue
        for i, (task, name) in enumerate(mapping.items()):
            slot = directory / f'{i:02d}-{task}'
            slot.mkdir(exist_ok=True)
            export_slot = root / 'teachers/exports' / name
            for asset in ('hf-final', 'weights_only_complete.json'):
                link = slot / asset
                if not link.exists():
                    link.symlink_to(export_slot / asset, target_is_directory=asset == 'hf-final')
        temporary = directory / f'.bank-{os.getpid()}.json'
        temporary.write_text(json.dumps(mapping, indent=2) + '\n')
        temporary.replace(bank_path)
        print('Ready teacher bank:', bank, mapping, flush=True)


def export(root, master, repo):
    report = json.loads((master / 'final-report.json').read_text())
    rows = {r['name']: r for r in report['runs']}
    registry = json.loads((master / 'initializers.json').read_text())
    required = set().union(*(set(bank.values()) for bank in BANKS.values()))
    jobs = root / 'rl/jobs.json'
    if jobs.exists():
        for job in json.loads(jobs.read_text()):
            directory = Path(job['output']) / 'policy'
            completed = directory / 'completed.json'
            if completed.exists() and json.loads(completed.read_text()).get('completed'):
                rows[job['name']] = dict(name=job['name'], initializer='web',
                    native_checkpoint=str(directory / 'ckpt'), run_directory=str(directory),
                    train_rollouts=json.loads(completed.read_text())['train_rollouts'])
    for name in sorted(required & rows.keys()):
        row = rows[name]
        slot = root / 'teachers/exports' / name
        marker = slot / 'weights_only_complete.json'
        if marker.exists():
            identity = json.loads(marker.read_text())
            # The converter copies tokenizer assets from the initializer. Its
            # untrained-Base identity must not label these trained RL weights.
            (slot / 'hf-final/pretrained_model_identity.json').unlink(missing_ok=True)
            (slot / 'hf-final/rl_teacher_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
            continue
        checkpoint = Path(row['native_checkpoint'])
        step = int((checkpoint / 'latest_checkpointed_iteration.txt').read_text())
        if step != 999 or row['train_rollouts'] != 8000:
            raise ValueError('Require a completed full-budget teacher: ' + name)
        source = Path(registry[row['initializer']]['hf'])
        slot.mkdir(parents=True, exist_ok=True)
        target = slot / 'hf-final'
        if not target.exists():
            command = ['python3', str(repo / 'tools/convert_torch_dist_to_hf.py'),
                       '--input-dir', str(checkpoint / f'iter_{step:07d}'),
                       '--output-dir', str(target), '--origin-hf-dir', str(source)]
            print('Exporting', name, flush=True)
            subprocess.run(command, check=True)
        verification = audit(target, source)
        record = {'name': name, 'checkpoint_step': step, 'format': 'hf-safetensors', 'path': 'hf-final',
                  'native_checkpoint': str(checkpoint), 'original_hf': str(source),
                  'teacher_run_directory': row['run_directory'], 'train_rollouts': 8000,
                  'common_sha256': digest(checkpoint / f'iter_{step:07d}' / 'common.pt'), **verification}
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(marker)
        (target / 'pretrained_model_identity.json').unlink(missing_ok=True)
        (target / 'rl_teacher_identity.json').write_text(json.dumps(record, indent=2) + '\n')
        print('Verified', name, verification, flush=True)
        build_banks(root)
    build_banks(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--master', type=Path, required=True)
    args = parser.parse_args()
    export(args.root.resolve(), args.master.resolve(), Path(__file__).resolve().parents[1])
