"""Export completed MOPD students to HF while retaining their full checkpoints."""

import argparse
import json
import subprocess
from pathlib import Path

from modular_rl.mopd_export import audit, digest


def export(root, names=None):
    experiment = root / 'students'
    registry = json.loads((experiment / 'initializers.json').read_text())
    original = Path(registry['base']['hf'])
    jobs = json.loads((experiment / 'jobs.json').read_text())
    if names is not None:
        selected = [job for job in jobs if job['name'] in names]
        if len(names) != len(set(names)) or {job['name'] for job in selected} != set(names):
            raise ValueError('Select distinct known student names')
        jobs = selected
    receipts = []
    for job in jobs:
        directory = Path(job['output']) / 'policy'
        result = json.loads((directory / 'completed.json').read_text())
        if not result['completed']:
            raise ValueError('Wait for MOPD completion: ' + job['name'])
        step = result['updates'] - 1
        checkpoint = directory / 'ckpt' / f'iter_{step:07d}'
        state = json.loads((directory / 'ckpt/rlve_checkpoint_state.json').read_text())
        if not state['completed'] or state['rollout_id'] != step:
            raise ValueError('MOPD checkpoint is not committed')
        marker = directory / 'weights_only_complete.json'
        if marker.exists():
            receipts.append(json.loads(marker.read_text()))
            continue
        target = directory / 'hf-final'
        if not target.exists():
            subprocess.run(['python3', str(Path(__file__).resolve().parents[1] / 'tools/convert_torch_dist_to_hf.py'),
                '--input-dir', str(checkpoint), '--output-dir', str(target),
                '--origin-hf-dir', str(original)], check=True)
        verified = audit(target, original)
        record = {'name': job['name'], 'kind': 'mopd-student', 'format': 'hf-safetensors',
                  'path': 'hf-final', 'checkpoint_step': step, 'train_rollouts': result['train_rollouts'],
                  'native_checkpoint': str(checkpoint), 'original_hf': str(original),
                  'teachers': job['teachers'], 'common_sha256': digest(checkpoint / 'common.pt'), **verified}
        (target / 'pretrained_model_identity.json').unlink(missing_ok=True)
        (target / 'mopd_identity.json').write_text(json.dumps(record, indent=2) + '\n')
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(marker)
        receipts.append(record)
        print('Exported and verified MOPD student:', job['name'], target, flush=True)
    (root / 'student-hf-exports.json').write_text(json.dumps(receipts, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--names', nargs='+')
    args = parser.parse_args()
    export(args.root.resolve(), args.names)
