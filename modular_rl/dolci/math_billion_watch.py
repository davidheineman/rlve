"""Monitor the 1B short-math SFT and its paired full math RL comparison."""

import argparse
import fcntl
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .config import ACTUAL_TOKENS, atomic_json
from .math_billion import PARENT, prepare_rl
from .report import events, scores, finalize_wandb
from .watch import states, TERMINAL_FAILURES


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def retry(root, record, row, state):
    history = read(root / 'retry-history.json', [])
    previous = [v for v in history if v['name'] == row['name']]
    if len(previous) >= 4:
        raise RuntimeError('Four retries exhausted: ' + row['name'])
    command = list(row['command'])
    if state['state'] == 'OUT_OF_MEMORY':
        command = ['--constraint=80G|141G' if v.startswith('--constraint=') else v for v in command]
        command = ['--mem=192G' if v.startswith('--mem=') else v for v in command]
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    history.append({'name': row['name'], 'previous_job': row['job_id'], 'job_id': job,
                    'previous_state': state, 'utc': datetime.now(timezone.utc).isoformat()})
    atomic_json(root / 'retry-history.json', history)
    value = read(record)
    rows = value if isinstance(value, list) else [value]
    for entry in rows:
        if entry['job_id'] == row['job_id']:
            entry.update(job_id=job, command=command)
    atomic_json(record, rows if isinstance(value, list) else rows[0])
    print('RETRIED', row['name'], job, flush=True)


def audit_rl(directory):
    job, = read(directory / 'jobs.json')
    policy = Path(job['output']) / 'policy'
    complete = read(policy / 'completed.json')
    if not complete:
        return None
    assert complete['completed'] and complete['train_rollouts'] == complete['updates'] * 8
    assert complete['max_train_rollouts'] == 8000 and complete['sampled_token_budget'] == 2048
    assert complete['reason'] == 'validation_accuracy' or complete['train_rollouts'] == 8000
    assert complete['optimizer_updates'] > 0
    selection = read(directory / 'selection/math/selection.json')
    train_keys, val_keys = [set(selection['keys'][split]) for split in ('train', 'val')]
    assert not train_keys & val_keys and len(val_keys) == 64
    train = sorted((policy / 'rollouts/train').glob('*.jsonl'))
    assert [int(p.stem) for p in train] == list(range(complete['updates']))
    for path in train:
        rows = events(path)
        assert len(rows) == 8
        keys = {json.loads(v['metadata_json'])['key'] for v in rows}
        assert len(keys) == 1 and keys <= train_keys
        for row in rows:
            metadata = json.loads(row['metadata_json'])
            prefix = metadata.get('_modular_v2_forcing', {}).get('prefix_tokens', 0)
            assert 0 < row['response_tokens'] - prefix <= 2048
    curve = []
    paths = sorted((policy / 'rollouts/eval').glob('*.jsonl'))
    expected = sorted({0, complete['updates'] - 1, *range(49, complete['updates'], 50)})
    assert [int(p.stem) for p in paths] == expected
    for path in paths:
        curve.append({'rollout_id': int(path.stem), **scores(events(path), val_keys)})
    checkpoint = read(policy / 'ckpt/rlve_checkpoint_state.json')
    assert checkpoint['completed'] and checkpoint['rollout_id'] == complete['updates'] - 1
    run_id = (policy / 'wandb_run_id.txt').read_text().strip()
    return {'passed': True, 'name': job['name'], 'completed': complete, 'curve': curve,
            'run_directory': str(policy), 'wandb_run_id': run_id,
            'wandb_url': 'https://forge.coreweave.com/wandb/david-heineman/modular-mopd-rl-v1/runs/' + run_id}


def maybe_submit_rl(root, alias):
    from modular_rl.suite import submit
    directory = root / 'rl' / alias
    if not (directory / 'jobs.json').exists():
        submit(directory)


def check_rl(root, alias, status):
    directory = root / 'rl' / alias
    jobs = read(directory / 'jobs.json', [])
    if not jobs:
        return False
    slurm = states(jobs)
    job, = jobs
    state = slurm.get(job['job_id'], {'state': 'UNKNOWN'})
    policy = Path(job['output']) / 'policy'
    status['rl'][alias] = {'job_id': job['job_id'], **state,
                         'completed': read(policy / 'completed.json')}
    if state['state'] in TERMINAL_FAILURES:
        retry(root, directory / 'jobs.json', job, state)
        return False
    if state['state'] == 'COMPLETED':
        proof_path = directory / 'completion-audit.json'
        proof = read(proof_path)
        if proof is None:
            proof = audit_rl(directory)
            if proof is None:
                return False  # Shared-file visibility can lag Slurm completion.
            atomic_json(proof_path, proof)
        try:
            finalize_wandb(job)
        except Exception as exc:
            status['rl'][alias]['wandb_summary_pending'] = str(exc)
        return True
    return False


def tick(root):
    root = Path(root)
    job = read(root / 'sft-job.json')
    slurm = states([job])
    state = slurm.get(job['job_id'], {'state': 'UNKNOWN'})
    status = {'utc': datetime.now(timezone.utc).isoformat(), 'sft': {'job_id': job['job_id'], **state},
              'rl': {}, 'scratch_free_gib': os.statvfs(root).f_bavail * os.statvfs(root).f_frsize / 2**30}
    timing = read(root / 'sft/math/timing.json')
    status['sft']['progress'] = timing
    if state['state'] in TERMINAL_FAILURES:
        retry(root, root / 'sft-job.json', job, state)
    baseline_done = check_rl(root, 'baseline', status)
    result = read(root / 'sft/math/results.json')
    if state['state'] == 'COMPLETED' and result and result['train_tokens'] == ACTUAL_TOKENS:
        from .export import finalize_generation
        from .storage import retire
        finalize_generation(root / 'sft/math')
        # Every final exported tensor is checked exactly against the completed
        # SFT state before optimizer space is released for downstream RL.
        retire(root / 'sft/math')
        directory = prepare_rl(root, 'math')
        native = root / 'initializers/math'
        if not (native / 'conversion-audit.json').exists():
            record = directory / 'conversion-jobs.json'
            conversions = read(record)
            if conversions is None:
                from modular_rl.suite_conversions import prepare
                prepare(directory, submit=True)
                conversions = read(record)
            conversion, = conversions
            conversion.setdefault('name', 'math-refresh-conversion')
            cs = states([conversion]).get(conversion['job_id'], {'state': 'UNKNOWN'})
            status['conversion'] = cs
            if cs['state'] in TERMINAL_FAILURES:
                value = read(record)
                value[0].setdefault('name', 'math-refresh-conversion')
                atomic_json(record, value)
                retry(root, record, value[0], cs)
        else:
            audit = read(native / 'conversion-audit.json')
            assert audit['all_native_parameters_exactly_match_hf_after_dtype_cast']
            maybe_submit_rl(root, 'math')
    math_done = check_rl(root, 'math', status)
    status['complete'] = baseline_done and math_done
    atomic_json(root / 'monitor.json', status)
    if status['complete']:
        original = next(v for v in read(PARENT / 'final-report.json')['rl'] if v['name'] == 'dolci-baseline-math')
        report = {'passed': True, 'sft': result, 'baseline_matched': read(root / 'rl/baseline/completion-audit.json'),
                  'refreshed_math': read(root / 'rl/math/completion-audit.json'), 'baseline_original_protocol': original,
                  'repetition': read(root / 'repetition-proof.json'),
                  'comparison_note': 'Primary pair shares prompt/decoding/training settings; original baseline uses its original protocol.'}
        atomic_json(root / 'final-report.json', report)
        print('COMPLETE', root, flush=True)
    return status


def run(root):
    root = Path(root).resolve()
    with (root / 'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        errors = 0
        while True:
            try:
                status = tick(root)
                # Refresh the separate comparison figure when a panel arrives.
                files = sorted((root / 'rl').glob('*/runs/*/policy/rollouts/eval/*.jsonl'))
                signature = [(str(p), p.stat().st_size) for p in files]
                if read(root / 'plot-inputs.json', []) != [list(v) for v in signature]:
                    output = Path('/juice5b/u/dhei/tatsu/rlve/plot/mopd-modular-v2/math_refresh_1b.png')
                    subprocess.run(['/juice5b/u/dhei/tatsu/nano-omni/.venv-midtraining/bin/python',
                        '-m', 'modular_rl.dolci.math_billion_plot', '--root', str(root),
                        '--output', str(output)], check=True)
                    atomic_json(root / 'plot-inputs.json', signature)
                errors = 0
                print(json.dumps(status), flush=True)
                if status['complete']:
                    return
            except Exception:
                errors += 1
                error = {'utc': datetime.now(timezone.utc).isoformat(), 'consecutive_errors': errors,
                         'traceback': traceback.format_exc()}
                atomic_json(root / 'controller-error.json', error)
                traceback.print_exc()
                if errors >= 8:
                    raise
            time.sleep(45)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.once:
        print(json.dumps(tick(args.root), indent=2))
    else:
        run(args.root)
