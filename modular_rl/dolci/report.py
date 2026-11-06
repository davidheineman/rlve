"""Audit completed SFT budgets, raw RL rollouts, held-out panels, and checkpoints."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .config import ACTUAL_TOKENS, atomic_json
from .watch import all_jobs, inspect, read


def events(path):
    # Unicode paragraph separators can occur inside JSON strings. JSONL records
    # are delimited by physical newlines, not every separator in str.splitlines.
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def scores(rows, expected_keys):
    groups = defaultdict(list)
    for row in rows:
        metadata = json.loads(row['metadata_json'])
        groups[metadata['key']].append(bool(row['correct']))
    if set(groups) != expected_keys or any(len(v) != 8 for v in groups.values()):
        raise ValueError('Validation panel differs from the frozen 64 x 8 selection')
    return {'pass@1': sum(map(sum, groups.values())) / len(rows),
            'pass@8': sum(any(v) for v in groups.values()) / len(groups)}


def finalize_wandb(job, api=None):
    """Restore a finished run's summary after all shared writers have closed."""
    directory = Path(job['output']) / 'policy'
    completed = read(directory / 'completed.json')
    responses_per_update = completed.get('prompts_per_update', 1) * completed.get('samples_per_prompt', 8)
    assert completed['completed'] and completed['train_rollouts'] == completed['updates'] * responses_per_update
    run_id = (directory / 'wandb_run_id.txt').read_text().strip()
    marker = directory / 'wandb-final-summary.json'
    cached = read(marker, {})
    if cached.get('run_id') == run_id and cached.get('updates') == completed['updates']:
        return True
    phase = Path(job['command'][-1]).parents[2]
    selection = read(phase / 'selection' / job['task'] / 'selection.json')
    rows = events(directory / 'rollouts/eval' / f"{completed['updates'] - 1:04d}.jsonl")
    assert len(rows) == 512
    metrics = scores(rows, set(selection['keys']['val']))
    summary = {'david/eval/val_envs/' + k: v for k, v in metrics.items()}
    summary.update({'david/rollouts': completed['train_rollouts'],
                    'david/step': completed['updates'] - 1,
                    'david/v2/completed_updates': completed['updates']})
    if api is None:
        import wandb
        api = wandb.Api(timeout=60, overrides={'base_url': 'https://api.forge.coreweave.com'})
    remote = api.run('david-heineman/modular-mopd-rl-v1/' + run_id)
    if remote.state != 'finished':
        return False
    remote.summary.update(summary)
    remote.load(force=True)
    assert all(abs(remote.summary[k] - value) < 1e-12 for k, value in summary.items())
    atomic_json(marker, {'run_id': run_id, 'updates': completed['updates'], 'summary': summary,
                         'all_shared_writers_finished': True, 'readback_verified': True})
    print('FINALIZED_WANDB', job['name'], run_id, flush=True)
    return True


def report(root, repair_wandb=False):
    root = Path(root)
    status = inspect(root)
    if not status['all_complete']:
        raise ValueError('The six SFT and fourteen RL runs are not all complete')
    result = {'passed': True, 'sft': [], 'rl': [], 'web_selection': read(root / 'web-selection.json')}
    for job in all_jobs(root):
        if job['kind'] == 'sft':
            trained = read(Path(job['output']) / 'results.json')
            assert trained['train_tokens'] == ACTUAL_TOKENS
            assert trained['config']['objective'] == 'assistant'
            assert trained['config']['model']['repo'] == 'Qwen/Qwen3-0.6B-Base'
            result['sft'].append({'name': job['name'], 'job_id': job['job_id'], 'train_tokens': trained['train_tokens'],
                                  'c4_bpb': trained['eval']['paloma_c4_en']['bpb'],
                                  'model': str(Path(job['output']) / 'model'), 'wandb': read(Path(job['output']) / 'wandb.json')})
        if job['kind'] != 'rl':
            continue
        directory = Path(job['output']) / 'policy'
        completed = read(directory / 'completed.json')
        updates = completed['updates']
        if completed['train_rollouts'] != updates * 8 or completed['max_train_rollouts'] != 8000:
            raise ValueError('Incorrect RL response budget')
        if completed['reason'] != 'validation_accuracy' and completed['train_rollouts'] != 8000:
            raise ValueError('RL run stopped before exhausting its budget')
        checkpoint = directory / 'ckpt'
        state = read(checkpoint / 'rlve_checkpoint_state.json')
        assert state['completed'] and state['rollout_id'] == updates - 1
        iteration = checkpoint / f'iter_{updates - 1:07d}'
        assert (iteration / '.metadata').exists() and (iteration / 'common.pt').exists()
        size = sum(p.stat().st_size for p in iteration.rglob('*') if p.is_file())
        assert size > 7000000000
        task = job['task']
        if task == 'chat':
            proof = read(Path(job['command'][-1]).parents[2] / 'chat-reward-normalization.json')
            assert proof['reference_and_generation_same_processing']
        selection = read(Path(job['record']).parent / 'selection' / task / 'selection.json')
        train_keys, val_keys = (set(selection['keys'][s]) for s in ('train', 'val'))
        assert not train_keys & val_keys and len(val_keys) == 64
        files = sorted((directory / 'rollouts/train').glob('*.jsonl'))
        assert [int(p.stem) for p in files] == list(range(updates))
        for path in files:
            rows = events(path)
            assert len(rows) == 8
            keys = {json.loads(row['metadata_json'])['key'] for row in rows}
            assert len(keys) == 1 and keys <= train_keys
            for row in rows:
                metadata = json.loads(row['metadata_json'])
                prefix = metadata.get('_modular_v2_forcing', {}).get('prefix_tokens', 0)
                assert 0 < row['response_tokens'] - prefix <= 2048
        expected = sorted({0, updates - 1, *range(49, updates, 50)})
        panels = sorted((directory / 'rollouts/eval').glob('*.jsonl'))
        assert [int(p.stem) for p in panels] == expected
        metrics = []
        for path in panels:
            rows = events(path)
            assert len(rows) == 512
            metrics.append({'rollout_id': int(path.stem), **scores(rows, val_keys)})
        if completed['reason'] == 'validation_accuracy':
            assert metrics[-1]['pass@1'] >= .98
        run_id = (directory / 'wandb_run_id.txt').read_text().strip()
        summary = {'david/eval/val_envs/' + k: v for k, v in metrics[-1].items() if k.startswith('pass@')}
        summary.update({'david/rollouts': completed['train_rollouts'], 'david/step': updates - 1,
                        'david/v2/completed_updates': updates})
        if repair_wandb:
            import wandb
            api = wandb.Api(timeout=60, overrides={'base_url': 'https://api.forge.coreweave.com'})
            remote = api.run('david-heineman/modular-mopd-rl-v1/' + run_id)
            if remote.state != 'finished':
                raise ValueError('Wait for all shared W&B writers to finish before summary repair')
            remote.summary.update(summary)
            remote.load(force=True)
            for key, value in summary.items():
                assert abs(remote.summary[key] - value) < 1e-12
        result['rl'].append({'name': job['name'], 'task': task, 'job_id': job['job_id'],
                             'train_rollouts': completed['train_rollouts'], 'prompt_steps': updates,
                             'baseline': metrics[0], 'final': metrics[-1], 'evaluation_panels': len(metrics),
                             'checkpoint': str(checkpoint), 'checkpoint_bytes': size,
                             'run_directory': str(directory), 'wandb_run_id': run_id, 'summary': summary,
                             'wandb_url': 'https://forge.coreweave.com/wandb/david-heineman/modular-mopd-rl-v1/runs/' + run_id})
    assert len(result['sft']) == 6 and len(result['rl']) == 14
    atomic_json(root / 'final-report.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repair-wandb', action='store_true')
    args = parser.parse_args()
    report(args.root, args.repair_wandb)
