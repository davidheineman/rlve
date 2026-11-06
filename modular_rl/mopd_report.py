"""Audit every completed follow-up trajectory, teacher route, and validation panel."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from modular_rl.mopd_export import BANKS
from slime.utils.eval_metrics import pass_at_k


def read(path):
    return [json.loads(line) for line in path.open()]


def scores(rows):
    groups = defaultdict(list)
    for row in rows:
        metadata = json.loads(row['metadata_json'])
        groups[(metadata['task'], metadata['key'])].append(int(row['correct']))
    if not groups or any(len(group) != 8 for group in groups.values()):
        raise ValueError('Validation must contain eight samples for every prompt')
    estimates = pass_at_k([sum(group) for group in groups.values()], 8)
    return {'pass@1': sum(sum(group) for group in groups.values()) / len(rows),
            'pass@8': estimates[8]}


def language_fallbacks(rows):
    """Count successes accepted by the inherited IF language-exception rule."""
    import langdetect

    affected = []
    for row in rows:
        reward = json.loads(row['reward_json'])
        if not row['correct'] or 'language:response_language' not in reward.get('constraints', []):
            continue
        # Match grade_if preprocessing; EOS text otherwise looks like language.
        text = row['response'].rsplit('</think>', 1)[-1].strip()
        for marker in ('<|endoftext|>', '<|im_end|>'):
            text = text.removesuffix(marker).rstrip()
        try:
            langdetect.detect(text)
        except langdetect.LangDetectException:
            affected.append(json.loads(row['metadata_json'])['key'])
    return {'accepted_language_detection_failures': len(affected),
            'validation_responses': len(rows),
            'pass@1_contribution': len(affected) / len(rows),
            'affected_prompt_keys': sorted(set(affected)), 'grader_unchanged': True}


def report(root, repair_wandb=False):
    monitor = json.loads((root / 'monitor.json').read_text())
    states = {r['name']: r for r in monitor['runs']}
    if not monitor['all_five_complete']:
        raise ValueError('Wait until all five jobs finish before the final audit')
    rows = []
    for kind in ('rl', 'students'):
        plan = json.loads((root / kind / 'launch-plan.json').read_text())
        updates = plan['train_steps']
        budget = plan['train_rollouts']
        interval = plan['evaluation_interval']
        for job in json.loads((root / kind / 'jobs.json').read_text()):
            directory = Path(job['output']) / 'policy'
            result = json.loads((directory / 'completed.json').read_text())
            if result['train_rollouts'] != budget or result['updates'] != updates:
                raise ValueError('Run did not finish its full response budget: ' + job['name'])
            checkpoint = directory / 'ckpt'
            state = json.loads((checkpoint / 'rlve_checkpoint_state.json').read_text())
            if not state['completed'] or state['rollout_id'] != updates - 1:
                raise ValueError('Incomplete final checkpoint')
            iteration = checkpoint / f'iter_{updates - 1:07d}'
            if not (iteration / 'common.pt').exists() or not (iteration / '.metadata').exists():
                raise ValueError('Missing native checkpoint files')
            checkpoint_bytes = sum(p.stat().st_size for p in iteration.rglob('*') if p.is_file())
            if checkpoint_bytes < 7_000_000_000:
                raise ValueError('Expected full model and optimizer checkpoint')
            train_files = sorted((directory / 'rollouts/train').glob('*.jsonl'))
            if [int(p.stem) for p in train_files] != list(range(updates)):
                raise ValueError('Incomplete or duplicate training rollout steps')
            task_counts = Counter()
            for path in train_files:
                samples = read(path)
                if len(samples) != 8 or any(s['rollout_id'] != int(path.stem) for s in samples):
                    raise ValueError('Incomplete training rollout group')
                tasks = Counter(s['source_environment'] for s in samples)
                if kind == 'students' and tasks != Counter(dict.fromkeys(('math', 'code', 'if', 'chat'), 2)):
                    raise ValueError('Unbalanced MOPD training step')
                task_counts.update(tasks)
                for sample in samples:
                    metadata = json.loads(sample['metadata_json'])
                    forced = metadata.get('_modular_v2_forcing', {}).get('prefix_tokens', 0)
                    if not 0 < sample['response_tokens'] - forced <= 2048:
                        raise ValueError('Invalid sampled response token budget')
            eval_files = sorted((directory / 'rollouts/eval').glob('*.jsonl'))
            expected_steps = [0] + list(range(interval - 1, updates, interval))
            if updates - 1 not in expected_steps:
                expected_steps.append(updates - 1)
            if [int(p.stem) for p in eval_files] != expected_steps:
                raise ValueError('Missing scheduled validation panels')
            metrics = []
            for path in eval_files:
                samples = read(path)
                by_task = defaultdict(list)
                for sample in samples:
                    by_task[sample['source_environment']].append(sample)
                if kind == 'students' and set(by_task) != {'math', 'code', 'if', 'chat'}:
                    raise ValueError('Missing MOPD validation task')
                if any(len(task_rows) != 512 for task_rows in by_task.values()):
                    raise ValueError('Expected 64 validation prompts times eight samples per task')
                for task, task_rows in by_task.items():
                    expected = {json.loads(line)['metadata']['key'] for line in
                                (root / kind / 'selection' / task / 'val.jsonl').open()}
                    actual = {json.loads(row['metadata_json'])['key'] for row in task_rows}
                    if actual != expected:
                        raise ValueError('Validation panel changed')
                metrics.append({'rollout_id': int(path.stem), 'aggregate': scores(samples),
                                'tasks': {task: scores(rs) for task, rs in by_task.items()}})
            diagnostics = {}
            probes = read(directory / 'update-probes.jsonl')
            if [p['rollout_id'] for p in probes] != list(range(0, updates, 10)):
                raise ValueError('Incomplete update diagnostics')
            if kind == 'students':
                routes = read(directory / 'teacher-scores.jsonl')
                if [r['rollout_id'] for r in routes] != list(range(updates)):
                    raise ValueError('Incomplete teacher routing records')
                bank = BANKS[job['bank']]
                for record in routes:
                    if len(record['samples']) != 8:
                        raise ValueError('Missing teacher scores')
                    for sample in record['samples']:
                        if sample['teacher'] != bank[sample['task']] or not math.isfinite(sample['mean_teacher_minus_student_logp']):
                            raise ValueError('Wrong teacher or nonfinite teacher scores')
                gains = [p['metrics']['david/opd/surrogate_improvement'] for p in probes]
                if not all(math.isfinite(v) for v in gains):
                    raise ValueError('Nonfinite MOPD update diagnostics')
                diagnostics = {'teacher_scored_responses': sum(len(r['samples']) for r in routes),
                    'all_routes_match_requested_teacher_bank': True, 'update_probes': len(probes),
                    'probes_with_positive_surrogate_improvement': sum(v > 0 for v in gains),
                    'mean_surrogate_improvement': sum(gains) / len(gains)}
            else:
                decisions = read(directory / 'optimizer-decisions.jsonl')
                if [r['rollout_id'] for r in decisions] != list(range(updates)):
                    raise ValueError('Incomplete RL optimizer accounting')
                diagnostics = {'optimizer_updates': sum(r['optimizer_update_applied'] for r in decisions),
                               'update_probes': len(probes)}
            if 'if' in by_task:
                diagnostics['if_language_verifier'] = language_fallbacks(by_task['if'])
            run_id = (directory / 'wandb_run_id.txt').read_text().strip()
            summary = {'david/step': updates - 1, 'david/rollouts': budget,
                       'david/eval/val_envs/pass@1': metrics[-1]['aggregate']['pass@1'],
                       'david/eval/val_envs/pass@8': metrics[-1]['aggregate']['pass@8']}
            for task, values in metrics[-1]['tasks'].items():
                summary.update({f'david/eval/val_envs/{task}/{key}': value for key, value in values.items()})
            if repair_wandb:
                import wandb
                api = wandb.Api(timeout=60, overrides={'base_url': 'https://api.forge.coreweave.com'})
                remote = api.run('david-heineman/modular-mopd-rl-v1/' + run_id)
                remote.load(force=True)
                if remote.state != 'finished':
                    raise ValueError('Wait for W&B writers to finish: ' + job['name'])
                remote.summary.update(summary)
                remote.load(force=True)
                if any(remote.summary.get(key) != value for key, value in summary.items()):
                    raise ValueError('W&B final summary readback disagrees with raw responses')
                metric_keys = [key for key in summary if key.startswith('david/eval/')]
                history = remote.history(keys=['david/step', *metric_keys], samples=10000, pandas=False)
                if [point.get('david/step') for point in history] != expected_steps:
                    raise ValueError('W&B validation history has missing or extra evaluation panels')
                for point, panel in zip(history, metrics, strict=True):
                    expected_values = {'david/eval/val_envs/' + key: value
                                       for key, value in panel['aggregate'].items()}
                    for task, values in panel['tasks'].items():
                        expected_values.update({f'david/eval/val_envs/{task}/{key}': value
                                                for key, value in values.items()})
                    if any(point.get(key) is None or not math.isclose(point[key], value, abs_tol=1e-12)
                           for key, value in expected_values.items()):
                        raise ValueError('W&B validation history disagrees with retained responses')
            rows.append({'name': job['name'], 'job_id': job['job_id'], 'state': states[job['name']]['state'],
                'train_rollouts': budget, 'updates': updates, 'task_rollouts': dict(task_counts),
                'gpus': job['gpus'], 'baseline': metrics[0], 'final': metrics[-1],
                'validation_panels': len(metrics), 'validation_responses': len(eval_files) * 512 * len(metrics[-1]['tasks']),
                'checkpoint': str(iteration), 'checkpoint_bytes': checkpoint_bytes, 'diagnostics': diagnostics,
                'wandb_run_id': run_id, 'wandb_url': 'https://forge.coreweave.com/wandb/david-heineman/modular-mopd-rl-v1/runs/' + run_id,
                'summary': summary})
    result = {'passed': True, 'runs': rows, 'all_five_completed_full_budgets': True,
              'wandb_summaries_repaired_and_verified': repair_wandb,
              'wandb_histories_verified_against_raw_responses': repair_wandb}
    (root / 'final-report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repair-wandb', action='store_true')
    args = parser.parse_args()
    report(args.root.resolve(), args.repair_wandb)
