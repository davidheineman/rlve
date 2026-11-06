"""Restore final code metrics after all shared W&B writers have finished."""
import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from .base_models import BASE_MODELS
from .inspect_rollouts import read_events


def final_monitor_summary(run, *, task='code', validation_prompts=32):
    if task not in ('math', 'code', 'if', 'chat') or validation_prompts not in (32, 64):
        raise ValueError('Select a supported task and the declared 32/64-prompt panel')
    done = json.loads((run / 'completed.json').read_text())
    updates = done['updates']
    if not done['completed'] or done['train_rollouts'] != updates * 8:
        raise ValueError('Expected completed one-prompt, eight-sample training')
    path = run / f'rollouts/eval/{max(0, updates - 1):04d}.jsonl'
    rows = read_events(path)
    groups = defaultdict(list)
    for row in rows:
        meta = json.loads(row['metadata_json'])
        if meta['task'] != task:
            raise ValueError('Summary repair must not modify a different task')
        groups[meta['key']].append(bool(row['correct']))
    if (len(groups) != validation_prompts or len(rows) != validation_prompts * 8
            or any(len(xs) != 8 for xs in groups.values())):
        raise ValueError('Incomplete final monitoring export')
    pass1 = mean(bool(row['correct']) for row in rows)
    final = [e for e in read_events(run / 'progress.jsonl') if e['phase'] == 'eval'][-1]
    if final['completed_updates'] != updates or abs(final['accuracy'] - pass1) > 1e-10:
        raise ValueError('Final raw grades and recorded progress disagree')
    return {'david/eval/val_envs/pass@1': pass1,
            'david/eval/val_envs/pass@8': mean(any(xs) for xs in groups.values()),
            'david/rollouts': done['train_rollouts'], 'david/step': max(0, updates - 1),
            'david/v2/completed_updates': updates}


def repair(review, extra_roots=()):
    import wandb

    roots = [Path(e['root']) for e in json.loads((review / 'code-training-roots.json').read_text())]
    roots += list(extra_roots)
    receipt = review / 'code-wandb-summary-repairs.json'
    repairs = json.loads(receipt.read_text()) if receipt.exists() else []
    repaired = {r['run_id'] for r in repairs}
    api = None
    for root in roots:
        for job in json.loads((root / 'jobs.json').read_text()):
            if job['task'] != 'code' or job['model'] not in BASE_MODELS:
                raise ValueError('Only explicitly requested pretrained Base code conditions may be repaired')
            run = Path(job['run'])
            if not (run / 'completed.json').exists():
                continue
            run_id = (run / 'wandb_run_id.txt').read_text().strip()
            if run_id in repaired:
                continue
            summary = final_monitor_summary(run)
            if api is None:
                api = wandb.Api(timeout=30)
            remote = api.run('david-heineman/modular-mopd-rl-initial/' + run_id)
            if remote.state != 'finished':
                print('Wait for shared writers to finish:', run_id, remote.state, flush=True)
                continue
            previous = {key: remote.summary.get(key) for key in summary}
            remote.summary.update(summary)
            row = {'utc': datetime.now(UTC).isoformat(), 'job_id': job['job_id'],
                   'run_id': run_id, 'source_run': str(run), 'before': previous, 'after': summary,
                   'history_modified': False, 'run_state_modified': False}
            repairs.append(row)
            temporary = receipt.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(repairs, indent=2) + '\n')
            temporary.replace(receipt)
            repaired.add(run_id)
            print('Restored final summary:', run_id, json.dumps(summary), flush=True)
    return repairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--extra-root', type=Path, action='append', default=[])
    args = parser.parse_args()
    repair(args.review, args.extra_root)
