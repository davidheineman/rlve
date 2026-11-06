"""Monitor six SFT and fourteen RL jobs; release each task group when its model exists."""

import argparse
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .config import ACTUAL_TOKENS, QUEUES, atomic_json, apportion, fingerprint, mixtures

TASKS = {'base': ('math', 'code', 'if', 'chat'),
         'baseline': ('math', 'code', 'if', 'chat'),
         'math': ('math',), 'code': ('code',), 'chat': ('chat', 'if'), 'web': ('chat', 'if')}
TERMINAL_FAILURES = {'FAILED', 'CANCELLED', 'NODE_FAIL', 'TIMEOUT', 'OUT_OF_MEMORY', 'BOOT_FAIL', 'PREEMPTED'}


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def latest(path):
    path = Path(path)
    if not path.exists():
        return None
    for line in reversed(path.read_text().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue


def all_jobs(root):
    jobs = []
    path = root / 'prepare-job.json'
    if path.exists():
        jobs.append({**read(path), 'kind': 'prepare', 'record': str(path)})
    for row in read(root / 'sft-jobs.json', []):
        jobs.append({**row, 'kind': 'sft', 'record': str(root / 'sft-jobs.json')})
    for alias in TASKS:
        directory = root / 'rl' / alias
        for name, kind in (('jobs.json', 'rl'), ('conversion-jobs.json', 'conversion')):
            for row in read(directory / name, []):
                jobs.append({**row, 'kind': kind, 'alias': alias, 'record': str(directory / name),
                             'gpus': row.get('gpus', 1 if kind == 'conversion' else 0),
                             'name': row.get('name', 'convert-dolci-' + alias)})
    return jobs


def states(jobs):
    if not jobs:
        return {}
    ids = {j['job_id'] for j in jobs}
    output = subprocess.check_output(['sacct', '-n', '-P', '-X', '-j', ','.join(sorted(ids)),
                                     '--format=JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,NodeList'], text=True)
    result = {}
    for line in output.splitlines():
        values = line.split('|')
        if values[0] in ids:
            result[values[0]] = dict(zip(('job_id', 'state', 'exit_code', 'seconds', 'start', 'end', 'node'), values))
    # sacct can lag a newly submitted/requeued job. Live Slurm state takes precedence.
    output = subprocess.check_output(['squeue', '-h', '-u', os.environ.get('USER', 'dhei'), '-o', '%i|%T|%N|%R'], text=True)
    for line in output.splitlines():
        job, state, node, reason = line.split('|', 3)
        if job in ids:
            result[job] = {**result.get(job, {}), 'job_id': job, 'state': state, 'node': node, 'reason': reason}
    return result


def inspect(root):
    jobs = all_jobs(root)
    status = states(jobs)
    rows = []
    for job in jobs:
        row = {k: job.get(k) for k in ('job_id', 'name', 'kind', 'gpus', 'alias', 'mix')}
        row.update(status.get(job['job_id'], {'state': 'UNKNOWN'}))
        if job['kind'] == 'rl':
            directory = Path(job['output']) / 'policy'
            row['progress'] = latest(directory / 'progress.jsonl')
            row['completed'] = read(directory / 'completed.json')
        if job['kind'] == 'sft':
            directory = Path(job['output'])
            row['progress'] = read(directory / 'timing.json')
            row['completed'] = read(directory / 'results.json', {}).get('train_tokens') == ACTUAL_TOKENS
        rows.append(row)
    value = {'utc': datetime.now(timezone.utc).isoformat(), 'runs': rows,
             'prepared': (root / 'manifest.json').exists(),
             'running_gpus': sum((r['gpus'] or 0) for r in rows if r['state'] == 'RUNNING'),
             'scratch_free_gib': os.statvfs(root).f_bavail * os.statvfs(root).f_frsize / 2**30,
             'sft_complete': sum(r['kind'] == 'sft' and r.get('completed', False) and r['state'] == 'COMPLETED' for r in rows),
             'rl_complete': sum(r['kind'] == 'rl' and bool(r.get('completed')) and r['state'] == 'COMPLETED' for r in rows)}
    value['all_complete'] = value['sft_complete'] == 6 and value['rl_complete'] == 14
    atomic_json(root / 'monitor.json', value)
    return value


def prepare_rl(root, alias):
    from modular_rl.initializers import inspect_hf
    from modular_rl.suite import prepare
    from modular_rl.suite_conversions import prepare as conversions
    repo = Path(__file__).resolve().parents[2]
    previous = root / 'rl/base'
    directory = root / 'rl' / alias
    if not (directory / 'launch-plan.json').exists():
        base = read(previous / 'initializers.json')['base']
        mix = read(root / 'web-selection.json')['chosen_mix'] if alias == 'web' else alias
        spec = inspect_hf(alias, root / 'sft' / mix / 'model', root / 'initializers' / alias, training_mix=mix)
        registry = root / 'registries' / (alias + '.json')
        atomic_json(registry, {'base': base, alias: spec})
        prepare(repo, Path('/nlp/scr/dhei/modular-mopd-rl/v2'), directory, root / 'sft', previous / 'selection',
                policy_gpus=2, reward_gpus=1, initializers_from=registry, selection_from=previous / 'selection',
                runs=[('dolci-' + alias + '-' + task, alias, task) for task in TASKS[alias]],
                queues=QUEUES, group='dolci-mix-1b', allow_unconverted=True, time_limit='08:00:00',
                frozen_from=root / 'rl/base', data_from=root / 'rl-data', clean_chat_rewards=True)
    if not (directory / 'conversion-jobs.json').exists():
        conversions(directory, submit=True, queues=QUEUES)
    return directory


def add_web_candidate(root):
    """Both user-requested natural definitions share the same tokenized corpus."""
    import csv
    manifest = read(root / 'manifest.json')
    if 'web-knowledge' in manifest['recipe']['mixes']:
        return
    weights = mixtures(manifest['sources'], natural='knowledge')['web']
    manifest['recipe']['mixes']['web-knowledge'] = weights
    manifest['quotas']['web-knowledge'] = apportion(ACTUAL_TOKENS // manifest['seq_len'], weights)
    manifest['natural_definitions'] = {'web': 'complement', 'web-knowledge': 'knowledge'}
    manifest.pop('signature')
    manifest['signature'] = fingerprint(manifest)
    path = root / 'mixture-ratios.csv'
    rows = list(csv.DictReader(path.open()))
    names = {(s['source_dataset'], s['domain']): n for n, s in manifest['sources'].items()}
    for row in rows:
        name = names[(row['source_dataset'], row['domain'])]
        row['web-knowledge_token_share'] = weights[name]
        row['web-knowledge_upsample'] = weights[name] / float(row['token_share'])
        row['web-knowledge_actual_train_tokens'] = manifest['quotas']['web-knowledge'][name] * manifest['seq_len']
    with path.open('w') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(root / 'manifest.json', manifest)


def select_web(root, jobs, slurm):
    if (root / 'web-selection.json').exists():
        return True
    candidates = [job for job in jobs if job['kind'] == 'sft' and job['mix'] in ('web', 'web-knowledge')]
    if len(candidates) != 2 or any(slurm.get(j['job_id'], {}).get('state') != 'COMPLETED' for j in candidates):
        return False
    rows = []
    for job in candidates:
        result = read(Path(job['output']) / 'results.json')
        assert result['train_tokens'] == ACTUAL_TOKENS
        scores = result['eval']['paloma_c4_en']
        rows.append({'mix': job['mix'], 'c4_bpb': scores['bpb'], 'c4_documents': scores['n_docs'],
                     'c4_tokens': scores['n_tokens'], 'model': str(Path(job['output']) / 'model'), 'job_id': job['job_id']})
    winner = min(rows, key=lambda r: (r['c4_bpb'], r['mix']))
    atomic_json(root / 'web-selection.json', {'criterion': 'lowest final full-panel held-out C4 BPB',
                                             'chosen_mix': winner['mix'], 'candidates': rows,
                                             'downstream_tasks': ['chat', 'if'], 'actual_input_tokens_each': ACTUAL_TOKENS})
    print('SELECTED_WEB', winner, flush=True)
    return True


def retry(root, job, state):
    """Retry bounded transient failures; retain all previous Slurm attempts and logs."""
    attempts = read(root / 'retry-history.json', [])
    old = [r for r in attempts if r['name'] == job['name']]
    if len(old) >= 3:
        raise RuntimeError('Three retries exhausted for ' + job['name'])
    command = list(job['command'])
    if state['state'] == 'OUT_OF_MEMORY':
        # Change placement, keeping the actual optimization recipe unchanged.
        command = ['--constraint=80G|141G' if c.startswith('--constraint=') else c for c in command]
        for i, part in enumerate(command):
            if part.startswith('--mem=') and part.endswith('G'):
                command[i] = '--mem=' + str(int(int(part[6:-1]) * 1.5)) + 'G'
    new = subprocess.check_output(command, text=True).strip().split(';')[0]
    record = Path(job['record'])
    value = read(record)
    if isinstance(value, list):
        for row in value:
            if row['job_id'] == job['job_id']:
                row.update(job_id=new, command=command)
    else:
        value.update(job_id=new, command=command)
    atomic_json(record, value)
    attempts.append({'name': job['name'], 'old_job_id': job['job_id'], 'new_job_id': new,
                     'failure': state, 'utc': datetime.now(timezone.utc).isoformat()})
    atomic_json(root / 'retry-history.json', attempts)
    print('RELAUNCHED', job['name'], job['job_id'], new, flush=True)


def tick(root):
    from .training import submit_ready
    status = inspect(root)
    if status['prepared']:
        add_web_candidate(root)
    if status['prepared'] and not read(root / 'validation.json', {}).get('passed'):
        from .validate import validate
        validate(root)
    if status['prepared'] and read(root / 'validation.json', {}).get('passed'):
        submit_ready(root)
    jobs = all_jobs(root)
    slurm = states(jobs)
    for job in jobs:
        current = slurm.get(job['job_id'], {})
        state = current.get('state', '').split()[0]
        if state in TERMINAL_FAILURES:
            retry(root, job, current)
            continue
        if job['kind'] == 'rl' and state == 'COMPLETED':
            from .report import finalize_wandb
            finalize_wandb(job)
        if job['kind'] == 'sft' and state == 'COMPLETED':
            result = read(Path(job['output']) / 'results.json', {})
            if result.get('train_tokens') != ACTUAL_TOKENS:
                raise ValueError('Slurm completed without the full SFT budget: ' + job['name'])
            from .export import finalize_generation
            finalize_generation(Path(job['output']))
            free_gib = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize / 2**30
            if free_gib < 45:
                from .storage import retire
                retire(Path(job['output']))
            if job['mix'] not in ('web', 'web-knowledge'):
                prepare_rl(root, job['mix'])
        if job['kind'] == 'conversion' and state == 'COMPLETED':
            from modular_rl.initializers import verify
            from modular_rl.suite import submit
            directory = root / 'rl' / job['alias']
            submitted = {r['name'] for r in read(directory / 'jobs.json', [])}
            missing = [r['name'] for r in read(directory / 'launch-plan.json')['runs'] if r['name'] not in submitted]
            if missing:
                registry = read(directory / 'initializers.json')
                verify(registry[job['alias']], full_weights=True)
                submit(directory, missing)
    if select_web(root, jobs, slurm):
        prepare_rl(root, 'web')
    status = inspect(root)
    print(json.dumps({k: v for k, v in status.items() if k != 'runs'}), flush=True)
    return status


def monitor(root, interval=45):
    import fcntl
    lock = (root / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    while not (root / 'STOP').exists():
        try:
            value = tick(root)
            (root / 'attention-required.json').unlink(missing_ok=True)
            if value['all_complete']:
                from .report import report
                report(root, repair_wandb=True)
                from .sensitivity import sensitivity
                sensitivity(root)
                return
        except Exception:
            message = traceback.format_exc()
            print(message, flush=True)
            atomic_json(root / 'attention-required.json', {'utc': datetime.now(timezone.utc).isoformat(), 'traceback': message})
        time.sleep(interval)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--once', action='store_true')
    args = p.parse_args()
    if args.once:
        print(json.dumps(tick(args.root.resolve()), indent=2))
    else:
        monitor(args.root.resolve())
