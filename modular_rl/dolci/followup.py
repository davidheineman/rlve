"""Launch four-GPU code reward controls and genuine Precise-IF SFT -> RL runs."""

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .config import ACTUAL_TOKENS, QUEUES, atomic_json, apportion, fingerprint
from .watch import read, states, TERMINAL_FAILURES

GROUP = 'dolci-code-if-followup'
INITIAL_RUNS = [('dolci-base-code-fixed', 'base', 'code'),
                ('dolci-baseline-code-fixed', 'baseline', 'code'),
                ('dolci-code-code-fixed', 'code', 'code'),
                ('dolci-baseline-if-dp4', 'baseline', 'if')]


def if_mixtures(manifest):
    baseline = manifest['recipe']['mixes']['baseline']
    is_if = {n: s['domain'] == 'Precise IF' for n, s in manifest['sources'].items()}
    mass = sum(w for n, w in baseline.items() if is_if[n])
    if not 0 < mass < 1:
        raise ValueError('Expected a nonempty Precise IF domain and remainder')
    return {f'if{int(100 * share)}': {n: w * (share / mass if is_if[n] else (1 - share) / (1 - mass))
                                     for n, w in baseline.items()} for share in (.2, .5)}


def safe_command(command):
    command = [c for c in command if not c.startswith('--export=')]
    command.insert(-1, '--export=ALL,WANDB_SERVICE=')
    for i, value in enumerate(command):
        if value.startswith('--exclude=') and 'pasteur6' not in value.split('=', 1)[1].split(','):
            command[i] = value + ',pasteur6'
    return command


def submit_job(row):
    command = safe_command(row['command'])
    env = {k: v for k, v in os.environ.items() if k != 'WANDB_SERVICE'}
    job = subprocess.check_output(command, text=True, env=env).strip().split(';')[0]
    print('SUBMITTED', row['name'], job, flush=True)
    return {**row, 'command': command, 'job_id': job}


def prepare_phase(root, alias):
    from modular_rl.initializers import inspect_hf
    from modular_rl.suite import prepare
    parent = Path(read(root / 'followup-plan.json')['parent'])
    repo = Path(__file__).resolve().parents[2]
    directory = root / 'rl' / alias
    if (directory / 'launch-plan.json').exists():
        return directory
    base = read(parent / 'rl/base/initializers.json')['base']
    if alias == 'corrected':
        specs = {'base': base,
                 'baseline': read(parent / 'rl/baseline/initializers.json')['baseline'],
                 'code': read(parent / 'rl/code/initializers.json')['code']}
        runs, allow_unconverted = INITIAL_RUNS, False
    else:
        specs = {'base': base, alias: inspect_hf(alias, root / 'sft' / alias / 'model',
                                               root / 'initializers' / alias, training_mix=alias)}
        runs, allow_unconverted = [(f'dolci-{alias}-if', alias, 'if')], True
    registry = root / 'registries' / (alias + '.json')
    atomic_json(registry, specs)
    prepare(repo, Path('/nlp/scr/dhei/modular-mopd-rl/v2'), directory, root / 'sft', parent / 'rl/base/selection',
            policy_gpus=4, reward_gpus=1, initializers_from=registry,
            selection_from=parent / 'rl/base/selection', runs=runs, queues=QUEUES, group=GROUP,
            allow_unconverted=allow_unconverted, time_limit='08:00:00', frozen_from=parent / 'rl/base',
            data_from=parent / 'rl-data', clean_code_rewards=True, resize_frozen_policy=True)
    plan = read(directory / 'launch-plan.json')
    for row in plan['runs']:
        row['command'] = safe_command(row['command'])
    atomic_json(directory / 'launch-plan.json', plan)
    return directory


def prepare(root, parent):
    if root.exists():
        raise ValueError('Use a fresh follow-up directory')
    root.mkdir(parents=True)
    for name in ('sources', 'eval', 'eval_inloop_64', 'model', 'model_revision.json'):
        (root / name).symlink_to(parent / name)
    manifest = read(parent / 'manifest.json')
    weights = if_mixtures(manifest)
    manifest['recipe']['mixes'] = weights
    manifest['quotas'] = {mix: apportion(ACTUAL_TOKENS // manifest['seq_len'], w) for mix, w in weights.items()}
    manifest['parent_experiment'] = str(parent)
    manifest['mixture_definition'] = '20% or 50% Precise IF input tokens; remainder proportional to empirical baseline'
    manifest.pop('signature')
    manifest['signature'] = fingerprint(manifest)
    atomic_json(root / 'manifest.json', manifest)
    plan = {'parent': str(parent), 'gpus_per_training_run': 4, 'initial_training_jobs': 6,
            'initial_requested_gpus': 24, 'sft_runs': ['if20', 'if50'], 'rl_runs': INITIAL_RUNS,
            'followup_rl_runs': ['dolci-if20-if', 'dolci-if50-if'],
            'sft_input_tokens': ACTUAL_TOKENS, 'sft_lr': .0001, 'rl_responses': 8000,
            'rl_lr': .000002, 'max_response_tokens': 2048, 'validation_prompts': 64,
            'validation_samples': 8, 'queues': QUEUES, 'group': GROUP,
            'only_required_handoffs': 'Each IF RL run waits for its own SFT and native initializer conversion.'}
    atomic_json(root / 'followup-plan.json', plan)
    source = root / 'sft-source'
    shutil.copytree(parent / 'sft-source', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.sbatch'))
    shutil.copy2(parent / 'sft-source/run.sbatch', source / 'run.sbatch')
    train = source / 'train.py'
    text = train.read_text()
    old = 'choices=["baseline", "code", "math", "chat", "web", "web-knowledge"]'
    assert text.count(old) == 1
    train.write_text(text.replace(old, 'choices=["if20", "if50"]'))
    environment = {}
    for line in (parent / 'sft-source/chat.sbatch').read_text().splitlines():
        if line.startswith('export '):
            key, value = line[7:].split('=', 1)
            environment[key] = shlex.split(value)[0]
    environment.update(NANO_DATA_ROOT=str(root), NANO_OUT_ROOT=str(root / 'sft'),
                       NANO_TRAIN_SCRIPT=str(train), PYTHONPATH=str(source),
                       NANO_WANDB_GROUP=GROUP, NANO_GPUS='4', NANO_GRAD_ACCUM='16')
    prototype = read(parent / 'sft-jobs.json')[0]['command']
    rows = []
    for mix in weights:
        wrapper = source / (mix + '.sbatch')
        wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\n' +
            ''.join(f'export {key}={shlex.quote(value)}\n' for key, value in {**environment, 'NANO_MIX': mix}.items()) +
            'exec bash ' + shlex.quote(str(source / 'run.sbatch')) + '\n')
        command = list(prototype)
        for i, value in enumerate(command):
            if value.startswith('--partition='): command[i] = '--partition=' + QUEUES
            elif value.startswith('--job-name='): command[i] = '--job-name=dolci-sft-' + mix
            elif value.startswith('--output='): command[i] = '--output=' + str(root / 'logs' / (mix + '-%j.out'))
            elif value.startswith('--error='): command[i] = '--error=' + str(root / 'logs' / (mix + '-%j.err'))
        command[-1] = str(wrapper)
        rows.append({'name': 'dolci-sft-' + mix, 'mix': mix, 'gpus': 4,
                     'command': safe_command(command), 'output': str(root / 'sft' / mix),
                     'script_sha256': hashlib.sha256(wrapper.read_bytes()).hexdigest()})
        subprocess.run(['bash', '-n', str(wrapper)], check=True)
    (root / 'logs').mkdir()
    atomic_json(root / 'sft-plan.json', {'runs': rows, 'source_sha256': {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}})
    prepare_phase(root, 'corrected')


def submit_initial(root):
    from modular_rl.suite import submit
    directory = root / 'rl/corrected'
    existing = {j['name'] for j in read(directory / 'jobs.json', [])}
    missing = [r['name'] for r in read(directory / 'launch-plan.json')['runs'] if r['name'] not in existing]
    if missing:
        submit(directory, missing)
    jobs = read(root / 'sft-jobs.json', [])
    for row in read(root / 'sft-plan.json')['runs']:
        if any(j['mix'] == row['mix'] for j in jobs):
            continue
        assert hashlib.sha256(Path(row['command'][-1]).read_bytes()).hexdigest() == row['script_sha256']
        jobs.append(submit_job(row))
        atomic_json(root / 'sft-jobs.json', jobs)


def jobs(root):
    result = [{**r, 'kind': 'sft', 'record': str(root / 'sft-jobs.json')} for r in read(root / 'sft-jobs.json', [])]
    for alias in ('corrected', 'if20', 'if50'):
        for file, kind in (('jobs.json', 'rl'), ('conversion-jobs.json', 'conversion')):
            path = root / 'rl' / alias / file
            result.extend({**r, 'kind': kind, 'alias': alias, 'record': str(path)}
                          for r in read(path, []) if r.get('job_id'))
    return result


def retry(root, row, state):
    attempts = read(root / 'retry-history.json', [])
    if sum(r['name'] == row['name'] for r in attempts) >= 5:
        raise RuntimeError('Retries exhausted for ' + row['name'])
    replacement = submit_job(row)
    value = read(Path(row['record']))
    for old in value:
        if old['job_id'] == row['job_id']:
            old.update(job_id=replacement['job_id'], command=replacement['command'])
    atomic_json(row['record'], value)
    attempts.append({'name': row['name'], 'old_job_id': row['job_id'],
                     'new_job_id': replacement['job_id'], 'failure': state,
                     'utc': datetime.now(timezone.utc).isoformat()})
    atomic_json(root / 'retry-history.json', attempts)


def prepare_conversion(root, alias):
    from modular_rl.suite_conversions import prepare as conversions
    directory = prepare_phase(root, alias)
    path = directory / 'conversion-jobs.json'
    if not path.exists():
        conversions(directory, submit=False, queues=QUEUES)
    records = read(path)
    for row in records:
        if row.get('job_id'):
            continue
        row.update(name=f'dolci-{alias}-convert', gpus=4)
        row['command'] = ['--gres=gpu:4' if c.startswith('--gres=') else c for c in row['command']]
        row.update(submit_job(row))
        atomic_json(path, records)


def completed_report(root):
    from .report import events, scores
    report = {'passed': True, 'sft': [], 'rl': []}
    for row in jobs(root):
        if row['kind'] == 'sft':
            result = read(Path(row['output']) / 'results.json')
            assert result['train_tokens'] == ACTUAL_TOKENS and result['config']['objective'] == 'assistant'
            report['sft'].append({'name': row['name'], 'job_id': row['job_id'], 'result': result})
        if row['kind'] != 'rl': continue
        policy = Path(row['output']) / 'policy'
        complete = read(policy / 'completed.json')
        updates = complete['updates']
        assert complete['completed'] and complete['train_rollouts'] == updates * 8
        assert updates == 1000 or complete['reason'] == 'validation_accuracy'
        state = read(policy / 'ckpt/rlve_checkpoint_state.json')
        assert state['completed'] and state['rollout_id'] == state['evaluated_rollout_id'] == updates - 1
        selection = read(Path(row['record']).parent / 'selection' / row['task'] / 'selection.json')
        assert not set(selection['keys']['train']) & set(selection['keys']['val'])
        train = sorted((policy / 'rollouts/train').glob('*.jsonl'))
        assert [int(p.stem) for p in train] == list(range(updates))
        for path in train:
            samples = events(path)
            assert len(samples) == 8
            assert all(json.loads(s['metadata_json'])['key'] in selection['keys']['train'] for s in samples)
        panels = sorted((policy / 'rollouts/eval').glob('*.jsonl'))
        assert [int(p.stem) for p in panels] == sorted({0, updates - 1, *range(49, updates, 50)})
        curve = [{'train_rollouts': 0 if int(p.stem) == 0 else (int(p.stem) + 1) * 8,
                  **scores(events(p), set(selection['keys']['val']))} for p in panels]
        if updates < 1000:
            assert curve[-1]['pass@1'] >= .98
        report['rl'].append({'name': row['name'], 'task': row['task'], 'job_id': row['job_id'],
                             'curve': curve, 'output': row['output'],
                             'wandb_run_id': (policy / 'wandb_run_id.txt').read_text().strip()})
    assert len(report['sft']) == 2 and len(report['rl']) == 6
    atomic_json(root / 'followup-report.json', report)


def monitor(root):
    from .export import finalize_generation
    from .report import finalize_wandb
    from .storage import retire
    from modular_rl.initializers import load_registry, verify
    from modular_rl.suite import submit
    lock = (root / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while not (root / 'STOP').exists():
        rows = jobs(root)
        status = states(rows)
        failures = []
        for row in rows:
            try:
                os.environ.pop('WANDB_SERVICE', None)
                current = status.get(row['job_id'], {})
                state = current.get('state', '').split()[0]
                if state in TERMINAL_FAILURES:
                    retry(root, row, current)
                elif row['kind'] == 'sft' and state == 'COMPLETED':
                    finalize_generation(Path(row['output']))
                    prepare_conversion(root, row['mix'])
                    retire(Path(row['output']))
                elif row['kind'] == 'conversion' and state == 'COMPLETED':
                    directory = root / 'rl' / row['alias']
                    if not (directory / 'jobs.json').exists():
                        verify(load_registry(directory / 'initializers.json')[row['alias']], full_weights=True)
                        submit(directory)
                elif row['kind'] == 'rl' and state == 'COMPLETED':
                    finalize_wandb(row)
            except Exception:
                message = traceback.format_exc()
                failures.append({'name': row['name'], 'traceback': message})
                print(message, flush=True)
        rows = jobs(root)
        status = states(rows)
        value = {'utc': datetime.now(timezone.utc).isoformat(),
                 'jobs': [{**{k: r.get(k) for k in ('name', 'job_id', 'kind', 'gpus', 'output')},
                           **status.get(r['job_id'], {})} for r in rows],
                 'sft_complete': sum(r['kind'] == 'sft' and status.get(r['job_id'], {}).get('state') == 'COMPLETED' for r in rows),
                 'rl_complete': sum(r['kind'] == 'rl' and status.get(r['job_id'], {}).get('state') == 'COMPLETED' for r in rows),
                 'scratch_free_gib': os.statvfs(root).f_bavail * os.statvfs(root).f_frsize / 2**30}
        atomic_json(root / 'monitor.json', value)
        print(json.dumps({k: v for k, v in value.items() if k != 'jobs'}), flush=True)
        if failures:
            atomic_json(root / 'attention-required.json', failures)
        else:
            (root / 'attention-required.json').unlink(missing_ok=True)
            if value['sft_complete'] == 2 and value['rl_complete'] == 6:
                try:
                    completed_report(root)
                    return
                except Exception:
                    message = traceback.format_exc()
                    print(message, flush=True)
                    atomic_json(root / 'attention-required.json', [{'name': 'final-audit', 'traceback': message}])
        time.sleep(45)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=Path('/nlp/scr/dhei/dolci-mix-20261029'))
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--monitor', action='store_true')
    args = parser.parse_args()
    root, parent = args.root.resolve(), args.parent.resolve()
    if args.prepare: prepare(root, parent)
    if args.submit: submit_initial(root)
    if args.monitor: monitor(root)
