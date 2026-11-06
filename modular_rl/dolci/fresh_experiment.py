"""Monitor paired GRPO/MaxRL and a fresh unique-data SFT -> code RL run."""

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

from .config import ACTUAL_TOKENS, QUEUES, atomic_json
from .followup import safe_command, submit_job
from .watch import read, states, TERMINAL_FAILURES


def prepare_sft(root, parent):
    if (root / 'sft-plan.json').exists(): return
    source = root / 'sft-source'
    shutil.copytree(parent / 'sft-source', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.sbatch'))
    shutil.copy2(parent / 'sft-source/run.sbatch', source / 'run.sbatch')
    repo = Path(__file__).resolve().parents[2]
    shutil.copy2(repo / 'modular_rl/dolci/data.py', source / 'modular_rl/dolci/data.py')
    train = source / 'train.py'
    text = train.read_text()
    text = text.replace('"dataset_repo": "allenai/Dolci-Instruct-SFT",',
                        '"dataset_repo": "dolci-fresh-code-unique",')
    text = text.replace('name=f"dolci-{args.mix}-{budget}"', 'name=f"dolci-fresh-{args.mix}-{budget}"')
    train.write_text(text)
    environment = {}
    for line in (parent / 'sft-source/code.sbatch').read_text().splitlines():
        if line.startswith('export '):
            key, value = line[7:].split('=', 1)
            environment[key] = shlex.split(value)[0]
    environment.update(NANO_DATA_ROOT=str(root), NANO_OUT_ROOT=str(root / 'sft'),
                       NANO_TRAIN_SCRIPT=str(train), PYTHONPATH=str(source),
                       NANO_WANDB_GROUP='dolci-fresh-code-1b', NANO_GPUS='4',
                       NANO_MICRO_BATCH='1', NANO_GRAD_ACCUM='8', NANO_GRADIENT_CHECKPOINTING='1')
    wrapper = source / 'code.sbatch'
    wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\n' +
                      ''.join(f'export {k}={shlex.quote(v)}\n' for k, v in environment.items()) +
                      'exec bash ' + shlex.quote(str(source / 'run.sbatch')) + '\n')
    command = list(read(parent / 'sft-jobs.json')[0]['command'])
    for i, value in enumerate(command):
        if value.startswith('--partition='): command[i] = '--partition=' + QUEUES
        elif value.startswith('--job-name='): command[i] = '--job-name=dolci-sft-fresh-code'
        elif value.startswith('--output='): command[i] = '--output=' + str(root / 'logs/sft-%j.out')
        elif value.startswith('--error='): command[i] = '--error=' + str(root / 'logs/sft-%j.err')
        elif value.startswith('--mem='): command[i] = '--mem=96G'
        elif value.startswith('--gres='): command[i] = '--gres=gpu:4'
    command[-1] = str(wrapper)
    subprocess.run(['bash', '-n', str(wrapper)], check=True)
    atomic_json(root / 'sft-plan.json', {'runs': [{'name': 'dolci-sft-fresh-code', 'mix': 'code', 'gpus': 4,
        'command': safe_command(command), 'output': str(root / 'sft/code'),
        'script_sha256': hashlib.sha256(wrapper.read_bytes()).hexdigest()}],
        'source_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source.rglob('*') if p.is_file()},
        'input_tokens': ACTUAL_TOKENS, 'seq_len': 8192, 'tokens_per_step': 262144,
        'lr': .0001, 'objective': 'assistant', 'no_example_repetition': True})


def prepare_fresh_rl(root, parent):
    from modular_rl.initializers import inspect_hf
    from modular_rl.suite import prepare
    from modular_rl.suite_conversions import prepare as conversions
    directory = root / 'rl/fresh'
    if not (directory / 'launch-plan.json').exists():
        specs = {'base': read(parent / 'rl/base/initializers.json')['base'],
                 'code': inspect_hf('code', root / 'sft/code/model', root / 'initializers/code')}
        registry = root / 'fresh-initializers.json'
        atomic_json(registry, specs)
        prepare(Path(__file__).resolve().parents[2], Path('/nlp/scr/dhei/modular-mopd-rl/v2'), directory,
                root / 'sft', root / 'selection', policy_gpus=4, initializers_from=registry,
                selection_from=root / 'selection', runs=[('dolci-fresh-code-grpo', 'code', 'code')],
                queues=QUEUES, group='dolci-fresh-code-1b', allow_unconverted=True,
                time_limit='08:00:00', frozen_from=parent / 'rl/base', data_from=parent / 'rl-data',
                clean_code_rewards=True, resize_frozen_policy=True)
        plan = read(directory / 'launch-plan.json')
        for row in plan['runs']: row['command'] = safe_command(row['command'])
        atomic_json(directory / 'launch-plan.json', plan)
    if not (directory / 'conversion-jobs.json').exists(): conversions(directory, queues=QUEUES)
    records = read(directory / 'conversion-jobs.json')
    for row in records:
        if row.get('job_id'): continue
        row.update(name='dolci-fresh-convert', gpus=4)
        row['command'] = ['--gres=gpu:4' if c.startswith('--gres=') else c for c in row['command']]
        row.update(submit_job(row)); atomic_json(directory / 'conversion-jobs.json', records)


def select_sft_memory_mode(root):
    """Use the faster mode only after a real four-GPU 8K optimizer preflight."""
    if (root / 'sft-jobs.json').exists(): return
    directory = root / 'preflight-no-checkpoint/outputs/code'
    result, benchmark = read(directory / 'results.json'), read(directory / 'benchmark.json')
    if not result or not benchmark: return
    config = result['config']
    if not (config['world_size'] == 4 and config['seq_len'] == 8192
            and config['tokens_per_step'] == 262144 and config['steps'] == 4
            and not config['gradient_checkpointing'] and result['train_tokens'] == 1048576
            and benchmark['peak_memory_bytes'] < 32 * 2**30):
        return
    wrapper = root / 'sft-source/code.sbatch'
    before = wrapper.read_text()
    if 'export NANO_GRADIENT_CHECKPOINTING=1\n' not in before: return
    after = before.replace('export NANO_GRADIENT_CHECKPOINTING=1\n',
                           'export NANO_GRADIENT_CHECKPOINTING=0\n')
    wrapper.write_text(after)
    plan = read(root / 'sft-plan.json')
    plan['runs'][0]['script_sha256'] = hashlib.sha256(wrapper.read_bytes()).hexdigest()
    plan['source_sha256']['code.sbatch'] = plan['runs'][0]['script_sha256']
    plan['gradient_checkpointing'] = False
    atomic_json(root / 'sft-plan.json', plan)
    atomic_json(root / 'sft-memory-preflight.json', {
        'passed': True, 'gradient_checkpointing': False, 'benchmark': benchmark,
        'reason': 'Four real optimizer steps fit with at least 8 GiB margin on every eligible GPU.',
        'global_batch_optimizer_objective_unchanged': True})


def jobs(root):
    result = [{**read(root / 'data-job.json'), 'kind': 'data', 'record': str(root / 'data-job.json')}]
    result.extend({**r, 'kind': 'sft', 'record': str(root / 'sft-jobs.json')} for r in read(root / 'sft-jobs.json', []))
    for phase in ('paired', 'fresh'):
        for file, kind in (('jobs.json', 'rl'), ('conversion-jobs.json', 'conversion')):
            p = root / 'rl' / phase / file
            result.extend({**r, 'kind': kind, 'phase': phase, 'record': str(p)} for r in read(p, []) if r.get('job_id'))
    return result


def retry(root, row, state):
    history = read(root / 'retry-history.json', [])
    if sum(r['name'] == row['name'] for r in history) >= 5:
        raise RuntimeError('Retry budget exhausted for ' + row['name'])
    replacement = submit_job(row)
    p = Path(row['record']); records = read(p)
    if isinstance(records, dict): records = replacement
    else:
        for old in records:
            if old['job_id'] == row['job_id']: old.update(job_id=replacement['job_id'], command=replacement['command'])
    atomic_json(p, records)
    history.append({'name': row['name'], 'old_job_id': row['job_id'], 'new_job_id': replacement['job_id'],
                    'state': state, 'utc': datetime.now(timezone.utc).isoformat()})
    atomic_json(root / 'retry-history.json', history)


def report(root):
    from .report import events, scores
    result = {'passed': True, 'sft': read(root / 'sft/code/results.json'), 'rl': []}
    assert result['sft']['train_tokens'] == ACTUAL_TOKENS
    for row in jobs(root):
        if row['kind'] != 'rl': continue
        policy = Path(row['output']) / 'policy'; complete = read(policy / 'completed.json')
        assert complete['completed'] and complete['train_rollouts'] == complete['updates'] * 8
        assert complete['updates'] == 1000 or complete['reason'] == 'validation_accuracy'
        selection = read(Path(row['record']).parent / 'selection/code/selection.json')
        assert selection['validation_contract_version'] == 'mbpp-contracts-20261101'
        train = sorted((policy / 'rollouts/train').glob('*.jsonl'))
        assert [int(p.stem) for p in train] == list(range(complete['updates']))
        panels = sorted((policy / 'rollouts/eval').glob('*.jsonl'))
        assert int(panels[-1].stem) == complete['updates'] - 1
        curve = [{'train_rollouts': 0 if int(p.stem) == 0 else (int(p.stem) + 1) * 8,
                  **scores(events(p), set(selection['keys']['val']))} for p in panels]
        result['rl'].append({'name': row['name'], 'algorithm': row.get('algorithm', 'grpo'),
                             'job_id': row['job_id'], 'curve': curve, 'output': row['output']})
    assert len(result['rl']) == 7
    atomic_json(root / 'final-report.json', result)


def monitor(root, parent):
    from modular_rl.suite import submit
    from .export import finalize_generation
    from .storage import retire
    from .report import finalize_wandb
    lock = (root / 'controller.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prepare_sft(root, parent)
    while not (root / 'STOP').exists():
        failures = []
        try:
            select_sft_memory_mode(root)
            paired = root / 'rl/paired'
            # Storage, not another training run, is the only launch gate. Submit
            # every independent pair together once old optimizer retirement
            # has left room for final states and checkpoint replacement files.
            if (paired / 'launch-plan.json').exists() and shutil.disk_usage(root).free >= 85 * 2**30:
                existing = {r['name'] for r in read(paired / 'jobs.json', [])}
                missing = [r['name'] for r in read(paired / 'launch-plan.json')['runs'] if r['name'] not in existing]
                if missing: submit(paired, missing)
            rows = jobs(root); status = states(rows)
            for row in rows:
                try:
                    os.environ.pop('WANDB_SERVICE', None)
                    state = status.get(row['job_id'], {}).get('state', '').split()[0]
                    if state in TERMINAL_FAILURES: retry(root, row, state)
                    elif row['kind'] == 'data' and state == 'COMPLETED' and not (root / 'sft-jobs.json').exists():
                        manifest = read(root / 'manifest.json')
                        assert manifest['require_unique_blocks'] and manifest['pool_prediction_tokens'] == ACTUAL_TOKENS
                        plan = read(root / 'sft-plan.json')
                        for name, h in plan['source_sha256'].items():
                            assert hashlib.sha256((root / 'sft-source' / name).read_bytes()).hexdigest() == h
                        atomic_json(root / 'sft-jobs.json', [submit_job(plan['runs'][0])])
                    elif row['kind'] == 'sft' and state == 'COMPLETED':
                        finalize_generation(Path(row['output'])); prepare_fresh_rl(root, parent); retire(Path(row['output']))
                    elif row['kind'] == 'conversion' and state == 'COMPLETED':
                        phase = root / 'rl' / row['phase']
                        if not (phase / 'jobs.json').exists(): submit(phase)
                    elif row['kind'] == 'rl' and state == 'COMPLETED': finalize_wandb(row)
                except Exception:
                    failures.append({'name': row['name'], 'traceback': traceback.format_exc()})
            rows = jobs(root); status = states(rows)
            value = {'utc': datetime.now(timezone.utc).isoformat(),
                     'jobs': [{**{k: r.get(k) for k in ('name', 'job_id', 'kind', 'gpus', 'output')}, **status.get(r['job_id'], {})} for r in rows],
                     'scratch_free_gib': shutil.disk_usage(root).free / 2**30,
                     'paired_submission': 'submitted' if (paired / 'jobs.json').exists() else 'waiting_for_85_gib_free'}
            atomic_json(root / 'monitor.json', value)
            if not failures and len([r for r in rows if r['kind'] == 'rl']) == 7 and all(
                    status.get(r['job_id'], {}).get('state') == 'COMPLETED' for r in rows):
                report(root); return
        except Exception:
            failures.append({'name': 'controller', 'traceback': traceback.format_exc()})
        if failures:
            atomic_json(root / 'attention-required.json', failures)
            for failure in failures: print(failure['name'], failure['traceback'], flush=True)
        else: (root / 'attention-required.json').unlink(missing_ok=True)
        time.sleep(45)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=Path('/nlp/scr/dhei/dolci-mix-20261029'))
    args = parser.parse_args()
    monitor(args.root.resolve(), args.parent.resolve())
