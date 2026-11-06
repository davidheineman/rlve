"""Run a small four-GPU SFT -> conversion -> math RL integration preflight.

This validates the training path, not whether a full epoch will improve MATH.
The full training run is prepared for review, not automatically launched.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from .config import atomic_json, QUEUES
from .math_refresh_build import PARENT

PILOT_STEPS = 4
PILOT_TOKENS = PILOT_STEPS * 262144


def prepare_trainer(root):
    root = Path(root)
    source = root / 'sft-source'
    if source.exists(): return source
    shutil.copytree(PARENT / 'sft-source', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.sbatch'))
    shutil.copy2(PARENT / 'sft-source/run.sbatch', source / 'run.sbatch')
    repo = Path(__file__).resolve().parents[2]
    for filename in ('math_epoch.py', 'config.py'):
        shutil.copy2(repo / 'modular_rl/dolci' / filename, source / 'modular_rl/dolci' / filename)
    train = source / 'train.py'; text = train.read_text()
    old = 'from modular_rl.dolci.data import MixedBlocks'
    if text.count(old) != 1: raise ValueError('Inspect the reference trainer import')
    text = text.replace(old, 'from modular_rl.dolci.math_epoch import OneEpochBlocks as MixedBlocks')
    text = text.replace('"dataset_repo": "allenai/Dolci-Instruct-SFT",', '"dataset_repo": "math-refresh-20261101",')
    text = text.replace('stream.loss_tokens(0, len(stream.labels))', 'stream.loss_tokens(0, steps * per_step // seq_len)')
    text = text.replace('name=f"dolci-{args.mix}-{budget}"', 'name=f"math-refresh-{args.mix}-{budget}"')
    train.write_text(text)
    return source


def sft_script(root, pilot=True):
    source = prepare_trainer(root)
    environment = {}
    for line in (PARENT / 'sft-source/math.sbatch').read_text().splitlines():
        if line.startswith('export '):
            key, value = line[7:].split('=', 1); environment[key] = shlex.split(value)[0]
    manifest = json.loads((root / 'manifest.json').read_text())
    environment.update(NANO_DATA_ROOT=str(root), NANO_OUT_ROOT=str(root / ('pilot/sft' if pilot else 'sft')),
                       NANO_TRAIN_SCRIPT=str(source / 'train.py'), NANO_MIX='math',
                       NANO_TOKENS=str(PILOT_TOKENS if pilot else manifest['training_tokens']),
                       NANO_GPUS='4', NANO_MICRO_BATCH='2', NANO_GRAD_ACCUM='16',
                       NANO_GRADIENT_CHECKPOINTING='0', PYTHONPATH=str(source),
                       NANO_WANDB_GROUP='math-refresh-preflight' if pilot else 'math-refresh-one-epoch')
    if pilot:
        environment.update(NANO_SKIP_EVAL='1', NANO_FULL_EVAL_AT_START='0')
    path = root / ('pilot/sft.sbatch' if pilot else 'sft-source/math.sbatch')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/usr/bin/env bash\nset -euo pipefail\n' +
                   ''.join(f'export {k}={shlex.quote(v)}\n' for k, v in environment.items()) +
                   'exec bash ' + shlex.quote(str(source / 'run.sbatch')) + '\n')
    subprocess.run(['bash', '-n', str(path)], check=True)
    return path


def prepare_rl(root):
    from modular_rl.initializers import inspect_hf
    from modular_rl.suite import prepare
    from modular_rl.suite_conversions import prepare as conversions
    directory = root / 'pilot/rl'
    if not (directory / 'launch-plan.json').exists():
        registry = root / 'pilot/initializers.json'
        atomic_json(registry, {'base': json.loads((PARENT / 'rl/base/initializers.json').read_text())['base'],
                             'math': inspect_hf('math', root / 'pilot/sft/math/model', root / 'pilot/native/math',
                                                expected_training_tokens=PILOT_TOKENS)})
        prepare(Path(__file__).resolve().parents[2], Path('/nlp/scr/dhei/modular-mopd-rl/v2'), directory,
                root / 'pilot/sft', PARENT / 'rl/base/selection', policy_gpus=4,
                initializers_from=registry, selection_from=PARENT / 'rl/base/selection',
                runs=[('math-refresh-pilot-rl', 'math', 'math')], queues=QUEUES,
                group='math-refresh-preflight', allow_unconverted=True, time_limit='02:00:00',
                frozen_from=PARENT / 'rl/base', data_from=PARENT / 'rl-data',
                resize_frozen_policy=True, train_steps=4, evaluation_interval=2)
        plan = json.loads((directory / 'launch-plan.json').read_text())
        # Only a final checkpoint is needed for this four-step integration test.
        # All optimizer updates and objective settings remain production defaults.
        for row in plan['runs']:
            script = Path(row['command'][-1]); body = script.read_text()
            body = body.replace('MODULAR_RL_CHECKPOINT_INTERVAL=2\n', 'MODULAR_RL_CHECKPOINT_INTERVAL=1000\n')
            script.write_text(body); row['script_sha256'] = hashlib.sha256(script.read_bytes()).hexdigest()
        atomic_json(directory / 'launch-plan.json', plan)
    if not (directory / 'conversion-jobs.json').exists(): conversions(directory, queues=QUEUES)
    return directory


def verify(root):
    from .report import events, scores
    sft = root / 'pilot/sft/math'
    result = json.loads((sft / 'results.json').read_text())
    if result['train_tokens'] != PILOT_TOKENS: raise ValueError('Incorrect pilot SFT budget')
    import torch
    from safetensors import safe_open
    key = 'model.layers.0.self_attn.q_proj.weight'
    with safe_open(sft / 'model/model.safetensors', framework='pt') as trained, \
            safe_open(PARENT / 'model/model.safetensors', framework='pt') as original:
        before, after = original.get_tensor(key).float(), trained.get_tensor(key).float()
        change = float((after - before).norm())
        if not change > 0 or not torch.isfinite(after).all(): raise ValueError('SFT did not update finite parameters')
    policy = root / 'pilot/rl/runs/math-refresh-pilot-rl/policy'
    complete = json.loads((policy / 'completed.json').read_text())
    if not complete['completed'] or complete['train_rollouts'] != 32 or complete['optimizer_updates'] < 1:
        raise ValueError('RL did not perform a real policy-gradient update')
    selection = json.loads((root / 'pilot/rl/selection/math/selection.json').read_text())
    panels = sorted((policy / 'rollouts/eval').glob('*.jsonl'))
    curve = [{'rollout_id': int(p.stem), **scores(events(p), set(selection['keys']['val']))} for p in panels]
    for p in (policy / 'rollouts/train').glob('*.jsonl'):
        for row in events(p):
            md = json.loads(row['metadata_json']); prefix = md.get('_modular_v2_forcing', {}).get('prefix_tokens', 0)
            if not 0 < row['response_tokens'] - prefix <= 2048: raise ValueError('RL generation cap changed')
    proof = {'passed': True, 'sft_steps': PILOT_STEPS, 'sft_input_token_slots': PILOT_TOKENS,
             'sft_parameter_change_l2': change, 'rl': complete, 'rl_eval_curve': curve,
             'heldout_math_loss': json.loads((root / 'pilot-heldout-loss.json').read_text()),
             'rl_response_cap': 2048, 'base_model': 'Qwen/Qwen3-0.6B-Base',
             'scope': 'Integration preflight; not a full-epoch performance claim.'}
    atomic_json(root / 'pilot-validation.json', proof)
    print('PILOT_PASSED', json.dumps(proof), flush=True)


def run(root):
    root = Path(root).resolve()
    if (root / 'pilot-validation.json').exists(): return
    os.environ.pop('WANDB_SERVICE', None)
    if len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) != 4:
        raise ValueError('Allocate four GPUs to the SFT -> RL pilot')
    if not (root / 'model').exists(): (root / 'model').symlink_to(PARENT / 'model')
    if not (root / 'model_revision.json').exists(): shutil.copy2(PARENT / 'model_revision.json', root / 'model_revision.json')
    for name in ('eval', 'eval_inloop_64'):
        if not (root / name).exists(): (root / name).symlink_to(PARENT / name)
    script = sft_script(root)
    subprocess.run(['bash', str(script)], check=True)
    from .export import finalize_generation
    from .storage import retire
    finalize_generation(root / 'pilot/sft/math', expected_tokens=PILOT_TOKENS)
    subprocess.run(['/juice5b/u/dhei/tatsu/nano-omni/.venv-midtraining/bin/python', '-u',
                    '-m', 'modular_rl.dolci.math_refresh_loss_probe', '--root', str(root)], check=True)
    directory = prepare_rl(root)
    conversion, = json.loads((directory / 'conversion-jobs.json').read_text())
    subprocess.run(['bash', conversion['command'][-1]], check=True)
    retire(root / 'pilot/sft/math', expected_tokens=PILOT_TOKENS)
    job, = json.loads((directory / 'launch-plan.json').read_text())['runs']
    subprocess.run(['bash', job['command'][-1]], check=True)
    verify(root)
    script = sft_script(root, pilot=False)
    manifest = json.loads((root / 'manifest.json').read_text())
    atomic_json(root / 'full-sft-job.json', {
        'status': 'prepared-not-submitted', 'gpus': 4,
        'training_tokens': manifest['training_tokens'], 'steps': manifest['train_steps'],
        'one_epoch': True, 'command': ['sbatch', '--parsable', '--account=miso',
            '--partition=' + QUEUES, '--nodes=1', '--ntasks=1', '--gres=gpu:4',
            '--constraint=40G|48G|80G|141G', '--cpus-per-task=28', '--mem=160G',
            '--time=06:00:00', '--requeue', '--signal=B:USR1@180',
            '--exclude=viscam10,jagupard33,jagupard34,pasteur6', '--job-name=math-refresh-sft',
            '--output=' + str(root / 'logs/sft-%j.log'), str(script)],
        'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
        'wandb_project': 'modular-mopd-midtrain-initial'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); run(args.root)
