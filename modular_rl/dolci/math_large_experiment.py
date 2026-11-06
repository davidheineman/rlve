"""Capacity-gated SFT/RL verification and full 1B math experiment launchers."""

import argparse
import hashlib
import importlib.util
import json
import os
import math
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .config import ACTUAL_TOKENS, atomic_json, fingerprint, apportion
from .math_large import PARENT, EARLIER

PILOT_TOKENS = 16_777_216
QUEUES = 'jag-standard,miso,sphinx,miso-lo,sphinx-lo,jag-lo,sc-loprio'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_orchestrator_source(repo):
    repo = Path(repo)
    required = ('slime/utils/checkpoint_state.py', 'slime/utils/eval_metrics.py',
                'tools/convert_torch_dist_to_hf.py',
                'slime/backends/megatron_utils/megatron_to_hf/qwen2.py')
    missing = [name for name in required if not (repo / name).exists()]
    if missing:
        raise ValueError('Incomplete frozen launch dependencies: ' + ', '.join(missing))
    subprocess.run([sys.executable, '-c',
                    'import modular_rl.suite; import modular_rl.suite_conversions; '
                    'from slime.utils.checkpoint_state import atomic_write; '
                    'from slime.utils.eval_metrics import pass_at_k'],
                   cwd=repo, env={**os.environ, 'PYTHONPATH': str(repo)}, check=True,
                   capture_output=True, text=True)


def sft_resources(root, *, pilot=False):
    path = Path(root) / 'sft-resources.json'
    requested = json.loads(path.read_text()) if path.exists() else {}
    gpus = 4 if pilot else requested.get('gpus', 4)
    if gpus not in (4, 8):
        raise ValueError('Math midtraining supports four or eight GPUs per node')
    sequences = 32 if pilot else 128
    accumulation = sequences // (gpus * 2)
    return {'gpus': gpus, 'micro_batch': 2, 'grad_accum': accumulation,
            'global_sequences': sequences, 'tokens_per_step': sequences * 2048,
            'cpus': 16, 'memory_gib': 128,
            'allow_24gb': bool(requested.get('allow_24gb', False)) and not pilot,
            'gradient_checkpointing': bool(requested.get('gradient_checkpointing', False)) and not pilot}


def training_startup(root, *, minimum_steps=5):
    root = Path(root)
    config_path, timing_path = root / 'sft/math/config.json', root / 'sft/math/timing.json'
    if not config_path.exists() or not timing_path.exists():
        return None
    config, timing = [json.loads(p.read_text()) for p in (config_path, timing_path)]
    requested = sft_resources(root)
    assert config['world_size'] == requested['gpus']
    assert config['tokens_per_step'] == requested['tokens_per_step']
    assert config['micro_batch'] * config['grad_accum'] * config['world_size'] == 128
    assert config['model']['repo'] == 'Qwen/Qwen3-0.6B-Base'
    steps = timing['train_tokens'] // config['tokens_per_step']
    if steps < minimum_steps:
        return None
    job = json.loads((root / 'sft-job.json').read_text())
    log = root / f"sft-{job['job_id']}.log"
    losses = [float(v) for v in re.findall(r'ce=([\d.eE+-]+)', log.read_text())] if log.exists() else []
    if not losses:
        return None
    assert all(math.isfinite(v) for v in losses)
    proof = {'passed': True, 'job_id': job['job_id'], 'actual_training_steps_observed': steps,
             'world_size': config['world_size'], 'global_sequences': 128,
             'input_tokens': timing['train_tokens'], 'latest_logged_loss': losses[-1],
             'tokens_per_second': timing['tokens_per_s'],
             'remaining_training_hours': timing['remaining_training_hours'],
             'real_curated_data': config['objective'] == 'assistant',
             'raw_base_student': True, 'synthetic': False}
    assert proof['real_curated_data']
    atomic_json(root / 'full-midtraining-startup.json', proof)
    return proof


def extend_gpu_types(path):
    path = Path(path)
    text = path.read_text()
    if 'return "a5000"' not in text:
        assert text.count('    return None\n') == 1
        text = text.replace('    return None\n',
                            '    if re.search(r"\\ba5000\\b", words):\n'
                            '        return "a5000"\n'
                            '    if re.search(r"\\bh100\\b", words):\n'
                            '        return "h100"\n'
                            '    return None\n')
        path.write_text(text)


def pilot_sft_completed(pilot):
    pilot = Path(pilot)
    path = pilot / 'sft/math/results.json'
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    assert result['train_tokens'] == PILOT_TOKENS
    assert result['config']['objective'] == 'assistant'
    assert result['config']['model']['repo'] == 'Qwen/Qwen3-0.6B-Base'
    return (pilot / 'sft/math/model/model.safetensors').exists()


def pilot_checks(completed, curve, loss):
    accuracies = [point['pass@1'] for point in curve]
    return {'rl_completed': completed['completed'] and completed['train_rollouts'] == 128,
            'real_rl_updates': completed['optimizer_updates'] > 0,
            'heldout_loss_improved': loss['nll_change_sft_minus_base'] < 0,
            'no_generation_collapse': bool(accuracies) and min(accuracies) >= .1,
            'no_large_short_rl_regression': bool(accuracies) and accuracies[-1] >= accuracies[0] - .1}


def preflight_data(root):
    root = Path(root).resolve()
    audit = json.loads((root / 'curation-audit.json').read_text())
    assert audit['passed'] and max(audit['source_epochs'].values()) <= 2
    pilot = root / 'preflight'
    if (pilot / 'manifest.json').exists():
        return pilot
    pilot.mkdir(exist_ok=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    labels = PILOT_TOKENS // 2048
    manifest['quotas']['math'] = apportion(labels, manifest['quotas']['math'])
    manifest.update(pool_prediction_tokens=PILOT_TOKENS, require_unique_blocks=True,
                    training_tokens=PILOT_TOKENS, train_steps=PILOT_TOKENS // 65536,
                    parent_corpus=str(root), preflight=True)
    earlier = json.loads((EARLIER / 'final/manifest.json').read_text())
    manifest['splits'] = {'val': earlier['splits']['val']}
    manifest.pop('signature', None)
    manifest['signature'] = fingerprint(manifest)
    (pilot / 'sources').mkdir(exist_ok=True)
    for path in (root / 'sources').iterdir():
        (pilot / 'sources' / path.name).symlink_to(path)
    for extension in ('.bin', '.mask', '.counts.npy'):
        (pilot / 'sources' / ('val' + extension)).symlink_to(EARLIER / ('final/sources/val' + extension))
    for name in ('model', 'eval', 'eval_inloop_64'):
        (pilot / name).symlink_to(PARENT / name)
    shutil.copy2(PARENT / 'model_revision.json', pilot / 'model_revision.json')
    atomic_json(pilot / 'manifest.json', manifest)
    from .data import MixedBlocks
    stream = MixedBlocks(pilot, 'math')
    for label in range(len(stream.names)):
        values = stream.indices[stream.labels == label]
        assert len(values) == len(set(values.tolist()))
    return pilot


def trainer(root, *, pilot=False):
    from .math_billion import prepare_sft
    root = Path(root).resolve()
    resources = sft_resources(root, pilot=pilot)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['repo'] == 'math-large-20261101' and manifest['maximum_epochs'] <= 2
    unsigned = {k: v for k, v in manifest.items() if k != 'signature'}
    assert fingerprint(unsigned) == manifest['signature']
    if pilot:
        assert manifest['preflight'] and manifest['require_unique_blocks']
    else:
        audit = json.loads((root / 'curation-audit.json').read_text())
        verification = json.loads((root / 'preflight/validation.json').read_text())
        review = json.loads((root / 'REVIEWED.json').read_text())
        assert audit['passed'] and max(audit['source_epochs'].values()) <= 2
        assert audit['input_token_slots'] == ACTUAL_TOKENS
        assert max(audit['maximum_block_exposures'].values()) <= 2
        assert verification['passed'] and review['passed']
    prepare_sft(root)  # The already-audited manifest bypasses its legacy pool builder.
    marker = root / 'large-trainer-prepared.json'
    if marker.exists():
        return
    train = root / 'sft-source/train.py'
    text = train.read_text()
    text = text.replace('"dataset_repo": "math-refresh-1b-20261101",',
                        '"dataset_repo": "math-large-20261101",')
    name = 'math-large-sft-preflight' if pilot else 'math-large-sft-1b'
    text = text.replace('name="math-refresh-sft-1b"', 'name=' + repr(name))
    train.write_text(text)
    extend_gpu_types(root / 'sft-source/gpu_policy.py')
    script = root / 'sft-source/math.sbatch'
    lines = []
    values = {'NANO_WANDB_GROUP': 'math-large-preflight' if pilot else 'math-large-1b',
              'NANO_VARIANT': 'math-large', 'NANO_TOKENS': str(PILOT_TOKENS if pilot else 1_000_000_000),
              'NANO_GPUS': str(resources['gpus']), 'NANO_MICRO_BATCH': '2',
              'NANO_GRAD_ACCUM': str(resources['grad_accum'])}
    if resources['gradient_checkpointing']:
        values['NANO_GRADIENT_CHECKPOINTING'] = '1'
    if resources['allow_24gb']:
        values['NANO_ALLOWED_GPU_TYPES'] = 'a5000,l40s,a100,h100,h200,rtx6000ada,a6000'
    if pilot:
        values.update(NANO_GRAD_ACCUM='4', NANO_SKIP_EVAL='1', NANO_FULL_EVAL_AT_START='0',
                      NANO_CHECKPOINT_EVERY_TOKENS='1000000000')
    for line in script.read_text().splitlines():
        if line.startswith('export '):
            key = line[7:].split('=', 1)[0]
            if key in values:
                line = 'export ' + key + '=' + shlex.quote(values[key])
        lines.append(line)
    script.write_text('\n'.join(lines) + '\n')
    subprocess.run(['bash', '-n', str(script)], check=True)
    job_path = root / 'sft-job.json'
    job = json.loads(job_path.read_text())
    job['name'] = name
    job['command'] = [v.replace('--job-name=math-refresh-sft-1b', '--job-name=' + name) for v in job['command']]
    replacements = {'--gres=': f"--gres=gpu:{resources['gpus']}",
                    '--cpus-per-task=': f"--cpus-per-task={resources['cpus']}",
                    '--mem=': f"--mem={resources['memory_gib']}G"}
    for prefix, replacement in replacements.items():
        job['command'] = [replacement if item.startswith(prefix) else item for item in job['command']]
    if resources['allow_24gb']:
        job['command'] = ['--constraint=24G|40G|48G|80G|141G' if item.startswith('--constraint=')
                          else item for item in job['command']]
    policy_spec = importlib.util.spec_from_file_location('math_sft_gpu_policy', root / 'sft-source/gpu_policy.py')
    policy = importlib.util.module_from_spec(policy_spec)
    policy_spec.loader.exec_module(policy)
    policy.UNHEALTHY_NODES.update({node: 'Existing production startup exclusion'
                                  for node in ('viscam10', 'jagupard33', 'jagupard34', 'pasteur6')})
    allowed = 'a5000,l40s,a100,h100,h200,rtx6000ada,a6000' if resources['allow_24gb'] else 'l40s,a100,h100,h200,rtx6000ada,a6000'
    gpu_inventory = policy.cluster_policy(tuple(allowed.split(',')))
    job['command'] = ['--exclude=' + ','.join(gpu_inventory['excluded_nodes']) if item.startswith('--exclude=')
                      else item for item in job['command']]
    job['gpu_policy'] = gpu_inventory
    job['script_sha256'] = sha256(script)
    job['source_sha256']['train.py'] = sha256(train)
    job['source_sha256']['gpu_policy.py'] = sha256(root / 'sft-source/gpu_policy.py')
    job.update(input_tokens=PILOT_TOKENS if pilot else ACTUAL_TOKENS, capacity_gate_passed=True,
               unique_pool_epoch_limit=2, gpus=resources['gpus'],
               global_sequences=resources['global_sequences'], resources=resources)
    atomic_json(job_path, job)
    atomic_json(marker, {'name': name, 'script': str(script), 'sha256': sha256(script)})


def run_pilot(root):
    root = Path(root).resolve()
    pilot = preflight_data(root)
    if (pilot / 'validation.json').exists():
        return
    if len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) != 4:
        raise ValueError('The preflight requires a four-GPU allocation')
    trainer(pilot, pilot=True)
    if not pilot_sft_completed(pilot):
        subprocess.run(['bash', str(pilot / 'sft-source/math.sbatch')], check=True)
    from .export import finalize_generation
    finalize_generation(pilot / 'sft/math', expected_tokens=PILOT_TOKENS)
    # Reuse the independently tested paired-loss probe on the same 32 held-out
    # blocks. Every one of its 512 questions was excluded during curation.
    (pilot / 'pilot').mkdir(exist_ok=True)
    link = pilot / 'pilot/sft'
    if not link.exists():
        link.symlink_to(pilot / 'sft')
    if not (pilot / 'pilot-heldout-loss.json').exists():
        subprocess.run(['/juice5b/u/dhei/tatsu/nano-omni/.venv-midtraining/bin/python', '-u',
                        '-m', 'modular_rl.dolci.math_refresh_loss_probe', '--root', str(pilot)], check=True)
    from .math_billion import prepare_rl
    directory = prepare_rl(pilot, 'math', expected_training_tokens=PILOT_TOKENS,
                           train_steps=16, evaluation_interval=8,
                           run_name='math-large-rl-preflight', group='math-large-preflight')
    plan_path = directory / 'launch-plan.json'
    plan = json.loads(plan_path.read_text())
    for job in plan['runs']:
        script = Path(job['command'][-1])
        body = script.read_text()
        body = body.replace('MODULAR_RL_CHECKPOINT_INTERVAL=8\n', 'MODULAR_RL_CHECKPOINT_INTERVAL=1000\n')
        script.write_text(body)
        job['script_sha256'] = sha256(script)
    atomic_json(plan_path, plan)
    from modular_rl.suite_conversions import prepare as conversions
    if not (directory / 'conversion-jobs.json').exists():
        conversions(directory, submit=False)
    conversion, = json.loads((directory / 'conversion-jobs.json').read_text())
    if not (pilot / 'initializers/math/conversion-audit.json').exists():
        subprocess.run(['bash', conversion['command'][-1]], check=True)
    from .storage import retire
    retire(pilot / 'sft/math', expected_tokens=PILOT_TOKENS)
    job, = plan['runs']
    subprocess.run(['bash', job['command'][-1]], check=True)
    policy = Path(job['output']) / 'policy'
    completed = json.loads((policy / 'completed.json').read_text())
    from .report import events, scores
    selection = json.loads((directory / 'selection/math/selection.json').read_text())
    curve = [{'rollout_id': int(p.stem), **scores(events(p), set(selection['keys']['val']))}
             for p in sorted((policy / 'rollouts/eval').glob('*.jsonl'))]
    for p in (policy / 'rollouts/train').glob('*.jsonl'):
        for row in events(p):
            metadata = json.loads(row['metadata_json'])
            prefix = metadata.get('_modular_v2_forcing', {}).get('prefix_tokens', 0)
            assert 0 < row['response_tokens'] - prefix <= 2048
    loss = json.loads((pilot / 'pilot-heldout-loss.json').read_text())
    checks = pilot_checks(completed, curve, loss)
    proof = {'passed': all(checks.values()), 'checks': checks, 'sft_tokens': PILOT_TOKENS,
             'sft_global_sequences': 32, 'sft_question_repetitions': False,
             'rl': completed, 'curve': curve, 'heldout_loss': loss,
             'conversion': json.loads((pilot / 'initializers/math/conversion-audit.json').read_text()),
             'scope': 'Small data/format/integration check; full 8K endpoint remains to be measured.'}
    atomic_json(pilot / 'validation.json', proof)
    if not proof['passed']:
        raise ValueError('Preflight quality/integration gate did not pass: ' + repr(checks))


def prepare_pilot_job(root):
    root = Path(root).resolve()
    repo = Path(__file__).resolve().parents[2]
    validate_orchestrator_source(repo)
    script = root / 'preflight.sbatch'
    script.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH={repo}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL=https://api.forge.coreweave.com
unset WANDB_SERVICE
cd {repo}
exec /nlp/scr/dhei/tinker-mopd/venv-rl/bin/python -u -m modular_rl.dolci.math_large_experiment --root {root} --pilot
''')
    cmd = ['sbatch', '--parsable', '--account=miso', '--partition=' + QUEUES,
           '--nodes=1', '--ntasks=1', '--gres=gpu:4', '--constraint=40G|48G|80G|141G',
           '--cpus-per-task=28', '--mem=160G', '--time=02:00:00', '--requeue',
           '--exclude=viscam10,jagupard33,jagupard34,pasteur6', '--job-name=math-large-preflight',
           '--output=' + str(root / 'preflight-%j.log'), str(script)]
    atomic_json(root / 'preflight-job.json', {'name': 'math-large-preflight', 'command': cmd,
                'gpus': 4, 'script_sha256': sha256(script), 'status': 'prepared'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.pilot:
        run_pilot(args.root)
    else:
        trainer(args.root)
