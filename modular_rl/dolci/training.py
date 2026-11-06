"""Freeze the proven 1B trainer, adding assistant masks and explicit dataset provenance."""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from .config import ACTUAL_TOKENS, DATASET, QUEUES, STEPS, TOKENS_PER_STEP, atomic_json


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Inspect upstream trainer before patching {old[:100]!r}')
    return text.replace(old, new)


def trainer_source(original):
    text = replace_once(original, 'from data import MixedBlocks', 'from modular_rl.dolci.data import MixedBlocks')
    text = replace_once(text, 'choices=["code", "math", "chat", "web"]',
                        'choices=["baseline", "code", "math", "chat", "web", "web-knowledge"]')
    text = replace_once(text, 'def forward(self, x, y):', 'def forward(self, x, y, normalization=None):')
    text = replace_once(text, 'return loss / targets.numel()',
                        'return loss / (normalization if normalization is not None else targets.ne(-100).sum().clamp_min(1))')
    text = replace_once(text, '        for micro in range(args.grad_accum):',
                        '        loss_normalization = (stream.loss_tokens(step * args.grad_accum * world * args.micro_batch,\n'
                        '                                  args.grad_accum * world * args.micro_batch) / (world * args.grad_accum)\n'
                        '                              if stream is not None else None)\n'
                        '        for micro in range(args.grad_accum):')
    text = replace_once(text, 'loss = ddp(x, y)', 'loss = ddp(x, y, normalization=loss_normalization)')
    text = replace_once(text, '"dolmino_revision": stream.manifest["revision"] if stream else None,',
                        '"dataset_revision": stream.manifest["revision"] if stream else None,\n'
                        '              "dataset_repo": "allenai/Dolci-Instruct-SFT",\n'
                        '              "objective": stream.manifest["objective"] if stream else "smoke",\n'
                        '              "actual_supervised_tokens": stream.loss_tokens(0, len(stream.labels)) if stream else 0,\n'
                        '              "natural_definition": stream.manifest.get("natural_definitions", {}).get(args.mix, stream.manifest["natural_definition"]) if stream else None,')
    text = replace_once(text, 'name=f"qwen3-0.6b-{args.mix}-{budget}{label}"', 'name=f"dolci-{args.mix}-{budget}"')
    text = replace_once(text, 'tags=["dolmino", "midtraining", args.mix] + ([args.variant, "code-100m-ablations"] if args.variant else [])',
                        'tags=["dolci", "sft", args.mix, config["objective"]]')
    return text


def prepare(root, rlve, nano):
    root, rlve, nano = Path(root).resolve(), Path(rlve).resolve(), Path(nano).resolve()
    if (root / 'sft-jobs.json').exists():
        raise ValueError('Submitted SFT source is immutable; use a new experiment root')
    original = nano / 'experiments/midtraining'
    source = root / 'sft-source'
    source.mkdir(exist_ok=True)
    train = trainer_source((original / 'train.py').read_text())
    hashes = {}
    for name in ('config.py', 'mfu.py', 'gpu_policy.py', 'run.sbatch', 'validate_smoke.py', 'requirements.txt'):
        data = (original / name).read_bytes()
        if name == 'gpu_policy.py':
            policy_text = replace_once(data.decode(),
                'ALLOWED_GPU_TYPES = ("l40s", "a100", "h200", "rtx6000ada")',
                'ALLOWED_GPU_TYPES = ("l40s", "a100", "h200", "rtx6000ada", "a6000")')
            policy_text = replace_once(policy_text, '    if "rtx6000ada" in compact:',
                '    if re.search(r"\\ba6000\\b", words):\n        return "a6000"\n'
                '    if "rtx6000ada" in compact:')
            data = policy_text.encode()
        (source / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    (source / 'train.py').write_text(train)
    hashes['train.py'] = hashlib.sha256(train.encode()).hexdigest()
    # Keep the data implementation immutable too; PYTHONPATH points at this copy.
    package = source / 'modular_rl/dolci'
    package.mkdir(parents=True, exist_ok=True)
    (package.parent / '__init__.py').write_text('')
    for name in ('__init__.py', 'config.py', 'data.py'):
        shutil.copy2(rlve / 'modular_rl/dolci' / name, package / name)
        hashes['modular_rl/dolci/' + name] = hashlib.sha256((package / name).read_bytes()).hexdigest()
    # The existing run.sbatch handles signals, transactional optimizer checkpoints,
    # workstation NCCL settings, and clean Slurm requeues without extra dependencies.
    common = {'NANO_REPO_ROOT': nano, 'NANO_DATA_ROOT': root, 'NANO_OUT_ROOT': root / 'sft',
              'NANO_TRAIN_SCRIPT': source / 'train.py', 'NANO_TOKENS': 1000000000, 'NANO_GPUS': 4,
              'NANO_MICRO_BATCH': 2, 'NANO_GRAD_ACCUM': 16, 'NANO_CE_CHUNK': 1024,
              'NANO_LR': .0001, 'NANO_LR_SCHEDULE': 'cosine', 'NANO_WARMUP_RATIO': .02,
              'NANO_MIN_LR_RATIO': .1, 'NANO_VARIANT': 'dolci', 'NANO_FULL_EVAL_AT_START': 1,
              'NANO_EVAL_EVERY_TOKENS': 10000000, 'NANO_CHECKPOINT_EVERY_TOKENS': 50000000,
              'NANO_EARLY_EVAL_EVERY_TOKENS': 1000000, 'NANO_EARLY_EVAL_UNTIL_TOKENS': 10000000,
              'NANO_WANDB_PROJECT': 'modular-mopd-midtrain-initial', 'NANO_WANDB_ENTITY': 'david-heineman',
              'NANO_WANDB_GROUP': 'dolci-mix-1b', 'NANO_ALLOWED_GPU_TYPES': 'l40s,a100,h200,rtx6000ada,a6000',
              'WANDB_BASE_URL': 'https://api.forge.coreweave.com', 'PYTHONPATH': source,
              'NANO_SMOKE': 0, 'NANO_SKIP_EVAL': 0, 'NANO_CE_PRESERVE_RNG_STATE': 0,
              'NANO_GRADIENT_CHECKPOINTING': 0, 'PYTHONUNBUFFERED': 1}
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location('dolci_gpu_policy', source / 'gpu_policy.py')
    policy = module_from_spec(spec)
    spec.loader.exec_module(policy)
    policy.UNHEALTHY_NODES.update({name: 'Excluded by the existing production RL startup policy'
                                  for name in ('viscam10', 'jagupard33', 'jagupard34')})
    inventory = policy.cluster_policy(policy.ALLOWED_GPU_TYPES)
    runs = []
    for mix in ('baseline', 'web', 'web-knowledge', 'chat', 'math', 'code'):
        gpus = 2 if mix in ('math', 'code') else 4
        wrapper = source / f'{mix}.sbatch'
        body = '#!/usr/bin/env bash\nset -euo pipefail\n'
        environment = {**common, 'NANO_MIX': mix, 'NANO_GPUS': gpus, 'NANO_GRAD_ACCUM': 64 // gpus}
        body += ''.join(f'export {k}={shlex.quote(str(v))}\n' for k, v in environment.items())
        body += 'exec bash ' + shlex.quote(str(source / 'run.sbatch')) + '\n'
        wrapper.write_text(body)
        command = ['sbatch', '--parsable', '--account=miso', '--partition=miso,' + QUEUES,
                   '--nodes=1', '--ntasks=1', f'--gres=gpu:{gpus}', '--constraint=40G|48G|80G|141G',
                   f'--cpus-per-task={gpus * 4}', f'--mem={48 if gpus == 2 else 64}G',
                   '--time=04:00:00', '--time-min=01:00:00', '--requeue',
                   '--signal=B:USR1@180', '--open-mode=append', '--job-name=dolci-sft-' + mix,
                   '--exclude=' + ','.join(inventory['excluded_nodes']),
                   '--output=' + str(logs / f'sft-{mix}-%j.out'), '--error=' + str(logs / f'sft-{mix}-%j.err'), str(wrapper)]
        subprocess.run(['bash', '-n', str(wrapper)], check=True)
        runs.append({'name': 'dolci-sft-' + mix, 'mix': mix, 'gpus': gpus, 'command': command,
                     'output': str(root / 'sft' / mix), 'script_sha256': hashlib.sha256(body.encode()).hexdigest()})
    plan = {'dataset_repo': DATASET, 'project': 'modular-mopd-midtrain-initial',
            'runs': runs, 'source_sha256': hashes, 'source': str(source), 'steps': STEPS,
            'actual_input_tokens': ACTUAL_TOKENS, 'tokens_per_step': TOKENS_PER_STEP,
            'optimizer': 'AdamW', 'lr': .0001, 'schedule': 'cosine', 'warmup_ratio': .02,
            'minimum_lr_ratio': .1, 'seed': 20261027, 'gpus_per_run': {r['mix']: r['gpus'] for r in runs}, 'independent_jobs': True,
            'gpu_policy': inventory,
            'hardware_extension': 'A6000 supported by real two-GPU masked-SFT preflight18184157; same global batch/optimizer',
            'reference_trainer': str(original / 'train.py'),
            'reference_trainer_sha256': hashlib.sha256((original / 'train.py').read_bytes()).hexdigest()}
    atomic_json(root / 'sft-plan.json', plan)
    return plan


def submit_ready(root):
    root = Path(root)
    if not (root / 'manifest.json').exists():
        return []
    plan = json.loads((root / 'sft-plan.json').read_text())
    records = json.loads((root / 'sft-jobs.json').read_text()) if (root / 'sft-jobs.json').exists() else []
    for row in plan['runs']:
        if any(r['name'] == row['name'] for r in records):
            continue
        script = Path(row['command'][-1])
        if hashlib.sha256(script.read_bytes()).hexdigest() != row['script_sha256']:
            raise ValueError('SFT submission script changed')
        job = subprocess.check_output(row['command'], text=True).strip().split(';')[0]
        records.append({**row, 'job_id': job})
        atomic_json(root / 'sft-jobs.json', records)
        print('SFT_SUBMITTED', row['name'], job, flush=True)
    return records
