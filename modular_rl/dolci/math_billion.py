"""Repeat the audited short-math mix to the baseline's 1B-token budget.

The unique question pool stays immutable. Repetition is explicit, with a fresh
permutation each pass, using the same MixedBlocks loader as the baseline SFT.
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .config import ACTUAL_TOKENS, SEED, atomic_json, fingerprint

PARENT = Path('/nlp/scr/dhei/dolci-mix-20261029').resolve()
RELEASE = Path('/nlp/scr/dhei/math-refresh-20261101/final').resolve()
QUEUES = 'jag-standard,miso,sphinx,miso-lo,sphinx-lo,jag-lo,sc-loprio'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def copy_prefix(source, destination, size):
    with Path(source).open('rb') as inp, Path(destination).open('wb') as out:
        remaining = size
        while remaining:
            chunk = inp.read(min(remaining, 8 * 1024**2))
            if not chunk:
                raise ValueError('Truncated source block file')
            out.write(chunk)
            remaining -= len(chunk)


def prepare_data(root):
    root = Path(root)
    if (root / 'manifest.json').exists():
        return
    release = json.loads((RELEASE / 'manifest.json').read_text())
    audit = json.loads((RELEASE / 'validation.json').read_text())
    assert audit['passed'] and release['one_epoch'] and release['no_truncation']
    real = release['splits']['train']['real_blocks']
    width = release['seq_len'] + 1
    sources = root / 'sources'
    sources.mkdir(parents=True, exist_ok=True)
    for extension, itemsize in (('.bin', 4), ('.mask', 1)):
        copy_prefix(RELEASE / ('sources/train' + extension), sources / ('curated' + extension), real * width * itemsize)
    counts = np.load(RELEASE / 'sources/train.counts.npy')[:real]
    assert counts.shape == (real,) and counts.min() > 0
    np.save(sources / 'curated.counts.npy', counts)
    for name in ('model', 'eval', 'eval_inloop_64'):
        (root / name).symlink_to(PARENT / name)
    shutil.copy2(PARENT / 'model_revision.json', root / 'model_revision.json')
    manifest = {
        'repo': 'math-refresh-1b-20261101', 'revision': release['revision'],
        'seq_len': 2048, 'seed': SEED, 'objective': 'assistant',
        'sources': {'curated': {'packed_tokens': real * width, 'real_blocks': real}},
        'quotas': {'math': {'curated': ACTUAL_TOKENS // 2048}},
        'recipe': release['recipe'], 'natural_definition': release['natural_definition'],
        'pool_prediction_tokens': ACTUAL_TOKENS, 'one_epoch': False,
        'require_unique_blocks': False, 'whole_conversations': True,
        'response_limit': 1536, 'no_truncation': True,
        'unique_data_release': str(RELEASE), 'unique_release_signature': release['signature'],
        'training_tokens': ACTUAL_TOKENS, 'train_steps': ACTUAL_TOKENS // 262144,
        'realized_block_epochs': (ACTUAL_TOKENS // 2048) / real,
        'repetition_authorized_by': 'User requested a 1B midtrain after the audited one-epoch experiment.',
        'unique_train_questions': audit['unique_train_questions'],
        'unique_nonpadding_input_tokens': audit['nonpadding_training_tokens'],
        'math_input_share_in_unique_pool': audit['math_input_share'],
        'source_sha256': {suffix: sha256(sources / ('curated' + suffix)) for suffix in ('.bin', '.mask')},
    }
    manifest['signature'] = fingerprint(manifest)
    atomic_json(root / 'manifest.json', manifest)
    from .data import MixedBlocks
    stream = MixedBlocks(root, 'math')
    assert len(stream.indices) == ACTUAL_TOKENS // 2048
    for offset in range(0, len(stream.indices), real):
        chunk = stream.indices[offset:offset + real]
        assert len(set(chunk.tolist())) == len(chunk)
    atomic_json(root / 'repetition-proof.json', {
        'passed': True, 'unique_real_blocks': real, 'sampled_blocks': len(stream.indices),
        'full_passes': len(stream.indices) // real, 'last_partial_pass_blocks': len(stream.indices) % real,
        'epochs': manifest['realized_block_epochs'], 'fresh_permutation_per_pass': True,
        'no_duplicate_blocks_within_a_pass': True, 'masked_only_padding_blocks_omitted': True,
        'indices_sha256': hashlib.sha256(stream.indices.tobytes()).hexdigest(),
        'supervised_tokens': stream.loss_tokens(0, len(stream.indices)),
        'input_token_slots': ACTUAL_TOKENS, 'assistant_only': True,
    })


def heldout_math_nll(model, rank, world, device):
    """Fixed 32-block assistant-target panel used by the original loss probe."""
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    manifest = json.loads((RELEASE / 'manifest.json').read_text())
    n = manifest['splits']['val']['blocks']
    ids = np.memmap(RELEASE / 'sources/val.bin', mode='r', dtype='<u4').reshape(n, 2049)
    mask = np.memmap(RELEASE / 'sources/val.mask', mode='r', dtype='u1').reshape(n, 2049)
    selected = np.random.default_rng(20261101).permutation(n)[:32][rank::world]
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for block in selected:
            sequence = torch.tensor(np.asarray(ids[block], dtype=np.int64), device=device)
            keep = torch.tensor(np.asarray(mask[block, 1:], dtype=bool), device=device)
            hidden = model.model(input_ids=sequence[None, :-1], use_cache=False).last_hidden_state[0, keep]
            targets = sequence[1:][keep]
            for start in range(0, len(targets), 256):
                logits = F.linear(hidden[start:start + 256], model.lm_head.weight).float()
                totals[0] += F.cross_entropy(logits, targets[start:start + 256], reduction='sum').double()
            totals[1] += len(targets)
    if world > 1:
        dist.all_reduce(totals)
    model.train()
    assert int(totals[1]) == 55825 and torch.isfinite(totals).all()
    return float(totals[0] / totals[1])


def save_best(model, root, out, step, nll):
    """Preserve the best intermediate weights; downstream primary uses final 1B."""
    record = out / 'best-math-validation.json'
    if record.exists() and json.loads(record.read_text())['assistant_nll'] <= nll:
        return
    target = out / 'model-best'
    temporary = out / 'model-best.partial'
    if temporary.exists():
        shutil.rmtree(temporary)
    model.save_pretrained(temporary, safe_serialization=True)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(root / 'model', local_files_only=True).save_pretrained(temporary)
    previous = out / 'model-best.previous'
    if previous.exists():
        shutil.rmtree(previous)
    if target.exists():
        target.rename(previous)
    temporary.rename(target)
    atomic_json(record, {'step': step, 'input_token_slots': step * 262144,
                         'assistant_nll': nll, 'model': str(target),
                         'selection': 'Internal SFT validation only; primary downstream RL starts from final 1B weights.'})
    if previous.exists():
        shutil.rmtree(previous)


def prepare_sft(root):
    root = Path(root).resolve()
    prepare_data(root)
    source = root / 'sft-source'
    if source.exists():
        return
    shutil.copytree(PARENT / 'sft-source', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.sbatch'))
    shutil.copy2(PARENT / 'sft-source/run.sbatch', source / 'run.sbatch')
    repo = Path(__file__).resolve().parents[2]
    shutil.copy2(__file__, source / 'modular_rl/dolci/math_billion.py')
    train = source / 'train.py'
    text = train.read_text().replace('"dataset_repo": "allenai/Dolci-Instruct-SFT",',
                                   '"dataset_repo": "math-refresh-1b-20261101",')
    text = text.replace('name=f"dolci-{args.mix}-{budget}"', 'name="math-refresh-sft-1b"')
    old = '        point = {"step": step, "train_tokens": step * per_step, "eval": scores}'
    assert text.count(old) == 1
    text = text.replace(old, '''        from modular_rl.dolci.math_billion import heldout_math_nll, save_best
        math_nll = heldout_math_nll(model, rank, world, device)
        point = {"step": step, "train_tokens": step * per_step, "eval": scores, "short_math_nll": math_nll}
        if rank == 0 and step > 0 and directory == "eval_inloop_64":
            save_best(model, root, out, step, math_nll)
        if world > 1:
            dist.barrier()''')
    text = text.replace('            payload = {"train/tokens": step * per_step}',
                        '            payload = {"train/tokens": step * per_step, f"{prefix}/short_math_nll": math_nll}')
    train.write_text(text)
    env = {}
    for line in (PARENT / 'sft-source/math.sbatch').read_text().splitlines():
        if line.startswith('export '):
            key, value = line[7:].split('=', 1)
            env[key] = shlex.split(value)[0]
    env.update(NANO_DATA_ROOT=str(root), NANO_OUT_ROOT=str(root / 'sft'),
               NANO_TRAIN_SCRIPT=str(train), PYTHONPATH=str(source),
               NANO_WANDB_GROUP='math-refresh-1b', NANO_VARIANT='math-refresh',
               NANO_ALLOWED_GPU_TYPES='l40s,a100,h100,h200,rtx6000ada,a6000',
               NANO_FULL_EVAL_AT_START='1')
    script = source / 'math.sbatch'
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\n' +
                      ''.join(f'export {k}={shlex.quote(v)}\n' for k, v in env.items()) +
                      'exec bash ' + shlex.quote(str(source / 'run.sbatch')) + '\n')
    subprocess.run(['bash', '-n', str(script)], check=True)
    command = ['sbatch', '--parsable', '--account=miso', '--partition=' + QUEUES,
               '--nodes=1', '--ntasks=1', '--gres=gpu:4', '--constraint=40G|48G|80G|141G',
               '--cpus-per-task=16', '--mem=128G', '--time=04:00:00', '--time-min=01:00:00',
               '--requeue', '--signal=B:USR1@180', '--exclude=viscam10,jagupard33,jagupard34,pasteur6',
               '--job-name=math-refresh-sft-1b', '--output=' + str(root / 'sft-%j.log'), str(script)]
    atomic_json(root / 'sft-job.json', {'name': 'math-refresh-sft-1b', 'command': command,
                'script_sha256': sha256(script), 'gpus': 4, 'input_tokens': ACTUAL_TOKENS,
                'status': 'prepared', 'learning_settings_match_baseline': True,
                'source_sha256': {str(p.relative_to(source)): sha256(p) for p in
                                  (train, source / 'modular_rl/dolci/math_billion.py')}})


def prepare_rl(root, alias, *, expected_training_tokens=ACTUAL_TOKENS, train_steps=1000,
               evaluation_interval=50, run_name=None, group='math-refresh-1b-paired'):
    """Use identical new RL settings for baseline-SFT and refreshed-SFT models."""
    from modular_rl.initializers import inspect_hf
    from modular_rl.suite import prepare
    root = Path(root).resolve()
    directory = root / 'rl' / alias
    if (directory / 'launch-plan.json').exists():
        existing = json.loads((directory / 'launch-plan.json').read_text())
        if (existing.get('enable_thinking') is False
                and existing.get('evaluation_temperature') == .7
                and existing.get('evaluation_top_p') == .95):
            return directory
    if directory.exists():
        # A preempted controller may leave preparation half finished. Preserve
        # its files for diagnosis and rebuild before any policy is submitted.
        if (directory / 'jobs.json').exists():
            raise ValueError('Refuse to replace an already submitted RL suite')
        quarantine = root / 'incomplete-preparations' / f'{alias}-{time.time_ns()}'
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(quarantine)
    control = json.loads((PARENT / 'rl/baseline/initializers.json').read_text())
    if alias == 'baseline':
        spec = control['baseline']
        name = run_name or 'math-baseline-sft-rl'
    elif alias == 'math':
        spec = inspect_hf('math', root / 'sft/math/model', root / 'initializers/math', training_mix='math',
                          expected_training_tokens=expected_training_tokens)
        name = run_name or 'math-refresh-1b-rl'
    else:
        raise ValueError('Only baseline and refreshed math are part of this pair')
    registry = root / 'registries' / (alias + '.json')
    atomic_json(registry, {'base': control['base'], alias: spec})
    repo = Path(__file__).resolve().parents[2]
    prepare(repo, Path('/nlp/scr/dhei/modular-mopd-rl/v2'), directory,
            root / 'sft', PARENT / 'rl/base/selection', policy_gpus=4,
            initializers_from=registry, selection_from=PARENT / 'rl/base/selection',
            runs=[(name, alias, 'math')], queues=QUEUES, group=group,
            allow_unconverted=alias == 'math', time_limit='08:00:00',
            frozen_from=PARENT / 'rl/base', data_from=PARENT / 'rl-data',
            resize_frozen_policy=True, train_steps=train_steps, evaluation_interval=evaluation_interval)
    plan_path = directory / 'launch-plan.json'
    plan = json.loads(plan_path.read_text())
    prompt = directory / 'source/modular_rl/v2/rollout.py'
    text = prompt.read_text()
    assert text.count('enable_thinking=True,') == 1
    prompt.write_text(text.replace('enable_thinking=True,', 'enable_thinking=False,'))
    plan['source_sha256']['modular_rl/v2/rollout.py'] = sha256(prompt)
    for row in plan['runs']:
        script = Path(row['command'][-1])
        lines = []
        for line in script.read_text().splitlines():
            if line.startswith(('export MODULAR_RL_EXTRA_ARGS=', 'export APPTAINERENV_MODULAR_RL_EXTRA_ARGS=')):
                key, value = line.split('=', 1)
                extra = shlex.split(value)[0]
                assert '--eval-temperature' not in extra
                line = key + '=' + shlex.quote(extra + ' --eval-temperature .7 --eval-top-p .95')
            lines.append(line)
        # Keep one committed resumable checkpoint; atomic saves can momentarily
        # coexist with the previous one. Never disable intermediate recovery.
        lines[lines.index('set -euo pipefail') + 1:lines.index('set -euo pipefail') + 1] = [
            'export RLVE_CHECKPOINT_KEEP=1', 'export APPTAINERENV_RLVE_CHECKPOINT_KEEP=1']
        script.write_text('\n'.join(lines) + '\n')
        subprocess.run(['bash', '-n', str(script)], check=True)
        row['script_sha256'] = sha256(script)
    plan.update(enable_thinking=False, evaluation_temperature=.7, evaluation_top_p=.95,
                training_temperature=1., training_top_p=1., checkpoint_keep=1,
                comparison='Paired full 8K math RL; only SFT initializer/data changes.',
                legacy_baseline_run='b001e8d6',
                legacy_baseline_protocol='Open thinking prefix, temperature 1/top-p 1; shown separately.')
    atomic_json(plan_path, plan)
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    prepare_sft(args.root)
    if args.submit:
        path = args.root / 'sft-job.json'
        job = json.loads(path.read_text())
        if not job.get('job_id'):
            job['job_id'] = subprocess.check_output(job['command'], text=True).strip().split(';')[0]
            job['status'] = 'submitted'
            atomic_json(path, job)
        print(json.dumps(job), flush=True)
