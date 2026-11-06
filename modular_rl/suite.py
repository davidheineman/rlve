"""Prepare, review, and explicitly submit the modular-mopd-rl-v1 suite."""

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from .initializers import inspect_hf, load_registry, verify
from .prepare import REWARD_REVISION, digest, write_json
from .v2.ablate import freeze
from .v2.base_models import sha256
from .v2.base_sweep import Cell, arguments
from .v2.data import stratified_sample, stratum
from .v2.inspect_rollouts import read_events

PROJECT = 'modular-mopd-rl-v1'
QUEUES = 'jag-standard,miso,sphinx,miso-lo,sphinx-lo,jag-lo,sc-loprio'
RUNS = [('base-math', 'base', 'math'), ('base-code', 'base', 'code'),
        ('base-if', 'base', 'if'), ('base-chat', 'base', 'chat'),
        ('mid-code', 'code', 'code'), ('mid-math', 'math', 'math'),
        ('mid-chat-chat', 'chat', 'chat'), ('mid-chat-if', 'chat', 'if'),
        ('mid-web-chat', 'web', 'chat'), ('mid-web-if', 'web', 'if')]


def shell_exports(values):
    return ''.join(f'export {k}={shlex.quote(str(v))}\n' for k, v in values.items())


def reuse_frozen_source(parent, experiment, repo, policy_gpus, reward_gpus, *, resize_policy=False):
    """Clone an already instrumented control snapshot without instrumenting twice."""
    parent, experiment, repo = Path(parent), Path(experiment), Path(repo)
    plan = json.loads((parent / 'launch-plan.json').read_text())
    if ((plan['policy_gpus_per_run'] != policy_gpus and not resize_policy)
            or plan['reward_gpus_per_chat_run'] != reward_gpus or plan.get('preflight_only')):
        raise ValueError('Frozen controls must use the same production GPU arrangement')
    if resize_policy and (plan['policy_gpus_per_run'] not in (1, 2, 4) or policy_gpus not in (1, 2, 4)):
        raise ValueError('Unsupported frozen-policy data parallelism')
    original = Path(plan['source'])
    for name, expected in plan['source_sha256'].items():
        if sha256(original / name) != expected:
            raise ValueError('Frozen control source changed: ' + name)
    source = experiment / 'source'
    for name in plan['source_sha256']:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / name, target)
    if plan['policy_gpus_per_run'] != policy_gpus:
        config = source / 'modular_rl/v2/config.yaml'
        before = config.read_text()
        after = before
        for key in ('actor-num-gpus-per-node', 'num-gpus-per-node'):
            old = f"{key}: {plan['policy_gpus_per_run']}"
            after, count = re.subn(r'^' + re.escape(old) + r'$', f'{key}: {policy_gpus}', after, flags=re.MULTILINE)
            if count != 1:
                raise ValueError('Inspect frozen GPU counts before resizing')
        if 'tensor-model-parallel-size: 1' not in after:
            raise ValueError('Only data-parallel resizing is supported')
        config.write_text(after)
        write_json(experiment / 'policy-gpu-resize.json', {
            'from': plan['policy_gpus_per_run'], 'to': policy_gpus,
            'tensor_parallel_size': 1, 'global_response_batch': 8,
            'before_sha256': hashlib.sha256(before.encode()).hexdigest(),
            'after_sha256': hashlib.sha256(after.encode()).hexdigest()})
    # These modules only select/verify initializers and orchestrate submission.
    # All rollout, grading, normalization, loss, optimizer, actor and probe code
    # stays byte-for-byte identical to the raw-Base control snapshot.
    for name in ('initializers.py', 'initializer_convert.py', 'suite.py', 'suite_conversions.py'):
        shutil.copy2(repo / 'modular_rl' / name, source / 'modular_rl' / name)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    shutil.copytree(parent / 'runtime', experiment / 'runtime', ignore=ignore)
    shutil.copytree(parent / 'selection', experiment / 'selection', ignore=ignore)
    extras = {}
    for row in plan['runs']:
        script = Path(row['command'][-1])
        if sha256(script) != row['script_sha256']:
            raise ValueError('Frozen control launcher changed: ' + row['name'])
        prefix = 'export MODULAR_RL_EXTRA_ARGS='
        line, = [line for line in script.read_text().splitlines() if line.startswith(prefix)]
        value, = shlex.split(line[len(prefix):])
        if row['task'] in extras and extras[row['task']] != value:
            raise ValueError('Controls have inconsistent per-task arguments')
        extras[row['task']] = value
    if set(extras) != {'math', 'code', 'if', 'chat'}:
        raise ValueError('Need all four frozen task controls')
    return source, extras


def prepare_selection(base, output, code_selection, tokenizers, rm):
    manifest = {}
    for task in ('math', 'code', 'if', 'chat'):
        previous = code_selection / task if task == 'code' else base / 'selection' / task
        train = read_events(previous / 'train.jsonl')
        old = read_events(previous / 'val.jsonl')
        assert len(old) == 32
        train_keys = {r['metadata']['key'] for r in train}
        train_groups = {r['metadata']['split_group'] for r in train}
        seen = {r['metadata']['key'] for r in old}
        pool = read_events(base / 'data' / task / 'val.jsonl')
        if task == 'code':
            pool += read_events(base / 'data/code/train.jsonl')
        candidates, excluded = [], Counter()
        for row in pool:
            meta = row['metadata']
            if meta['key'] in seen or meta['key'] in train_keys or meta['split_group'] in train_groups:
                continue
            meta['environment'] = task
            prompt = tokenizers[0].apply_chat_template(meta['messages'], tokenize=False,
                add_generation_prompt=True, enable_thinking=True)
            if len(tokenizers[0](prompt, add_special_tokens=False)['input_ids']) > 2048:
                excluded['input_over_2048'] += 1
                continue
            if task == 'chat':
                rejected = meta['messages'] + [{'role': 'assistant', 'content': meta['target']['rejected_reference']}]
                if len(rm.apply_chat_template(rejected)) > 8192:
                    excluded['rm_reference_over_8192'] += 1
                    continue
                if len(rm.apply_chat_template(meta['messages'], add_generation_prompt=True)) > 4096:
                    excluded['rm_prompt_over_4096'] += 1
                    continue
            candidates.append(row)
            seen.add(meta['key'])
        val = old + stratified_sample(candidates, 32, seed=20261029)
        for row in val:
            meta = row['metadata']
            meta['environment'] = task
            encodings = [t(t.apply_chat_template(meta['messages'], tokenize=False,
                add_generation_prompt=True, enable_thinking=True), add_special_tokens=False)['input_ids']
                for t in tokenizers]
            if any(ids != encodings[0] or len(ids) > 2048 for ids in encodings):
                raise ValueError('Initializers encode validation differently or exceed the prompt cap')
        val_keys = {r['metadata']['key'] for r in val}
        val_groups = {r['metadata']['split_group'] for r in val}
        assert len(val_keys) == 64 and not val_keys & train_keys and not val_groups & train_groups
        directory = output / task
        directory.mkdir(parents=True)
        shas = {}
        for split, rows in (('train', train), ('val', val)):
            path = directory / (split + '.jsonl')
            path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
            shas[split] = sha256(path)
        info = {'task': task, 'validation_prompts': 64, 'seed': 20261029,
                'counts': {'train': len(train), 'val': 64}, 'sha256': shas,
                'keys': {'train': [r['metadata']['key'] for r in train],
                         'val': [r['metadata']['key'] for r in val]},
                'strata': {'val': dict(Counter('/'.join(stratum(r)) for r in val))},
                'previous_32_retained': True, 'additional_32_eligible': len(candidates),
                'excluded': dict(excluded), 'train_val_disjoint_by_key_and_group': True,
                'all_five_tokenizer_encodings_identical': True, 'source_selection': str(previous)}
        write_json(directory / 'selection.json', info)
        manifest[task] = {k: v for k, v in info.items() if k != 'keys'}
    return manifest


def prepare(repo, base, experiment, midtraining_root, code_selection, policy_gpus=1, reward_gpus=1,
            initializers_from=None, selection_from=None, preflight=False, *, runs=None,
            queues=None, group=None, allow_unconverted=False, time_limit='16:00:00',
            frozen_from=None, data_from=None, clean_chat_rewards=False,
            clean_code_rewards=False, resize_frozen_policy=False, maxrl_runs=(),
            train_steps=1000, evaluation_interval=50):
    from transformers import AutoTokenizer

    repo, base, experiment = repo.resolve(), base.resolve(), experiment.resolve()
    if experiment.exists():
        raise ValueError('Use a fresh suite directory')
    experiment.mkdir(parents=True)
    if policy_gpus not in (1, 2, 4) or reward_gpus not in (1, 2):
        raise ValueError('Use one, two, or four policy GPUs and one or two reward replicas')
    if frozen_from and preflight:
        raise ValueError('Frozen control reuse is for matching production runs')
    if train_steps <= 0 or evaluation_interval <= 0:
        raise ValueError('Training and evaluation intervals must be positive')
    if initializers_from:
        specs = load_registry(initializers_from)
        for spec in specs.values():
            verify(spec, require_native=not allow_unconverted)
    else:
        models = base.parents[1] / 'models'
        specs = {'base': inspect_hf('base', models / 'Qwen3-0.6B-Base', models / 'Qwen3-0.6B-Base_torch_dist')}
        for alias in ('code', 'math', 'chat', 'web'):
            specs[alias] = inspect_hf(alias, midtraining_root / alias / 'model', experiment / 'initializers' / alias)
    registry = experiment / 'initializers.json'
    write_json(registry, specs)
    tokenizers = [AutoTokenizer.from_pretrained(s['hf'], local_files_only=True) for s in specs.values()]
    contracts = [{'alias': alias, 'vocab_sha256': digest(t.get_vocab()), 'eos_token_id': t.eos_token_id,
                  'chat_template_sha256': hashlib.sha256(t.chat_template.encode()).hexdigest()}
                 for alias, t in zip(specs, tokenizers, strict=True)]
    if len({(c['vocab_sha256'], c['eos_token_id'], c['chat_template_sha256']) for c in contracts}) != 1:
        raise ValueError('Supplied models have different tokenizer vocabularies or templates')
    write_json(experiment / 'tokenizer-contract.json', contracts)
    rm_path = base.parents[1] / 'tinker-mopd/hf/hub/models--Skywork--Skywork-Reward-V2-Llama-3.2-1B/snapshots' / REWARD_REVISION
    rm = AutoTokenizer.from_pretrained(rm_path, local_files_only=True)
    if selection_from:
        selection = {}
        for task in ('math', 'code', 'if', 'chat'):
            directory = Path(selection_from) / task
            info = json.loads((directory / 'selection.json').read_text())
            if info['validation_prompts'] != 64:
                raise ValueError('Reuse requires the declared 64-prompt validation panel')
            for split, expected in info['sha256'].items():
                if sha256(directory / (split + '.jsonl')) != expected:
                    raise ValueError('Reused task data changed')
            shutil.copytree(directory, experiment / 'selection-prepared' / task)
            selection[task] = {k: v for k, v in info.items() if k != 'keys'}
    else:
        selection = prepare_selection(base, experiment / 'selection-prepared', code_selection.resolve(), tokenizers, rm)
    frozen_arguments = None
    if frozen_from:
        source, frozen_arguments = reuse_frozen_source(frozen_from, experiment, repo, policy_gpus, reward_gpus,
                                                      resize_policy=resize_frozen_policy)
    else:
        source = freeze(repo, experiment, base, ['math', 'code', 'if', 'chat'], skip_zero=True)
    if clean_chat_rewards:
        from .dolci.chat_reward import install
        proof = install(source, repo / 'modular_rl/dolci/chat_reward.py')
        write_json(experiment / 'chat-reward-normalization.json', proof)
    if clean_code_rewards:
        from .dolci.code_reward import install
        proof = install(source, repo / 'modular_rl/graders.py')
        write_json(experiment / 'code-reward-normalization.json', proof)
    if maxrl_runs:
        declared = {name for name, _, _ in (RUNS if runs is None else runs)}
        if not set(maxrl_runs) <= declared:
            raise ValueError('MaxRL names must belong to this suite')
        shutil.copy2(repo / 'modular_rl/maxrl.py', source / 'modular_rl/maxrl.py')
    for task in selection:
        shutil.rmtree(experiment / 'selection' / task)
        shutil.move(experiment / 'selection-prepared' / task, experiment / 'selection' / task)
    (experiment / 'selection-prepared').rmdir()
    if preflight:
        for task in selection:
            directory = experiment / 'selection' / task
            path = directory / 'val.jsonl'
            rows = read_events(path)[:4]
            path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
            info = json.loads((directory / 'selection.json').read_text())
            info.update(preflight_only=True, validation_prompts=4)
            info['counts']['val'] = 4
            info['keys']['val'] = [row['metadata']['key'] for row in rows]
            info['sha256']['val'] = sha256(path)
            write_json(directory / 'selection.json', info)
            selection[task] = {k: v for k, v in info.items() if k != 'keys'}
    config = source / 'modular_rl/v2/config.yaml'
    if not frozen_from:
        text = config.read_text()
        assert text.count('use-dynamic-batch-size: true') == 1
        config.write_text(text.replace('use-dynamic-batch-size: true', 'use-dynamic-batch-size: false'))
    if policy_gpus > 1 and not frozen_from:
        text = config.read_text()
        for old, new in [('actor-num-gpus-per-node: 1', f'actor-num-gpus-per-node: {policy_gpus}'),
                         ('num-gpus-per-node: 1\n', f'num-gpus-per-node: {policy_gpus}\n'),
                         ('balance-data: true', 'balance-data: false')]:
            text = text.replace(old, new)
        config.write_text(text)
    registry_sha = sha256(registry)
    jobs = []
    for name, initializer, task in (RUNS if runs is None else runs):
        if preflight and initializer != 'base':
            continue
        if preflight:
            name += '-dp2-preflight'
        assert re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', name)
        output = experiment / 'runs' / name
        output.mkdir(parents=True)
        reserve = {'math': 256, 'code': 512, 'if': 128, 'chat': 128}[task]
        if frozen_arguments is not None:
            extra = frozen_arguments[task]
        else:
            extra = arguments(Cell('Qwen3-0.6B-Base', 2048, 'control', task='code' if task == 'code' else 'math'))
            extra += (f' --num-rollout {4 if preflight else 1000} --eval-interval {2 if preflight else 50} --rollout-max-response-len 2048'
                      f' --eval-max-response-len 2048 --answer-forcing-extra-tokens {reserve}')
            if task == 'if':
                extra += (' --custom-generate-function-path modular_rl.v2.if_rollout.generate'
                          ' --custom-rm-path modular_rl.if_grader.reward')
        if name in maxrl_runs:
            if task not in ('math', 'code') or '--custom-reward-post-process-path' in extra:
                raise ValueError('This MaxRL comparison requires unmodified binary math/code rewards')
            extra += ' --custom-reward-post-process-path modular_rl.maxrl.post_process'
        if not preflight and (train_steps != 1000 or evaluation_interval != 50):
            for flag, value in (('num-rollout', train_steps), ('eval-interval', evaluation_interval)):
                extra, count = re.subn(r'--' + flag + r'\s+\d+', '--' + flag + ' ' + str(value), extra)
                if count != 1:
                    raise ValueError('Expected one explicit training/evaluation budget in control arguments')
        environment = {'RLVE_REPO': source, 'RLVE_SCRATCH': base.parents[1],
            'RLVE_IMAGE': base.parents[1] / 'images/slime-v0.5.0rc0-cu126.sif',
            'RLVE_WANDB_TEAM': 'david-heineman', 'WANDB_BASE_URL': 'https://api.forge.coreweave.com',
            'MODULAR_RL_PROFILE': 'v2',
            'MODULAR_RL_TASKS': task, 'MODULAR_RL_INITIALIZER': initializer,
            'MODULAR_RL_INITIALIZERS': registry, 'MODULAR_RL_INITIALIZERS_SHA256': registry_sha,
            'MODULAR_RL_RUN_NAME': name, 'MODULAR_RL_WANDB_PROJECT': PROJECT,
            'MODULAR_RL_RUNTIME': experiment / 'runtime', 'MODULAR_RL_SELECTION': experiment / 'selection',
            'MODULAR_RL_DATA': Path(data_from) if data_from else base / 'data', 'MODULAR_RL_OUTPUT': output,
            'MODULAR_RL_HF_HOME': base.parents[1] / 'tinker-mopd/hf',
            'MODULAR_RL_VALIDATION_PROMPTS': 4 if preflight else 64, 'MODULAR_RL_PROMPTS_PER_UPDATE': 1,
            'MODULAR_RL_CHECKPOINT_INTERVAL': 2 if preflight else min(50, evaluation_interval),
            'MODULAR_RL_MAX_RESPONSE_TOKENS': 2048, 'MODULAR_RL_ANSWER_FORCING_TOKENS': reserve,
            'MODULAR_RL_BASE_TOKENIZER': 1, 'MODULAR_RL_SKIP_ZERO_ADVANTAGE': 1,
            'MODULAR_RL_POLICY_GPUS': policy_gpus, 'MODULAR_RL_REWARD_GPUS': reward_gpus,
            'MODULAR_RL_GROUP': group or ('rl-v1-dp2-preflight' if preflight else ('rl-v1-dp2' if policy_gpus == 2 else 'rl-v1')),
            'PYTORCH_CUDA_ALLOC_CONF': 'garbage_collection_threshold:0.7,max_split_size_mb:128',
            'MODULAR_RL_EVAL_ONLY': 0, 'MODULAR_RL_FILTER_UNIFORM_REFERENCE_BATCHES': 0,
            'MODULAR_RL_ALLOW_OBJECTIVE_REUSE': 1,
            'MODULAR_RL_CUDA_CACHE_CLEAR_INTERVAL': 1, 'MODULAR_RL_UPDATE_PROBE_INTERVAL': 1 if preflight else 10,
            'MODULAR_RL_EXTRA_ARGS': extra}
        script = output / 'run.sbatch'
        body = ('#!/usr/bin/env bash\n#SBATCH --account=miso\n'
                f'#SBATCH --partition={queues or QUEUES}\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n'
                '#SBATCH --requeue\n#SBATCH --open-mode=append\nset -euo pipefail\n')
        body += shell_exports(environment)
        body += shell_exports({f'APPTAINERENV_{key}': value for key, value in environment.items()})
        body += 'export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY:?}"\n'
        body += ('source "$RLVE_REPO/scripts/slurm/common.sh"\n'
                 'rlve_apptainer bash modular_rl/suite_allocation.sh "$@"\n')
        script.write_text(body)
        cards = policy_gpus + (reward_gpus if task == 'chat' else 0)
        command = ['sbatch', '--parsable', f'--job-name={name}', f'--gres=gpu:{cards}',
                   '--constraint=40G|48G|80G|141G', f'--cpus-per-task={6 * cards + 4}',
                   f'--mem={64 + 32 * (cards - 1)}G', f'--time={time_limit}',
                   '--exclude=viscam10,jagupard33,jagupard34', f'--chdir={source}',
                   f'--output={output}/slurm-%j.out', str(script)]
        jobs.append({'name': name, 'initializer': initializer, 'task': task, 'gpus': cards,
                     'algorithm': 'maxrl' if name in maxrl_runs else 'grpo',
                     'hf': specs[initializer]['hf'], 'native': specs[initializer]['native'],
                     'command': command, 'script_sha256': sha256(script), 'output': str(output)})
    plan = {'project': PROJECT, 'status': 'prepared-not-submitted', 'runs': jobs,
            'frozen_training_source_from': str(Path(frozen_from).resolve()) if frozen_from else None,
            'learning_arguments_copied_from_base_controls': frozen_from is not None,
            'clean_chat_reward_endings': clean_chat_rewards,
            'clean_code_reward_endings': clean_code_rewards,
            'resize_frozen_policy': resize_frozen_policy, 'maxrl_runs': list(maxrl_runs),
            'preflight_only': preflight,
            'policy_runs': len(jobs), 'simultaneous_gpus': sum(j['gpus'] for j in jobs), 'dependencies': [],
            'policy_gpus_per_run': policy_gpus, 'reward_gpus_per_chat_run': reward_gpus,
            'source': str(source), 'initializers': str(registry), 'initializers_sha256': registry_sha,
            'selection': selection, 'train_rollouts': 32 if preflight else train_steps * 8, 'prompts_per_step': 1, 'samples_per_prompt': 8,
            'train_steps': 4 if preflight else train_steps, 'max_response_tokens': 2048, 'lr': 2e-6, 'objective_passes': 1,
            'validation_prompts': 4 if preflight else 64, 'validation_samples_per_prompt': 8, 'evaluation_interval': 2 if preflight else evaluation_interval,
            'maximum_evaluation_rollouts_per_run': 96 if preflight else len({0, train_steps - 1, *range(evaluation_interval - 1, train_steps, evaluation_interval)}) * 512, 'early_stop_pass1': .98,
            'math_code_answer_forcing': True, 'if_answer_forcing': False,
            'source_sha256': {str(p.relative_to(source)): sha256(p)
                              for p in sorted(source.rglob('*')) if p.is_file()}}
    write_json(experiment / 'launch-plan.json', plan)
    (experiment / 'submit-all.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
        + f'export PYTHONPATH={shlex.quote(str(source))}\n'
        + f'exec /nlp/scr/dhei/tinker-mopd/venv-rl/bin/python -m modular_rl.suite submit {shlex.quote(str(experiment))}\n')
    print(json.dumps({k: v for k, v in plan.items() if k not in ('source_sha256', 'selection')}), flush=True)


def submit(experiment, names=None):
    plan = json.loads((experiment / 'launch-plan.json').read_text())
    jobs_path = experiment / 'jobs.json'
    if jobs_path.exists() and names is None:
        raise ValueError('Suite already submitted or partially submitted; inspect before retrying')
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    selected = plan['runs'] if names is None else [r for r in plan['runs'] if r['name'] in names]
    if names is not None and (len(names) != len(set(names)) or {r['name'] for r in selected} != set(names)):
        raise ValueError('Select unique run names from the prepared plan')
    if {r['name'] for r in selected} & {r['name'] for r in jobs}:
        raise ValueError('Selected run already submitted; inspect its existing job before retrying')
    registry = load_registry(plan['initializers'], plan['initializers_sha256'])
    for alias in {row['initializer'] for row in selected}:
        verify(registry[alias])
    for name, expected in plan['source_sha256'].items():
        assert sha256(Path(plan['source']) / name) == expected
    checks = []
    with tempfile.TemporaryDirectory(prefix='suite-shell-check-') as temporary:
        for row in selected:
            script = Path(row['command'][-1])
            assert sha256(script) == row['script_sha256']
            assert not any('dependency' in part for part in row['command'])
            body = script.read_text()
            marker = 'rlve_apptainer bash modular_rl/suite_allocation.sh'
            if body.count(marker) != 1:
                raise ValueError('Inspect the container entrypoint before submitting')
            check = Path(temporary) / (row['name'] + '.sh')
            check.write_text(body.replace(marker, 'rlve_apptainer_cpu bash modular_rl/suite_allocation.sh'))
            result = subprocess.run(['bash', str(check), '--check-environment'],
                                    capture_output=True, text=True, check=False)
            checks.append({'name': row['name'], 'returncode': result.returncode,
                           'output': result.stdout + result.stderr})
            write_json(experiment / 'container-shell-preflight.json', checks)
            if result.returncode:
                raise RuntimeError('Container shell preflight failed for ' + row['name'])
    for row in selected:
        assert sha256(Path(row['command'][-1])) == row['script_sha256']
        assert not any('dependency' in p for p in row['command'])
        job = subprocess.check_output(row['command'], text=True).strip().split(';')[0]
        jobs.append({**row, 'job_id': job})
        write_json(experiment / 'jobs.json', jobs)
        print(row['name'], job, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--base', type=Path, required=True)
    prep.add_argument('--experiment', type=Path, required=True)
    prep.add_argument('--midtraining-root', type=Path, required=True)
    prep.add_argument('--code-selection', type=Path, required=True)
    prep.add_argument('--policy-gpus', type=int, choices=[1, 2, 4], default=1)
    prep.add_argument('--reward-gpus', type=int, choices=[1, 2], default=1)
    prep.add_argument('--initializers-from', type=Path)
    prep.add_argument('--selection-from', type=Path)
    prep.add_argument('--preflight', action='store_true')
    launch = sub.add_parser('submit')
    launch.add_argument('experiment', type=Path)
    launch.add_argument('--runs', nargs='+', help='Explicitly submit only previously unsubmitted names')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(Path.cwd(), args.base, args.experiment, args.midtraining_root, args.code_selection,
                args.policy_gpus, args.reward_gpus, args.initializers_from, args.selection_from, args.preflight)
    else:
        submit(args.experiment.resolve(), args.runs)
