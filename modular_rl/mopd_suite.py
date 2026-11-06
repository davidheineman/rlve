"""Prepare and independently launch the three requested modular MOPD students."""

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

import yaml

from modular_rl.mopd_export import BANKS
from modular_rl.prepare import write_json
from modular_rl.suite import PROJECT, shell_exports
from modular_rl.v2.ablate import freeze
from modular_rl.v2.base_models import sha256
from modular_rl.v2.base_sweep import Cell, arguments

TASKS = ('math', 'code', 'if', 'chat')


def mixed_data(selection, updates, tokenizer=None, *, enable_thinking=None):
    pools = {task: [json.loads(line) for line in (selection / task / 'train.jsonl').open()] for task in TASKS}
    original_counts = {task: len(rows) for task, rows in pools.items()}
    if tokenizer is not None:
        # Match native Dataset filtering BEFORE interleaving tasks. Filtering the
        # mixed file afterward would break its exact two-prompts/task batches.
        def eligible(row):
            prompt = row['user_prompt']
            if not (isinstance(prompt, list) and prompt and isinstance(prompt[0], dict) and 'role' in prompt[0]):
                prompt = [{'role': 'user', 'content': prompt}]
            native = tokenizer.apply_chat_template(prompt, None, tokenize=False, add_generation_prompt=True)
            if len(tokenizer(native)['input_ids']) > 2048:
                return False
            if enable_thinking is not None:
                generated = tokenizer.apply_chat_template(prompt, tokenize=False,
                    add_generation_prompt=True, enable_thinking=enable_thinking)
                return len(tokenizer(generated)['input_ids']) <= 2048
            return True
        pools = {task: [row for row in rows if eligible(row)] for task, rows in pools.items()}
        if any(not rows for rows in pools.values()):
            raise ValueError('A task has no prompts within the native input cap')
    rng = {task: random.Random(42 + i) for i, task in enumerate(TASKS)}
    remaining = {task: [] for task in TASKS}
    train = []
    for _ in range(updates):
        for _ in range(2):
            for task in TASKS:
                if not remaining[task]:
                    remaining[task] = list(pools[task])
                    rng[task].shuffle(remaining[task])
                train.append(remaining[task].pop())
    val = [json.loads(line) for task in TASKS for line in (selection / task / 'val.jsonl').open()]
    for split, rows in (('train', train), ('val', val)):
        (selection / f'mixed-{split}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    write_json(selection / 'mixed-selection.json', {'seed': 42, 'train_rollouts': updates * 8,
        'original_pool_counts': original_counts, 'eligible_pool_counts': {task: len(rows) for task, rows in pools.items()},
        'counts': {task: {'train': updates * 2, 'validation_prompts': len(val) // 4} for task in TASKS},
        'prompts_per_step': 8, 'prompts_per_task_per_step': 2, 'samples_per_prompt': 1,
        'generation_enable_thinking': enable_thinking,
        'sha256': {split: sha256(selection / f'mixed-{split}.jsonl') for split in ('train', 'val')}})


def prepare(root, master, repo, updates=500, eval_interval=64):
    experiment = root / 'students'
    if experiment.exists():
        raise ValueError('Use a fresh student directory')
    experiment.mkdir(parents=True)
    base = Path('/juice5b/scr5b/dhei/modular-mopd-rl/v2')
    source = freeze(repo, experiment, base, list(TASKS), skip_zero=False)
    for task in TASKS:
        shutil.rmtree(experiment / 'selection' / task)
        shutil.copytree(master / 'selection' / task, experiment / 'selection' / task)
    from transformers import AutoTokenizer
    registry_info = json.loads((master / 'initializers.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(registry_info['base']['hf'], local_files_only=True)
    mixed_data(experiment / 'selection', updates, tokenizer)
    registry = experiment / 'initializers.json'
    shutil.copy2(master / 'initializers.json', registry)
    config = yaml.safe_load((source / 'modular_rl/v2/config.yaml').read_text())
    config.update({'actor-num-gpus-per-node': 4, 'num-gpus-per-node': 4,
        'use-dynamic-batch-size': False, 'balance-data': False, 'rollout-shuffle': False,
        'rollout-batch-size': 8, 'n-samples-per-prompt': 1, 'over-sampling-batch-size': 8,
        'num-rollout': updates, 'eval-interval': eval_interval, 'rollout-max-response-len': 2048,
        'eval-max-response-len': 2048, 'rollout-function-path': 'modular_rl.mopd_runtime.generate_rollout',
        'custom-generate-function-path': 'modular_rl.mopd_runtime.generate',
        'custom-rm-path': 'modular_rl.mopd_runtime.reward', 'weight-decay': 0.01,
        'loss-type': 'custom_loss', 'custom-loss-function-path': 'slime.backends.megatron_utils.opd_loss.opd_loss_function',
        'disable-compute-advantages-and-returns': True})
    (source / 'modular_rl/mopd-config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    actor = source / 'slime/backends/megatron_utils/actor.py'
    text = actor.read_text().replace('from modular_rl.v2.update_probe import probe', 'from modular_rl.mopd_runtime import probe')
    marker = '            # Train\n'
    if text.count(marker) != 1:
        raise ValueError('Inspect native actor before inserting the MOPD objective probe')
    text = text.replace(marker, '            from modular_rl.mopd_runtime import before_probe\n'
        '            before_probe(self, rollout_id, rollout_data, data_iterator, num_microbatches)\n\n' + marker)
    actor.write_text(text)
    buffer = source / 'slime/ray/buffer.py'
    text = buffer.read_text()
    marker = '            teacher_time = time() - teacher_start_time\n'
    if text.count(marker) != 1:
        raise ValueError('Inspect native buffer before recording teacher routing')
    buffer.write_text(text.replace(marker, '            from modular_rl.mopd_runtime import record_teacher_scores\n'
        '            record_teacher_scores(self.args, rollout_id, data)\n' + marker))
    jobs = []
    for bank, mapping in BANKS.items():
        name = 'mopd-' + bank
        output = experiment / 'runs' / name
        output.mkdir(parents=True)
        teacher_bank = root / 'teachers/banks' / bank
        reward_gpus = 1 if bank == 'mid' else 2
        cards = 6 + reward_gpus
        extra = arguments(Cell('Qwen3-0.6B-Base', 2048, 'control', task='math'))
        extra += (f' --num-rollout {updates} --eval-interval {eval_interval} --rollout-batch-size 8 --n-samples-per-prompt 1'
                  ' --over-sampling-batch-size 8 --rollout-max-response-len 2048 --eval-max-response-len 2048'
                  ' --custom-generate-function-path modular_rl.mopd_runtime.generate'
                  ' --custom-rm-path modular_rl.mopd_runtime.reward'
                  ' --rollout-function-path modular_rl.mopd_runtime.generate_rollout'
                  ' --loss-type custom_loss --custom-loss-function-path slime.backends.megatron_utils.opd_loss.opd_loss_function'
                  ' --disable-compute-advantages-and-returns --weight-decay 0.01'
                  f' --opd-teacher-dir {teacher_bank} --opd-teacher-gpus 2 --environment-list math code if chat')
        env = {'RLVE_REPO': source, 'RLVE_SCRATCH': base.parents[1],
            'RLVE_IMAGE': base.parents[1] / 'images/slime-v0.5.0rc0-cu126.sif',
            'RLVE_WANDB_TEAM': 'david-heineman', 'WANDB_BASE_URL': 'https://api.forge.coreweave.com',
            'MODULAR_RL_PROFILE': 'v2', 'MODULAR_RL_TASKS': 'math,code,if,chat',
            'MODULAR_RL_INITIALIZER': 'base', 'MODULAR_RL_INITIALIZERS': registry,
            'MODULAR_RL_INITIALIZERS_SHA256': sha256(registry),
            'MODULAR_RL_RUN_NAME': name, 'MODULAR_RL_WANDB_PROJECT': PROJECT,
            'MODULAR_RL_RUNTIME': experiment / 'runtime', 'MODULAR_RL_SELECTION': experiment / 'selection',
            'MODULAR_RL_DATA': base / 'data', 'MODULAR_RL_OUTPUT': output,
            'MODULAR_RL_HF_HOME': base.parents[1] / 'tinker-mopd/hf',
            'MODULAR_RL_VALIDATION_PROMPTS': 64, 'MODULAR_RL_PROMPTS_PER_UPDATE': 8,
            'MODULAR_RL_CHECKPOINT_INTERVAL': eval_interval,
            'MODULAR_RL_MAX_RESPONSE_TOKENS': 2048, 'MODULAR_RL_BASE_TOKENIZER': 1,
            'MODULAR_RL_SKIP_ZERO_ADVANTAGE': 0, 'MODULAR_RL_GROUP': 'modular-mopd',
            'MODULAR_RL_TEACHER_BANK': teacher_bank, 'MODULAR_RL_CUDA_CACHE_CLEAR_INTERVAL': 1,
            'MODULAR_RL_REWARD_GPUS': reward_gpus,
            'MODULAR_RL_UPDATE_PROBE_INTERVAL': 10,
            'PYTORCH_CUDA_ALLOC_CONF': 'garbage_collection_threshold:0.7,max_split_size_mb:128',
            'MODULAR_RL_EXTRA_ARGS': extra}
        script = output / 'run.sbatch'
        body = ('#!/usr/bin/env bash\n#SBATCH --account=miso\n'
                '#SBATCH --partition=miso,jag-lo,miso-lo,sphinx-lo,sc-loprio\n'
                '#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --requeue\n#SBATCH --open-mode=append\nset -euo pipefail\n')
        body += shell_exports(env) + shell_exports({f'APPTAINERENV_{k}': v for k, v in env.items()})
        body += ('export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY:?}"\n'
                 'source "$RLVE_REPO/scripts/slurm/common.sh"\n'
                 'rlve_apptainer bash modular_rl/mopd_allocation.sh "$@"\n')
        script.write_text(body)
        command = ['sbatch', '--parsable', f'--job-name={name}', f'--gres=gpu:{cards}',
            '--constraint=40G|48G|80G|141G', f'--cpus-per-task={6 * cards + 4}',
            f'--mem={64 + 32 * (cards - 1)}G', '--time=23:50:00',
            '--exclude=viscam10,jagupard33,jagupard34', f'--chdir={source}',
            f'--output={output}/slurm-%j.out', str(script)]
        jobs.append({'name': name, 'bank': bank, 'teachers': mapping, 'teacher_bank': str(teacher_bank),
            'gpus': cards, 'student_gpus': 4, 'teacher_gpus': 2, 'reward_gpus': reward_gpus, 'output': str(output),
            'command': command, 'script_sha256': sha256(script)})
    plan = {'runs': jobs, 'source': str(source), 'initializers': str(registry),
        'initializers_sha256': sha256(registry), 'train_steps': updates, 'train_rollouts': updates * 8,
        'prompts_per_step': 8, 'samples_per_prompt': 1, 'evaluation_interval': eval_interval,
        'evaluation_interval_train_rollouts': eval_interval * 8,
        'validation_prompts_per_task': 64, 'validation_samples_per_prompt': 8, 'max_response_tokens': 2048,
        'student_initializer': 'Qwen/Qwen3-0.6B-Base', 'lr': 2e-6, 'weight_decay': .01,
        'objective': 'native sampled reverse KL MOPD; per-task frozen teachers; no environment reward in loss',
        'selection_sha256': {name: sha256(experiment / 'selection' / name)
                            for name in ('mixed-train.jsonl', 'mixed-val.jsonl', 'mixed-selection.json')},
        'source_sha256': {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob('*')) if p.is_file()}}
    write_json(experiment / 'launch-plan.json', plan)
    print('Prepared', [r['name'] for r in jobs], 'with', updates * 8, 'rollouts each', flush=True)


def submit_ready(root):
    experiment = root / 'students'
    plan = json.loads((experiment / 'launch-plan.json').read_text())
    jobs_file = experiment / 'jobs.json'
    jobs = json.loads(jobs_file.read_text()) if jobs_file.exists() else []
    submitted = {j['name'] for j in jobs}
    ready = [r for r in plan['runs'] if r['name'] not in submitted and (Path(r['teacher_bank']) / 'bank.json').exists()]
    if not ready:
        return
    for name, expected in plan.get('selection_sha256', {}).items():
        if sha256(experiment / 'selection' / name) != expected:
            raise ValueError('Prepared mixed-task data changed: ' + name)
    for name, expected in plan['source_sha256'].items():
        if sha256(Path(plan['source']) / name) != expected:
            raise ValueError('Frozen MOPD source changed: ' + name)
    for row in ready:
        script = Path(row['command'][-1])
        if sha256(script) != row['script_sha256'] or any('dependency' in p for p in row['command']):
            raise ValueError('MOPD launch script changed or has an unnecessary dependency')
        bank = Path(row['teacher_bank'])
        if json.loads((bank / 'bank.json').read_text()) != row['teachers']:
            raise ValueError('Teacher bank mapping changed')
        for i, (task, teacher) in enumerate(row['teachers'].items()):
            marker = json.loads((bank / f'{i:02d}-{task}/weights_only_complete.json').read_text())
            if marker['name'] != teacher or marker['checkpoint_step'] != 999 or marker['train_rollouts'] != 8000:
                raise ValueError('Wrong/incomplete MOPD teacher')
        job_id = subprocess.check_output(row['command'], text=True).strip().split(';')[0]
        jobs.append({**row, 'job_id': job_id})
        write_json(jobs_file, jobs)
        print('Submitted', row['name'], job_id, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'submit-ready'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--master', type=Path)
    parser.add_argument('--updates', type=int, default=500)
    parser.add_argument('--eval-interval', type=int, default=64)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root.resolve(), args.master.resolve(), Path(__file__).resolve().parents[1], args.updates, args.eval_interval)
    else:
        submit_ready(args.root.resolve())
