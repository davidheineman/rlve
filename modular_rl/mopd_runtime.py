"""Mixed-task adapters and diagnostics for the native MOPD trainer."""

import copy
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from modular_rl.v2 import rollout

TASKS = ('math', 'code', 'if', 'chat')


def record(filename, value):
    directory = Path(os.environ['MODULAR_RL_RUN_DIR'])
    with (directory / filename).open('a') as stream:
        stream.write(json.dumps(value) + '\n')


async def generate(args, sample, sampling_params):
    task = sample.metadata['task']
    if task == 'if':
        from modular_rl.v2.if_rollout import generate as generate_if
        return await generate_if(args, sample, sampling_params)
    local = copy.copy(args)
    local.answer_forcing_extra_tokens = {'math': 256, 'code': 512, 'chat': 128}[task]
    return await rollout.generate(local, sample, sampling_params)


async def reward(args, sample, **kwargs):
    if sample.metadata['task'] == 'if':
        from modular_rl.if_grader import reward as reward_if
        return await reward_if(args, sample, **kwargs)
    return await rollout.reward(args, sample, **kwargs)


def generation_health(samples, counts):
    """Track termination separately from task correctness."""
    grouped = {task: [] for task in TASKS}
    for sample, count in zip(samples, counts, strict=True):
        grouped[sample.metadata['task']].append((sample, count))
    result = {}
    for task, rows in grouped.items():
        if not rows:continue
        n = len(rows)
        result[task] = {'responses': n,
            'mean_tokens': sum(count['sampled_tokens'] for _, count in rows) / n,
            'truncated_fraction': sum(count['truncated'] for _, count in rows) / n,
            'forced_fraction': sum(count['forced'] for _, count in rows) / n,
            'endoftext_fraction': sum(sample.tokens[-1] == 151643 for sample, _ in rows) / n,
            'im_end_fraction': sum(sample.tokens[-1] == 151645 for sample, _ in rows) / n}
    return result


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    from slime.rollout.sglang_rollout import generate_rollout as original

    rollout._COUNTS.clear()
    rollout._SAMPLES.clear()
    started = time.monotonic()
    result = original(args, rollout_id, data_buffer, evaluation=evaluation)
    samples = rollout._SAMPLES
    per_task = Counter(s.metadata['task'] for s in samples)
    expected = (int(os.environ['MODULAR_RL_VALIDATION_PROMPTS']) * 8 if evaluation else 2)
    if per_task != Counter({task: expected for task in TASKS}):
        raise ValueError(f'MOPD must retain a balanced complete panel: {per_task}')
    if len(rollout._COUNTS) != len(samples):
        raise ValueError('Missing MOPD generation accounting')
    if evaluation:
        if len(result['val_envs']['rewards']) != 4 * expected:
            raise ValueError('Incomplete MOPD validation rewards')
        rollout.export_evaluation_samples(Path(os.environ['MODULAR_RL_RUN_DIR']), rollout_id,
                                         samples, args.wandb_run_name, args.wandb_group)
    phase = 'eval' if evaluation else 'train'
    health = generation_health(samples, rollout._COUNTS)
    record('generation.jsonl', {'phase': phase, 'rollout_id': rollout_id,
        'responses': len(samples), 'per_task': dict(per_task), 'wall_seconds': time.monotonic() - started,
        'per_task_generation': health,
        'sampled_token_budget': 2048, 'sampled_tokens': sum(r['sampled_tokens'] for r in rollout._COUNTS),
        'max_sampled_tokens': max(r['sampled_tokens'] for r in rollout._COUNTS),
        'forced_responses': sum(r['forced'] for r in rollout._COUNTS)})
    if args.use_wandb:
        import wandb
        from slime.ray.buffer import wandb_step
        wandb.log({'david/step': wandb_step(rollout_id, args),
                   **{f'david/{phase}/generation/{task}/{key}': value
                      for task, values in health.items() for key, value in values.items()}})
    return result


def record_teacher_scores(args, rollout_id, samples):
    import math

    rows = []
    bank = json.loads((Path(args.opd_teacher_dir) / 'bank.json').read_text())
    for sample in samples:
        task = sample.metadata['environment']
        old, teacher = sample.rollout_log_probs, sample.teacher_log_probs
        mask = sample.loss_mask or [1] * sample.response_length
        if len(teacher) != sample.response_length or not all(math.isfinite(v) for v in teacher):
            raise ValueError('Teacher produced invalid response log probabilities')
        advantages = [t - p for t, p, keep in zip(teacher, old, mask, strict=True) if keep]
        rows.append({'sample': sample.index, 'task': task, 'teacher': bank[task],
                     'tokens': len(advantages), 'mean_teacher_minus_student_logp': sum(advantages) / len(advantages),
                     'last_token_id': sample.tokens[-1], 'last_teacher_logp': teacher[-1],
                     'last_student_logp': old[-1], 'last_teacher_minus_student_logp': teacher[-1] - old[-1]})
    record('teacher-scores.jsonl', {'rollout_id': rollout_id, 'samples': rows,
        'student_eos_token_id': getattr(args, 'opd_student_eos_token_id', None),
        'teacher_eos_token_id': getattr(args, 'opd_teacher_eos_token_id', None)})


def before_probe(actor, rollout_id, rollout_data, data_iterator, num_microbatches):
    if rollout_id % int(os.environ.get('MODULAR_RL_UPDATE_PROBE_INTERVAL', '10')):
        return
    from slime.backends.megatron_utils.model import forward_only

    actor._mopd_before = [v.detach().float().cpu() for v in
        forward_only(actor.args, actor.model, data_iterator, num_microbatches)['log_probs']]


def probe(actor, rollout_id, rollout_data, data_iterator, num_microbatches):
    if rollout_id % int(os.environ.get('MODULAR_RL_UPDATE_PROBE_INTERVAL', '10')):
        return
    import torch
    from megatron.core import mpu
    from slime.backends.megatron_utils.model import forward_only

    after = forward_only(actor.args, actor.model, data_iterator, num_microbatches)['log_probs']
    rows = []
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    for i, (before, new, old, teacher, mask) in enumerate(zip(actor._mopd_before, after,
            rollout_data['rollout_log_probs'], rollout_data['teacher_log_probs'], rollout_data['loss_masks'], strict=True)):
        new, old, teacher = (v.detach().float().cpu() for v in (new, old, teacher))
        keep = torch.as_tensor(mask, dtype=torch.bool).cpu()
        before, new, old, teacher = (v[keep] for v in (before, new, old, teacher))
        gain = ((new - old).exp() - (before - old).exp()) * (teacher - old)
        if not torch.isfinite(gain).all() or not gain.numel():
            raise ValueError('Nonfinite MOPD objective probe')
        rows.append({'sample': rank + i * world, 'tokens': gain.numel(),
                     'surrogate_improvement': float(gain.mean()),
                     'mean_abs_delta_logp': float((new - before).abs().mean())})
    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, rows,
        group=mpu.get_data_parallel_group_gloo(with_context_parallel=True))
    if rank == 0:
        rows = sorted([r for shard in gathered for r in shard], key=lambda r: r['sample'])
        if [r['sample'] for r in rows] != list(range(8)):
            raise ValueError('MOPD update probe omitted or duplicated samples')
        metrics = {'david/step': rollout_id, 'david/opd/surrogate_improvement':
                   sum(r['surrogate_improvement'] for r in rows) / len(rows)}
        record('update-probes.jsonl', {'rollout_id': rollout_id, 'samples': rows, 'metrics': metrics})
        if actor.args.use_wandb:
            import wandb
            wandb.log(metrics)


def main():
    repo = Path(os.environ['RLVE_REPO'])
    spec = importlib.util.spec_from_file_location('mopd_native_training_driver', repo / 'train.py')
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    args = native.parse_args()
    from modular_rl.initializers import validate_training
    from modular_rl.v2.driver import configured_run_name
    initializer = os.environ.get('MODULAR_RL_INITIALIZER', 'base')
    identity = validate_training(args, os.environ['MODULAR_RL_INITIALIZERS'], initializer,
        os.environ['MODULAR_RL_RUN_DIR'], os.environ['MODULAR_RL_INITIALIZERS_SHA256'])
    if (args.rollout_batch_size != 8 or args.n_samples_per_prompt != 1
            or args.n_samples_per_eval_prompt != 8 or args.compute_advantages_and_returns
            or args.loss_type != 'custom_loss' or args.balance_data or args.rollout_shuffle
            or args.environment_list != list(TASKS) or args.rollout_temperature != 1
            or args.rollout_max_response_len != 2048 or args.eval_max_response_len != 2048
            or os.environ.get('MODULAR_RL_SKIP_ZERO_ADVANTAGE') == '1'):
        raise ValueError('MOPD settings differ from the balanced on-policy distillation recipe')
    args.sglang_dtype = 'bfloat16'
    args.wandb_run_name = configured_run_name('mopd')
    if os.environ.get('MODULAR_RL_EOS_ALIGNMENT_REQUIRED') == '1':
        student_eos = int(os.environ.get('MODULAR_RL_STUDENT_EOS_TOKEN_ID', '151643'))
        teacher_eos = int(os.environ.get('MODULAR_RL_TEACHER_EOS_TOKEN_ID', '151645'))
        if (getattr(args, 'opd_student_eos_token_id', None) != student_eos
                or getattr(args, 'opd_teacher_eos_token_id', None) != teacher_eos
                or set(args.rollout_stop_token_ids or []) != {151643, 151645}
                or os.environ.get('MODULAR_RL_CODE_FORCING_FALLBACK') != '1'):
            raise ValueError('The v2 MOPD rerun requires both EOS aliases, teacher label alignment, and code fallback')
        record('training-protocol.jsonl', {'name': args.wandb_run_name, 'wandb_project': args.wandb_project,
            'student_initializer': initializer, 'student_initializer_kind': identity['kind'],
            'student_native_initializer': identity['native'],
            'student_eos_token_id': args.opd_student_eos_token_id,
            'teacher_eos_token_id': args.opd_teacher_eos_token_id, 'stop_token_ids': args.rollout_stop_token_ids,
            'code_forcing_fallback': True, 'tasks': list(TASKS), 'train_steps': args.num_rollout,
            'teacher_bank': args.opd_teacher_dir, 'hf_checkpoint': args.hf_checkpoint})
    original_step, original_eval = native.log_step_perf, native.evaluate_student
    evaluated = False

    def step(args, rollout_id, seconds, tokens):
        original_step(args, rollout_id, seconds, tokens)
        record('progress.jsonl', {'phase': 'train', 'completed_updates': rollout_id + 1,
                                'train_rollouts': (rollout_id + 1) * 8, 'wall_seconds': seconds})

    def evaluate(args, actor, manager, rollout_id):
        nonlocal evaluated
        start = time.monotonic()
        accuracy = original_eval(args, actor, manager, rollout_id)
        updates = 0 if not evaluated and rollout_id == 0 and args.start_rollout_id == 0 else rollout_id + 1
        record('progress.jsonl', {'phase': 'eval', 'completed_updates': updates,
            'train_rollouts': updates * 8, 'accuracy': accuracy, 'wall_seconds': time.monotonic() - start})
        evaluated = True
        return accuracy

    native.log_step_perf, native.evaluate_student = step, evaluate
    native.train(args)
    from slime.utils.checkpoint_state import read_checkpoint_state
    state = read_checkpoint_state(args.save)
    if not state['completed']:
        raise ValueError('MOPD training did not complete')
    result = {'completed': True, 'updates': state['rollout_id'] + 1,
              'train_rollouts': (state['rollout_id'] + 1) * 8, 'samples_per_prompt': 1,
              'prompts_per_update': 8, 'checkpoint': state, 'teacher_bank': args.opd_teacher_dir}
    (Path(os.environ['MODULAR_RL_RUN_DIR']) / 'completed.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Completed MOPD:', result, flush=True)


if __name__ == '__main__':
    main()
