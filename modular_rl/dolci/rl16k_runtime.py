"""Enforce the approved fresh-initializer 16K GRPO protocol before training."""

import json
import os
from pathlib import Path


def validate(args):
    config = json.loads(Path(os.environ['MODULAR_RL_RUN_CONFIG']).read_text())
    task = os.environ['MODULAR_RL_TASKS']
    group = 32 if task == 'math' else 16
    checks = {
        'n_samples_per_prompt': group, 'rollout_batch_size': 1,
        'global_batch_size': group, 'num_rollout': 16000 // group,
        'eval_interval': 512 // group, 'n_samples_per_eval_prompt': 8,
        'rollout_max_response_len': 2048, 'eval_max_response_len': 2048,
        'lr': 2e-6, 'weight_decay': 0.0, 'train_reuse_epochs': 1,
        'advantage_estimator': 'grpo', 'compute_advantages_and_returns': True,
        'custom_loss_function_path': 'modular_rl.v2.losses.fixed_token_policy_loss',
        'eval_temperature': .7, 'eval_top_p': .95,
        'rollout_temperature': 1.0, 'rollout_top_p': 1.0,
        'early_stop_val_accuracy': None, 'early_stop_patience_evals': None,
        'actor_num_gpus_per_node': 2 if task == 'chat' else 4,
    }
    for key, expected in checks.items():
        if getattr(args, key, None) != expected:
            raise ValueError(f'16K protocol mismatch: {key}: {getattr(args,key,None)!r} != {expected!r}')
    if set(args.rollout_stop_token_ids or []) != {151643, 151645}:
        raise ValueError('Both EOS aliases must terminate generation')
    if config['group_size'] != group or config['training_responses'] != 16000:
        raise ValueError('Run config disagrees with the approved budget')
    if Path(args.hf_checkpoint).resolve() != Path(config['initializer_hf']).resolve():
        raise ValueError('Wrong fresh initializer')
    protocol = {key: getattr(args, key, None) for key in checks}
    protocol.update(run_name=config['name'], task=task, project=args.wandb_project,
                    initializer=config['initializer_alias'], hf_checkpoint=args.hf_checkpoint,
                    native_initializer=args.ref_load, stop_token_ids=args.rollout_stop_token_ids,
                    training_responses=16000, config=os.environ['MODULAR_RL_RUN_CONFIG'])
    if args.wandb_project != 'modular-mopd-rl-v1':
        raise ValueError('Unexpected W&B project')
    with (Path(os.environ['MODULAR_RL_RUN_DIR'])/'training-protocol.jsonl').open('a') as stream:
        stream.write(json.dumps(protocol)+'\n')


def main():
    from modular_rl.v2 import driver
    original = driver.validate_run_settings

    def checked(args):
        original(args)
        validate(args)

    driver.validate_run_settings = checked
    driver.main()


if __name__ == '__main__':
    main()
