"""Scheduling and path selection for the final three OOD evaluations."""


def is_final_ood_update(rollout_id, num_rollout, final_ood_eval_path):
    # Eval after the last, tenth-last, and twentieth-last updates.
    return final_ood_eval_path is not None and rollout_id + 1 in (
        num_rollout - 20, num_rollout - 10, num_rollout
    )


def eval_path(name, path, rollout_id, num_rollout, final_ood_eval_path):
    if name == "test_envs" and is_final_ood_update(rollout_id, num_rollout, final_ood_eval_path):
        return final_ood_eval_path
    return path


def should_run_dataset(name, rollout_id, num_rollout, eval_interval, final_ood_eval_path, final_ood_only):
    if not final_ood_only:
        return True
    if name == "test_envs":
        return is_final_ood_update(rollout_id, num_rollout, final_ood_eval_path)
    if name == "val_envs":
        return rollout_id == 0 or (rollout_id + 1) % eval_interval == 0
    return True
