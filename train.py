import os
import time

import ray
import wandb
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.ray.buffer import wandb_step
from slime.ray.placement_group import create_actor_group, create_placement_groups, create_rollout_manager
from slime.ray.opd_teachers import create_teacher_scorers
from slime.utils.arguments import parse_args
from slime.utils.checkpoint_state import commit_checkpoint, mark_completed, mark_early_stopped, mark_evaluated, mark_plateau_progress, read_checkpoint_state
from slime.utils.final_ood_eval import is_final_ood_update
from slime.utils.early_stopping import update_plateau
from slime.utils.rollout_annealing import rollouts_for_accuracy
from slime.utils.wandb_utils import init_wandb_primary


def log_step_perf(args, rollout_id, step_time, tokens):
    """Wall-clock cost of one training step, measured from the driver.

    The actor-side perf/* timers each cover one phase; this is the whole step - generation,
    log-probs, the optimizer step and the weight sync - and deliberately excludes eval, so
    the number stays comparable on the steps where an eval happens to run.
    """
    log_dict = {
        "david/perf/step_time": step_time,
        "david/perf/tps": tokens / step_time if step_time > 0 else 0.0,
        "david/step": wandb_step(rollout_id, args),
    }
    # These fixed-size runs consume one complete prompt group per update. Keep the
    # cumulative count on a common axis for 8x1 MOPD and 8x16 RL comparisons.
    if args.rollout_anneal_thresholds is None:
        log_dict["david/rollouts"] = (
            (rollout_id + 1) * args.rollout_batch_size * args.n_samples_per_prompt
        )
    print(f"david {rollout_id}: {log_dict}", flush=True)
    if args.use_wandb:
        wandb.log(log_dict)


def evaluate_student(args, actor_model, rollout_manager, rollout_id):
    if not args.fixed_rollout_policy:
        return ray.get(rollout_manager.async_eval(rollout_id))

    # The inference engine normally holds pi_0. Swap only for evaluation, then
    # restore pi_0 before the next training rollout, including on eval failure.
    ray.get(actor_model.async_update_weights(model_tag="actor"))
    try:
        return ray.get(rollout_manager.async_eval(rollout_id))
    finally:
        ray.get(actor_model.async_update_weights(model_tag="fixed_rollout_policy"))


def train(args):
    resumable = getattr(args, "transactional_checkpoints", False)
    final_weights_only = os.environ.get("RLVE_FINAL_WEIGHTS_ONLY") == "1"
    if final_weights_only:
        assert not resumable and args.save and args.save_interval
    if resumable:
        assert (args.rlve or args.rollout_global_dataset) and args.save and args.save_interval
        assert args.save_interval > 1, "Megatron cannot reload a checkpoint at zero-based iteration 0"
        assert not getattr(args, "async_save", False), "Transactional checkpoints require synchronous saves"
    # allocate the GPUs
    pgs = create_placement_groups(args)
    wandb_run_id = init_wandb_primary(args)

    actor_model = create_actor_group(args, pgs["actor"], wandb_run_id=wandb_run_id)

    # create the rollout manager, with sglang engines inside.
    rollout_manager = create_rollout_manager(args, pgs["rollout"], wandb_run_id=wandb_run_id)
    if args.opd_teacher_dir is not None:
        teacher_scorers = create_teacher_scorers(args, pgs["teacher"])
        ray.get(rollout_manager.controller.set_teacher_scorers.remote(teacher_scorers))
    if args.dpo_online:
        from slime.ray.dpo_base import create_base_generator

        base_generator = create_base_generator(args, pgs["teacher"])
        ray.get(rollout_manager.controller.set_dpo_base_generator.remote(base_generator))

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.controller.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
    assert args.num_rollout > 0

    # sync the initialization (model initalization, load checkpoint, etc.)
    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss or args.loss_type == "dpo_loss")
    )
    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if args.rollout_global_dataset:
        ray.get(rollout_manager.controller.load.remote(args.start_rollout_id - 1))
    elif args.rlve:
        ray.get(rollout_manager.controller.load.remote(args.start_rollout_id - 1))
    else :
        assert False, "None of args.rollout_global_dataset or args.rlve is set - there is no data source."

    # initialize the connection for weight update during training
    ray.get(actor_model.async_init_weight_update_connections(rollout_manager))

    if args.offload:
        ray.get(rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    # Initialize the inference engine with the policy used for training rollouts.
    initial_model_tag = "fixed_rollout_policy" if args.fixed_rollout_policy else "actor"
    ray.get(actor_model.async_update_weights(model_tag=initial_model_tag))

    if args.offload:
        ray.get(rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

    if resumable and args.start_rollout_id > 0:
        checkpoint_state = read_checkpoint_state(args.save)
        completed_rollout = checkpoint_state["rollout_id"]
        assert args.start_rollout_id == completed_rollout + 1
        if (
            args.eval_interval is not None
            and (completed_rollout + 1) % args.eval_interval == 0
            and checkpoint_state["evaluated_rollout_id"] < completed_rollout
        ):
            evaluate_student(args, actor_model, rollout_manager, completed_rollout)
            ray.get(rollout_manager.controller.save_runtime.remote(completed_rollout))
            mark_evaluated(args.save, completed_rollout)

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    stop_state = read_checkpoint_state(args.save) if resumable else {}
    early_stopped = stop_state.get("early_stopped", False)
    best_val_accuracy = stop_state.get("plateau_best_accuracy")
    stale_evals = stop_state.get("plateau_stale_evals", 0)
    if early_stopped:
        print("Previous run reached its validation early-stop threshold; finalizing", flush=True)
    rollout_ids = () if early_stopped else range(args.start_rollout_id, args.num_rollout)
    reached_accuracy = False
    reached_plateau = False
    initial_rollout_count = args.n_samples_per_prompt
    current_rollout_count = initial_rollout_count

    def anneal_from_validation(accuracy, rollout_id):
        nonlocal current_rollout_count
        if args.rollout_anneal_thresholds is None:
            return
        selected = rollouts_for_accuracy(
            initial_rollout_count, args.rollout_anneal_thresholds, args.rollout_anneal_sizes, accuracy
        )
        if selected >= current_rollout_count:
            return
        ray.get(rollout_manager.controller.set_rollouts_per_prompt.remote(selected))
        ray.get(actor_model.async_set_rollouts_per_prompt(selected))
        args.n_samples_per_prompt = selected
        args.global_batch_size = args.rollout_batch_size * selected
        current_rollout_count = selected
        print(f"Rollout annealing after validation at update {rollout_id}: "
              f"accuracy={accuracy:.4f}, rollouts_per_prompt={selected}", flush=True)
        if args.use_wandb:
            wandb.log({"david/rollout_anneal/rollouts_per_prompt": selected,
                       "david/rollout_anneal/validation_accuracy": accuracy,
                       "rollout/step": rollout_id})

    if args.rollout_anneal_thresholds is not None and args.start_rollout_id > 0 and not early_stopped:
        # Reconstruct the schedule from the loaded model when resuming.
        anneal_from_validation(evaluate_student(args, actor_model, rollout_manager, args.start_rollout_id - 1),
                               args.start_rollout_id)
    for rollout_id in rollout_ids:
        # TODO extract the duplicated eval logic
        if args.eval_interval is not None and rollout_id == 0:
            initial_val_accuracy = evaluate_student(args, actor_model, rollout_manager, rollout_id)
            anneal_from_validation(initial_val_accuracy, rollout_id)
            if args.early_stop_patience_evals is not None:
                best_val_accuracy, stale_evals, _ = update_plateau(
                    best_val_accuracy, stale_evals, initial_val_accuracy, args.early_stop_min_delta
                )

        step_start_time = time.time()
        rollout_data_ref = ray.get(rollout_manager.async_generate(rollout_id))

        if args.offload:
            ray.get(rollout_manager.async_offload())

        ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if not final_weights_only and args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or rollout_id + 1 == args.num_rollout
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(actor_model.async_save_model(rollout_id, with_optimizer=not final_weights_only))
            if args.rollout_global_dataset:
                ray.get(rollout_manager.controller.save.remote(rollout_id))
            elif args.rlve:
                ray.get(rollout_manager.controller.save.remote(rollout_id))
            else:
                assert False, "None of args.rollout_global_dataset or args.rlve is set - there is no data source."
            if resumable:
                commit_checkpoint(args.save, rollout_id)

        if args.offload:
            ray.get(actor_model.async_offload())
            ray.get(rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

        if not args.fixed_rollout_policy:
            ray.get(actor_model.async_update_weights())

        if args.offload:
            ray.get(rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

        log_step_perf(
            args,
            rollout_id,
            time.time() - step_start_time,
            ray.get(rollout_manager.controller.get_last_rollout_tokens.remote()),
        )

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (getattr(args, "eval_on_last_rollout", False) and rollout_id + 1 == args.num_rollout)
            or is_final_ood_update(rollout_id, args.num_rollout,
                                   getattr(args, "rlve_final_ood_eval_path", None))
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            val_accuracy = evaluate_student(args, actor_model, rollout_manager, rollout_id)
            anneal_from_validation(val_accuracy, rollout_id + 1)
            if resumable and read_checkpoint_state(args.save)["rollout_id"] == rollout_id:
                ray.get(rollout_manager.controller.save_runtime.remote(rollout_id))
                mark_evaluated(args.save, rollout_id)
            if args.early_stop_patience_evals is not None:
                best_val_accuracy, stale_evals, _ = update_plateau(
                    best_val_accuracy, stale_evals, val_accuracy, args.early_stop_min_delta
                )
                if resumable and read_checkpoint_state(args.save)["rollout_id"] == rollout_id:
                    mark_plateau_progress(args.save, rollout_id, best_val_accuracy, stale_evals)
            reached_accuracy = (args.early_stop_val_accuracy is not None and val_accuracy is not None
                                and val_accuracy >= args.early_stop_val_accuracy)
            reached_plateau = (args.early_stop_patience_evals is not None
                               and stale_evals >= args.early_stop_patience_evals
                               and rollout_id + 1 >= args.early_stop_min_rollouts)
            if reached_accuracy or reached_plateau:
                if resumable:
                    if read_checkpoint_state(args.save)["rollout_id"] != rollout_id:
                        # Evaluation runs with the training model asleep when GPUs are
                        # shared with rollout engines. Restore it before checkpointing.
                        if args.offload:
                            ray.get(rollout_manager.async_offload())
                            ray.get(actor_model.async_onload())
                        ray.get(actor_model.async_save_model(rollout_id))
                        ray.get(rollout_manager.controller.save.remote(rollout_id))
                        commit_checkpoint(args.save, rollout_id)
                        ray.get(rollout_manager.controller.save_runtime.remote(rollout_id))
                        mark_evaluated(args.save, rollout_id)
                    if args.early_stop_patience_evals is not None:
                        mark_plateau_progress(args.save, rollout_id, best_val_accuracy, stale_evals)
                reason = "accuracy" if reached_accuracy else "plateau"
                if resumable:
                    mark_early_stopped(args.save, rollout_id, val_accuracy, reason)
                print(f"Early stop at update {rollout_id + 1}: reason={reason}, "
                      f"validation={val_accuracy:.4f}, best={best_val_accuracy}, "
                      f"stale_evals={stale_evals}", flush=True)
                break

    if args.require_early_stop and not (early_stopped or reached_accuracy or reached_plateau):
        ray.get(rollout_manager.controller.finish_wandb.remote())
        if args.use_wandb:
            wandb.finish(exit_code=1)
        raise RuntimeError("Maximum updates reached before validation saturation; refusing to advance the next turn")

    if final_weights_only:
        # Save exactly once, at the terminal trained update (including early stops).
        # No dataloader, rollout, optimizer, or RNG checkpoint is needed in this mode.
        if args.offload:
            ray.get(rollout_manager.async_offload())
            ray.get(actor_model.async_onload())
        ray.get(actor_model.async_save_model(rollout_id, with_optimizer=False))

    # Flush the rollout actor's wandb writer before this process exits and takes the actors
    # with it, otherwise the last eval of the run is silently dropped.
    ray.get(rollout_manager.controller.finish_wandb.remote())
    if args.use_wandb:
        wandb.finish()
    if resumable:
        mark_completed(args.save)


if __name__ == "__main__":
    args = parse_args()
    train(args)
