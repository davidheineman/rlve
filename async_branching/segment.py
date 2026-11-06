"""A resumable student segment or fresh specialist branch on the existing Slime backend.

This module is deliberately separate from train.py and the running sweep
launchers. Teacher GRPO and student sampled reverse KL use their existing
configurations, actor implementations, rollout generation, and reward code.
"""

import os
from pathlib import Path
import subprocess
import sys
import time

from .io import read_json, write_json


def configure_args(args, task):
    student = task["kind"] == "student"
    start = task.get("start_rollouts", 0) // 8 if student else 0
    count = task["cap"] // 8
    args.num_rollout = start + count
    args.save = task["checkpoint"]
    args.save_interval = max(2, count)
    args.no_save_optim = not student
    args.no_save_rng = not student
    args.no_load_optim = not (student and start > 0)
    args.no_load_rng = not (student and start > 0)
    # Each merge has a different end update, so Megatron derives a different
    # scheduler horizon from num_rollout. Resume the saved scheduler and its
    # sample counter together with optimizer/RNG state instead of comparing
    # its original horizon to this segment's end. The experiment uses a
    # constant learning rate and weight decay throughout the student run.
    args.use_checkpoint_opt_param_scheduler = student and start > 0
    args.override_opt_param_scheduler = False
    if student and start > 0:
        args.load = task["parent"]["resume"]
        args.finetune = False
    args.early_stop_val_accuracy = task["target"] if student else .98
    if not 0 <= args.early_stop_val_accuracy <= 1:
        raise ValueError("Async recovery accuracy must be between zero and one")
    args.early_stop_eval_names = ["gate_envs"] if student else ["val_envs"]
    args.rollout_function_path = "async_branching.rollout.generate_rollout"
    args.eval_function_path = "slime.rollout.sglang_rollout.generate_rollout"
    args.seed = task["train_seed"] % (2**31 - 1)
    replay = task.get("opd_replay") if student else None
    args.opd_replay_fraction = replay["fraction"] if replay else 0.0
    args.opd_replay_capacity_per_env = replay["capacity_per_env"] if replay else 400
    args.opd_replay_clip_epsilon = replay["clip_epsilon"] if replay else 0.2
    args.opd_replay_load = task["parent"].get("replay") if replay else None
    return args, start, count


def worker_arguments(task):
    """Supply segment settings before Slime and Megatron validate arguments.

    The shell launcher and GPU runtime probe share this list. configure_args
    still applies optimizer continuation after parsing, before actor creation.
    """
    from types import SimpleNamespace
    args, _, _ = configure_args(SimpleNamespace(), task)
    # Slime's shared parser requires a positive stopping threshold. A teacher
    # with zero validation accuracy has an exact zero recovery target; use a
    # positive placeholder only for parsing. configure_args restores zero
    # before baseline evaluation, so any measured accuracy recovers it.
    parser_accuracy = args.early_stop_val_accuracy if args.early_stop_val_accuracy > 0 else 1e-12
    flags = ["--num-rollout", str(args.num_rollout),
            "--save", args.save, "--save-interval", str(args.save_interval),
            "--transactional-checkpoints",
            "--early-stop-val-accuracy", str(parser_accuracy),
            "--early-stop-eval-names", *args.early_stop_eval_names,
            "--seed", str(args.seed)]
    if args.use_checkpoint_opt_param_scheduler:
        # Megatron's public CLI uses an underscore in opt_param.
        flags.append("--use-checkpoint-opt_param-scheduler")
    if args.opd_replay_fraction:
        flags.extend(["--opd-replay-fraction", str(args.opd_replay_fraction),
                      "--opd-replay-capacity-per-env", str(args.opd_replay_capacity_per_env),
                      "--opd-replay-clip-epsilon", str(args.opd_replay_clip_epsilon)])
        if args.opd_replay_load:
            flags.extend(["--opd-replay-load", args.opd_replay_load])
    return flags


def validate_loaded_updates(starts, expected, finetune=False):
    # Megatron resets a release/finetune checkpoint's iteration to zero;
    # RayTrainGroup reports that plus one. Fresh branches explicitly begin
    # at rollout zero, as train.py does, while student continuations must
    # match their actual saved global update exactly.
    loaded = set(starts)
    allowed = {0, 1} if expected == 0 and finetune else {expected}
    if len(loaded) != 1 or not loaded <= allowed:
        raise RuntimeError(f"Continuation loaded update {starts}, expected {expected}")


def run(args, task):
    import ray
    import wandb
    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
    from slime.ray.placement_group import create_actor_group, create_placement_groups, create_rollout_manager
    from slime.ray.opd_teachers import create_teacher_scorers
    from slime.utils.wandb_utils import init_wandb_primary
    from train import log_step_perf
    from .exports import validation_functions

    # Resolve export dependencies before spending GPU time on a branch.
    tensor_shapes, validate_export = validation_functions()

    args, start, count = configure_args(args, task)
    student = task["kind"] == "student"
    groups = create_placement_groups(args)
    run_id = init_wandb_primary(args)
    actor = create_actor_group(args, groups["actor"], wandb_run_id=run_id)
    rollout = create_rollout_manager(args, groups["rollout"], wandb_run_id=run_id)
    scorers = []
    if student:
        scorers = create_teacher_scorers(args, groups["teacher"])
        ray.get(rollout.controller.set_teacher_scorers.remote(scorers))
    starts = ray.get(actor.async_init(args, role="actor", with_ref=False))
    validate_loaded_updates(starts, start, getattr(args, "finetune", False))
    ray.get(actor.async_init_weight_update_connections(rollout))
    if args.offload:
        ray.get(rollout.async_onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
    ray.get(actor.async_update_weights())
    if args.offload:
        ray.get(rollout.async_onload(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

    def evaluate(step):
        return ray.get(rollout.async_eval(step))

    def finish():
        ray.get(rollout.controller.finish_wandb.remote())
        if args.use_wandb:
            wandb.finish()
        # CPU conversion temporarily holds another model copy. Release the
        # training optimizer/model and student scorers before that allocation
        # instead of retaining them until the worker process exits.
        for handle in [*actor._actor_handlers, *scorers]:
            ray.kill(handle, no_restart=True)

    accuracy = evaluate(max(0, start - 1))
    if student and accuracy >= task["target"]:
        finish()
        result = {"rollouts": 0, "accuracy": accuracy, "hf": task["parent"]["hf"],
                  "resume": task["parent"]["resume"], "reason": "recovered_before_training"}
        if args.opd_replay_fraction:
            result["replay"] = task["parent"].get("replay")
        return result
    if not student and accuracy >= .98:
        finish()
        import shutil
        parent = Path(task["parent"]["hf"])
        output = Path(task["directory"]) / "hf-final"

        def link_or_copy(source, destination):
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
            return destination

        shutil.copytree(parent, output, copy_function=link_or_copy)
        tensors, size = validate_export(output, Path(task["base_hf"]), tensor_shapes(Path(task["base_hf"]))[0])
        write_json(output.parent / "weights_only_complete.json", {
            "format": "hf-safetensors", "path": "hf-final", "checkpoint_step": 0,
            "tensor_count": tensors, "weight_bytes": size})
        return {"rollouts": 0, "accuracy": accuracy, "hf": str(output),
                "resume": None, "reason": "teacher_already_at_threshold"}
    if not student:
        write_json(Path(task["directory"]) / "progress.json", {
            "rollouts": 0, "accuracy": accuracy, "next_validation_rollouts": min(400, task["cap"])})
    last_eval = None
    for step in range(start, start + count):
        began = time.monotonic()
        data = ray.get(rollout.async_generate(step))
        if args.offload:
            ray.get(rollout.async_offload())
        ray.get(actor.async_train(step, data))
        if args.offload:
            ray.get(actor.async_offload())
            ray.get(rollout.async_onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
        ray.get(actor.async_update_weights())
        if args.offload:
            ray.get(rollout.async_onload(tags=[GPU_MEMORY_TYPE_KV_CACHE]))
        log_step_perf(args, step, time.monotonic() - began,
                      ray.get(rollout.controller.get_last_rollout_tokens.remote()))
        used = (step - start + 1) * 8
        if args.use_wandb:
            wandb.log({"david/step": step, "async/simulated_clock": task["start_clock"] + used,
                       "async/student_rollouts" if student else "async/teacher_device_rollouts":
                       (task["start_rollouts"] if student else task["teacher_start_rollouts"]) + used,
                       "async/teacher_generation": task["generation"]})
        if (step + 1) % 50 == 0 or used == task["cap"]:
            accuracy = evaluate(step)
            last_eval = step
            write_json(Path(task["directory"]) / "progress.json",
                       {"rollouts": used, "accuracy": accuracy, "clock": task["start_clock"] + used,
                        "next_validation_rollouts": (min(task["cap"], used + 400)
                            if not student and used < task["cap"] and accuracy < .98 else None)})
            if accuracy >= args.early_stop_val_accuracy:
                break
    if last_eval != step:
        accuracy = evaluate(step)
    if args.offload:
        ray.get(rollout.async_offload())
        ray.get(actor.async_onload())
    # The single latest student continuation retains optimizer/RNG state.
    # Branch parents are exported model-only HF snapshots; teachers always
    # start fresh optimizer state from their selected student weights.
    ray.get(actor.async_save_model(step, with_optimizer=student))
    replay_path = None
    if args.opd_replay_fraction:
        replay_path = ray.get(rollout.controller.save_opd_replay.remote(
            str(Path(task["directory"]) / "opd-replay.json.gz")))
    finish()
    checkpoint = Path(task["checkpoint"])
    iteration = int((checkpoint / "latest_checkpointed_iteration.txt").read_text())
    output = Path(task["directory"]) / "hf-final"
    temporary = output.with_name("hf.incomplete")
    if temporary.exists():
        import shutil
        shutil.rmtree(temporary)
    import json
    vocab = json.loads((Path(task["base_hf"]) / "config.json").read_text())["vocab_size"]
    subprocess.run([sys.executable, "tools/convert_torch_dist_to_hf.py",
                    "--input-dir", str(checkpoint / f"iter_{iteration:07d}"),
                    "--output-dir", str(temporary), "--origin-hf-dir", task["base_hf"],
                    "--vocab-size", str(vocab)], check=True)
    tensors, weight_bytes = validate_export(temporary, Path(task["base_hf"]), tensor_shapes(Path(task["base_hf"]))[0])
    os.replace(temporary, output)
    write_json(output.parent / "weights_only_complete.json", {
        "format": "hf-safetensors", "path": "hf-final", "checkpoint_step": iteration,
        "tensor_count": tensors, "weight_bytes": weight_bytes,
    })
    used = (step - start + 1) * 8
    result = {"rollouts": used, "accuracy": accuracy, "hf": str(output),
              "resume": str(checkpoint) if student else None,
              "reason": "recovered" if accuracy >= args.early_stop_val_accuracy else "rollout_cap"}
    if replay_path:
        result["replay"] = replay_path
    return result


def main():
    from slime.utils.arguments import parse_args

    task = read_json(os.environ["RLVE_ASYNC_TASK"])
    args = parse_args()
    if os.environ.get("RLVE_ASYNC_PARSE_ONLY") == "1":
        import json
        args, start, count = configure_args(args, task)
        print(json.dumps({"kind": task["kind"], "start_update": start, "updates": count,
                          "actor_gpus": args.actor_num_gpus_per_node, "scorer_gpus": args.opd_teacher_gpus,
                          "load_optimizer": not args.no_load_optim, "save_optimizer": not args.no_save_optim,
                          "loss": args.loss_type, "stop_metric": args.early_stop_eval_names,
                          "stop_accuracy": args.early_stop_val_accuracy,
                          "checkpoint": args.save}))
        return
    result = run(args, task)
    write_json(Path(task["directory"]) / "result.json", result)
    print(f"ASYNC_RESULT {result}", flush=True)


if __name__ == "__main__":
    main()
