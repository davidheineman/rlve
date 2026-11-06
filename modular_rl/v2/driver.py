"""Run the native trainer, recording budgets and handling an already-solved baseline."""

import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

from modular_rl.v2.rollout import validation_prompt_count


def training_updates(rollouts=8000, samples=8):
    if rollouts <= 0 or samples <= 0 or rollouts % samples:
        raise ValueError("Response budget must be a positive multiple of group size")
    return rollouts // samples


def validate_generation_budget(total, reserve):
    if total <= 0 or not 0 < reserve < total:
        raise ValueError("Answer reserve must fit inside the response token budget")


def validate_run_settings(args):
    if registry := os.environ.get('MODULAR_RL_INITIALIZERS'):
        from modular_rl.initializers import validate_training

        validate_training(args, registry, os.environ['MODULAR_RL_INITIALIZER'],
                          os.environ['MODULAR_RL_RUN_DIR'], os.environ['MODULAR_RL_INITIALIZERS_SHA256'])
        # Slime intentionally omits dtype from the SGLang CLI, but its engine
        # forwards this attribute to ServerArgs. Match the BF16 policy actor.
        args.sglang_dtype = 'bfloat16'
        policy_gpus = int(os.environ.get('MODULAR_RL_POLICY_GPUS', '1'))
        if policy_gpus not in (1, 2, 4) or args.actor_num_gpus_per_node != policy_gpus:
            raise ValueError('Actor GPU count differs from the declared suite topology')
        if policy_gpus > 1:
            if (args.tensor_model_parallel_size != 1 or args.pipeline_model_parallel_size != 1
                    or args.context_parallel_size != 1 or args.balance_data):
                raise ValueError('Multi-GPU suite requires pure data parallelism with strided samples')
    elif os.environ.get("MODULAR_RL_REQUIRE_PRETRAINED_BASE") == "1":
        from modular_rl.v2.base_models import verify_initialization

        identity = verify_initialization(args.hf_checkpoint, args.ref_load)
        expected = os.environ.get("MODULAR_RL_EXPECTED_BASE_MODEL")
        if expected and identity["repo_id"] != "Qwen/" + expected:
            raise ValueError("Parsed model does not match the explicitly requested pretrained Base checkpoint")
        if Path(args.load).resolve() != Path(args.ref_load).resolve():
            run = Path(os.environ.get("MODULAR_RL_RUN_DIR", ".")).resolve()
            previous = run / "initial-model-identity.json"
            if (Path(args.load).resolve() != run / "ckpt" or not previous.exists()
                    or json.loads(previous.read_text()).get("repo_id") != identity["repo_id"]):
                raise ValueError("Fresh Base runs require an untrained Base seed; only their own Base checkpoint may resume")
        if directory := os.environ.get("MODULAR_RL_RUN_DIR"):
            Path(directory).mkdir(parents=True, exist_ok=True)
            (Path(directory) / "initial-model-identity.json").write_text(json.dumps({
                **identity, "actor_load_for_this_attempt": args.load,
                "untrained_reference": args.ref_load,
                "prompt_format": "official tokenizer assets from the same pinned pretrained Base repository",
            }, indent=2) + "\n")
    prompts = int(os.environ.get("MODULAR_RL_PROMPTS_PER_UPDATE", "1"))
    samples = int(os.environ.get("MODULAR_RL_SAMPLES_PER_PROMPT", "8"))
    if (
        prompts <= 0
        or samples not in (8, 16, 32)
        or args.rollout_batch_size != prompts
        or args.n_samples_per_prompt != samples
        or args.n_samples_per_eval_prompt != 8
        or args.rollout_max_response_len != args.eval_max_response_len
    ):
        raise ValueError(
            "v2 requires the declared train prompt count and group size (defaults 1 and 8), "
            "8 validation samples per prompt, and equal train/eval token caps"
        )
    validate_generation_budget(
        args.rollout_max_response_len, args.answer_forcing_extra_tokens
    )
    if (
        getattr(args, "advantage_estimator", "grpo") in ("grpo", "gspo")
        and getattr(args, "kl_coef", 0) != 0
    ):
        raise ValueError(
            "GRPO/GSPO ignores kl_coef in this trainer; use use_kl_loss, "
            "kl_loss_coef and kl_loss_type=k3 for an explicit reference penalty"
        )
    if getattr(args, "use_kl_loss", False) and getattr(args, "kl_loss_type", "kl") == "kl":
        raise ValueError(
            "The direct linear kl loss has a reference-independent gradient; "
            "select kl_loss_type=k3 for reference anchoring"
        )
    for name, actual in (
        ("MODULAR_RL_MAX_RESPONSE_TOKENS", args.rollout_max_response_len),
        ("MODULAR_RL_ANSWER_FORCING_TOKENS", args.answer_forcing_extra_tokens),
    ):
        if name in os.environ and int(os.environ[name]) != actual:
            raise ValueError(f"Parsed generation settings disagree with {name}")


def record(directory, value):
    with (directory / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")


def commit_validation(checkpoint, rollout_id, accuracy, threshold):
    """Persist a solved evaluation together with its checkpoint before advancing."""
    from slime.utils.checkpoint_state import mark_early_stopped, mark_evaluated

    mark_evaluated(checkpoint, rollout_id)
    if accuracy is not None and threshold is not None and accuracy >= threshold:
        mark_early_stopped(checkpoint, rollout_id, accuracy)


class SolvedAtBaseline(Exception):
    def __init__(self, manager, accuracy):
        self.manager, self.accuracy = manager, accuracy


def configured_run_name(default):
    name = os.environ.get('MODULAR_RL_RUN_NAME')
    if not name:
        return default
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', name):
        raise ValueError('Suite run names must be lowercase words/numbers separated by hyphens')
    return name


def main():
    repo = Path(os.environ["RLVE_REPO"])
    spec = importlib.util.spec_from_file_location(
        "rlve_native_training_driver", repo / "train.py"
    )
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    args = native.parse_args()
    validate_run_settings(args)
    if os.environ.get('MODULAR_RL_TASKS') == 'if':
        args.answer_forcing = False
    responses_per_update = args.rollout_batch_size * args.n_samples_per_prompt
    directory = Path(os.environ["MODULAR_RL_RUN_DIR"])
    kind = "FULL" if args.num_rollout * responses_per_update in (8000, 16000) else "PREFLIGHT"
    args.wandb_run_name = (
        f"{kind} {args.num_rollout * responses_per_update} responses | "
        f"{directory.name} | "
        f"gen{args.rollout_max_response_len} answer{args.answer_forcing_extra_tokens}"
    )
    if variant := os.environ.get("MODULAR_RL_VARIANT"):
        args.wandb_run_name = f"ABLATION {variant} | {args.wandb_run_name}"
    args.wandb_run_name = configured_run_name(args.wandb_run_name)
    original_step, original_eval = native.log_step_perf, native.evaluate_student
    window = {"train_seconds": 0.0, "updates": 0}
    evaluated_once = False
    accuracies = {}

    def step_perf(args, rollout_id, step_time, tokens):
        original_step(args, rollout_id, step_time, tokens)
        if args.use_wandb:
            native.wandb.log(
                {
                    "david/step": rollout_id,
                    "david/v2/completed_updates": rollout_id + 1,
                }
            )
        window["train_seconds"] += step_time
        window["updates"] += 1
        record(
            directory,
            {
                "phase": "train",
                "completed_updates": rollout_id + 1,
                "train_rollouts": (rollout_id + 1) * responses_per_update,
                "wall_seconds": step_time,
            },
        )

    def evaluate(args, actor, manager, rollout_id):
        nonlocal evaluated_once
        started = time.monotonic()
        accuracy = original_eval(args, actor, manager, rollout_id)
        accuracies[rollout_id] = accuracy
        seconds = time.monotonic() - started
        baseline = not evaluated_once and rollout_id == 0 and args.start_rollout_id == 0
        updates = 0 if baseline else rollout_id + 1
        metrics = {
            "david/step": rollout_id,
            "david/v2/completed_updates": updates,
            "david/budget/eval_wall_seconds": seconds,
        }
        if window["updates"]:
            metrics["david/budget/eval_wall_fraction_window"] = seconds / (
                seconds + window["train_seconds"]
            )
            metrics["david/budget/eval_to_train_response_ratio_window"] = (
                validation_prompt_count(os.environ.get('MODULAR_RL_TASKS', 'math')) * args.n_samples_per_eval_prompt
                / (window["updates"] * responses_per_update)
            )
        record(
            directory,
            {
                "phase": "eval",
                "completed_updates": updates,
                "train_rollouts": updates * responses_per_update,
                "accuracy": accuracy,
                "wall_seconds": seconds,
                "metrics": metrics,
            },
        )
        if args.use_wandb:
            native.wandb.log(metrics)
        evaluated_once = True
        window.update(train_seconds=0.0, updates=0)
        if (
            baseline
            and args.early_stop_val_accuracy is not None
            and accuracy >= args.early_stop_val_accuracy
        ):
            raise SolvedAtBaseline(manager, accuracy)
        return accuracy

    native.log_step_perf, native.evaluate_student = step_perf, evaluate
    # The native recovery path reevaluates an interrupted checkpoint before reading
    # its stopping state. Commit the threshold there too, so it cannot train onward.
    native.mark_evaluated = lambda checkpoint, rollout_id: commit_validation(
        checkpoint,
        rollout_id,
        accuracies.get(rollout_id),
        args.early_stop_val_accuracy,
    )
    try:
        native.train(args)
    except SolvedAtBaseline as solved:
        from slime.utils.checkpoint_state import mark_completed, mark_early_stopped

        mark_early_stopped(args.save, -1, solved.accuracy)
        mark_completed(args.save)
        native.ray.get(solved.manager.controller.finish_wandb.remote())
        if args.use_wandb:
            native.wandb.finish()
        print(
            f"Early stop before training: validation={solved.accuracy:.6f}", flush=True
        )
    from slime.utils.checkpoint_state import read_checkpoint_state

    state = read_checkpoint_state(args.save)
    actual = state["rollout_id"] + 1
    if not state["completed"] or actual > args.num_rollout:
        raise RuntimeError("Training did not finish within its response budget")
    result = {
        "completed": True,
        "updates": actual,
        "train_rollouts": actual * responses_per_update,
        "max_train_rollouts": args.num_rollout * responses_per_update,
        "prompts_per_update": args.rollout_batch_size,
        "samples_per_prompt": args.n_samples_per_prompt,
        "eval_samples_per_prompt": args.n_samples_per_eval_prompt,
        "sampled_token_budget": args.rollout_max_response_len,
        "answer_forcing_reserved_tokens": args.answer_forcing_extra_tokens,
        "checkpoint": state,
        "reason": "validation_accuracy"
        if state.get("early_stopped")
        else "rollout_budget",
    }
    decisions_path = directory / "optimizer-decisions.jsonl"
    if decisions_path.exists():
        decisions = {row["rollout_id"]: row for row in (
            json.loads(line) for line in decisions_path.open()
        ) if row["rollout_id"] < actual}
        if set(decisions) != set(range(actual)):
            raise RuntimeError("Skip-update trial has incomplete optimizer accounting")
        result["optimizer_updates"] = sum(d["optimizer_update_applied"] for d in decisions.values())
        result["optimizer_updates_skipped"] = actual - result["optimizer_updates"]
    (directory / "completed.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
