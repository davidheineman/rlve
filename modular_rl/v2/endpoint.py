"""Evaluate a saved policy on an independent panel without taking an update."""

import importlib.util
import json
import os
import sys
from pathlib import Path


class EvaluationComplete(Exception):
    pass


def validate_endpoint_identity(args):
    if not hasattr(args, 'hf_checkpoint') or not Path(args.hf_checkpoint).name.endswith('-Base'):
        return None
    from modular_rl.v2.base_models import verify_initialization
    from modular_rl.v2.snapshot import MANIFEST, validate_snapshot

    identity = verify_initialization(args.hf_checkpoint, args.ref_load)
    checkpoint = Path(args.load)
    tracker = (checkpoint / 'latest_checkpointed_iteration.txt').read_text().strip()
    if tracker == 'release':
        if checkpoint.resolve() != Path(args.ref_load).resolve():
            raise ValueError('Base endpoint baseline must be its verified untrained release')
    elif (checkpoint / MANIFEST).exists():
        if validate_snapshot(checkpoint)['model'] != identity['repo_id'].split('/')[-1]:
            raise ValueError('Endpoint snapshot has a different Base initializer')
    else:
        previous = json.loads((checkpoint.parent / 'initial-model-identity.json').read_text())
        if previous['repo_id'] != identity['repo_id']:
            raise ValueError('Endpoint policy has a different Base initializer')
    return {**identity, 'actor_load_for_this_attempt': str(checkpoint),
            'untrained_reference': str(args.ref_load)}


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    from modular_rl import rollout as v1

    from . import rollout

    if not evaluation:
        raise RuntimeError("Endpoint evaluation must never generate training data")
    rollout._COUNTS.clear()
    rollout._SAMPLES.clear()
    data = v1.generate_rollout(args, rollout_id, data_buffer, evaluation=True)
    count = int(os.environ["MODULAR_RL_ENDPOINT_PROMPTS"]) * args.n_samples_per_eval_prompt
    if len(data["val_envs"]["rewards"]) != count or len(rollout._COUNTS) != count:
        raise RuntimeError("Endpoint evaluation lost prompts or samples")
    if any(row["sampled_tokens"] > args.eval_max_response_len for row in rollout._COUNTS):
        raise RuntimeError("Endpoint generation exceeded the sampled-token cap")
    directory = Path(os.environ["MODULAR_RL_RUN_DIR"])
    rollout.export_evaluation_samples(directory, rollout_id, rollout._SAMPLES,
                                      args.wandb_run_name, args.wandb_group)
    with (directory / "generation.jsonl").open("a") as stream:
        stream.write(json.dumps({"phase": "eval", "responses": count,
                                 "sampled_tokens": sum(r["sampled_tokens"] for r in rollout._COUNTS),
                                 "max_sampled_tokens": max(r["sampled_tokens"] for r in rollout._COUNTS)}) + "\n")
    return data


def main():
    repo = Path(os.environ["RLVE_REPO"])
    spec = importlib.util.spec_from_file_location("rlve_endpoint_native", repo / "train.py")
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    args = native.parse_args()
    identity = validate_endpoint_identity(args)
    tracker = Path(args.load) / "latest_checkpointed_iteration.txt"
    expected_tracker = os.environ["MODULAR_RL_ENDPOINT_TRACKER"]
    if tracker.read_text().strip() != expected_tracker:
        raise ValueError("Selected policy checkpoint changed before endpoint evaluation")
    from modular_rl.v2.snapshot import MANIFEST, validate_snapshot

    if (Path(args.load) / MANIFEST).exists():
        validate_snapshot(args.load)
    args.start_rollout_id, args.num_rollout, args.eval_interval = 0, 1, 1
    args.transactional_checkpoints = False
    args.save, args.save_interval = None, None
    args.no_load_optim, args.no_load_rng = True, True
    if os.environ.get("MODULAR_RL_TASKS") == "if":
        args.answer_forcing = False
    args.wandb_run_name = "ENDPOINT EVAL ONLY | " + os.environ["MODULAR_RL_ENDPOINT_NAME"]
    original_eval = native.evaluate_student
    directory = Path(os.environ["MODULAR_RL_RUN_DIR"])
    if identity is not None:
        (directory / 'initial-model-identity.json').write_text(json.dumps(identity, indent=2) + '\n')

    def evaluate(args, actor, manager, rollout_id):
        score = original_eval(args, actor, manager, rollout_id)
        if tracker.read_text().strip() != expected_tracker:
            raise ValueError("Selected policy checkpoint changed during endpoint evaluation")
        if (Path(args.load) / MANIFEST).exists():
            validate_snapshot(args.load)
        result = {
            "evaluation_only": True, "optimizer_updates": 0, "policy_checkpoint": args.load,
            "prompts": int(os.environ["MODULAR_RL_ENDPOINT_PROMPTS"]), "samples_per_prompt": 8,
            "pass@1": score, "name": os.environ["MODULAR_RL_ENDPOINT_NAME"],
            "wandb_id": native.wandb.run.id if args.use_wandb else None,
        }
        (directory / "evaluated.json").write_text(json.dumps(result, indent=2) + "\n")
        native.ray.get(manager.controller.finish_wandb.remote())
        if args.use_wandb:
            native.wandb.finish()
        raise EvaluationComplete

    native.evaluate_student = evaluate
    try:
        native.train(args)
    except EvaluationComplete:
        pass
    if not (directory / "evaluated.json").exists() or list((directory / "rollouts/train").glob("*.jsonl")):
        raise RuntimeError("Endpoint evaluation did not finish without training")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--report":
        root = Path(sys.argv[2])
        results = [json.loads(p.read_text()) for p in root.glob("Qwen*/evaluated.json")]
        if not results or not all(r["evaluation_only"] and r["optimizer_updates"] == 0 for r in results):
            raise RuntimeError("No valid completed endpoint evaluation")
        (root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results), flush=True)
    else:
        main()
