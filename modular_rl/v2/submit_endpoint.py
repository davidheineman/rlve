"""Prepare unused holdout problems and submit read-only checkpoint evaluations."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from modular_rl.prepare import write_json

from .ablate import freeze
from .base_models import BASE_MODELS, verify_initialization
from .data import stratified_sample, stratum
from .inspect_rollouts import read_events
from .snapshot import MANIFEST, validate_snapshot


def prepare_if_panel(experiment, tokenizer, size):
    """Keep the declared Dolci validation panel identical across model/cap jobs."""
    directory = experiment / 'selection/if'
    info = json.loads((directory / 'selection.json').read_text())
    path = directory / 'val.jsonl'
    rows = read_events(path)
    training = read_events(directory / 'train.jsonl')
    train_keys = {r['metadata']['key'] for r in training}
    train_groups = {r['metadata']['split_group'] for r in training}
    if (len(rows) != size or len({r['metadata']['key'] for r in rows}) != size
            or hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']['val']):
        raise ValueError('Dolci validation panel differs from its declared selection')
    for row in rows:
        meta = row['metadata']
        if (meta['key'] in train_keys or meta['split_group'] in train_groups
                or meta['task'] != 'if' or meta['environment'] != 'if'):
            raise ValueError('Dolci panel must have IF metadata and disjoint training prompts')
        prompt = tokenizer.apply_chat_template(meta['messages'], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True)
        if len(tokenizer(prompt, add_special_tokens=False)['input_ids']) > 2048:
            raise ValueError('Declared Dolci prompt exceeds the prompt budget')
    info['endpoint_panel'] = {'kind': 'fixed-Dolci-validation', 'answer_forcing': False,
                              'strict_prompt_success': True, 'prompts': size}
    write_json(directory / 'selection.json', info)


def prepare_reserved_panel(experiment, task, tokenizer, panel, size, expected_sha256):
    """Copy a declared panel only after checking the actual train/monitor split."""
    panel = Path(panel)
    if hashlib.sha256(panel.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Reserved endpoint panel SHA changed; no jobs submitted')
    directory = experiment / 'selection' / task
    development = [r for name in ('train', 'val') for r in read_events(directory / (name + '.jsonl'))]
    used_keys = {r['metadata']['key'] for r in development}
    used_groups = {r['metadata']['split_group'] for r in development}
    rows = read_events(panel)
    if (len(rows) != size or len({r['metadata']['key'] for r in rows}) != size
            or len({r['metadata']['split_group'] for r in rows}) != size):
        raise ValueError('Reserved panel must contain the declared number of independent problems')
    for row in rows:
        meta = row['metadata']
        if meta['key'] in used_keys or meta['split_group'] in used_groups:
            raise ValueError('Reserved endpoint overlaps training or monitoring by problem group')
        if meta['task'] != task or meta.get('environment') != task:
            raise ValueError('Reserved endpoint task/environment metadata changed')
        rendered = tokenizer.apply_chat_template(meta['messages'], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=True)
        if len(tokenizer(rendered, add_special_tokens=False)['input_ids']) > 2048:
            raise ValueError('Reserved endpoint exceeds the prompt budget')
    shutil.copyfile(panel, directory / 'val.jsonl')
    info = json.loads((directory / 'selection.json').read_text())
    info['keys']['val'] = [r['metadata']['key'] for r in rows]
    info['sha256']['val'] = expected_sha256
    info.setdefault('strata', {})['val'] = dict(Counter('/'.join(stratum(r)) for r in rows))
    info.update(validation_prompts=size, endpoint_panel={
        'kind': 'prospectively-reserved', 'source': str(panel), 'sha256': expected_sha256,
        'disjoint_from_training_and_monitoring_by_key_and_group': True})
    write_json(directory / 'selection.json', info)


def prepare_panel(base, experiment, task, tokenizer, size, excluded_panels=(), seed=20261027):
    directory = experiment / "selection" / task
    monitored = {r["metadata"]["key"] for r in read_events(base / "selection" / task / "val.jsonl")}
    training = {r["metadata"]["key"] for r in read_events(directory / "train.jsonl")}
    excluded = {r["metadata"]["key"] for panel in excluded_panels for r in read_events(panel)}
    eligible = []
    for row in read_events(base / "data" / task / "val.jsonl"):
        metadata = row["metadata"]
        if metadata["key"] in monitored or metadata["key"] in excluded:
            continue
        if metadata["key"] in training:
            raise ValueError("Held-out pool overlaps training")
        rendered = tokenizer.apply_chat_template(metadata["messages"], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=True)
        if len(tokenizer(rendered, add_special_tokens=False)['input_ids']) > 2048:
            continue
        metadata["environment"] = task
        eligible.append(row)
    rows = stratified_sample(eligible, size, seed=seed)
    if len(rows) != size:
        raise ValueError("Not enough eligible unused holdout problems")
    path = directory / "val.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    info = json.loads((directory / "selection.json").read_text())
    info["keys"]["val"] = [r["metadata"]["key"] for r in rows]
    info["sha256"]["val"] = hashlib.sha256(path.read_bytes()).hexdigest()
    info["strata"]["val"] = dict(Counter("/".join(stratum(row)) for row in rows))
    info.update(validation_prompts=size, endpoint_panel={"unused_before_selection": len(eligible),
                "disjoint_from_monitored_panel": True, "seed": seed,
                "excluded_panels": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                    for p in excluded_panels]})
    write_json(directory / "selection.json", info)


def submit(repo, base, experiment, policies, afterok=(), model="Qwen3-0.6B-Base",
           math_prompts=128, code_prompts=64, excluded_math_panels=(), panel_seed=20261027,
           max_concurrent=24, expected_math_panel_sha256=None, max_sampled_tokens=2048,
           primary_family_size=1, answer_reserve=256, selection_source=None,
           reserved_panels=None, expected_reserved_panel_sha256=None, code_format="chat", if_prompts=32):
    from transformers import AutoTokenizer

    repo, base, experiment = repo.resolve(), base.resolve(), experiment.resolve()
    if model not in (*BASE_MODELS, "Qwen3-0.6B", "Qwen3-1.7B"):
        raise ValueError("Choose an explicit Qwen3 checkpoint")
    if not isinstance(max_concurrent, int) or not 1 <= max_concurrent <= 24:
        raise ValueError("Use one to 24 endpoint GPU lanes")
    if max_sampled_tokens not in (2048, 4096):
        raise ValueError("Use the declared 2k or 4k endpoint generation cap")
    if type(answer_reserve) is not int or not 0 < answer_reserve < max_sampled_tokens:
        raise ValueError('Endpoint answer reserve must fit inside the sampled-token cap')
    if not isinstance(primary_family_size, int) or primary_family_size < 1:
        raise ValueError("Declare a positive primary comparison family size")
    identity = None
    if model in BASE_MODELS:
        models = base.parents[1] / "models"
        identity = verify_initialization(models / model, models / (model + "_torch_dist"))
    if experiment.exists():
        raise ValueError("Use a fresh endpoint experiment directory")
    tasks = sorted({p["task"] for p in policies})
    if not tasks or set(tasks) - {"math", "code", "if"}:
        raise ValueError("Choose math, code, and/or IF endpoint policies")
    if code_format not in ("chat", "prefix") or (code_format == "prefix" and tasks != ["code"]):
        raise ValueError('A Python prefix requires a code-only endpoint')
    for policy in policies:
        if any(not str(job).isdigit() or int(job) <= 0 for job in policy.get("afterok", [])):
            raise ValueError("Policy dependencies must be positive Slurm job IDs")
        policy["checkpoint"] = str(Path(policy["checkpoint"]).resolve())
        if not (Path(policy["checkpoint"]) / "latest_checkpointed_iteration.txt").exists():
            raise ValueError("Policy checkpoint is not committed")
        checkpoint = Path(policy["checkpoint"])
        policy["expected_tracker"] = (checkpoint / "latest_checkpointed_iteration.txt").read_text().strip()
        if identity and policy["expected_tracker"] == "release" and checkpoint != models / (model + "_torch_dist"):
            raise ValueError("Base endpoint baseline must use its verified untrained release")
        if policy["expected_tracker"] != "release":
            state = json.loads((checkpoint / "rlve_checkpoint_state.json").read_text())
            if (checkpoint / MANIFEST).exists():
                if validate_snapshot(checkpoint)["model"] != model:
                    raise ValueError("Preserved checkpoint belongs to a different model")
            elif not state["completed"]:
                raise ValueError("Use completed policies or validated frozen evaluation snapshots")
            elif identity:
                source_identity = json.loads((checkpoint.parent / "initial-model-identity.json").read_text())
                if source_identity["repo_id"] != identity["repo_id"]:
                    raise ValueError("Endpoint checkpoint belongs to a different Base model")
    source = freeze(repo, experiment, base, tasks)
    if selection_source is not None:
        for task in tasks:
            destination = experiment / 'selection' / task
            shutil.rmtree(destination)
            shutil.copytree(Path(selection_source).resolve() / task, destination)
    tokenizer = AutoTokenizer.from_pretrained(base.parents[1] / "models" / model, local_files_only=True)
    sizes = {"math": math_prompts, "code": code_prompts, "if": if_prompts}
    if any(not isinstance(size, int) or size <= 0 for size in sizes.values()):
        raise ValueError("Endpoint panel sizes must be positive integers")
    for task in tasks:
        if task == "if":
            prepare_if_panel(experiment, tokenizer, sizes[task])
        elif reserved_panels and task in reserved_panels:
            if not expected_reserved_panel_sha256 or task not in expected_reserved_panel_sha256:
                raise ValueError('A reserved panel requires its prospectively declared SHA')
            prepare_reserved_panel(experiment, task, tokenizer, reserved_panels[task], sizes[task],
                                   expected_reserved_panel_sha256[task])
        else:
            prepare_panel(base, experiment, task, tokenizer, sizes[task],
                          excluded_math_panels if task == "math" else (), panel_seed)
    if expected_math_panel_sha256 is not None:
        panel = experiment / "selection/math/val.jsonl"
        if "math" not in tasks or hashlib.sha256(panel.read_bytes()).hexdigest() != expected_math_panel_sha256:
            raise ValueError("Endpoint math panel differs from the reserved holdout; no jobs submitted")
    write_json(experiment / "plan.json", {"kind": "independent-endpoint-evaluation", "policies": policies,
                                         "prompts": sizes, "samples": 8, "afterok": list(afterok),
                                         "optimizer_updates": 0, "max_sampled_tokens": max_sampled_tokens,
                                         "answer_reserve": answer_reserve, "model": f"Qwen/{model}",
                                         "selection_source": str(selection_source) if selection_source else None,
                                         "reserved_panel_sha256": expected_reserved_panel_sha256,
                                         "code_format": code_format,
                                         "if_generation": "unforced-full-budget" if "if" in tasks else None,
                                         "pretrained_base_identity": identity,
                                         "primary_family_size": primary_family_size,
                                         "max_concurrent_gpu_lanes": max_concurrent,
                                         "sglang_mem_fraction_static": 0.3,
                                         "source_sha256": {
                                             str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in sorted(source.rglob("*")) if p.is_file()
                                         }})
    jobs = []
    for index, policy in enumerate(policies):
        task, name = policy["task"], policy["name"]
        output = experiment / task / name
        output.mkdir(parents=True)
        dependencies = list(afterok) + list(policy.get("afterok", []))
        # Later jobs follow completed work in the same requested GPU lane.
        if index >= max_concurrent:
            dependencies.append(jobs[index - max_concurrent]["job_id"])
        environment = {
            **os.environ, "RLVE_REPO": str(source), "MODULAR_RL_TASKS": task,
            "MODULAR_RL_MODELS": model, "MODULAR_RL_EVAL_ONLY": "1",
            "MODULAR_RL_ENDPOINT_NAME": f"{model} gen{max_sampled_tokens} {task} {name}",
            "MODULAR_RL_ENDPOINT_PROMPTS": str(sizes[task]),
            "MODULAR_RL_ENDPOINT_TRACKER": policy["expected_tracker"],
            "MODULAR_RL_GROUP": experiment.name, "MODULAR_RL_DATA": str(base / "data"),
            "MODULAR_RL_SELECTION": str(experiment / "selection"),
            "MODULAR_RL_RUNTIME": str(experiment / "runtime"), "MODULAR_RL_OUTPUT": str(output),
            "MODULAR_RL_STEPS": "1", "MODULAR_RL_EVAL_INTERVAL": "1",
            "MODULAR_RL_MAX_RESPONSE_TOKENS": str(max_sampled_tokens),
            "MODULAR_RL_ANSWER_FORCING_TOKENS": str(answer_reserve),
            "MODULAR_RL_EXTRA_ARGS": ("--sglang-mem-fraction-static 0.3 "
                                       f"--seq-length {4096 if max_sampled_tokens == 2048 else 8192} "
                                       "--rollout-function-path modular_rl.v2.endpoint.generate_rollout "
                                       f"--load {policy['checkpoint']} --no-load-optim --no-load-rng"),
        }
        if code_format == 'prefix':
            environment['MODULAR_RL_EXTRA_ARGS'] += (
                ' --custom-generate-function-path modular_rl.v2.code_generation.generate'
                ' --custom-rm-path modular_rl.v2.code_generation.reward')
        if task == 'if':
            environment['MODULAR_RL_EXTRA_ARGS'] += (
                ' --custom-generate-function-path modular_rl.v2.if_rollout.generate'
                ' --custom-rm-path modular_rl.if_grader.reward')
        command = ["sbatch", "--parsable", f"--job-name=modular-endpoint-{task}-{name}",
                   "--gres=gpu:a6000:1", "--cpus-per-task=8",
                   "--mem=128G" if "1.7B" in model else "--mem=64G",
                   "--time=03:00:00" if max_sampled_tokens == 4096 else
                   ("--time=01:30:00" if "1.7B" in model and sizes[task] >= 256 else "--time=01:00:00"),
                   "--constraint=48G&ampere", "--exclude=viscam10,jagupard33,jagupard34",
                   f"--chdir={source}", f"--output={output}/slurm-%j.out"]
        if dependencies:
            command.append("--dependency=afterok:" + ":".join(map(str, dependencies)))
        command.append(str(source / "modular_rl/v2/run.sbatch"))
        job = subprocess.check_output(command, env=environment, text=True).strip().split(";")[0]
        jobs.append({**policy, "job_id": job, "output": str(output), "command": command})
        write_json(experiment / "jobs.json", jobs)
        print(json.dumps(jobs[-1]), flush=True)
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/nlp/scr/dhei/modular-mopd-rl/v2"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True, help="JSON list of task/name/checkpoint records")
    parser.add_argument("--afterok", nargs="*", default=[])
    parser.add_argument("--model", choices=list(BASE_MODELS), default="Qwen3-0.6B-Base")
    parser.add_argument("--math-prompts", type=int, default=128)
    parser.add_argument("--code-prompts", type=int, default=64)
    parser.add_argument("--if-prompts", type=int, default=32)
    parser.add_argument("--exclude-math-panel", type=Path, nargs="*", default=[])
    parser.add_argument("--panel-seed", type=int, default=20261027)
    parser.add_argument("--max-concurrent", type=int, choices=range(1, 25), default=24)
    parser.add_argument("--max-sampled-tokens", type=int, choices=[2048, 4096], default=2048)
    parser.add_argument("--primary-family-size", type=int, default=1)
    parser.add_argument("--expected-math-panel-sha256", help="Reject a changed reserved panel before submitting jobs")
    parser.add_argument("--answer-reserve", type=int, default=256)
    parser.add_argument("--selection-source", type=Path)
    parser.add_argument("--reserved-code-panel", type=Path)
    parser.add_argument("--expected-code-panel-sha256")
    parser.add_argument("--code-format", choices=["chat", "prefix"], default="chat")
    args = parser.parse_args()
    submit(Path.cwd(), args.base, args.experiment, json.loads(args.policies.read_text()), args.afterok, args.model,
           args.math_prompts, args.code_prompts, args.exclude_math_panel, args.panel_seed,
           args.max_concurrent, args.expected_math_panel_sha256, args.max_sampled_tokens, args.primary_family_size,
           answer_reserve=args.answer_reserve, selection_source=args.selection_source,
           reserved_panels={"code": args.reserved_code_panel} if args.reserved_code_panel else None,
           expected_reserved_panel_sha256={"code": args.expected_code_panel_sha256}
           if args.expected_code_panel_sha256 else None, code_format=args.code_format,
           if_prompts=args.if_prompts)
