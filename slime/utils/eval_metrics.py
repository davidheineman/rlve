import json
from math import prod


def group_eval_results(name, rewards, truncated, metadata):
    results = {name: {"rewards": rewards, "truncated": truncated}}
    for reward, is_truncated, sample_metadata in zip(rewards, truncated, metadata, strict=True):
        if isinstance(sample_metadata, str):
            try:
                sample_metadata = json.loads(sample_metadata)
            except json.JSONDecodeError:
                continue
        if not isinstance(sample_metadata, dict) or not sample_metadata.get("environment"):
            continue
        key = f"{name}/{sample_metadata['environment']}"
        environment_results = results.setdefault(key, {"rewards": [], "truncated": []})
        environment_results["rewards"].append(reward)
        environment_results["truncated"].append(is_truncated)
    return results


def early_stop_accuracy(results, names):
    missing = [name for name in names if name not in results]
    if missing:
        raise ValueError(f"Early-stop eval dataset missing: {missing}")
    rewards = [reward for name in names for reward in results[name]["rewards"]]
    if not rewards:
        raise ValueError("Early-stop evals contain no samples")
    return sum(float(reward >= 1.0 - 1e-6) for reward in rewards) / len(rewards)


def pass_at_k(correct_per_prompt, group_size):
    """Unbiased pass@k, averaged over prompts, for powers of two up to group_size."""
    estimates = {}
    sample_count = 1
    while sample_count <= group_size:
        per_prompt = []
        for num_correct in correct_per_prompt:
            if group_size - num_correct < sample_count:
                per_prompt.append(1.0)
            else:
                per_prompt.append(
                    1.0 - prod(
                        1.0 - sample_count / remaining
                        for remaining in range(group_size - num_correct + 1, group_size + 1)
                    )
                )
        estimates[sample_count] = sum(per_prompt) / len(per_prompt)
        sample_count *= 2
    return estimates
