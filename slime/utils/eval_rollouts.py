"""Fixed-prompt rollout selection and incremental evaluation tables."""

import hashlib
import json

EXAMPLE_COLUMNS = [
    "environment",
    "prompt_id",
    "prompt_index",
    "rollout_index",
    "problem_seed",
    "prompt",
    "response",
    "reference_answer",
    "score",
    "reward",
    "status",
    "response_tokens",
]
TABLE_COLUMNS = ["david/step", *EXAMPLE_COLUMNS]


def select_eval_rollouts(samples, samples_per_prompt, reward_key):
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be positive")
    selected = {}
    for sample in sorted(samples, key=lambda item: item.index):
        metadata = sample.metadata
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if not isinstance(metadata, dict) or not metadata.get("environment"):
            raise ValueError(
                "Evaluation rollout tables require RLVE environment metadata"
            )
        environment = metadata["environment"]
        prompt_index, rollout_index = divmod(sample.index, samples_per_prompt)
        selected_prompt, rows = selected.setdefault(environment, (prompt_index, []))
        if prompt_index != selected_prompt:
            continue
        config = metadata.get("config", {})
        prompt = (
            sample.prompt
            if isinstance(sample.prompt, str)
            else json.dumps(sample.prompt, ensure_ascii=False)
        )
        identity = json.dumps(
            [environment, config, prompt], sort_keys=True, ensure_ascii=False
        )
        reference = config.get("parameter", {}).get("reference_answer", sample.label)
        reference = "" if reference is None else reference
        score = sample.reward if not reward_key else sample.reward[reward_key]
        rows.append(
            {
                "environment": environment,
                "prompt_id": hashlib.sha256(identity.encode()).hexdigest()[:16],
                "prompt_index": prompt_index,
                "rollout_index": rollout_index,
                "problem_seed": config.get("seed"),
                "prompt": prompt,
                "response": sample.response,
                "reference_answer": reference
                if isinstance(reference, str)
                else json.dumps(reference, ensure_ascii=False),
                "score": float(score),
                "reward": json.dumps(sample.reward, ensure_ascii=False),
                "status": sample.status.value,
                "response_tokens": sample.response_length,
            }
        )
    for environment, (_, rows) in selected.items():
        if [row["rollout_index"] for row in rows] != list(range(samples_per_prompt)):
            raise ValueError(f"Incomplete or duplicate rollout group for {environment}")
    return [row for _, rows in selected.values() for row in rows]


def update_eval_rollout_tables(data, step, tables, table_factory):
    logged = {}
    for name, result in data.items():
        examples = result.get("rollout_examples", [])
        if not examples:
            continue
        if name not in tables:
            tables[name] = table_factory(columns=TABLE_COLUMNS, log_mode="INCREMENTAL")
        table = tables[name]
        for example in examples:
            table.add_data(step, *(example[column] for column in EXAMPLE_COLUMNS))
        logged[f"eval_rollouts/{name}"] = table
    return logged
