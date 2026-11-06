"""Write scored training trajectories as resumable, HF-friendly JSONL shards."""

import hashlib
import json
import os
from pathlib import Path


def export_training_rollouts(path_template, rollout_id, samples, run_name, run_group):
    path = Path(path_template.format(rollout_id=rollout_id))
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as output:
        for row_index, sample in enumerate(samples):
            metadata = sample.metadata or {}
            source_environment = metadata.get("environment") or metadata.get("run_id")
            if not source_environment:
                raise ValueError(f"Missing source environment for training rollout {rollout_id}, row {row_index}")
            config = metadata.get("config") or {}
            reward = sample.reward
            accuracy = reward.get("accuracy") if isinstance(reward, dict) else reward
            if accuracy is None:
                raise ValueError(f"Missing accuracy for training rollout {rollout_id}, row {row_index}")
            prompt = sample.prompt if isinstance(sample.prompt, str) else json.dumps(sample.prompt, ensure_ascii=False)
            identity = json.dumps([source_environment, config, prompt], sort_keys=True, ensure_ascii=False)
            reference = config.get("parameter", {}).get("reference_answer", getattr(sample, "label", None))
            row = {
                "run_name": run_name,
                "run_group": run_group,
                "rollout_id": rollout_id,
                "row_index": row_index,
                "source_environment": source_environment,
                "problem_seed": config.get("seed"),
                "problem_difficulty": metadata.get("problem_difficulty"),
                "prompt_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
                "prompt": prompt,
                "response": sample.response,
                "reference_answer": None if reference is None else str(reference),
                "accuracy": float(accuracy),
                "correct": float(accuracy) >= 1.0 - 1e-6,
                "reward_json": json.dumps(reward, ensure_ascii=False),
                "metadata_json": json.dumps(metadata, ensure_ascii=False),
                "status": sample.status.value,
                "response_tokens": sample.response_length,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    return path
