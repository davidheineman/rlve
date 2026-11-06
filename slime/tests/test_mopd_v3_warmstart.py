import importlib.util
import json
from pathlib import Path

import torch
import pytest


spec = importlib.util.spec_from_file_location(
    "prepare_mopd_v3_warmstart",
    Path(__file__).resolve().parents[2] / "tools" / "prepare_mopd_v3_warmstart.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(char) for char in text]


def test_uses_correct_completed_traces_from_every_teacher_and_ignores_stale_shards(tmp_path):
    names = [f"Env{index}" for index in range(16)]
    environment_file = tmp_path / "n16.txt"
    environment_file.write_text(" ".join(names))
    teacher_dir = tmp_path / "teachers"
    for index, environment in enumerate(names):
        run = teacher_dir / f"{index:02d}-{environment}"
        (run / "hf-final").mkdir(parents=True)
        (run / "hf-final" / "model.safetensors.index.json").write_text("{}")
        (run / "weights_only_complete.json").write_text(json.dumps({"checkpoint_step": 0}))
        shards = run / "rollouts" / "train"
        shards.mkdir(parents=True)
        rows = [
            {"source_environment": environment, "rollout_id": 0, "correct": True,
             "status": "completed", "prompt": "P", "response": "R", "prompt_id": f"{index}"},
            {"source_environment": environment, "rollout_id": 0, "correct": False,
             "status": "completed", "prompt": "P", "response": "W", "prompt_id": f"{index}"},
            {"source_environment": environment, "rollout_id": 0, "correct": True,
             "status": "truncated", "prompt": "P", "response": "T", "prompt_id": f"{index}"},
        ]
        (shards / "0000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (shards / "0001.jsonl").write_text("stale and should not be read\n")

    output_dir = tmp_path / "replay"
    manifest = module.prepare(teacher_dir, environment_file, output_dir, FakeTokenizer())
    assert manifest["num_batches"] == 2
    assert manifest["selected_traces"] == 16
    assert manifest["discarded_remainder"] == 0
    samples = [
        sample
        for batch_path in sorted(output_dir.glob("*.pt"))
        for sample in torch.load(batch_path, weights_only=False)["samples"]
    ]
    assert len(samples) == 16
    assert {sample["metadata"]["environment"] for sample in samples} == set(names)
    assert all(sample["tokens"] == [ord("P"), ord("R"), 99] for sample in samples)
    assert all(sample["response_length"] == 2 for sample in samples)


@pytest.mark.parametrize("size", [4, 8])
def test_subset_replay_contains_only_selected_teacher_traces(tmp_path, size):
    all_names = [f"Env{index}" for index in range(16)]
    environment_file = tmp_path / f"n{size}.txt"
    environment_file.write_text(" ".join(all_names[:size]))
    teacher_dir = tmp_path / "teachers"
    for index, environment in enumerate(all_names):
        run = teacher_dir / f"{index:02d}-{environment}"
        (run / "hf-final").mkdir(parents=True)
        (run / "hf-final" / "model.safetensors.index.json").write_text("{}")
        (run / "weights_only_complete.json").write_text(json.dumps({"checkpoint_step": 0}))
        shards = run / "rollouts" / "train"
        shards.mkdir(parents=True)
        row = {"source_environment": environment, "rollout_id": 0, "correct": True,
               "status": "completed", "prompt": "P", "response": "R",
               "prompt_id": environment}
        (shards / "0000.jsonl").write_text((json.dumps(row) + "\n") * 4)

    output_dir = tmp_path / "replay"
    manifest = module.prepare(teacher_dir, environment_file, output_dir, FakeTokenizer(),
                              expected_num_environments=size)
    assert manifest["source_environments"] == all_names[:size]
    assert set(manifest["per_environment"]) == set(all_names[:size])
    samples = [sample for batch_path in sorted(output_dir.glob("*.pt"))
               for sample in torch.load(batch_path, weights_only=False)["samples"]]
    assert {sample["metadata"]["environment"] for sample in samples} == set(all_names[:size])
    assert len(samples) == size * 4
