import json
import importlib.util
from types import SimpleNamespace
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / "utils" / "rollout_export.py"
spec = importlib.util.spec_from_file_location("rollout_export", module_path)
rollout_export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rollout_export)
export_training_rollouts = rollout_export.export_training_rollouts


def test_training_rollout_export_is_scored_and_replay_safe(tmp_path):
    sample = SimpleNamespace(
        prompt="question",
        response=r"\boxed{42}",
        reward={"accuracy": 1.0, "reward": 1.0},
        metadata={
            "environment": "AdditionTable",
            "problem_difficulty": 0,
            "config": {"seed": 7, "parameter": {"reference_answer": 42}},
        },
        status=SimpleNamespace(value="completed"),
        response_length=5,
    )
    template = str(tmp_path / "{rollout_id:04d}.jsonl")
    path = export_training_rollouts(template, 3, [sample], "teacher-AdditionTable", "sweep")
    export_training_rollouts(template, 3, [sample], "teacher-AdditionTable", "sweep")

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["source_environment"] == "AdditionTable"
    assert rows[0]["problem_seed"] == 7
    assert rows[0]["accuracy"] == 1.0 and rows[0]["correct"] is True
    assert rows[0]["prompt"] == "question"
    assert rows[0]["response"] == r"\boxed{42}"
    assert json.loads(rows[0]["metadata_json"])["config"]["parameter"]["reference_answer"] == 42


def test_training_rollout_export_accepts_program_tree_metadata(tmp_path):
    sample = SimpleNamespace(
        prompt="predict main('abc')",
        response='{"output": "abc"}',
        label="abc",
        reward={"accuracy": 1.0, "reward": 1.0},
        metadata={"run_id": "A-depth4", "family": "depth", "_answer_forcing": {"initial_tokens": 1024, "prefix_tokens": 7}},
        status=SimpleNamespace(value="completed"),
        response_length=1159,
    )
    template = str(tmp_path / "{rollout_id:04d}.jsonl")
    path = export_training_rollouts(template, 0, [sample], "tree-depth-b1d4-answerforcing", "trees")
    row = json.loads(path.read_text().strip())
    assert row["source_environment"] == "A-depth4"
    assert row["reference_answer"] == "abc"
    assert row["accuracy"] == 1.0
