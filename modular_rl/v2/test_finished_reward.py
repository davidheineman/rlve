import asyncio
from types import SimpleNamespace

import pytest

from slime.utils.types import Sample

from . import math_reward


@pytest.mark.parametrize("status,correct,expected", [
    (Sample.Status.COMPLETED, 1, 1), (Sample.Status.TRUNCATED, 1, 0),
    (Sample.Status.COMPLETED, 0, 0), (Sample.Status.TRUNCATED, 0, 0),
])
def test_finished_reward_keeps_boxed_accuracy_for_validation(monkeypatch, status, correct, expected):
    async def original(response, answer):
        assert response == "example" and answer == "42"
        return {"reward": float(correct), "accuracy": float(correct)}

    monkeypatch.setattr(math_reward, "reward", original)
    sample = SimpleNamespace(response="example", status=status,
                             metadata={"task": "math", "target": {"answer": "42"}})
    result = asyncio.run(math_reward.correct_and_finished_reward(None, sample))
    assert result["reward"] == expected
    assert result["accuracy"] == correct
    assert result["boxed_reward"] == correct
    assert sample.metadata["reward_variant"] == "correct_and_finished"


@pytest.mark.parametrize("task,status", [
    ("code", Sample.Status.COMPLETED), ("math", Sample.Status.ABORTED),
    ("math", Sample.Status.PENDING),
])
def test_finished_reward_rejects_wrong_task_or_unfinished_generation(task, status):
    sample = SimpleNamespace(status=status, metadata={"task": task})
    with pytest.raises(ValueError):
        asyncio.run(math_reward.correct_and_finished_reward(None, sample))
