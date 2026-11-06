import asyncio
from types import SimpleNamespace

import pytest

from . import rollout
from .code_reward import partial_test_reward


@pytest.mark.parametrize("passes,accuracy,expected", [
    ([True, True, True], 1, 1), ([True, False, False], 0, 1 / 3),
    ([True], 0, 1 / 3), ([], 0, 0), ([False, False, False], 0, 0),
])
def test_partial_reward_preserves_strict_accuracy_and_full_test_denominator(monkeypatch, passes, accuracy, expected):
    async def original(args, sample, **kwargs):
        return {"reward": float(accuracy), "accuracy": float(accuracy), "passes": passes}

    monkeypatch.setattr(rollout, "reward", original)
    sample = SimpleNamespace(metadata={"task": "code", "target": {"tests": ["a", "b", "c"]}})
    result = asyncio.run(partial_test_reward(None, sample))
    assert result["reward"] == pytest.approx(expected)
    assert result["accuracy"] == accuracy
    assert result["all_tests_reward"] == accuracy
    assert sample.metadata["reward_variant"] == "fraction_tests_passed"


def test_partial_reward_rejects_a_non_code_task():
    sample = SimpleNamespace(metadata={"task": "math"})
    with pytest.raises(ValueError, match="only defined for code"):
        asyncio.run(partial_test_reward(None, sample))
