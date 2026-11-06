"""CPU symbolic grading, with persistent spawned workers and signal timeouts."""

import asyncio
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

from .data import boxed_answer

_POOL = None


def initialize_worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    runtime = os.environ.get("MODULAR_RL_RUNTIME")
    if runtime:
        sys.path.insert(0, runtime)


@lru_cache(maxsize=8192)
def parse_gold(answer):
    from math_verify import LatexExtractionConfig, parse

    return parse(
        "$" + answer + "$",
        extraction_config=[LatexExtractionConfig()],
        parsing_timeout=3,
    )


def grade(response, answer):
    from math_verify import LatexExtractionConfig, parse, verify

    response = response.rsplit("</think>", 1)[-1]
    prediction = boxed_answer(response)
    if prediction is None:
        return {"reward": 0.0, "accuracy": 0.0}
    gold = parse_gold(answer)
    if not gold:
        raise ValueError(f"Unparseable MATH target: {answer!r}")
    parsed = parse(
        "$" + prediction + "$",
        extraction_config=[LatexExtractionConfig()],
        parsing_timeout=3,
    )
    correct = bool(parsed) and bool(verify(gold, parsed, timeout_seconds=3))
    return {"reward": float(correct), "accuracy": float(correct)}


async def reward(response, answer):
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(
            max_workers=4,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
        )
    return await asyncio.get_running_loop().run_in_executor(
        _POOL, grade, response, answer
    )


async def correct_and_finished_reward(args, sample, **kwargs):
    """Opt-in training reward; boxed-answer validation accuracy stays unchanged."""
    from slime.utils.types import Sample

    if sample.metadata["task"] != "math":
        raise ValueError("Correct-and-finished reward is only defined for math")
    if sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
        raise ValueError("Grade only completed or truncated generations")
    result = dict(await reward(sample.response, sample.metadata["target"]["answer"]))
    result["boxed_reward"] = result["reward"]
    result["reward"] = float(bool(result["accuracy"]) and sample.status == Sample.Status.COMPLETED)
    sample.metadata["reward_variant"] = "correct_and_finished"
    return result
