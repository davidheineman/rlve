"""Keep RLVE training seeds independent across branches, using the existing rollout path."""

import os
import random

from .io import read_json


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.rollout.sglang_rollout import generate_rollout as original

    if not evaluation and not getattr(data_source, "async_branch_initialized", False):
        task = read_json(os.environ["RLVE_ASYNC_TASK"])
        seed = task["train_seed"]
        random.seed(seed)
        data_source.rlve_manager.problem_generation_seed = seed
        data_source.async_branch_initialized = True
    return original(args, rollout_id, data_source, evaluation=evaluation)
