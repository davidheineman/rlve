"""Opt-in MBPP partial credit; held-out accuracy still requires every test."""


async def partial_test_reward(args, sample, **kwargs):
    from .rollout import reward

    if sample.metadata["task"] != "code":
        raise ValueError("Partial-test reward is only defined for code")
    total = len(sample.metadata["target"]["tests"])
    if total == 0:
        raise ValueError("Partial-test reward requires at least one test")
    result = dict(await reward(args, sample, **kwargs))
    passes = result.get("passes", [])
    if len(passes) > total:
        raise ValueError("Grader returned more test outcomes than the target")
    result["all_tests_reward"] = result["reward"]
    # Missing/unexecuted tests receive no credit; a partial timeout cannot earn 1.
    result["reward"] = 1.0 if result["accuracy"] == 1 else sum(bool(p) for p in passes) / total
    sample.metadata["reward_variant"] = "fraction_tests_passed"
    return result
