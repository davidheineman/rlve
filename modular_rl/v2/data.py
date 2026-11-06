"""Deterministic proportional sampling and symbolic MATH targets."""

import math
import random
import re
from collections import Counter, defaultdict

from modular_rl.prepare import digest


def boxed_answer(text):
    """Extract the final complete boxed/fboxed expression, including nested braces."""
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    for match in reversed(matches):
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index].strip()
    return None


def stratum(row):
    metadata = row["metadata"]
    return (metadata.get("subject", metadata["task"]), metadata.get("level", "all"))


def proportional_counts(counts, size):
    """Hamilton allocation: exact total, with proportions rounded by largest remainder."""
    if not 0 <= size <= sum(counts.values()):
        raise ValueError("Sample size is outside the available pool")
    if size == 0:
        return {key: 0 for key in counts}
    total = sum(counts.values())
    allocation = {
        key: math.floor(size * count / total) for key, count in counts.items()
    }
    order = sorted(
        counts, key=lambda key: (-(size * counts[key] / total - allocation[key]), key)
    )
    for key in order[: size - sum(allocation.values())]:
        allocation[key] += 1
    return allocation


def stratified_sample(rows, size, seed=42):
    groups = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    allocation = proportional_counts(
        Counter({key: len(value) for key, value in groups.items()}), size
    )
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        selected.extend(group[: allocation[key]])
    rng.shuffle(selected)
    return selected


def normalize_math(row, subject, index):
    answer = boxed_answer(row["solution"])
    if not answer:
        raise ValueError("MATH reference has no nonempty complete boxed answer")
    messages = [
        {
            "role": "user",
            "content": row["problem"]
            + "\n\nSolve the problem and put your final answer in \\boxed{}.",
        }
    ]
    key = digest(messages)
    return {
        "user_prompt": messages,
        "metadata": {
            "task": "math",
            "environment": "math",
            "id": f"{subject}:{index}",
            "key": key,
            "split_group": digest(row["problem"].strip()),
            "subject": subject,
            "level": row["level"],
            "messages": messages,
            "target": {"format": "hendrycks_math", "answer": answer},
        },
    }
