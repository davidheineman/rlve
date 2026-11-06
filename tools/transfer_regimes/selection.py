"""Reuse fitted quartets, with either random or near-zero net-transfer controls."""

import hashlib
import itertools
import json
import random
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def select_groups(
    positive_path, negative_path, transfer_path, *, seed=1234, controls="random"
):
    if controls not in {"random", "near-zero"}:
        raise ValueError("Expected random or near-zero controls")
    positive, negative, transfer = map(
        read, (positive_path, negative_path, transfer_path)
    )
    names = positive["environments"]
    if names != negative["environments"] or names != transfer["environments"]:
        raise ValueError("Transfer studies must use the identical task ordering")
    if len(names) != 64 or len(set(names)) != 64:
        raise ValueError("Expected the original 64-task study")
    matrix = transfer["transfer"]

    def mean(group):
        return sum(matrix[i][j] for i, j in itertools.permutations(group, 2)) / 12

    def quartets(study):
        candidate = next(c for c in study["candidates"] if c["size"] == 4)
        groups = candidate["fit"]["groups"]
        if sorted(i for g in groups for i in g) != list(range(64)):
            raise ValueError("Earlier quartets must partition the 64 tasks")
        return sorted(groups, key=lambda g: (mean(g), tuple(g)))

    best, worst = quartets(positive), quartets(negative)
    chosen = [
        ("best-1", best[-1]),
        ("best-2", best[-2]),
        ("worst-1", worst[0]),
        ("worst-2", worst[1]),
    ]
    if controls == "random":
        rng = random.Random(seed)
        permutation = rng.sample(range(64), 64)
        chosen.extend(
            (f"random-{i + 1}", sorted(permutation[4 * i : 4 * i + 4]))
            for i in range(2)
        )
    else:
        used = {tuple(sorted(group)) for _, group in chosen}
        candidates = {
            tuple(sorted(group))
            for group in best + worst
            if tuple(sorted(group)) not in used
        }

        # Match the signed mean used to rank the extremes. Record the mean
        # absolute directional effect too: a near-zero net can hide cancellation.
        def neutral_key(group):
            absolute = (
                sum(abs(matrix[i][j]) for i, j in itertools.permutations(group, 2)) / 12
            )
            return abs(mean(group)), absolute, group

        neutral = sorted(candidates, key=neutral_key)[:2]
        if len(neutral) != 2:
            raise ValueError("Need two distinct near-zero fitted quartets")
        chosen.extend((f"neutral-{i + 1}", list(g)) for i, g in enumerate(neutral))
    records = []
    for label, group in chosen:
        directions = [matrix[i][j] for i, j in itertools.permutations(group, 2)]
        records.append(
            {
                "label": label,
                "indices": list(group),
                "environments": [names[i] for i in group],
                "mean_transfer": mean(group),
                "mean_absolute_directional_transfer": sum(map(abs, directions)) / 12,
                "negative_direction_fraction": sum(x < 0 for x in directions) / 12,
                "selection": "uniform seeded draw; scores not consulted"
                if label.startswith("random")
                else "closest to zero mean net transfer among the earlier fitted four-task groups"
                if label.startswith("neutral")
                else "rank the earlier fitted four-task groups by mean off-diagonal transfer",
            }
        )
    if len({tuple(sorted(r["indices"])) for r in records}) != 6:
        raise ValueError("Selected controls duplicate an existing quartet")
    return {
        "groups": records,
        "random_seed": seed,
        "controls": controls,
        "neutral_interpretation": "Near-zero mean net discovery transfer; individual directions may have positive and negative effects, so this does not establish zero interaction",
        "random_controls_are_not_assumed_zero_transfer": True,
        "extremes_scope": "Earlier positive and negative balanced k=16 partitions; not a new search over all 64-choose-4 subsets",
        "score": "Mean of 12 directed other-task adapter gains versus the unchanged base; excludes self-transfer",
        "interpretation": "Discovery adapter-transfer proxy, not a measured effect of joint RL; different quartets also differ in task difficulty",
        "source_sha256": {
            str(p): digest(p) for p in (positive_path, negative_path, transfer_path)
        },
    }
