"""Score each on-policy response with its own task's fixed teacher."""

import asyncio


async def score_with_teachers(rows, completions, teachers, score):
    if len(rows) != len(completions):
        raise ValueError("Mixed OPD expects one completion per student prompt")
    groups = {}
    for index, row in enumerate(rows):
        environment = row[1]
        if environment not in teachers:
            raise ValueError(f"No teacher for {environment}")
        groups.setdefault(environment, []).append(index)

    async def run(environment, indices):
        traces = await score(
            [rows[i] for i in indices],
            [completions[i] for i in indices],
            teachers[environment],
        )
        if len(traces) != len(indices):
            raise ValueError("Teacher returned a different number of trajectories")
        return indices, traces

    ordered = [None] * len(rows)
    for indices, traces in await asyncio.gather(
        *(run(e, indices) for e, indices in groups.items())
    ):
        for index, trace in zip(indices, traces):
            ordered[index] = trace
    return ordered
