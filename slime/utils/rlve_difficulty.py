"""Checkpointable, exactly balanced RLVE environment/difficulty schedule."""

import random


class BalancedDifficultySchedule:
    def __init__(self, environments, max_difficulty, state=None):
        self.pairs = [(environment, difficulty)
                      for environment in environments
                      for difficulty in range(max_difficulty + 1)]
        if not self.pairs or len(set(self.pairs)) != len(self.pairs):
            raise ValueError("Expected distinct environments and nonnegative difficulty")
        self.pending = []
        self.counts = {pair: 0 for pair in self.pairs}
        if state is not None:
            self.pending = [tuple(pair) for pair in state["pending"]]
            self.counts = {tuple(row[:2]): row[2] for row in state["counts"]}
            if (len(set(self.pending)) != len(self.pending) or
                    not set(self.pending) <= set(self.pairs) or
                    set(self.counts) != set(self.pairs)):
                raise ValueError("Invalid balanced difficulty checkpoint")

    def peek(self):
        if not self.pending:
            self.pending = random.sample(self.pairs, len(self.pairs))
        return self.pending[-1]

    def accept(self):
        pair = self.pending.pop()
        self.counts[pair] += 1

    def get_state(self):
        return {
            "pending": self.pending,
            "counts": [[environment, difficulty, count]
                       for (environment, difficulty), count in self.counts.items()],
        }
