"""Bounded OPD trajectory replay; no model, GPU, or teacher weights are retained."""

from collections import deque
from copy import deepcopy
from fractions import Fraction
import gzip
import json
import math
import os
from pathlib import Path
import random


def replay_count(step, batch_size, fraction):
    """Fractional slots carry across updates, including segment boundaries."""
    if step < 0 or batch_size < 1 or not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("Replay requires a nonnegative step, positive batch, and fraction in [0, 1)")
    rate = Fraction(str(fraction)) * batch_size
    return min(batch_size - 1, ((step + 1) * rate).__floor__() - (step * rate).__floor__())


def validate_trace(trace):
    environment = trace.get("metadata", {}).get("environment")
    count = trace.get("response_length", 0)
    if not environment or not isinstance(count, int) or not 0 < count < len(trace.get("tokens", [])):
        raise ValueError("An OPD replay trace needs an environment, prompt and response tokens")
    for key in ("rollout_log_probs", "teacher_log_probs"):
        values = trace.get(key)
        if values is None or len(values) != count or not all(math.isfinite(x) for x in values):
            raise ValueError(f"OPD replay requires finite, aligned {key}")
    mask = trace.get("loss_mask")
    if mask is not None and (len(mask) != count or any(x not in (0, 1) for x in mask)):
        raise ValueError("OPD replay loss masks must align with response tokens")


class OPDReplayBuffer:
    def __init__(self, capacity_per_env=400):
        if type(capacity_per_env) is not int or capacity_per_env < 1:
            raise ValueError("Replay capacity per environment must be a positive integer")
        self.capacity = capacity_per_env
        self.entries = {}
        self.cursor = 0
        self.fresh_count = 0

    def add(self, samples):
        """Record fresh scored traces only. Replayed traces never refresh their age."""
        for sample in samples:
            trace = deepcopy(sample if isinstance(sample, dict) else sample.to_dict())
            if trace.get("opd_replay", False):
                continue
            validate_trace(trace)
            trace["opd_replay"] = False
            environment = trace["metadata"]["environment"]
            self.entries.setdefault(environment, deque(maxlen=self.capacity)).append(trace)
            self.fresh_count += 1

    def sample(self, count, excluded_environments=(), seed=0):
        eligible = sorted(e for e, traces in self.entries.items()
                          if traces and e not in excluded_environments)
        if not eligible:
            return []
        rng = random.Random(seed)
        result = []
        for index in range(count):
            environment = eligible[(self.cursor + index) % len(eligible)]
            trace = deepcopy(rng.choice(self.entries[environment]))
            trace["opd_replay"] = True
            # Keep replay IDs distinct from the data source's fresh indices.
            trace["index"] = -(seed * max(1, count) + index + 1)
            result.append(trace)
        self.cursor += count
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        state = {"version": 1, "capacity_per_env": self.capacity, "cursor": self.cursor,
                 "fresh_count": self.fresh_count,
                 "entries": {e: list(v) for e, v in sorted(self.entries.items())}}
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                compressed.write(json.dumps(state, allow_nan=False, separators=(",", ":")).encode())
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
        return str(path)

    @classmethod
    def load(cls, path, capacity_per_env=400):
        with gzip.open(path, "rt") as handle:
            state = json.load(handle)
        if state.get("version") != 1 or state.get("capacity_per_env") != capacity_per_env:
            raise ValueError("Replay snapshot version or per-environment capacity does not match")
        result = cls(capacity_per_env)
        if type(state.get("cursor")) is not int or state["cursor"] < 0:
            raise ValueError("Invalid replay sampling cursor")
        result.cursor = state["cursor"]
        if type(state.get("fresh_count")) is not int or state["fresh_count"] < 0:
            raise ValueError("Invalid replay fresh trajectory count")
        result.fresh_count = state["fresh_count"]
        for environment, traces in state["entries"].items():
            if len(traces) > capacity_per_env:
                raise ValueError("Replay snapshot exceeds its capacity")
            for trace in traces:
                validate_trace(trace)
                if trace["metadata"]["environment"] != environment or trace.get("opd_replay"):
                    raise ValueError("Replay snapshot contains misfiled or replayed traces")
            result.entries[environment] = deque(traces, maxlen=capacity_per_env)
        return result

    @property
    def size(self):
        return sum(len(v) for v in self.entries.values())
