"""Deterministic event scheduling; no GPU, Ray, or W&B imports."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Snapshot:
    rollouts: int
    clock: int
    hf: str
    resume: str | None = None
    replay: str | None = None


@dataclass
class Version:
    environment: str
    generation: int
    available_at: int
    teacher_rollouts: int
    accuracy: float
    hf: str
    used: int = 0
    recovered: bool = False
    student_accuracy: float | None = None

    @property
    def key(self):
        return f"{self.environment}/g{self.generation:04d}"


@dataclass
class Device:
    environment: str
    rollouts: int = 0
    generation: int = 0
    active: dict | None = None
    version: Version | None = None
    parked_parent: str | None = None


@dataclass
class State:
    environments: list[str]
    budget: int
    parent_policy: str
    teacher_horizon: int | None
    clock: int = 0
    student_rollouts: int = 0
    sequence: int = 0
    devices: list[Device] = field(default_factory=list)
    snapshots: list[Snapshot] = field(default_factory=list)
    versions: list[Version] = field(default_factory=list)
    stopped: str | None = None

    def teacher_rollout_counts(self):
        counts = {}
        for d in self.devices:
            active = d.active
            partial = min(active["cap"], max(0, self.clock - active["start_clock"])) if active else 0
            counts[d.environment] = d.rollouts + partial
        return counts

    def to_dict(self):
        return {**asdict(self), "teacher_device_rollouts": self.teacher_rollout_counts()}

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data.pop("teacher_device_rollouts", None)
        data["snapshots"] = [Snapshot(**s) for s in data["snapshots"]]
        data["versions"] = [Version(**v) for v in data.get("versions", [])]
        data["devices"] = [Device(**{**d, "version": Version(**d["version"]) if d["version"] else None})
                           for d in data["devices"]]
        by_key = {v.key: v for v in data["versions"]}
        for device in data["devices"]:
            if device.version:
                device.version = by_key.get(device.version.key, device.version)
        return cls(**data)


class Simulation:
    """Teacher counters count training; clocks additionally include device idle time.

    Branch parent bounds are the teacher's cumulative rollout count, exactly
    as requested. Snapshot availability is additionally checked against event
    time. Waiting for a closed student watermark is optional and explicit.
    Evaluation samples never increment training rollout counters.
    """

    def __init__(self, state, teacher_cap=4000, merge_cap=1000, recovery=.9, quantum=400):
        self.state = state
        self.teacher_cap, self.merge_cap = teacher_cap, merge_cap
        self.recovery, self.quantum = recovery, quantum
        if state.parent_policy not in {"wait", "latest-available"}:
            raise ValueError("Unknown checkpoint parent policy")
        if (state.budget <= 0 or state.budget % 8 or
                (state.teacher_horizon is not None and (state.teacher_horizon <= 0 or state.teacher_horizon % 8))):
            raise ValueError("Budgets must be positive multiples of eight training rollouts")
        if min(teacher_cap, merge_cap, quantum) <= 0 or any(n % 8 for n in (teacher_cap, merge_cap, quantum)):
            raise ValueError("All caps must be positive multiples of eight")

    def parent(self, teacher_rollouts):
        s = self.state
        if s.parent_policy == "wait" and s.student_rollouts < min(teacher_rollouts, s.budget):
            return None
        candidates = [p for p in s.snapshots if p.rollouts <= teacher_rollouts and p.clock <= s.clock]
        return max(candidates, key=lambda p: (p.rollouts, p.clock)) if candidates else None

    def branches(self):
        """Create all currently eligible branches in stable environment order."""
        s, tasks = self.state, []
        for index, d in enumerate(s.devices):
            if d.active or s.student_rollouts >= s.budget or (s.teacher_horizon is not None and d.rollouts >= s.teacher_horizon):
                continue
            parent = self.parent(d.rollouts)
            if parent is None:
                continue
            if d.parked_parent == parent.hf:
                continue
            d.generation += 1
            cap = self.teacher_cap if s.teacher_horizon is None else min(self.teacher_cap, s.teacher_horizon - d.rollouts)
            task = {"id": f"teacher-{index:02d}-g{d.generation:04d}", "kind": "teacher",
                    "environment": d.environment, "environment_index": index,
                    "generation": d.generation, "teacher_start_rollouts": d.rollouts,
                    "start_clock": s.clock, "cap": cap,
                    "minimum_rollouts": 0, "parent": asdict(parent)}
            d.active = task
            tasks.append(task)
        return tasks

    def materialize(self, task_id, result):
        """Record a computed result without publishing it before its event time."""
        d = next(d for d in self.state.devices if d.active and d.active["id"] == task_id)
        count, accuracy = result["rollouts"], result["accuracy"]
        if not isinstance(count, int) or not d.active["minimum_rollouts"] <= count <= d.active["cap"] or count % 8:
            raise ValueError("Invalid teacher rollout count")
        if not 0 <= accuracy <= 1 or (count < d.active["cap"] and accuracy < .98):
            raise ValueError("Teacher stopped without its cap or 98% validation")
        if not result.get("hf"):
            raise ValueError("Teacher result has no final weights")
        d.active["result"] = dict(result)
        d.active["end_clock"] = d.active["start_clock"] + count

    def unknown_due(self):
        s = self.state
        return [d.active for d in s.devices if d.active and "result" not in d.active
                and d.active["start_clock"] + d.active["minimum_rollouts"] <= s.clock]

    def publish(self):
        s, published = self.state, []
        due = [d for d in s.devices if d.active and "result" in d.active and d.active["end_clock"] <= s.clock]
        for d in sorted(due, key=lambda d: (d.active["end_clock"], s.environments.index(d.environment))):
            task = d.active
            result = task["result"]
            d.rollouts += result["rollouts"]
            d.parked_parent = task["parent"]["hf"] if result["rollouts"] == 0 else None
            d.version = Version(d.environment, d.generation, task["end_clock"],
                                result["rollouts"], result["accuracy"], result["hf"])
            s.versions.append(d.version)
            published.append({"task": task, "version": asdict(d.version)})
            d.active = None
        return published

    def next_boundary(self):
        pending = [d.active for d in self.state.devices if d.active]
        return min((t.get("end_clock", t["start_clock"] + t["minimum_rollouts"]) for t in pending), default=None)

    def student_task(self):
        s = self.state
        if s.student_rollouts >= s.budget:
            s.stopped = "student_budget"
            return None
        candidates = [v for v in s.versions if not v.recovered
                      and v.used < self.merge_cap and v.available_at <= s.clock]
        if not candidates:
            return None
        # Fairly spread the next quantum across the available specialist versions.
        v = min(candidates, key=lambda v: (v.used, v.available_at, s.environments.index(v.environment)))
        count = min(self.quantum, self.merge_cap - v.used, s.budget - s.student_rollouts)
        boundary = self.next_boundary()
        if boundary is not None:
            if boundary <= s.clock:
                raise RuntimeError("Resolve/publish due teacher events before student training")
            count = min(count, boundary - s.clock)
        if count % 8 or count < 8:
            raise RuntimeError("Event times must align with the eight-rollout update")
        return {"id": f"student-{s.sequence:06d}-{s.student_rollouts:08d}", "kind": "student",
                "environment": v.environment, "generation": v.generation, "version_key": v.key,
                "teacher_hf": v.hf, "teacher_accuracy": v.accuracy,
                "target": self.recovery * v.accuracy, "cap": count,
                "start_rollouts": s.student_rollouts, "start_clock": s.clock,
                "parent": asdict(s.snapshots[-1])}

    def merge(self, task, result):
        s = self.state
        if task["start_rollouts"] != s.student_rollouts or task["start_clock"] != s.clock:
            raise ValueError("Stale student task")
        v = next((v for v in s.versions if v.key == task["version_key"]), None)
        if v is None:
            raise ValueError("Student teacher version changed in flight")
        count, accuracy = result["rollouts"], result["accuracy"]
        if not isinstance(count, int) or not 0 <= count <= task["cap"] or count % 8 or not 0 <= accuracy <= 1:
            raise ValueError("Invalid student result")
        if count < task["cap"] and accuracy + 1e-12 < task["target"]:
            raise ValueError("Student stopped without recovering its teacher's performance")
        if count and (not result.get("hf") or not result.get("resume")):
            raise ValueError("Student continuation needs both weights and optimizer checkpoint")
        s.student_rollouts += count
        s.clock += count
        s.sequence += 1
        v.used += count
        v.student_accuracy = accuracy
        v.recovered = accuracy + 1e-12 >= task["target"]
        if v.used > self.merge_cap:
            raise RuntimeError("Per-version merge budget exceeded")
        if count:
            s.snapshots.append(Snapshot(s.student_rollouts, s.clock, result["hf"], result["resume"],
                                        result.get("replay")))

    def advance_idle(self):
        boundary = self.next_boundary()
        if boundary is None:
            self.state.stopped = "available_targets_recovered_or_exhausted"
            return False
        if boundary < self.state.clock:
            raise RuntimeError("A teacher event was skipped")
        self.state.clock = boundary
        return True


def initial_state(environments, base_hf, base_ref, budget=16000, parent_policy="latest-available", teacher_horizon=None):
    if not environments or len(environments) != len(set(environments)):
        raise ValueError("Expected distinct teacher environments")
    return State(list(environments), budget, parent_policy, teacher_horizon,
                 devices=[Device(e) for e in environments],
                 snapshots=[Snapshot(0, 0, str(base_hf), str(base_ref))])
