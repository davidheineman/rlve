"""Reuse causal rollout events with a fixed teacher-device pool and deadline."""

import random
from dataclasses import dataclass, field

from async_branching.schedule import Device, Simulation, Snapshot, State, Version


@dataclass
class ClusterState(State):
    clock_budget: int = 16000
    cluster_queue: list[str] = field(default_factory=list)
    cluster_generations: dict[str, int] = field(default_factory=dict)
    parked_parents: dict[str, list[str]] = field(default_factory=dict)
    fixed_lanes: bool = False

    def teacher_rollout_counts(self):
        return {
            f"lane-{i:02d}": d.rollouts
            + (
                min(d.active["cap"], max(0, self.clock - d.active["start_clock"]))
                if d.active
                else 0
            )
            for i, d in enumerate(self.devices)
        }


def initial_state(plan, base):
    labels = [c["id"] for c in plan["clusters"]]
    queue = list(labels)
    random.Random(plan["schedule_seed"]).shuffle(queue)
    return ClusterState(
        environments=labels,
        budget=plan["student_rollouts"],
        parent_policy=plan["parent_policy"],
        teacher_horizon=plan["teacher_device_rollout_horizon"],
        devices=[Device(e) for e in queue[: plan["teacher_worker_count"]]],
        snapshots=[Snapshot(0, 0, base["hf"], base["resume"])],
        clock_budget=plan["simulated_clock_budget"],
        cluster_queue=queue,
        cluster_generations={label: 0 for label in labels},
        fixed_lanes=len(labels) <= plan["teacher_worker_count"],
    )


class ClusterSimulation(Simulation):
    def branches(self):
        s, tasks = self.state, []
        if s.clock >= s.clock_budget or s.student_rollouts >= s.budget:
            return tasks
        active = {d.active["environment"] for d in s.devices if d.active}
        for lane, device in enumerate(s.devices):
            if device.active or (
                s.teacher_horizon is not None and device.rollouts >= s.teacher_horizon
            ):
                continue
            parent = self.parent(device.rollouts)
            if parent is None:
                continue
            chosen = None
            for cluster in s.cluster_queue:
                if cluster in active or (
                    s.fixed_lanes and cluster != device.environment
                ):
                    continue
                if parent.hf in s.parked_parents.get(cluster, []):
                    continue
                chosen = cluster
                break
            if chosen is None:
                continue
            s.cluster_queue.remove(chosen)
            device.environment = chosen
            s.cluster_generations[chosen] += 1
            device.generation = s.cluster_generations[chosen]
            cap = min(self.teacher_cap, s.clock_budget - s.clock)
            if s.teacher_horizon is not None:
                cap = min(cap, s.teacher_horizon - device.rollouts)
            task = {
                "id": f"teacher-{chosen}-g{device.generation:04d}",
                "kind": "teacher",
                "environment": chosen,
                "environment_index": lane,
                "lane": lane,
                "generation": device.generation,
                "teacher_start_rollouts": device.rollouts,
                "start_clock": s.clock,
                "cap": cap,
                "minimum_rollouts": 0,
                "parent": vars(parent).copy(),
            }
            device.active = task
            active.add(chosen)
            tasks.append(task)
        return tasks

    def publish(self):
        s, published = self.state, []
        due = [
            d
            for d in s.devices
            if d.active and "result" in d.active and d.active["end_clock"] <= s.clock
        ]
        for d in sorted(
            due,
            key=lambda d: (d.active["end_clock"], s.environments.index(d.environment)),
        ):
            task, result = d.active, d.active["result"]
            d.rollouts += result["rollouts"]
            if result["rollouts"] == 0:
                # Shared lanes can revisit earlier rollout bounds. Remember
                # every solved parent, otherwise two lanes with different
                # bounds can bounce a zero-rollout cluster forever.
                parents = s.parked_parents.setdefault(d.environment, [])
                if task["parent"]["hf"] not in parents:
                    parents.append(task["parent"]["hf"])
            d.version = Version(
                d.environment,
                d.generation,
                task["end_clock"],
                result["rollouts"],
                result["accuracy"],
                result["hf"],
            )
            s.versions.append(d.version)
            published.append({"task": task, "version": vars(d.version).copy()})
            s.cluster_queue.append(d.environment)
            d.active = None
        return published

    def student_task(self):
        if self.state.clock >= self.state.clock_budget:
            self.state.stopped = "simulated_wall_time_budget"
            return None
        task = super().student_task()
        if task:
            task["cap"] = min(task["cap"], self.state.clock_budget - self.state.clock)
        return task

    def merge(self, task, result):
        super().merge(task, result)
        if self.state.clock >= self.state.clock_budget:
            self.state.stopped = "simulated_wall_time_budget"

    def advance_idle(self):
        boundary = self.next_boundary()
        if boundary is not None and boundary >= self.state.clock_budget:
            self.state.clock = self.state.clock_budget
            self.state.stopped = "simulated_wall_time_budget"
            return True
        return super().advance_idle()
