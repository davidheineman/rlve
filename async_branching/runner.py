"""Replay causal rollout-clock events against either a fake or GPU backend."""

from .io import write_json


class Runner:
    def __init__(self, simulation, backend, directory):
        self.simulation, self.backend, self.directory = simulation, backend, directory
        self.futures = {}
        self.events = []

    def record(self, kind, value):
        from pathlib import Path
        import json
        s = self.simulation.state
        event = {"event": kind, "clock": s.clock, "student_rollouts": s.student_rollouts, **value}
        self.events.append(event)
        path = Path(self.directory) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        write_json(Path(self.directory) / "state.json", s.to_dict())

    def run(self, *, close_backend=True):
        sim, s = self.simulation, self.simulation.state
        try:
            # On restart, unfinished branch tasks replay from their immutable parent.
            for d in s.devices:
                if d.active and "result" not in d.active:
                    self.futures[d.active["id"]] = self.backend.teacher(d.active)
            while not s.stopped:
                if s.student_rollouts >= s.budget:
                    s.stopped = "student_budget"
                    break
                # Results are consumed by logical time and environment order,
                # never by which physical GPU happened to finish first.
                unknown = sim.unknown_due()
                if unknown:
                    # The student is waiting for causal information. Lend its
                    # two physical GPUs to queued teachers during that wait.
                    self.backend.reserve_student(False)
                for task in unknown:
                    future = self.futures[task["id"]]
                    result = self.backend.information(task, future)
                    if "lower_bound_rollouts" in result:
                        bound = result["lower_bound_rollouts"]
                        if bound <= task["minimum_rollouts"] or bound > task["cap"] or bound % 8:
                            raise RuntimeError("Invalid branch progress watermark")
                        task["minimum_rollouts"] = bound
                        self.record("teacher_progress", {"task_id": task["id"], "lower_bound_rollouts": bound})
                        continue
                    self.futures.pop(task["id"])
                    sim.materialize(task["id"], result)
                    self.record("teacher_computed", {"task_id": task["id"], "result": result})
                for item in sim.publish():
                    self.record("teacher_available", item)
                self.backend.reserve_student(any(not v.recovered and v.used < sim.merge_cap
                                                 for v in s.versions))
                for task in sim.branches():
                    self.record("branch", {"task": task})
                    self.futures[task["id"]] = self.backend.teacher(task)
                # Resolve initial validation before advancing time. A teacher
                # already at 98% publishes its parent weights with zero rollouts.
                if sim.unknown_due():
                    continue
                task = sim.student_task()
                if task:
                    self.record("merge_start", {"task": task})
                    result = self.backend.student(task)
                    sim.merge(task, result)
                    self.record("merge_end", {"task": task, "result": result})
                else:
                    if not sim.advance_idle():
                        break
                    self.record("idle", {})
                self.backend.collect(s)
            self.record("stopped", {"reason": s.stopped})
            return s
        finally:
            if close_backend:
                self.backend.close()
