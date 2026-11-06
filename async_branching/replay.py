"""CPU-only scheduler exercise with explicit synthetic outcomes, not training."""

from concurrent.futures import Future
from pathlib import Path


class ReplayBackend:
    def __init__(self, teacher_rollouts=4000, recover_after=None, teacher_accuracy=.8):
        self.teacher_rollouts = teacher_rollouts
        self.recover_after = recover_after
        self.teacher_accuracy = teacher_accuracy
        self.tasks = []
        self.teacher_training_rollouts = 0
        self.student_training_rollouts = 0

    def reserve_student(self, active):
        pass

    def information(self, task, future):
        return future.result()

    def teacher(self, task):
        self.tasks.append(dict(task))
        amount = min(task["cap"], self.teacher_rollouts)
        accuracy = .99 if amount < task["cap"] else self.teacher_accuracy
        self.teacher_training_rollouts += amount
        future = Future()
        future.set_result({"rollouts": amount, "accuracy": accuracy,
                           "hf": f"replay/{task['id']}/hf-final"})
        return future

    def student(self, task):
        self.tasks.append(dict(task))
        count = task["cap"]
        accuracy = 0.
        if self.recover_after is not None:
            count = min(count, self.recover_after)
            accuracy = task["target"]
        self.student_training_rollouts += count
        return {"rollouts": count, "accuracy": accuracy,
                "hf": f"replay/{task['id']}/hf-final", "resume": f"replay/{task['id']}/ckpt"}

    def collect(self, state):
        pass

    def close(self):
        pass
