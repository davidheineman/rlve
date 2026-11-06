"""Execute the clustered event graph, then evaluate the final published student."""

import fcntl
import time

from async_branching.io import read_json, write_json
from async_branching.runner import Runner
from clustering_mopd.backend import ClusterBackend, start_workers
from clustering_mopd.schedule import ClusterSimulation, ClusterState, initial_state


def run_clients(experiment, plan, *, base_url, model_path, gateway):
    with (experiment / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (experiment / "completed.json").exists():
            return
        (experiment / "workers.stop").unlink(missing_ok=True)
        processes, logs = start_workers(experiment, plan, model_path, base_url)
        backend = ClusterBackend(experiment, plan, processes, gateway)
        try:
            deadline = time.monotonic() + 1800
            while not all(
                (experiment / "workers" / name / "ready.json").exists()
                for name in processes
            ):
                backend.check()
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "Cluster adapter slots did not initialize within 30 minutes"
                    )
                time.sleep(1)
            path = experiment / "state.json"
            state = (
                ClusterState.from_dict(read_json(path))
                if path.exists()
                else initial_state(plan, read_json(experiment / "base.json"))
            )
            simulation = ClusterSimulation(
                state,
                teacher_cap=plan["teacher_branch_rollouts"],
                merge_cap=plan["student_rollouts_per_teacher_version"],
                recovery=plan["teacher_recovery_fraction"],
                quantum=plan["student_quantum_rollouts"],
            )
            runner = Runner(simulation, backend, experiment)
            state = runner.run(close_backend=False)
            # If all targets are exhausted early, the unchanged final policy
            # remains the policy at the common comparison deadline.
            trained_until = state.clock
            if state.clock < state.clock_budget:
                state.clock = state.clock_budget
                runner.record("terminal_idle", {"from_clock": trained_until})
            snapshot = state.snapshots[-1]
            backend.student(
                {
                    "id": "student-final-evaluation",
                    "kind": "student",
                    "evaluation_only": True,
                    "environment": state.environments[0],
                    "generation": 0,
                    "start_rollouts": state.student_rollouts,
                    "start_clock": state.clock,
                    "parent": vars(snapshot).copy(),
                    "cap": 0,
                }
            )
            backend.collect(state)
            write_json(
                experiment / "completed.json",
                {
                    "stage": "complete",
                    "student": vars(snapshot).copy(),
                    "state": state.to_dict(),
                    "cluster_count": plan["cluster_count"],
                    "assignment": plan["assignment"],
                    "simulated_clock": state.clock,
                    "last_training_clock": trained_until,
                    "physical_finished_at": time.time(),
                },
            )
        finally:
            backend.close()
            for log in logs:
                log.close()
