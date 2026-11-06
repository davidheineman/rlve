"""Run the original rollout-clock simulation with a single persistent Miles backend."""

import fcntl
import time

from async_branching.io import read_json, write_json
from async_branching.runner import Runner
from async_branching.schedule import Simulation, State, initial_state
from async_lora.backend import LoRABackend, start_workers


def run_clients(experiment, plan, *, base_url, model_path, gateway):
    lock = (experiment / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (experiment / "workers.stop").unlink(missing_ok=True)
    processes, logs = start_workers(experiment, plan, model_path, base_url)
    backend = LoRABackend(experiment, plan, processes, gateway)
    try:
        deadline = time.monotonic() + 1800
        while not all((experiment / "workers" / name / "ready.json").exists() for name in processes):
            backend.check()
            if time.monotonic() >= deadline:
                raise TimeoutError("LoRA slots and baseline did not initialize within 30 minutes")
            time.sleep(1)
        state_path = experiment / "state.json"
        if state_path.exists():
            state = State.from_dict(read_json(state_path))
        else:
            base = read_json(experiment / "base.json")
            state = initial_state(
                plan["environments"],
                base["hf"],
                base["resume"],
                plan["student_rollouts"],
                plan["parent_policy"],
                plan["teacher_device_rollout_horizon"],
            )
            state.snapshots[0].replay = base.get("replay")
        simulation = Simulation(
            state,
            teacher_cap=plan["teacher_branch_rollouts"],
            merge_cap=plan["student_rollouts_per_teacher_version"],
            recovery=plan["teacher_recovery_fraction"],
            quantum=plan["student_quantum_rollouts"],
        )
        print(f"All {len(processes)} LoRA slots ready; starting causal async MOPD", flush=True)
        final = Runner(simulation, backend, experiment).run()
        write_json(experiment / "completed.json", final.to_dict())
    finally:
        backend.close()
        for log in logs:
            log.close()
        lock.close()
