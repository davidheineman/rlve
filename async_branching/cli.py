"""Run or replay one allocation's simulated specialist devices and MOPD student."""

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess

from .io import read_json, write_json
from .schedule import Simulation, State, initial_state
from .runner import Runner


def finalize(directory, state):
    from .backend import SlimeBackend
    final_teachers = SlimeBackend.finalize_storage(directory, state)
    write_json(directory / "completed.json", {
        "reason": state.stopped, "student_rollouts": state.student_rollouts,
        "student_hf": state.snapshots[-1].hf, "simulated_clock": state.clock,
        "teacher_device_rollouts": state.teacher_rollout_counts(), "teacher_hf": final_teachers,
        "student_replay": state.snapshots[-1].replay,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--replay", action="store_true", help="CPU-only synthetic exercise; never trains or logs to W&B")
    parser.add_argument("--replay-recover-after", type=int)
    args = parser.parse_args()
    plan = read_json(args.plan)
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    plan_copy = directory / "plan.json"
    if plan_copy.exists() and read_json(plan_copy) != plan:
        raise RuntimeError("Cannot resume with a different plan")
    write_json(plan_copy, plan)
    with (directory / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / "state.json"
        if not path.exists() and not args.replay and plan.get("minimum_launch_free_gib"):
            from .storage import wait_for_launch_storage
            wait_for_launch_storage(directory, plan["minimum_launch_free_gib"])
        state = State.from_dict(read_json(path)) if path.exists() else initial_state(
            plan["environments"], plan["base_hf"], plan["base_ref"], plan["student_rollouts"],
            plan["parent_policy"], plan.get("teacher_device_rollout_horizon"))
        if state.stopped:
            if not args.replay:
                if plan.get("backend") == "slurm":
                    from .slurm import SlurmBackend
                    SlurmBackend.stop_workers(directory)
                finalize(directory, state)
            print(f"Already stopped: {state.stopped}; student rollouts={state.student_rollouts}")
            return
        if args.replay:
            from .replay import ReplayBackend
            backend = ReplayBackend(recover_after=args.replay_recover_after)
        elif plan.get("backend") == "slurm":
            from .slurm import SlurmBackend
            backend = SlurmBackend(plan, directory, state)
        else:
            from .backend import SlimeBackend
            devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if len(devices) != 4 or not all(devices):
                raise RuntimeError("This implementation expects exactly four allocated physical GPUs")
            backend = SlimeBackend(plan, directory, devices)
        simulation = Simulation(state, plan["teacher_branch_rollouts"], plan["student_rollouts_per_teacher_version"],
                                plan["teacher_recovery_fraction"], plan["student_quantum_rollouts"])

        def terminate(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        Runner(simulation, backend, directory).run()
        if not args.replay:
            finalize(directory, state)
        print(f"Stopped: {state.stopped}; student rollouts={state.student_rollouts}; clock={state.clock}")


if __name__ == "__main__":
    main()
