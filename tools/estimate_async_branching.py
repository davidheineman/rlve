"""Estimate physical elapsed time using measured specialist throughput and scheduler replay."""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from async_branching.io import write_json
from async_branching.schedule import Simulation, initial_state
from async_branching.replay import ReplayBackend
from async_branching.runner import Runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-experiment", type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference_experiment.resolve()
    with (reference / "jobs.tsv").open() as handle:
        teachers = [r for r in csv.DictReader(handle, delimiter="\t") if r["kind"] == "teacher"][:32]
    ids = ",".join(r["training_job"] for r in teachers)
    raw = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", ids,
                                   "-o", "JobIDRaw,ElapsedRaw"], text=True)
    elapsed = {p[0]: int(p[1]) for line in raw.splitlines() if len(p := line.split("|")) >= 2 and p[1].isdigit()}
    observations = []
    for row in teachers:
        marker = json.loads((reference / "teacher" / row["label"] / "weights_only_complete.json").read_text())
        count = (marker["checkpoint_step"] + 1) * 8
        duration = elapsed[row["training_job"]]
        observations.append({"environment": row["label"].split("-", 1)[1], "rollouts": count,
                             "elapsed_seconds": duration, "seconds_per_rollout": duration / count})
    results = []
    for size in (16, 32):
        plan = json.loads((ROOT / f"data/DATA-SCALING-ASYNC-V1/n{size}.json").read_text())
        state = initial_state(plan["environments"], plan["base_hf"], plan["base_ref"],
                              budget=16000, parent_policy="latest-available")
        simulation = Simulation(state)
        backend = ReplayBackend()
        runner = Runner(simulation, backend, "/unused-synthetic-replay")
        # The cost estimate needs the event graph, not files or model execution.
        runner.record = lambda kind, value: runner.events.append({"event": kind, **value})
        runner.run()
        compute = {name: 0 for name in plan["environments"]}
        branches = [task for task in backend.tasks if task["kind"] == "teacher"]
        for task in branches:
            compute[task["environment"]] += task["cap"]
        reference_rates = {r["environment"]: r["seconds_per_rollout"] for r in observations[:size]}
        teacher_gpu_hours = sum(compute[name] * reference_rates[name] for name in compute) / 3600
        student_segments = sum(t["kind"] == "student" for t in backend.tasks)
        # Same 8-rollout update, now one student GPU plus one scorer GPU.
        # The existing N16 runs measured roughly 9–17 sec with two student GPUs.
        # These are planning factors until a GPU smoke/benchmark is authorized.
        student_hours = [(2000 * step_seconds + student_segments * overhead_seconds) / 3600
                         for step_seconds, overhead_seconds in ((12, 200), (24, 600))]
        # Conservative physical schedule: all four GPUs when causality blocks
        # the student, otherwise two specialists beside its two-GPU segment.
        wall = [.75 * teacher_gpu_hours / 4 + student_hours[0],
                1.25 * teacher_gpu_hours / 4 + student_hours[1]]
        wall = [math.floor(wall[0] / 5) * 5, math.ceil(wall[1] / 5) * 5]
        results.append({"size": size, "physical_gpus": 4, "simulated_teacher_devices": size,
                        "simulated_student_devices": 1, "student_rollouts": state.student_rollouts,
                        "replay_final_clock": state.clock, "replay_teacher_branches_started": len(branches),
                        "replay_teacher_rollouts_requested": sum(compute.values()),
                        "reference_rate_teacher_gpu_hours": teacher_gpu_hours,
                        "student_segments": student_segments, "student_elapsed_hours_range": student_hours,
                        "physical_elapsed_hours_range": wall,
                        "allocation_gpu_hours_range": [4 * n for n in wall]})
    report = {"utc": datetime.now(timezone.utc).isoformat(), "reference_experiment": str(reference),
              "observations": observations, "estimates": results,
              "assumptions": [
                  "Planning estimate, not a measurement of the new backend orchestration",
                  "Four continuously allocated >=48 GiB GPUs; queue time is additional",
                  "16k actual student rollouts in both panels; validation samples are excluded",
                  "Replay assumes full 4k teacher branches and 1k per-version student caps, with no 90% recovery savings",
                  "Teacher rate uses the completed current-sweep specialists; warm-parent throughput may differ by 25% or more",
                  "Student uses the existing Slime loss and SGLang generation, one colocated actor/rollout GPU and one scorer GPU",
                  "Student range includes baseline/gate/ID/test eval, subprocess startup, optimizer checkpoints and HF export",
                  "Outstanding teacher tasks are conservatively charged their full requested budgets",
                  "Frequent early teacher stops create more branches; parent conversion/startup overhead and student idle gaps can extend the estimate",
                  "No guaranteed finite upper bound if successive specialists yield no student training progress; the estimate assumes continued useful teacher arrivals",
                  "One job self-requeues before its 24h limit; estimate excludes preemption/requeue waiting and interrupted-segment replay",
              ]}
    path = ROOT / "data/DATA-SCALING-ASYNC-V1/compute_estimate.json"
    write_json(path, report)
    print(json.dumps({"estimates": results}, indent=2))


if __name__ == "__main__":
    main()
