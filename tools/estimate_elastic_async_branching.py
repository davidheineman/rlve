"""Recompute elastic planning ranges from the measured local estimate inputs."""

import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from async_branching.io import read_json, write_json


def estimate(reference, cap):
    teachers = cap - 2
    hours = reference["reference_rate_teacher_gpu_hours"]
    student = reference["student_elapsed_hours_range"]
    # Sum rather than max: do not assume perfect teacher/student overlap.
    # The ±25% specialist factor includes variation across warm-start branches.
    low = .75 * hours / teachers + student[0]
    high = 1.25 * hours / teachers + student[1]
    return {"size": reference["size"], "physical_gpu_cap": cap,
            "maximum_teacher_jobs": teachers, "student_gpus": 2,
            "reference_teacher_gpu_hours": hours,
            "student_elapsed_hours_range": student,
            "physical_elapsed_hours_range": [round(low / 2) * 2, math.ceil(high / 2) * 2]}


def main():
    source = ROOT / "data/DATA-SCALING-ASYNC-V1/compute_estimate.json"
    reference = read_json(source)
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "reference": str(source.relative_to(ROOT)),
        "estimates": [estimate(row, cap) for cap in (8, 12, 16) for row in reference["estimates"]],
        "assumptions": [
            "Planning ranges, not GPU benchmarks of this new elastic backend",
            "The GPU cap includes the persistent two-GPU student service; teachers request one GPU each",
            "Teacher slots must be occupied on average to achieve the listed range; a cap is not an allocation guarantee",
            "Same full-cap 16k-student rollout event graph and teacher compute as compute_estimate.json",
            "Formula: 0.75 to 1.25 times measured teacher GPU-hours divided by (cap - 2), plus 9 to 20 student hours; lower endpoint rounded to two hours, upper endpoint rounded up",
            "Does not assume perfect overlap; actual causal barriers, heterogeneous GPUs and branch startup can extend elapsed time",
            "Excludes Slurm queue/preemption delays and replay of interrupted tasks",
            "GPU resource limits may change during execution; lowering a limit drains existing jobs without changing logical events",
            "Student work remains sequential; adding GPUs cannot divide total wall time proportionally",
            "Early stops/recovery can alter the number of branches and total compute",
        ]}
    write_json(ROOT / "data/DATA-SCALING-ASYNC-V1/compute_estimate_elastic.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
