"""Authorize removal of uploaded v3 source folders only after their consumers finish."""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(__file__).resolve().parent.parent / "data/HF-SCRATCH-ARCHIVE-20261004/release_authorization.json"
DEPENDENTS = {
    "mopd-v3-r1-p1r8-teachers-20261004-095049": [
        17720283, 17720472, 17720474,
        17721669, 17721671, 17721673, 17721675, 17721677, 17721679,
    ],
    "mopd-v3-r1-all16-warmstart-sft-20261004-095049": [17720471, 17720473, 17720475],
}


def job_state(job):
    result = subprocess.run(
        ["sacct", "-n", "-P", "--duplicates", "--allocations", "-j", str(job), "--format=JobID,State"],
        check=True, capture_output=True, text=True,
    )
    states = [line.split("|", 1)[1].split()[0] for line in result.stdout.splitlines()
              if line.split("|", 1)[0] == str(job)]
    return states


def main():
    queue = subprocess.run(["squeue", "-h", "-u", "dhei", "-o", "%i"],
                           check=True, capture_output=True, text=True).stdout.splitlines()
    queued = set(queue)
    groups = {}
    for group, jobs in DEPENDENTS.items():
        states = {str(job): job_state(job) for job in jobs}
        complete = all("COMPLETED" in states[str(job)] and str(job) not in queued for job in jobs)
        groups[group] = {"approved": complete, "dependent_jobs": states}
        print(group, "release approved" if complete else "retained: dependent jobs incomplete")
    record = {"checked_utc": datetime.now(timezone.utc).isoformat(), "groups": groups}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
