"""Record the first three training updates of the six submitted N=8 students."""

import argparse
import csv
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


STEP = re.compile(r"\bdavid (\d+): \{")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}


def jobs(root):
    items = []
    for manifest, label in (("jobs.tsv", "rl"), ("dpo-students.tsv", "dpo")):
        with (root / manifest).open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                kind = row["kind"]
                run_label = kind.replace("rl-idea", "idea") + "-rl" if label == "rl" else kind
                items.append({"kind": kind, "job": row["job"],
                              "log": str(root.parent / row["group"] / f"{run_label}.log")})
    return items


def slurm_states(job_ids):
    result = subprocess.run(
        ["sacct", "-j", ",".join(job_ids), "--format=JobIDRaw,State", "-P", "-n"],
        capture_output=True, text=True, check=False,
    )
    states = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[0] in job_ids:
            states[fields[0]] = fields[1].split()[0].split("+")[0]
    return states


def array_states(array_id):
    result = subprocess.run(
        ["sacct", "-j", array_id, "--format=JobIDRaw,State", "-P", "-n"],
        capture_output=True, text=True, check=False,
    )
    states = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[0].startswith(array_id + "_") and "." not in fields[0]:
            states[fields[0]] = fields[1].split()[0].split("+")[0]
    return states


def snapshot(root, items):
    states = slurm_states([item["job"] for item in items])
    results = []
    for item in items:
        path = Path(item["log"])
        contents = path.read_text(errors="replace") if path.exists() else ""
        updates = sorted({int(match.group(1)) for match in STEP.finditer(contents)})
        state = states.get(item["job"], "UNKNOWN")
        errors = [line[-300:] for line in contents.splitlines()
                  if "Traceback (most recent call last)" in line or "FAILED student" in line]
        results.append({**item, "state": state, "updates_seen": updates[:5],
                        "first_three_verified": all(step in updates for step in (0, 1, 2)),
                        "error_markers": errors[-3:],
                        "failed_before_three": state in TERMINAL and not all(step in updates for step in (0, 1, 2))})
    low_group = root.parent / "dpo-n8-low-teachers-20261006-232728" / "resumable"
    low_ready = len(list(low_group.glob("*/weights_only_complete.json")))
    pair_ready = len(list((root / "pairs").glob("*.jsonl")))
    teacher_states = array_states("17744602")
    pair_states = array_states("17744605")
    pack_state = slurm_states(["17744606"]).get("17744606", "UNKNOWN")
    preparation_failed = (
        any(state in TERMINAL - {"COMPLETED"} for state in teacher_states.values()) or
        any(state in TERMINAL - {"COMPLETED"} for state in pair_states.values()) or
        pack_state in TERMINAL - {"COMPLETED"}
    )
    preparation = {"low_snapshots_ready": low_ready, "teacher_pairs_ready": pair_ready,
                   "replay_ready": (root / "teacher-replay" / "manifest.json").exists(),
                   "teacher_states": teacher_states, "pair_states": pair_states,
                   "pack_state": pack_state, "failed": preparation_failed}
    return {"updated_utc": datetime.now(timezone.utc).isoformat(), "runs": results,
            "preparation": preparation,
            "all_first_three_verified": all(item["first_three_verified"] for item in results),
            "any_failed_before_three": preparation_failed or
            any(item["failed_before_three"] for item in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--hours", type=int, default=72)
    args = parser.parse_args()
    items = jobs(args.root)
    if len(items) != 6:
        raise RuntimeError(f"Expected six students, found {len(items)}")
    deadline = time.monotonic() + args.hours * 3600
    previous = None
    while True:
        status = snapshot(args.root, items)
        output = args.root / "monitor.json"
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
        progress = ([(item["kind"], item["state"], item["updates_seen"])
                     for item in status["runs"]],
                    status["preparation"]["low_snapshots_ready"],
                    status["preparation"]["teacher_pairs_ready"],
                    status["preparation"]["pack_state"])
        if progress != previous:
            print(status["updated_utc"], progress, flush=True)
            previous = progress
        if status["all_first_three_verified"] or status["any_failed_before_three"]:
            break
        if time.monotonic() >= deadline:
            print("Monitoring window expired before all students reached three updates", flush=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
