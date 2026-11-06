"""Wait on scratch headroom before the coordinator submits any GPU work."""

from pathlib import Path
import shutil
import time

from .io import write_json


def wait_for_launch_storage(directory, minimum_gib, interval=60):
    if type(minimum_gib) is not int or minimum_gib <= 0:
        raise ValueError("minimum_launch_free_gib must be a positive integer")
    directory = Path(directory)
    while True:
        free = shutil.disk_usage(directory).free
        ready = free >= minimum_gib * 1024**3
        now = time.time()
        write_json(directory / "coordinator-lease.json", {"timestamp": now})
        write_json(directory / "storage-wait.json", {
            "timestamp": now, "status": "ready" if ready else "waiting",
            "free_gib": free / 1024**3, "minimum_free_gib": minimum_gib,
        })
        print(f"Launch storage: {free / 1024**3:.1f}/{minimum_gib} GiB free; "
              f"{'ready' if ready else 'waiting without GPU workers'}", flush=True)
        if ready:
            return
        time.sleep(interval)
