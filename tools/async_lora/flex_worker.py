"""One optional generation GPU; never owns training state or joins core Ray."""

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

from lora_teachers.runner import stop_process, verify_gpu_visibility, verify_inputs
from submit_data_scaling_lora import container_argv, gpu_environment

from async_branching.io import read_json, write_json
from async_lora.flex_protocol import (
    check_probe,
    engine_argv,
    http,
    local_ip,
    router_workers,
    unregister,
    validate_engine,
)


def lease_valid(directory, generation):
    lease = read_json(directory / "lease.json")
    return lease["generation"] == generation and time.time() - lease["time"] < 60


def worker(args):
    verify_inputs(args.experiment)
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.total", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    os.environ.update(gpu_environment(rows, os.environ.get("CUDA_VISIBLE_DEVICES"), expected_count=1))
    command = container_argv(args.experiment, scratch=args.scratch, image=args.image, model_path=args.model_path)
    original = str(args.experiment / "source/rlve/tools/submit_data_scaling_lora.py")
    command[command.index(original)] = str(args.experiment / "source/rlve/tools/submit_async_lora.py")
    command[command.index("--inside")] = "--flex-inside"
    home = args.experiment / "container-home" / f"flex-{os.environ['SLURM_JOB_ID']}"
    home.mkdir(parents=True, exist_ok=True)
    command[command.index("--home") + 1] = str(home)
    command += ["--slot", str(args.slot), "--generation", args.generation]
    script = (
        'source "$1"; rlve_reserve_port_slot "$2"; export LORA_PORT_SLOT="$RLVE_RESERVED_PORT_SLOT"; shift 2; exec "$@"'
    )
    process = subprocess.Popen(
        [
            "bash",
            "-euc",
            script,
            "flex-port-lock",
            str(args.experiment / "source/rlve/scripts/slurm/port_slot.sh"),
            str(args.scratch / "locks/rlve-ports" / os.uname().nodename.split(".")[0]),
            *command,
        ],
        start_new_session=True,
    )
    requeue = False

    def terminate(signum, _frame):
        nonlocal requeue
        requeue = signum == signal.SIGUSR1
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGUSR1, terminate)
    signal.signal(signal.SIGTERM, terminate)
    status = process.wait()
    if requeue and lease_valid(args.experiment / "flex", args.generation):
        subprocess.run(["scontrol", "requeue", os.environ["SLURM_JOB_ID"]], check=True)
    elif status:
        raise SystemExit(status)


def inside(args):
    directory = args.experiment / "flex"
    core = read_json(directory / "core.json")
    if core["generation"] != args.generation or not lease_valid(directory, args.generation):
        return
    work = directory / f"gpu-{os.environ['SLURM_JOB_ID']}"
    work.mkdir(exist_ok=True)
    verify_gpu_visibility(work, expected_count=1)
    host, port = local_ip(), 6500 + 3000 * int(os.environ["LORA_PORT_SLOT"])
    url = f"http://{host}:{port}"
    record = {"job_id": os.environ["SLURM_JOB_ID"], "generation": args.generation, "url": url, "slot": args.slot}
    status_path = directory / f"worker-{args.slot}.json"

    def status(stage, **extra):
        write_json(status_path, record | {"time": time.time(), "stage": stage} | extra)

    process = None
    registered = False

    def terminate(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    try:
        with (work / "engine.log").open("a") as log:
            command = engine_argv(core["engine_config"], host=host, port=port, slot=args.slot)
            write_json(work / "engine_argv.json", command)
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status("starting")
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Flex SGLang engine exited; inspect its engine.log")
                if not lease_valid(directory, args.generation):
                    return
                try:
                    http(url + "/health", timeout=2)
                    break
                except (OSError, ValueError):
                    time.sleep(3)
            else:
                raise TimeoutError("Flex engine did not start within 20 minutes")
            plan = read_json(args.experiment / "plan.json")
            validate_engine(
                http(url + "/get_server_info"),
                model_path=args.model_path,
                slots=plan["adapter_slots"],
                rank=plan["lora_rank"],
            )
            status("probing")
            probe = check_probe(url, read_json(directory / "probe.json"))
            http(core["router_url"] + "/workers", {"url": url, "worker_type": "regular"}, timeout=30)
            registered = True
            deadline = time.monotonic() + 60
            while url not in {worker["url"] for worker in router_workers(core["router_url"])}:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Flex generator was not admitted by the router")
                time.sleep(1)
            status("ready", probe=probe)
            while lease_valid(directory, args.generation):
                if process.poll() is not None:
                    raise RuntimeError("Flex SGLang engine exited after admission")
                http(url + "/health", timeout=5)
                status("ready", probe=probe)
                time.sleep(5)
    finally:
        if registered:
            try:
                unregister(core["router_url"], url)
            except (OSError, ValueError):
                pass
        status("stopped")
        if process is not None:
            stop_process(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--worker", action="store_true")
    modes.add_argument("--inside", action="store_true")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--scratch", type=Path, default=Path("/juice5b/scr5b/dhei"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    (worker if args.worker else inside)(args)


if __name__ == "__main__":
    main()
