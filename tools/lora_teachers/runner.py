"""Own a private Ray cluster and concurrent teacher clients within one allocation."""

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from lora_teachers.plan import gateway_argv, sha256


def verify_inputs(experiment):
    hashes = json.loads((experiment / "input_sha256.json").read_text())
    for relative, expected in hashes.items():
        if sha256(experiment / relative) != expected:
            raise ValueError(f"Frozen experiment input changed: {relative}")
    preserved_results(json.loads((experiment / "plan.json").read_text()))


def preserved_results(plan):
    """Read completed teachers retained by a recovery, checking their identities and bytes."""
    results = []
    for record in plan.get("recovery", {}).get("preserved_teachers", []):
        completion = Path(record["completion_path"])
        if sha256(completion) != record["completion_sha256"]:
            raise ValueError(f"Preserved completion changed: {completion}")
        result = json.loads(completion.read_text())
        if result["environment"] != record["environment"]:
            raise ValueError("Preserved teacher identity changed")
        for name, expected in record["adapter_sha256"].items():
            if sha256(Path(result["adapter_path"]) / name) != expected:
                raise ValueError(f"Preserved adapter changed: {record['environment']}/{name}")
        results.append(result)
    return results


def verify_gpu_visibility(experiment, *, expected_count=4):
    # Torch is a container-only dependency; the host also imports verify_inputs.
    import torch

    expected = os.environ["LORA_EXPECTED_GPU_UUIDS"].split(",")
    properties = [torch.cuda.get_device_properties(index) for index in range(torch.cuda.device_count())]
    actual = [str(device.uuid).removeprefix("GPU-").lower() for device in properties]
    normalized = [uuid.removeprefix("GPU-").lower() for uuid in expected]
    if len(actual) != expected_count or len(expected) != expected_count or set(actual) != set(normalized):
        raise RuntimeError(f"CUDA GPU visibility disagrees with the allocation: expected {normalized}, got {actual}")
    (experiment / "gpu_mapping.json").write_text(
        json.dumps(
            {
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "cuda_device_order": os.environ["CUDA_DEVICE_ORDER"],
                "uuids": actual,
                "names": [device.name for device in properties],
            },
            indent=2,
        )
        + "\n"
    )


def ray_argv(port, temp_dir, gpus):
    flags = {
        "port": port,
        "dashboard-port": port + 1,
        "node-manager-port": port + 2,
        "object-manager-port": port + 3,
        "runtime-env-agent-port": port + 4,
        "dashboard-agent-grpc-port": port + 5,
        "dashboard-agent-listen-port": port + 6,
        "metrics-export-port": port + 7,
        "ray-client-server-port": port + 8,
        "min-worker-port": port + 20,
        "max-worker-port": port + 219,
        "node-ip-address": "127.0.0.1",
        "num-gpus": gpus,
        "num-cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", "24")),
        "temp-dir": temp_dir,
    }
    return ["ray", "start", "--head", "--disable-usage-stats", *[f"--{key}={value}" for key, value in flags.items()]]


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def wait_healthy(process, base_url):
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Miles gateway exited during startup; inspect gateway.log")
        try:
            with urllib.request.urlopen(base_url + "/api/v1/healthz", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError("Miles gateway did not become healthy within 30 minutes")


def run_clients(experiment, plan, *, base_url, model_path, gateway):
    processes, logs = [], []
    try:
        for index, teacher in enumerate(plan["teachers"]):
            directory = experiment / "teacher" / teacher["source_label"]
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / "train.log").open("a")
            logs.append(log)
            command = [
                sys.executable,
                "-m",
                "lora_teachers.client",
                "--experiment",
                str(experiment),
                "--index",
                str(index),
                "--model-path",
                str(model_path),
                "--base-url",
                base_url,
            ]
            # The SDK and custom-loss torch operations are CPU clients; only Ray
            # workers own the allocated GPUs.
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
                    start_new_session=True,
                )
            )
        deadline = time.monotonic() + 1800
        while not all((experiment / "teacher" / t["source_label"] / "ready").is_file() for t in plan["teachers"]):
            if gateway.poll() is not None or any(p.poll() is not None for p in processes):
                raise RuntimeError("A client or gateway failed before all adapters were ready")
            if time.monotonic() > deadline:
                raise TimeoutError("Not all teacher clients initialized within 30 minutes")
            time.sleep(2)
        (experiment / "clients.start").touch()
        print(f"All {len(plan['teachers'])} adapters initialized; starting concurrent teacher RL", flush=True)
        while any(process.poll() is None for process in processes):
            if gateway.poll() is not None:
                raise RuntimeError("Miles gateway died; inspect gateway.log")
            time.sleep(5)
        failed = [
            teacher["environment"]
            for teacher, process in zip(plan["teachers"], processes, strict=True)
            if process.returncode != 0
        ]
        if failed:
            raise RuntimeError(f"Teacher clients failed: {failed}")
        results = [
            json.loads((experiment / "teacher" / t["source_label"] / "completed.json").read_text())
            for t in plan["teachers"]
        ]
        results.extend(preserved_results(plan))
        if len({result["environment"] for result in results}) != len(results):
            raise ValueError("Recovery results contain duplicate environments")
        results.sort(key=lambda result: result["environment"])
        (experiment / "completed.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        for process in processes:
            stop_process(process)
        for log in logs:
            log.close()


def run(experiment, model_path, *, clients=run_clients):
    from miles.utils.external_utils.model_args_utils import load_model_args

    verify_inputs(experiment)
    plan = json.loads((experiment / "plan.json").read_text())
    verify_gpu_visibility(experiment, expected_count=plan["train_gpus"] + plan["sampling_gpus"])
    slot = int(os.environ["LORA_PORT_SLOT"])
    ray_port, service_port = 4000 + 220 * slot, 6000 + 3000 * slot
    temp_dir = Path("/tmp") / f"miles-lora-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(temp_dir).free < 8 * 2**30:
        raise RuntimeError("Need 8 GiB of node-local space for temporary adapter snapshots and Ray")
    checkpoint_root = temp_dir / "checkpoints"
    generation = uuid.uuid4().hex
    if plan.get("elastic_generation"):
        checkpoint_root = experiment / "shared-checkpoints" / generation
    os.environ.update(
        {
            "RAY_ADDRESS": f"127.0.0.1:{ray_port}",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "TRITON_CACHE_DIR": str(temp_dir / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(temp_dir / "inductor"),
            "LORA_CHECKPOINT_ROOT": str(checkpoint_root),
            # Each node-wide slot owns this otherwise unused part of its block.
            # Set it before Ray starts so managers and their children inherit it.
            "MILES_DYNAMIC_PORT_START": str(service_port + 1500),
            "MILES_DYNAMIC_PORT_END": str(service_port + 1999),
        }
    )
    subprocess.run(ray_argv(ray_port, temp_dir, plan["train_gpus"] + plan["sampling_gpus"]), check=True)
    gateway = None
    try:
        model_args = shlex.split(load_model_args(plan["model_args_type"]))
        # R1-Distill has the Qwen2.5-1.5B architecture, with an untied output head.
        model_args.append("--untie-embeddings-and-output-weights")
        args = gateway_argv(
            plan,
            model_path=model_path,
            checkpoint_root=checkpoint_root,
            port=service_port,
            router_port=service_port + 1000,
        )
        command = [sys.executable, str(experiment / "source/miles/serve_tinker.py"), *model_args, *args]
        (experiment / "gateway_argv.json").write_text(json.dumps(command, indent=2) + "\n")
        with (experiment / "gateway.log").open("a") as log:
            gateway = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            base_url = f"http://127.0.0.1:{service_port}"
            wait_healthy(gateway, base_url)
            if plan.get("elastic_generation"):
                from async_lora.flex_protocol import http, local_ip, router_workers

                from async_branching.io import write_json

                router_url = f"http://{local_ip()}:{service_port + 1000}"
                workers = router_workers(router_url)
                if len(workers) != plan["sampling_gpus"]:
                    raise RuntimeError("Unexpected core generator inventory before enabling flex GPUs")
                engine_config = http(workers[0]["url"] + "/get_server_info")
                write_json(
                    experiment / "flex/core.json",
                    {
                        "generation": generation,
                        "core_job_id": os.environ["SLURM_JOB_ID"],
                        "router_url": router_url,
                        "core_workers": workers,
                        "engine_config": engine_config,
                        "checkpoint_root": str(checkpoint_root),
                    },
                )
            clients(experiment, plan, base_url=base_url, model_path=model_path, gateway=gateway)
    finally:
        if gateway is not None:
            stop_process(gateway)
        # The container has a private PID namespace, so Ray teardown cannot
        # reach another allocation's processes on the same node.
        subprocess.run(["ray", "stop", "--force"], check=False, timeout=60)
