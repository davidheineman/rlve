"""Own an inference-only SGLang process and evaluate immutable group adapters."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def http(url, payload=None, timeout=180):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def start_server(model, native, root, *, port, max_rank, gpu=0):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--base-gpu-id",
        str(gpu),
        "--tp-size",
        "1",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--enable-lora",
        "--lora-paths",
        f"native={native}",
        "--max-lora-rank",
        str(max_rank),
        "--max-loras-per-batch",
        "1",
        "--max-loaded-loras",
        "2",
        "--lora-backend",
        "triton",
        "--mem-fraction-static",
        ".65",
        "--max-running-requests",
        "32",
        "--context-length",
        "8192",
        "--disable-cuda-graph",
        "--attention-backend",
        "flashinfer",
        "--random-seed",
        "1234",
    ]
    (root / "server_argv.json").write_text(json.dumps(command, indent=2) + "\n")
    log = (root / "server.log").open("a")
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log.close()
            raise RuntimeError(f"SGLang exited during startup: {root / 'server.log'}")
        try:
            http(url + "/get_server_info", timeout=2)
            return process, log, url
        except (OSError, ValueError):
            time.sleep(2)
    stop_server(process, log)
    raise TimeoutError("Inference server did not start within 15 minutes")


def stop_server(process, log):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    log.close()


async def request(url, payload):
    return await asyncio.to_thread(http, url, payload, 600)


def load_adapter(url, name, path):
    result = http(
        url + "/load_lora_adapter",
        {"lora_name": name, "lora_path": str(path)},
        timeout=300,
    )
    if not result.get("success"):
        raise RuntimeError(f"Adapter load failed: {result}")


def unload_adapter(url, name):
    result = http(url + "/unload_lora_adapter", {"lora_name": name}, timeout=180)
    if not result.get("success"):
        raise RuntimeError(f"Adapter unload failed: {result}")
