"""Exercise the pinned router's hot admission and transport-failure retry on CPU."""

import importlib.util
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from async_lora.flex_protocol import http, router_workers, unregister


class FakeEngine(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def reply(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(
            {
                "model_path": "/frozen-base",
                "is_generation": True,
                "served_model_name": "frozen-base",
                "context_len": 4096,
            }
        )

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(request)
        if self.server.fail:
            self.close_connection = True
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
        else:
            self.reply({"text": "ok", "meta_info": {"finish_reason": {"type": "length", "length": 1}}})


@unittest.skipUnless(importlib.util.find_spec("sglang_router"), "requires the pinned Miles image")
class FlexRouterIntegrationTest(unittest.TestCase):
    def test_hot_add_and_lost_worker_retry_preserve_immutable_adapter_payload(self):
        engines = []
        for _ in range(2):
            server = ThreadingHTTPServer(("127.0.0.1", 0), FakeEngine)
            server.requests, server.fail = [], False
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            engines.append(server)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        router = f"http://127.0.0.1:{port}"
        with tempfile.TemporaryFile(mode="w+") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "sglang_router.launch_router",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--policy",
                    "round_robin",
                    "--log-level",
                    "warn",
                    "--request-timeout-secs",
                    "10",
                    "--worker-startup-check-interval",
                    "1",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 60
                while True:
                    try:
                        router_workers(router)
                        break
                    except OSError:
                        if process.poll() is not None or time.monotonic() > deadline:
                            log.seek(0)
                            self.fail(log.read())
                        time.sleep(0.2)
                urls = [f"http://127.0.0.1:{server.server_port}" for server in engines]
                for url in urls:
                    http(router + "/workers", {"url": url, "worker_type": "regular"})
                while len(router_workers(router)) != 2:
                    if time.monotonic() > deadline:
                        log.seek(0)
                        self.fail(log.read())
                    time.sleep(0.2)
                request = {
                    "input_ids": [1, 2],
                    "lora_path": "immutable-v123",
                    "lora_backfill_paths": {"immutable-v123": "/shared/exact-v123"},
                    "sampling_params": {"max_new_tokens": 1},
                }
                self.assertEqual(http(router + "/generate", request)["text"], "ok")
                engines[1].fail = True
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(lambda _: http(router + "/generate", request, timeout=30), range(12)))
                self.assertTrue(all(result["text"] == "ok" for result in results))
                self.assertGreater(len(engines[1].requests), 0)
                for server in engines:
                    for payload in server.requests:
                        if payload.get("lora_path") == request["lora_path"]:
                            self.assertEqual({key: payload[key] for key in request}, request)
                unregister(router, urls[1])
                self.assertEqual([worker["url"] for worker in router_workers(router)], [urls[0]])
                self.assertEqual(http(router + "/generate", request)["text"], "ok")
            finally:
                process.terminate()
                process.wait(timeout=20)


if __name__ == "__main__":
    unittest.main()
