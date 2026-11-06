"""GPU Skywork scoring service, adapted from tinker-mopd/reward_server.py."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import secrets
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path

from .prepare import REWARD_MODEL, REWARD_REVISION, digest, write_json


class Skywork:
    def __init__(self, device='cuda:0'):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("Skywork requires its allocated GPU")
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            REWARD_MODEL, revision=REWARD_REVISION
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                REWARD_MODEL,
                revision=REWARD_REVISION,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                num_labels=1,
            )
            .to(self.device)
            .eval()
        )
        self.cache = {}

    def score(self, conversations):
        results = []
        with self.torch.cuda.device(self.device), self.torch.inference_mode():
            for messages in conversations:
                if any(m["role"] == "system" for m in messages):
                    raise ValueError(
                        "Skywork scores conversations without system prompts"
                    )
                key = digest(messages)
                if key not in self.cache:
                    ids = self.tokenizer.apply_chat_template(messages, tokenize=True)
                    if len(ids) > 16384:
                        raise ValueError(
                            f"Reward input too long: {len(ids)} > 16384; no truncation"
                        )
                    inputs = self.torch.tensor([ids], device=self.device)
                    score = float(
                        self.model(input_ids=inputs, use_cache=False)
                        .logits[0, 0]
                        .float()
                    )
                    if not math.isfinite(score):
                        raise RuntimeError("Nonfinite Skywork reward")
                    self.cache[key] = score
                results.append(self.cache[key])
        return results


def serve(endpoint_file, replicas=1):
    if replicas not in (1, 2):
        raise ValueError('Use one or two independent Skywork replicas')
    models = [Skywork(f'cuda:{i}') for i in range(replicas)]
    model = models[0]
    prompt = [
        {"role": "user", "content": "What is 2 + 2? Give a clear and correct answer."}
    ]
    good = prompt + [{"role": "assistant", "content": "2 + 2 = 4."}]
    bad = prompt + [{"role": "assistant", "content": "2 + 2 = 937. I do not know."}]
    controls = []
    for replica in models:
        positive, negative = replica.score([good, bad])
        if positive <= negative:
            raise RuntimeError(f'Skywork known-answer control failed: {positive} <= {negative}')
        controls.append({'device': replica.device, 'positive_score': positive, 'negative_score': negative})
    positive, negative = controls[0]['positive_score'], controls[0]['negative_score']
    health = {
        "model": REWARD_MODEL,
        "revision": REWARD_REVISION,
        "device": "cuda",
        "gpu": model.torch.cuda.get_device_name(),
        "positive_score": positive,
        "negative_score": negative,
        'replicas': replicas,
        'replica_controls': controls,
    }
    available = queue.Queue()
    for replica in models:
        available.put(replica)
    write_json(endpoint_file.parent / "reward-model-checks.json", health)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, payload, status=200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            if self.headers.get("Authorization") == "Bearer " + token:
                return True
            self.reply({"error": "Unauthorized"}, 401)
            return False

        def do_GET(self):
            if self.authorized():
                self.reply(health)

        def do_POST(self):
            if not self.authorized():
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16 * 1024**2:
                    raise ValueError("Invalid scoring request size")
                payload = json.loads(self.rfile.read(size))
                replica = available.get()
                try:
                    self.reply({'scores': replica.score(payload['conversations'])})
                finally:
                    available.put(replica)
            except Exception as exc:  # noqa: BLE001 - report scoring infrastructure failures to caller
                traceback.print_exc()
                self.reply({"error": str(exc), "type": type(exc).__name__}, 500)

    server_class = ThreadingHTTPServer if replicas > 1 else HTTPServer
    server = server_class(("127.0.0.1", 0), Handler)
    temporary = endpoint_file.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(
            {"url": f"http://127.0.0.1:{server.server_port}", "token": token, **health},
            stream,
        )
    temporary.replace(endpoint_file)
    print(json.dumps({"ready": health, "port": server.server_port}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-file", type=Path, required=True)
    parser.add_argument('--replicas', type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    serve(args.endpoint_file, args.replicas)
