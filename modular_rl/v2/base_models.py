"""Pinned identities for the requested pretrained Qwen3 Base experiments."""

import argparse
import hashlib
import json
from pathlib import Path

BASE_MODELS = {
    "Qwen3-0.6B-Base": {
        "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "config_sha256": "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59",
        "weights_sha256": {"model.safetensors": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"},
    },
    "Qwen3-1.7B-Base": {
        "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        "config_sha256": "1bb33a92c3548fbc68b889b490e810440435253598835bd71dff0396060c12db",
        "weights_sha256": {"model.safetensors": "6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841"},
    },
}


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_hf(directory, *, full_weights=False):
    directory = Path(directory).resolve()
    if directory.name not in BASE_MODELS:
        raise ValueError("Require exactly Qwen3-0.6B-Base or Qwen3-1.7B-Base; reasoning checkpoints are excluded")
    expected = BASE_MODELS[directory.name]
    identity = json.loads((directory / "pretrained_model_identity.json").read_text())
    if (identity.get("repo_id") != "Qwen/" + directory.name
            or identity.get("checkpoint_family") != "pretrained-base"
            or any(identity.get(key) != value for key, value in expected.items())):
        raise ValueError("Pretrained Base repository/revision/weight identity mismatch")
    if sha256(directory / "config.json") != expected["config_sha256"]:
        raise ValueError("Configuration differs from the pinned pretrained Base repository")
    for name, checksum in expected["weights_sha256"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Missing pretrained Base weights")
        if full_weights and sha256(path) != checksum:
            raise ValueError("Weights differ from the verified pretrained Base download")
    return identity


def verify_initialization(hf_directory, reference_directory):
    identity = verify_hf(hf_directory)
    reference_directory = Path(reference_directory).resolve()
    if reference_directory.name != Path(hf_directory).name + "_torch_dist":
        raise ValueError("Reference initialization must be the matching pretrained Base conversion")
    reference = json.loads((reference_directory / "pretrained_model_identity.json").read_text())
    if (any(reference.get(key) != identity[key] for key in ("repo_id", "revision", "weights_sha256"))
            or reference.get("optimizer_updates") != 0
            or (reference_directory / "latest_checkpointed_iteration.txt").read_text().strip() != "release"):
        raise ValueError("Base initialization must be an untrained conversion with zero optimizer updates")
    return identity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--full-weights", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_hf(args.directory, full_weights=args.full_weights)), flush=True)
