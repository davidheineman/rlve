"""Durable checkpoint bundles and bounded immutable sampling exports."""

import fcntl
import pickle
import shutil
import uuid
from pathlib import Path

from lora_teachers.client import persist_final_adapter, snapshot_directory

from async_branching.io import read_json, write_json
from async_lora.snapshot_cleanup import retire_snapshot


def uri_directory(root, uri):
    parts = uri.removeprefix("tinker://").split("/")
    if not uri.startswith("tinker://") or len(parts) != 3 or parts[1] not in {"weights", "sampler_weights"}:
        raise ValueError(f"Invalid checkpoint URI: {uri}")
    if any(p in {"", ".", ".."} for p in parts):
        raise ValueError("Checkpoint URI contains an invalid segment")
    return Path(root).joinpath(*parts)


def restore_reference(root, durable, *, sampling=False):
    """Copy into the gateway's confined root; never link outside its ownership boundary."""
    durable = Path(durable)
    reference = read_json(durable.parent / "reference.json")
    uri = reference["sampler_uri" if sampling else "state_uri"]
    if not uri:
        raise ValueError("This bundle has no training checkpoint")
    destination = uri_directory(root, uri)
    lock = Path(root) / ".restore.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if not destination.is_dir():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".restore")
            shutil.rmtree(temporary, ignore_errors=True)
            shutil.copytree(durable, temporary)
            temporary.replace(destination)
    return uri


async def export(training, service, root, name, previous=None, *, retain_previous=False):
    saved = await (await training.save_weights_for_sampler_async(name=name))
    source = snapshot_directory(root, saved.path)
    if not (source / "META.json").is_file() or not (source / "adapter_model.safetensors").is_file():
        raise ValueError("Incomplete immutable adapter export")
    sampler = await service.create_sampling_client_async(model_path=saved.path)
    if previous and not retain_previous and previous != saved.path:
        retire_snapshot(root, snapshot_directory(root, previous))
    return sampler, saved.path


async def save_bundle(
    training, sampler_uri, root, destination, plan, client_state=None, *, optimizer=True, replay=None
):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Checkpoint bundles are immutable: {destination}")
    temporary = destination.with_name(destination.name + ".incomplete")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    state_uri = None
    if optimizer:
        saved = await (await training.save_state_async(name=f"state-{uuid.uuid4().hex}"))
        state_uri = saved.path
        shutil.copytree(uri_directory(root, state_uri), temporary / "state")
    persist_final_adapter(snapshot_directory(root, sampler_uri), temporary / "adapter", plan)
    write_json(temporary / "reference.json", {"state_uri": state_uri, "sampler_uri": sampler_uri})
    if client_state is not None:
        with (temporary / "client-state.pkl").open("wb") as stream:
            pickle.dump(client_state, stream)
    if replay is not None:
        replay.save(temporary / "opd-replay.json.gz")
    temporary.replace(destination)
    # Training state is durable. Rehydrate only when another slot needs it.
    if state_uri:
        retire_snapshot(root, uri_directory(root, state_uri))
    result = {"hf": str(destination / "adapter"), "resume": str(destination / "state") if optimizer else None}
    if replay is not None:
        result["replay"] = str(destination / "opd-replay.json.gz")
    return result


def client_state(bundle):
    path = Path(bundle) / "client-state.pkl"
    with path.open("rb") as stream:
        return pickle.load(stream)


def adapter_identity(parent, child):
    """Byte identity proves the full branch delta matches before its first update."""
    from lora_teachers.plan import sha256

    name = "adapter_model.safetensors"
    expected, actual = sha256(Path(parent) / name), sha256(Path(child) / name)
    if expected != actual:
        # Serialization can differ while tensor values agree; compare tensors.
        import torch
        from safetensors.torch import load_file

        a, b = load_file(str(Path(parent) / name)), load_file(str(Path(child) / name))
        if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
            raise RuntimeError("Branch clone does not match its selected student adapter")
    return {"identical_adapter_tensors": True, "parent_sha256": expected, "branch_sha256": actual}
