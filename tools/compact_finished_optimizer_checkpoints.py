"""Retain byte-identical model/RNG checkpoint payloads; retire completed Adam state.

Only the explicit inventory of inactive, completed modular experiments is in scope.
Every retained DCP record is copied verbatim and rehashed with unchanged tensor
shape/dtype metadata. No weights, evaluations, rollouts, or logs are discarded.
"""

import copy
import fcntl
import hashlib
import json
import os
import pickle
import shutil
import time
from dataclasses import replace
from itertools import groupby
from pathlib import Path

import torch.distributed.checkpoint as dcp
from convert_torch_dist_to_hf import WrappedStorageReader

SCOPE = Path("/juice5b/scr5b/dhei/modular-mopd-rl")


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def identities(directory):
    result = {}
    for p in directory.iterdir():
        if p.is_symlink() or not p.is_file():
            raise ValueError(f"Unexpected checkpoint entry: {p}")
        st = p.stat()
        result[p.name] = [st.st_ino, st.st_size, st.st_mtime_ns]
    return result


def optimizer_key(key):
    return (
        key == "optimizer"
        or key.startswith("optimizer.")
        or key.startswith("optimizer/")
    )


def stream_hash(stream, offset, length):
    digest = hashlib.sha256()
    stream.seek(offset)
    while length:
        data = stream.read(min(length, 8 * 2**20))
        if not data:
            raise ValueError("Truncated checkpoint payload")
        digest.update(data)
        length -= len(data)
    return digest.hexdigest()


def retain_payloads(source, target):
    """Repack independent serialized DCP records without changing their bytes."""
    metadata = WrappedStorageReader(str(source)).read_metadata()
    kept = {k for k in metadata.state_dict_metadata if not optimizer_key(k)}
    removed = sorted(set(metadata.state_dict_metadata) - kept)
    if not removed or not any(
        isinstance(metadata.state_dict_metadata[k], dcp.metadata.TensorStorageMetadata)
        for k in kept
    ):
        raise ValueError(
            "Need both optimizer state to retire and model tensors to retain"
        )
    target.mkdir()
    records = {}
    storage = {}
    entries = sorted(
        (
            (index, info)
            for index, info in metadata.storage_data.items()
            if index.fqn in kept
        ),
        key=lambda pair: (pair[1].relative_path, pair[1].offset),
    )
    for filename, group in groupby(entries, key=lambda pair: pair[1].relative_path):
        relative = Path(filename)
        if relative.name != str(relative) or relative.suffix != ".distcp":
            raise ValueError("Unexpected DCP storage path")
        old, new = source / relative, target / relative
        file_records = []
        with old.open("rb") as reader, new.open("wb") as writer:
            for index, info in group:
                digest = hashlib.sha256()
                reader.seek(info.offset)
                offset = writer.tell()
                remaining = info.length
                while remaining:
                    data = reader.read(min(remaining, 8 * 2**20))
                    if not data:
                        raise ValueError("Truncated original checkpoint")
                    writer.write(data)
                    digest.update(data)
                    remaining -= len(data)
                storage[index] = replace(info, offset=offset)
                record = {
                    "key": index.fqn,
                    "file": str(relative),
                    "offset": offset,
                    "length": info.length,
                    "sha256": digest.hexdigest(),
                }
                records[str(index)] = record
                file_records.append(record)
            writer.flush()
            os.fsync(writer.fileno())
        with new.open("rb") as reader:
            for record in file_records:
                if (
                    stream_hash(reader, record["offset"], record["length"])
                    != record["sha256"]
                ):
                    raise ValueError("Retained checkpoint payload changed")
    updated = copy.copy(metadata)
    updated.state_dict_metadata = {
        k: v for k, v in metadata.state_dict_metadata.items() if k in kept
    }
    updated.storage_data = storage
    if isinstance(metadata.planner_data, dict):
        updated.planner_data = {
            k: v for k, v in metadata.planner_data.items() if k in kept
        }
    if hasattr(metadata, "mcore_data"):
        updated.mcore_data = {
            k: v for k, v in metadata.mcore_data.items() if not optimizer_key(k)
        }
    if hasattr(metadata, "all_local_plans"):
        # Megatron's save-plan cache includes optimizer buffers and is not part
        # of loading model state. Its loader explicitly accepts a missing cache.
        updated.all_local_plans = None
    with (target / ".metadata").open("wb") as stream:
        pickle.dump(updated, stream)
    for p in source.iterdir():
        if p.name != ".metadata" and p.suffix != ".distcp":
            shutil.copy2(p, target / p.name)
    verify_metadata(metadata, WrappedStorageReader(str(target)).read_metadata())
    return {
        "removed_optimizer_keys": removed,
        "retained_payloads": records,
        "model_tensor_count": sum(
            isinstance(v, dcp.metadata.TensorStorageMetadata)
            for v in updated.state_dict_metadata.values()
        ),
        "model_tensor_parity": True,
        "parity_method": "Byte-identical serialized DCP records and unchanged tensor dtype/shape metadata; no tensor arithmetic or conversion",
        "retained_rng_and_extra_state": True,
    }


def verify_metadata(original, retained):
    expected = {
        k: v for k, v in original.state_dict_metadata.items() if not optimizer_key(k)
    }
    if retained.state_dict_metadata != expected:
        raise ValueError("Retained tensor shape/dtype or RNG metadata changed")
    indices = {k for k in original.storage_data if k.fqn in expected}
    if retained.storage_data.keys() != indices:
        raise ValueError("Retained storage records differ")
    for index in indices:
        a, b = original.storage_data[index], retained.storage_data[index]
        if a.length != b.length or a.relative_path != b.relative_path:
            raise ValueError("Retained storage length or file changed")


def verify_existing(source, target):
    original = WrappedStorageReader(str(source)).read_metadata()
    retained = WrappedStorageReader(str(target)).read_metadata()
    verify_metadata(original, retained)
    records = {}
    entries = sorted(
        retained.storage_data.items(),
        key=lambda pair: (pair[1].relative_path, pair[1].offset),
    )
    for name, group in groupby(entries, key=lambda pair: pair[1].relative_path):
        if Path(name).name != name:
            raise ValueError("Invalid DCP storage path")
        with (source / name).open("rb") as a, (target / name).open("rb") as b:
            for index, info in group:
                old = original.storage_data[index]
                expected = stream_hash(a, old.offset, old.length)
                if stream_hash(b, info.offset, info.length) != expected:
                    raise ValueError("Retained payload changed")
                records[str(index)] = {
                    "key": index.fqn,
                    "file": name,
                    "offset": info.offset,
                    "length": info.length,
                    "sha256": expected,
                }
    for p in source.iterdir():
        if p.name != ".metadata" and p.suffix != ".distcp":
            if (
                hashlib.sha256(p.read_bytes()).digest()
                != hashlib.sha256((target / p.name).read_bytes()).digest()
            ):
                raise ValueError("Common checkpoint metadata changed")
    return {
        "removed_optimizer_keys": sorted(
            k for k in original.state_dict_metadata if optimizer_key(k)
        ),
        "retained_payloads": records,
        "model_tensor_count": sum(
            isinstance(v, dcp.metadata.TensorStorageMetadata)
            for v in retained.state_dict_metadata.values()
        ),
        "model_tensor_parity": True,
        "retained_rng_and_extra_state": True,
        "parity_method": "Byte-identical serialized DCP records and unchanged tensor dtype/shape metadata; no tensor arithmetic or conversion",
    }


def compact(
    item,
    audit,
    *,
    scope=SCOPE,
    resume_verified_copy=False,
    completion_file="completed.json",
):
    policy, native = Path(item["policy"]), Path(item["native"])
    if (
        not policy.resolve().is_relative_to(scope.resolve())
        or policy.is_symlink()
        or native.resolve().parent != policy.resolve() / "ckpt"
        or native.is_symlink()
        or not native.name.startswith("iter_")
    ):
        raise ValueError("Checkpoint is outside the approved completed-run scope")
    if completion_file not in ("completed.json", "ckpt/rlve_checkpoint_state.json"):
        raise ValueError("Unexpected completed-run marker")
    complete = json.loads((policy / completion_file).read_text())
    state = json.loads((policy / "ckpt/rlve_checkpoint_state.json").read_text())
    if (
        not complete.get("completed")
        or not state.get("completed")
        or native.name != f"iter_{state['rollout_id']:07d}"
    ):
        raise ValueError("Checkpoint is not a completed final checkpoint")
    target = native.with_name(native.name + ".model-only-incomplete")
    retired = native.with_name(native.name + ".optimizer-retired")
    if (target.exists() and not resume_verified_copy) or retired.exists():
        raise FileExistsError("Inspect an interrupted compaction before retrying")
    with (policy / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = identities(native)
        original_bytes = sum(v[1] for v in before.values())
        metadata = WrappedStorageReader(str(native)).read_metadata()
        needed = sum(
            v.length
            for k, v in metadata.storage_data.items()
            if not optimizer_key(k.fqn)
        )
        if shutil.disk_usage(native).free < needed + 2 * 2**30:
            raise RuntimeError(
                "Insufficient space to verify the retained checkpoint before cleanup"
            )
        receipt = {
            "policy": str(policy),
            "native": str(native),
            "started_at": time.time(),
            "original_files": before,
            "original_bytes": original_bytes,
            "status": "copying",
            "exact_optimizer_resume_retained": False,
        }
        key = hashlib.sha256(str(native).encode()).hexdigest()[:16]
        record_path = audit / f"{key}.json"
        atomic(record_path, receipt)
        proof = (
            verify_existing(native, target)
            if target.exists()
            else retain_payloads(native, target)
        )
        if identities(native) != before:
            raise ValueError("Original checkpoint changed during verification")
        receipt.update(
            proof,
            status="verified",
            retained_bytes=sum(v[1] for v in identities(target).values()),
        )
        atomic(record_path, receipt)
        # The unchanged model/RNG payloads and common metadata are committed
        # before retiring only the old optimizer-bearing directory.
        native.rename(retired)
        target.rename(native)
        receipt.update(status="model_committed", committed_at=time.time())
        atomic(record_path, receipt)
        shutil.rmtree(retired)
        receipt.update(
            status="complete",
            finished_at=time.time(),
            bytes_reclaimed=original_bytes - receipt["retained_bytes"],
        )
        atomic(record_path, receipt)
        atomic(
            policy / "optimizer_state_retired.json",
            {
                "status": "model_only",
                "checkpoint": str(native),
                "audit": str(record_path),
                "model_tensors_identical": True,
                "rng_and_extra_state_retained": True,
                "exact_adam_resume_available": False,
            },
        )
        return receipt
