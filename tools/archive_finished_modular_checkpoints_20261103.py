"""Archive an explicit inventory of completed modular checkpoints, then verify and evict tensors."""

import argparse
import fcntl
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCOPE = Path("/juice5b/scr5b/dhei/modular-mopd-rl/v1")
OWNER = "davidheineman"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def identity(path):
    stat = path.stat()
    return [stat.st_ino, stat.st_size, stat.st_mtime_ns]


def inspect_checkpoint(item, *, scope=SCOPE):
    policy, checkpoint = Path(item["source"]), Path(item["checkpoint"])
    if (
        policy.is_symlink()
        or checkpoint.is_symlink()
        or not policy.resolve().is_relative_to(scope.resolve())
        or checkpoint.resolve().parent != policy.resolve() / "ckpt"
        or not checkpoint.name.startswith("iter_")
    ):
        raise ValueError("Checkpoint escapes the selected archive scope")
    completed = json.loads((policy / "completed.json").read_text())
    state = json.loads((policy / "ckpt/rlve_checkpoint_state.json").read_text())
    latest = (policy / "ckpt/latest_checkpointed_iteration.txt").read_text().strip()
    if not completed.get("completed") or not state.get("completed"):
        raise ValueError("Only completed checkpoints may be archived")
    if checkpoint.name != f"iter_{state['rollout_id']:07d}" or latest != str(
        state["rollout_id"]
    ):
        raise ValueError("Checkpoint is not the completed latest iteration")
    files = list(checkpoint.iterdir())
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError("Unexpected linked or nested checkpoint payload")
    files += [
        policy / "ckpt/latest_checkpointed_iteration.txt",
        policy / "ckpt/rlve_checkpoint_state.json",
    ]
    return {str(p.relative_to(policy)): p for p in files}


def ensure_inactive(scope=SCOPE):
    """Protect queued jobs, current process arguments, mappings and open files."""
    needle = str(scope)
    queue = subprocess.check_output(
        ["squeue", "-h", "-u", os.environ.get("USER", "dhei"), "-o", "%i|%Z|%o"],
        text=True,
        timeout=30,
    )
    if needle in queue:
        raise RuntimeError("A queued or running job references the archive scope")
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal() or int(directory.name) == os.getpid():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            name = (directory / "comm").read_text().strip()
            if name in {"du", "rg", "find"}:
                continue
            command = (
                (directory / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
            maps = (directory / "maps").read_text(errors="replace")
            if needle in command or needle in maps:
                raise RuntimeError(
                    f"Process {directory.name} references the archive scope"
                )
            for fd in (directory / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except (FileNotFoundError, PermissionError):
                    continue
                if needle in target:
                    raise RuntimeError(
                        f"Process {directory.name} holds an archive file open"
                    )
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def verify_remote(api, repo, revision, records):
    info = api.model_info(repo, revision=revision, files_metadata=True)
    if not info.private:
        raise ValueError("Archive repository must remain private")
    siblings = {p.rfilename: p for p in info.siblings}
    for name, expected in records.items():
        remote = siblings.get(name)
        if remote is None or remote.size != expected["size"]:
            raise ValueError(f"Remote size mismatch: {name}")
        if remote.lfs is not None:
            if remote.lfs.sha256 != expected["sha256"]:
                raise ValueError(f"Remote SHA-256 mismatch: {name}")
        elif remote.blob_id != expected["git_blob_sha1"]:
            raise ValueError(f"Remote Git blob mismatch: {name}")


def evict_tensors(checkpoint, receipt):
    if receipt.get("status") not in {"verified", "evicting"}:
        raise ValueError("Cannot evict an unverified archive")
    expected = receipt["files"]
    for name, row in expected.items():
        path = Path(row["local_path"])
        if (
            receipt["status"] == "evicting"
            and path.suffix == ".distcp"
            and not path.exists()
        ):
            continue
        if path.is_symlink() or identity(path) != row["identity"]:
            raise ValueError("Local checkpoint changed after verification")
    tensors = [
        Path(row["local_path"])
        for row in expected.values()
        if Path(row["local_path"]).suffix == ".distcp"
    ]
    if not tensors or any(p.parent.resolve() != checkpoint.resolve() for p in tensors):
        raise ValueError("Tensor path escapes the selected checkpoint")
    known = {
        Path(row["local_path"]).name
        for row in expected.values()
        if Path(row["local_path"]).parent == checkpoint
    }
    actual = {p.name for p in checkpoint.iterdir()}
    if not actual <= known or (receipt["status"] == "verified" and actual != known):
        raise ValueError("Checkpoint acquired new files during archival")
    reclaimed = sum(
        p.stat().st_blocks * 512
        for p in tensors
        if p.exists() and p.stat().st_nlink == 1
    )
    for path in tensors:
        path.unlink(missing_ok=receipt["status"] == "evicting")
    return reclaimed


def finish_eviction(policy, checkpoint, receipt, receipt_path):
    if receipt["status"] == "verified":
        for row in receipt["files"].values():
            path = Path(row["local_path"])
            if path.is_symlink() or identity(path) != row["identity"]:
                raise ValueError("Local checkpoint changed before eviction")
        receipt["allocated_bytes_to_evict"] = sum(
            Path(row["local_path"]).stat().st_blocks * 512
            for row in receipt["files"].values()
            if Path(row["local_path"]).suffix == ".distcp"
            and Path(row["local_path"]).stat().st_nlink == 1
        )
        receipt["status"] = "evicting"
        write(receipt_path, receipt)
    write(
        policy / "checkpoint_hf_archive.json",
        {
            "repo_id": receipt["repo_id"],
            "revision": receipt["revision"],
            "collection": receipt["collection"],
            "receipt": str(receipt_path),
            "restore": {"allow_patterns": ["ckpt/**"], "local_dir": str(policy)},
            "full_optimizer_and_rng_retained_remotely": True,
        },
    )
    evict_tensors(checkpoint, receipt)
    receipt["allocated_bytes_evicted"] = receipt["allocated_bytes_to_evict"]
    receipt.update(status="evicted", finished_at=time.time())
    write(receipt_path, receipt)
    return receipt


def archive_one(api, item, output, collection):
    from huggingface_hub import CommitOperationAdd

    policy, checkpoint = Path(item["source"]), Path(item["checkpoint"])
    key = hashlib.sha256(str(checkpoint).encode()).hexdigest()[:12]
    slug = re.sub(r"[^a-z0-9]+", "-", str(policy.relative_to(SCOPE)).lower()).strip("-")
    repo = f"{OWNER}/modular-finished-{slug[-48:]}-{key}"
    receipt_path = output / "receipts" / f"{key}.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if receipt.get("status") == "evicted":
        verify_remote(api, repo, receipt["revision"], receipt["files"])
        return receipt
    with (policy / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_inactive()
        if receipt.get("status") in {"verified", "evicting"}:
            verify_remote(api, repo, receipt["revision"], receipt["files"])
            if repo not in {i.item_id for i in api.get_collection(collection).items}:
                raise ValueError("Verified archive is missing from its collection")
            return finish_eviction(policy, checkpoint, receipt, receipt_path)
        files = inspect_checkpoint(item)
        records, operations = {}, []
        print(f"Hashing {repo}", flush=True)
        for name, path in files.items():
            before = identity(path)
            operation = CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
            if identity(path) != before:
                raise ValueError("Checkpoint changed while hashing")
            blob = None
            if before[1] < 16 * 2**20:
                digest = hashlib.sha1(f"blob {before[1]}\0".encode())
                digest.update(path.read_bytes())
                blob = digest.hexdigest()
            records[name] = {
                "local_path": str(path),
                "identity": before,
                "size": before[1],
                "sha256": operation.upload_info.sha256.hex(),
                "git_blob_sha1": blob,
            }
            operations.append(operation)
        receipt = {
            "status": "uploading",
            "source": str(policy),
            "checkpoint": str(checkpoint),
            "repo_id": repo,
            "collection": collection,
            "files": records,
            "started_at": time.time(),
        }
        write(receipt_path, receipt)
        manifest = json.dumps(
            {
                "source": str(policy),
                "checkpoint_step": item["step"],
                "files": {
                    n: {k: v[k] for k in ["size", "sha256"]} for n, v in records.items()
                },
                "full_optimizer_and_rng_retained": True,
            },
            indent=2,
        ).encode()
        card = (
            "---\ntags:\n- checkpoint-archive\n- modular-mopd\n---\n\n"
            f"# Finished modular checkpoint: {policy.parent.name}\n\n"
            f"Original policy: `{policy}`\n\nFinal checkpoint step: {item['step']}. "
            "Native model weights, Adam optimizer state, RNG and checkpoint metadata are preserved exactly. "
            "Training logs and evaluations remain at the original local path.\n\n"
            "Restore only the checkpoint files with:\n\n```python\n"
            "from huggingface_hub import snapshot_download\n"
            f"snapshot_download({repo!r}, revision='<PINNED_COMMIT_FROM_RECEIPT>', allow_patterns=['ckpt/**'], local_dir={str(policy)!r})\n```\n"
        ).encode()
        operations += [
            CommitOperationAdd(
                path_in_repo="archive_manifest.json", path_or_fileobj=manifest
            ),
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card),
        ]
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        if not api.model_info(repo).private:
            raise ValueError("Refusing to publish an archive into a public repository")
        print(
            f"Uploading {sum(r['size'] for r in records.values()) / 2**30:.2f} GiB to {repo}",
            flush=True,
        )
        commit = api.create_commit(
            repo,
            operations=operations,
            commit_message="Archive completed native checkpoint with optimizer and RNG",
            num_threads=3,
        )
        revision = commit.oid
        verify_remote(api, repo, revision, records)
        api.add_collection_item(collection, repo, "model", exists_ok=True)
        if repo not in {i.item_id for i in api.get_collection(collection).items}:
            raise ValueError("Verified archive is missing from its collection")
        receipt.update(status="verified", revision=revision, verified_at=time.time())
        write(receipt_path, receipt)
        del operations
        gc.collect()
        ensure_inactive()
        receipt = finish_eviction(policy, checkpoint, receipt, receipt_path)
        print(
            f"Verified and evicted {repo}: {receipt['allocated_bytes_evicted'] / 2**30:.2f} GiB",
            flush=True,
        )
        return receipt


def execute(inventory, output, *, workers=1):
    from huggingface_hub import HfApi

    manifest = json.loads(inventory.read_text())
    if Path(manifest["scope"]) != SCOPE or manifest.get("private") is not True:
        raise ValueError("Unexpected archive scope or privacy setting")
    api = HfApi()
    if api.whoami().get("name") != OWNER:
        raise ValueError("Archive requires the owner's account")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "archive.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_inactive()
        path = output / "collection.json"
        if path.exists():
            collection = api.get_collection(json.loads(path.read_text())["slug"])
        else:
            collection = api.create_collection(
                manifest["collection_title"],
                namespace=OWNER,
                private=True,
                exists_ok=True,
            )
            write(
                path,
                {
                    "slug": collection.slug,
                    "url": collection.url,
                    "private": collection.private,
                },
            )
        if not collection.private:
            raise ValueError("Archive collection must be private")
        results, failures = [], []
        print("Collection:", collection.url, flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(archive_one, api, item, output, collection.slug): item
                for item in manifest["items"]
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    failure = {
                        "checkpoint": futures[future]["checkpoint"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    print(json.dumps(failure), flush=True)
                write(
                    output / "progress.json",
                    {
                        "time": time.time(),
                        "completed": len(results),
                        "total": len(manifest["items"]),
                        "failures": failures,
                        "allocated_gib_evicted": sum(
                            r["allocated_bytes_evicted"] for r in results
                        )
                        / 2**30,
                        "free_gib": shutil.disk_usage(SCOPE).free / 2**30,
                    },
                )
        # Reconcile membership sequentially after concurrent additions.
        for result in results:
            api.add_collection_item(
                collection.slug, result["repo_id"], "model", exists_ok=True
            )
        members = {i.item_id for i in api.get_collection(collection.slug).items}
        if any(r["repo_id"] not in members for r in results):
            raise ValueError("Archive collection membership is incomplete")
        if failures:
            raise RuntimeError(
                f"{len(failures)} checkpoints require a retry; verified receipts are retained"
            )
        report = {
            "completed_at": time.time(),
            "checkpoints": len(results),
            "collection": collection.url,
            "allocated_gib_evicted": sum(r["allocated_bytes_evicted"] for r in results)
            / 2**30,
            "free_gib": shutil.disk_usage(SCOPE).free / 2**30,
            "full_optimizer_and_rng_retained_remotely": True,
        }
        write(output / "completed.json", report)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args()
    if not args.execute:
        for entry in json.loads(args.inventory.read_text())["items"]:
            print(entry["source"], len(inspect_checkpoint(entry)))
    else:
        execute(args.inventory, args.output, workers=args.workers)
