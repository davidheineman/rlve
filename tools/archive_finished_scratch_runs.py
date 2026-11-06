"""Back up completed scratch run checkpoints to one HF collection, then evict verified runs.

Inventory is immutable once created. Upload progress is journaled outside scratch so
an interrupted upload can safely resume without trusting an absent local folder.
"""

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path


SCRATCH = Path("/juice5b/scr5b/dhei")
RUNS = SCRATCH / "runs"
ARCHIVE = Path(__file__).resolve().parent.parent / "data/HF-SCRATCH-ARCHIVE-20261004"
MANIFEST = ARCHIVE / "manifest.json"
JOURNAL = ARCHIVE / "journal.json"
RELEASE_AUDIT = ARCHIVE / "release_authorization.json"
COLLECTION = "davidheineman/rlve-finished-scratch-runs-6ac3388c8cb2137053d9717a"
NAMESPACE = "davidheineman"
PROTECTED_GROUPS = {
    "mopd-v3-r1-p1r8-teachers-20261004-095049": "used by the active v3 students and queued similarity-v2 MOPD jobs",
    "mopd-v3-r1-all16-warmstart-sft-20261004-095049": "used by active v3 warm-start students",
}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def slug(value):
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value.lower())).strip("-")


def record(source, kind, payload, checkpoint_step=None, existing_repo=None):
    source = source.resolve(strict=True)
    if source.is_symlink() or not source.is_dir() or not source.is_relative_to(SCRATCH):
        raise ValueError(f"Invalid source run: {source}")
    relative = source.relative_to(SCRATCH)
    key = hashlib.sha256(str(relative).encode()).hexdigest()[:12]
    group = relative.parts[1] if relative.parts[0] == "runs" else relative.parts[0]
    repo_name = f"rlve-archive-{slug(group)[:38]}-{slug(source.name)[:28]}-{key}"
    if len(repo_name) > 96:
        raise ValueError(repo_name)
    protected = PROTECTED_GROUPS.get(group)
    if payload is not None:
        payload = payload.resolve(strict=True)
        if not payload.is_relative_to(source) or not payload.is_dir():
            raise ValueError(f"Payload escapes source: {payload}")
    return {
        "id": key,
        "source": str(source),
        "relative_source": str(relative),
        "kind": kind,
        "payload": str(payload) if payload is not None else None,
        "checkpoint_step": checkpoint_step,
        "repo_id": existing_repo or f"{NAMESPACE}/{repo_name}",
        "protected_reason": protected,
    }


def inventory(refresh=False):
    if MANIFEST.exists() and not refresh:
        print(f"Existing immutable inventory: {MANIFEST}")
        return
    known_sources = ({item["source"] for item in json.loads(MANIFEST.read_text())["runs"]}
                     if refresh and MANIFEST.exists() else set())
    items = []
    seen = set()
    for marker in sorted(RUNS.rglob("weights_only_complete.json")):
        source = marker.parent
        if str(source.resolve()) in known_sources:
            continue
        metadata = json.loads(marker.read_text())
        export = source / "hf-final"
        index = export / "model.safetensors.index.json"
        if metadata.get("format") != "hf-safetensors" or not index.is_file():
            raise ValueError(f"Invalid completed export: {source}")
        weight_map = json.loads(index.read_text())["weight_map"]
        weights = {export / filename for filename in weight_map.values()}
        if len(weight_map) != metadata["tensor_count"] or not all(p.is_file() for p in weights):
            raise ValueError(f"Missing exported tensors: {source}")
        if sum(p.stat().st_size for p in weights) != metadata["weight_bytes"]:
            raise ValueError(f"Weight byte count differs: {source}")
        items.append(record(source, "hf-safetensors", export, metadata["checkpoint_step"]))
        seen.add(source.resolve())
    for state_file in sorted(RUNS.rglob("rlve_checkpoint_state.json")):
        source = state_file.parent.parent
        if source.resolve() in seen or str(source.resolve()) in known_sources:
            continue
        state = json.loads(state_file.read_text())
        if not state.get("completed"):
            continue
        step = state["rollout_id"]
        checkpoint = source / "ckpt"
        latest = (checkpoint / "latest_checkpointed_iteration.txt").read_text().strip()
        payload = checkpoint / f"iter_{step:07d}"
        if latest != str(step) or not (payload / ".metadata").is_file():
            raise ValueError(f"Completed state has no final checkpoint: {source}")
        items.append(record(source, "megatron-torch-dist", payload, step))
        seen.add(source.resolve())
    for backup in sorted(RUNS.rglob("uploaded_to_hf.json")):
        source = backup.parent
        if source.resolve() in seen or str(source.resolve()) in known_sources:
            continue
        data = json.loads(backup.read_text())
        if not (source / "weights_only_complete.uploaded.json").is_file():
            raise ValueError(f"Unverified previous archive marker: {source}")
        items.append(record(source, "previously-uploaded", None, existing_repo=data["repo_id"]))
        seen.add(source.resolve())
    # This run trained all 200 requested updates and evaluated update 199. Its Slurm
    # wrapper failed after training, so it has no transactional completed marker.
    spade = RUNS / "spade-qwen3-seedwide-03255-20261001/qwen3-final-game"
    checkpoint = spade / "ckpt"
    if str(spade.resolve()) not in known_sources and \
            (checkpoint / "latest_checkpointed_iteration.txt").is_file() and \
            (checkpoint / "latest_checkpointed_iteration.txt").read_text().strip() == "199":
        payload = checkpoint / "iter_0000199"
        if not (payload / ".metadata").is_file() or spade.resolve() in seen:
            raise ValueError("Invalid final SPADE checkpoint")
        items.append(record(spade, "megatron-torch-dist", payload, 199))
        seen.add(spade.resolve())
    nano = SCRATCH / "nano-omni-multiteacher"
    for source in sorted((nano / "runs").iterdir()):
        if str(source.resolve()) not in known_sources and source.is_dir() and (source / "model.pt").is_file():
            items.append(record(source, "pytorch-checkpoint", source))
            seen.add(source.resolve())
    inloop = nano / "runs-inloop-20260930/code"
    if str(inloop.resolve()) not in known_sources and (inloop / "train_state.pt").is_file():
        items.append(record(inloop, "pytorch-checkpoint", inloop))
        seen.add(inloop.resolve())
    for source in sorted((nano / "runs-inloop-300000000-2000000").iterdir()):
        if str(source.resolve()) not in known_sources and source.is_dir() and \
                (source / "model.pt").is_file() and (source / "train_state.pt").is_file():
            items.append(record(source, "pytorch-checkpoint", source))
            seen.add(source.resolve())
    if len({item["source"] for item in items}) != len(items) or len({item["repo_id"] for item in items}) != len(items):
        raise ValueError("Duplicate source or repository in inventory")
    # Free large, completed sweep exports first. Keep active pipeline sources local.
    def priority(item):
        group = Path(item["relative_source"]).parts[1] if item["relative_source"].startswith("runs/") else "nano"
        if item["protected_reason"]:
            return (9, group, item["source"])
        if group.startswith("mixing-r1-nofilter16"):
            return (0, group, item["source"])
        if group.startswith("1way-300step-mixing"):
            return (1, group, item["source"])
        if group.startswith("diversity-teachers-r1"):
            return (2, group, item["source"])
        if item["kind"] == "previously-uploaded":
            return (3, group, item["source"])
        if item["kind"] == "hf-safetensors":
            return (4, group, item["source"])
        if item["kind"] == "pytorch-checkpoint":
            return (5, group, item["source"])
        return (6, group, item["source"])
    items.sort(key=priority)
    if refresh and MANIFEST.exists():
        previous = json.loads(MANIFEST.read_text())
        if previous["collection"] != COLLECTION or previous["scratch"] != str(SCRATCH):
            raise ValueError("Existing archive manifest does not match this archive")
        known = {item["source"] for item in previous["runs"]}
        additions = [item for item in items if item["source"] not in known]
        items = previous["runs"] + additions
        print(f"Newly completed runs added: {len(additions)}", flush=True)
    atomic_json(MANIFEST, {"collection": COLLECTION, "scratch": str(SCRATCH), "runs": items})
    print(f"Inventory: {len(items)} finished runs: {MANIFEST}")
    for kind in sorted({item["kind"] for item in items}):
        print(f"  {kind}: {sum(item['kind'] == kind for item in items)}")
    print(f"  protected dependencies: {sum(bool(item['protected_reason']) for item in items)}")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_blob_sha1(path):
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def files_to_verify(item):
    source = Path(item["source"])
    payload = Path(item["payload"])
    if item["kind"] == "megatron-torch-dist":
        prefix = f"checkpoint/iter_{item['checkpoint_step']:07d}"
    else:
        prefix = ""
    files = {}
    for file in payload.rglob("*"):
        if file.is_file() and not file.is_symlink():
            relative = file.relative_to(payload).as_posix()
            files[f"{prefix}/{relative}".lstrip("/")] = file
    if item["kind"] == "megatron-torch-dist":
        files["checkpoint/rlve_checkpoint_state.json"] = source / "ckpt/rlve_checkpoint_state.json" if (source / "ckpt/rlve_checkpoint_state.json").is_file() else None
        files = {name: path for name, path in files.items() if path is not None}
        files["checkpoint/latest_checkpointed_iteration.txt"] = source / "ckpt/latest_checkpointed_iteration.txt"
    if not files or not any(path.stat().st_size > 100_000_000 for path in files.values()):
        raise ValueError(f"No model checkpoint payload: {source}")
    return files


def card(item):
    run = Path(item["source"])
    run_id = (run / "wandb_run_id.txt").read_text().strip() if (run / "wandb_run_id.txt").is_file() else None
    return ("---\ntags:\n- rlve\n- scratch-archive\n---\n\n"
            f"# Archived checkpoint: {run.name}\n\n"
            f"Original scratch path: `{item['relative_source']}`.\n\n"
            f"Checkpoint format: `{item['kind']}`.\n\n"
            f"Final checkpoint step: `{item['checkpoint_step']}`.\n\n"
            f"W&B run ID: `{run_id}`.\n\n"
            "This repository preserves the final checkpoint from a completed run. "
            "For Megatron distributed checkpoints, the `checkpoint/` directory "
            "contains the exact saved model state.\n")


def upload_payload(api, item):
    repo = item["repo_id"]
    source = Path(item["source"])
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    if item["kind"] == "megatron-torch-dist":
        api.upload_folder(repo_id=repo, folder_path=item["payload"],
                          path_in_repo=f"checkpoint/iter_{item['checkpoint_step']:07d}",
                          commit_message="Archive final Megatron checkpoint")
        state = source / "ckpt/rlve_checkpoint_state.json"
        if state.is_file():
            api.upload_file(repo_id=repo, path_or_fileobj=state,
                            path_in_repo="checkpoint/rlve_checkpoint_state.json",
                            commit_message="Record final training state")
        api.upload_file(repo_id=repo, path_or_fileobj=source / "ckpt/latest_checkpointed_iteration.txt",
                        path_in_repo="checkpoint/latest_checkpointed_iteration.txt",
                        commit_message="Record final checkpoint step")
    else:
        api.upload_folder(repo_id=repo, folder_path=item["payload"],
                          commit_message="Archive final run checkpoint")
    api.upload_file(repo_id=repo, path_or_fileobj=card(item).encode(),
                    path_in_repo="README.md", commit_message="Add checkpoint provenance")


def verify_repo(api, item):
    info = api.model_info(item["repo_id"], files_metadata=True)
    siblings = {s.rfilename: s for s in info.siblings}
    if item["kind"] == "previously-uploaded":
        backup = json.loads((Path(item["source"]) / "uploaded_to_hf.json").read_text())
        weight = siblings.get(backup["weight_file"])
        if weight is None or weight.lfs is None or weight.lfs.sha256 != backup["weight_sha256"]:
            raise ValueError(f"Previously uploaded weight mismatch: {item['repo_id']}")
        return info.sha
    files = files_to_verify(item)
    for name, path in files.items():
        remote = siblings.get(name)
        if remote is None or remote.size != path.stat().st_size:
            raise ValueError(f"Missing or size-mismatched remote file: {item['repo_id']}/{name}")
        if remote.lfs is not None:
            if remote.lfs.sha256 != sha256(path):
                raise ValueError(f"SHA-256 mismatch: {item['repo_id']}/{name}")
        elif remote.blob_id != git_blob_sha1(path):
            raise ValueError(f"Git blob SHA-1 mismatch: {item['repo_id']}/{name}")
    readme = siblings.get("README.md")
    if readme is None:
        raise ValueError(f"Missing model card: {item['repo_id']}")
    return info.sha


@contextmanager
def run_lock(source):
    lock_path = source / "run.lock"
    if not lock_path.exists():
        yield
        return
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Run is active: {source}") from error
        yield


def journal_update(journal, item, status, **extra):
    journal[item["id"]] = {"status": status, "repo_id": item["repo_id"], "source": item["source"], **extra}
    atomic_json(JOURNAL, journal)


def collection_has_repo(api, repo_id):
    collection = api.get_collection(COLLECTION)
    return repo_id in {x.item_id for x in collection.items if x.item_type == "model"}


def remove_source(source):
    # Lustre can briefly report ENOTEMPTY after unlinking the last file.
    for attempt in range(10):
        if not source.exists():
            return
        try:
            shutil.rmtree(source)
            return
        except OSError as error:
            if error.errno not in {errno.ENOTEMPTY, errno.EBUSY, errno.ENOENT} or attempt == 9:
                raise
            time.sleep(min(2 ** attempt, 8))


def process_one(api, item, journal, release_protected=False):
    source = Path(item["source"])
    previous = journal.get(item["id"], {})
    if previous.get("status") == "deleted":
        return "already deleted"
    if not source.is_dir():
        if previous.get("status") in {"verified", "retained"}:
            journal_update(journal, item, "deleted", revision=previous.get("revision"))
            return "already absent after verified backup"
        raise FileNotFoundError(f"Source absent without verified backup: {source}")
    if source.is_symlink() or not source.resolve().is_relative_to(SCRATCH):
        raise ValueError(f"Unexpected source path: {source}")
    with run_lock(source):
        if previous.get("status") in {"verified", "retained"}:
            # A prior process may have stopped partway through local deletion.
            revision = api.model_info(item["repo_id"]).sha
            if revision != previous.get("revision") or not collection_has_repo(api, item["repo_id"]):
                raise ValueError(f"Previously verified remote backup changed: {item['repo_id']}")
        else:
            if item["kind"] != "previously-uploaded":
                upload_payload(api, item)
            revision = verify_repo(api, item)
            api.add_collection_item(COLLECTION, item["repo_id"], "model", exists_ok=True)
            if not collection_has_repo(api, item["repo_id"]):
                raise ValueError(f"Missing collection item: {item['repo_id']}")
            journal_update(journal, item, "verified", revision=revision)
        if item["protected_reason"] and not release_protected:
            journal_update(journal, item, "retained", revision=revision,
                           reason=item["protected_reason"])
            return "uploaded; retained for dependent jobs"
        if item["protected_reason"] and release_protected:
            audit = json.loads(RELEASE_AUDIT.read_text())
            group = Path(item["relative_source"]).parts[1]
            if not audit["groups"].get(group, {}).get("approved"):
                journal_update(journal, item, "retained", revision=revision,
                               reason=f"Dependency audit does not release {group}")
                return "uploaded; retained for incomplete dependent jobs"
    remove_source(source)
    journal_update(journal, item, "deleted", revision=revision)
    return "uploaded, verified, and deleted"


def upload_locked(args):
    from huggingface_hub import HfApi

    manifest = json.loads(MANIFEST.read_text())
    if manifest["collection"] != COLLECTION:
        raise ValueError("Collection mismatch")
    api = HfApi()
    collection = api.get_collection(COLLECTION)
    print(f"Collection: https://huggingface.co/collections/{collection.slug}", flush=True)
    journal = json.loads(JOURNAL.read_text()) if JOURNAL.exists() else {}
    items = manifest["runs"]
    if args.command == "release":
        if not RELEASE_AUDIT.is_file():
            raise FileNotFoundError(f"Missing dependency audit: {RELEASE_AUDIT}")
        items = [item for item in items if item["protected_reason"]]
    if args.only:
        items = [item for item in items if args.only in item["relative_source"]]
        if not items:
            raise ValueError(f"No item matches --only {args.only}")
    failed = 0
    for index, item in enumerate(items, 1):
        for attempt in range(3):
            try:
                result = process_one(api, item, journal, release_protected=args.command == "release")
                print(f"{index}/{len(items)} {item['relative_source']}: {result}; {item['repo_id']}", flush=True)
                break
            except Exception as error:
                if attempt == 2:
                    failed += 1
                    if journal.get(item["id"], {}).get("status") not in {"verified", "retained"}:
                        journal_update(journal, item, "failed", error=f"{type(error).__name__}: {error}")
                    print(f"FAILED {item['relative_source']}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                else:
                    print(f"Retry {attempt + 1}/2 {item['relative_source']}: {type(error).__name__}: {error}", flush=True)
                    time.sleep(2 ** attempt)
    status = {item["id"]: journal.get(item["id"], {}).get("status", "pending") for item in manifest["runs"]}
    print(f"Archive status: {len(items)} selected, {sum(x == 'deleted' for x in status.values())} deleted, "
          f"{sum(x == 'retained' for x in status.values())} protected, "
          f"{sum(x == 'failed' for x in status.values())} failed", flush=True)
    if failed:
        raise SystemExit(1)


def upload(args):
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    with (ARCHIVE / "archive.lock").open("a") as lock:
        print("Waiting for archive writer lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        print("Archive writer lock acquired", flush=True)
        upload_locked(args)


def status():
    manifest = json.loads(MANIFEST.read_text())
    journal = json.loads(JOURNAL.read_text()) if JOURNAL.exists() else {}
    from collections import Counter
    counts = Counter(journal.get(item["id"], {}).get("status", "pending") for item in manifest["runs"])
    print(f"Collection: https://huggingface.co/collections/{COLLECTION}")
    print(f"Inventory: {len(manifest['runs'])}")
    for name, count in sorted(counts.items()):
        print(f"{name}: {count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inventory", "upload", "release", "status"])
    parser.add_argument("--only", help="Process only records whose source path contains this text")
    parser.add_argument("--refresh", action="store_true", help="Append newly completed runs to the inventory")
    args = parser.parse_args()
    if args.command == "inventory":
        inventory(refresh=args.refresh)
    elif args.command in {"upload", "release"}:
        upload(args)
    else:
        status()


if __name__ == "__main__":
    main()
