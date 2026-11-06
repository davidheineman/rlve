"""Archive the completed original scaling models, then evict verified local exports.

The immutable inventory, per-model upload receipts, and compressed provenance live
outside scratch. Only the two named original experiments are in deletion scope.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/juice5b/scr5b/dhei")
GROUPS = ("data-scaling-v1-20261022-180018", "data-scaling-v1-n128-20261022-225839")
SOURCES = tuple(SCRATCH / "runs" / name for name in GROUPS)
ARCHIVE = ROOT / "data/HF-OPD-ARCHIVE-20261025"
NAMESPACE = "davidheineman"
TITLES = {"teacher": "RLVE OPD Teachers (Oct 25 2026)",
          "student": "RLVE OPD Students (Oct 25 2026)"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fingerprint(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def checkpoint_metadata_only(source):
    checkpoint = Path(source) / "ckpt"
    if not checkpoint.exists():
        return not checkpoint.is_symlink()
    return (not checkpoint.is_symlink() and checkpoint.is_dir()
            and all(p.name == "rlve_checkpoint_state.json" and p.is_file()
                    and not p.is_symlink() and p.stat().st_size < 1024**2
                    for p in checkpoint.iterdir()))


def validate_export(source):
    """Check completion, weight-map membership, tensor bounds, and export sizes."""
    source = Path(source)
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"Expected an original physical run directory: {source}")
    if not (source / "completed").is_file() or not checkpoint_metadata_only(source):
        raise ValueError(f"Not a finalized weights-only run: {source}")
    marker = read(source / "weights_only_complete.json")
    export = source / "hf-final"
    if export.is_symlink() or not export.is_dir():
        raise ValueError(f"Unexpected export: {export}")
    index = read(export / "model.safetensors.index.json")["weight_map"]
    if marker["format"] != "hf-safetensors" or len(index) != marker["tensor_count"]:
        raise ValueError(f"Invalid tensor index: {source}")
    weights = [export / name for name in sorted(set(index.values()))]
    if any(p.parent != export or p.is_symlink() or not p.is_file() for p in weights):
        raise ValueError(f"Missing or unsafe weight shard: {source}")
    if sum(p.stat().st_size for p in weights) != marker["weight_bytes"]:
        raise ValueError(f"Wrong weight byte count: {source}")
    for weight in weights:
        with weight.open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            if not 0 < length < min(weight.stat().st_size, 16 * 1024**2):
                raise ValueError(f"Invalid safetensors header: {weight}")
            tensors = json.loads(f.read(length))
        tensors.pop("__metadata__", None)
        if set(tensors) != {key for key, name in index.items() if name == weight.name}:
            raise ValueError(f"Tensor header/index mismatch: {weight}")
        end = weight.stat().st_size - 8 - length
        if any(not 0 <= t["data_offsets"][0] <= t["data_offsets"][1] <= end for t in tensors.values()):
            raise ValueError(f"Tensor outside file: {weight}")
    return marker


def inventory():
    target = ARCHIVE / "inventory.json"
    if target.exists():
        return read(target)
    records = []
    seen = set()
    for root in SOURCES:
        plan = read(root / "run_plan.json")
        for run in plan["runs"]:
            source = root / run["kind"] / run["label"]
            if source.is_symlink():
                if source.resolve() not in seen:
                    raise ValueError(f"Unexpected alias preceding its original: {source}")
                continue
            if source.resolve() in seen:
                raise ValueError(f"Duplicate physical source: {source}")
            seen.add(source.resolve())
            marker = validate_export(source)
            role = "teacher" if run["kind"] == "teacher" else "student"
            if role == "teacher":
                index = int(run["label"].split("-", 1)[0])
                suffix = f"{index:03d}-" + run["environments"][0]
            else:
                suffix = run["label"]
            repo = f"{NAMESPACE}/rlve-opd-{role}-R1Distill-{suffix}-20261025"
            if len(repo.split("/")[1]) > 96:
                raise ValueError(f"Repository name too long: {repo}")
            key = hashlib.sha256(str(source).encode()).hexdigest()[:16]
            records.append({"id": key, "source": str(source), "group": root.name,
                            "relative_source": str(source.relative_to(SCRATCH)),
                            "role": role, "repo_id": repo, "run": run, "marker": marker,
                            "wandb_run_id": (source / "wandb_run_id.txt").read_text().strip()})
    counts = Counter(item["role"] for item in records)
    if counts != {"teacher": 128, "student": 8} or len({r["repo_id"] for r in records}) != 136:
        raise ValueError(f"Unexpected inventory: {counts}")
    value = {"created_utc": now(), "namespace": NAMESPACE, "collection_titles": TITLES,
             "groups": list(GROUPS), "models": records}
    atomic(target, value)
    print(f"Inventory validated: {dict(counts)}", flush=True)
    return value


@contextmanager
def lock_run(source):
    with (Path(source) / "run.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def receipt_path(item):
    return ARCHIVE / "receipts" / (item["id"] + ".json")


def model_files(item):
    """Stage only selected nonsecret metadata beside the model, not raw logs."""
    source = Path(item["source"])
    staging = ARCHIVE / "model-metadata" / item["id"]
    staging.mkdir(parents=True, exist_ok=True)
    run, marker = item["run"], item["marker"]
    project = run["project"]
    wandb = f"https://wandb.ai/david-heineman/{project}/runs/{item['wandb_run_id']}"
    method = "GRPO specialist" if item["role"] == "teacher" else ("MOPD" if run["kind"] == "mopd" else "joint RL (GRPO)")
    actual = (marker["checkpoint_step"] + 1) * (run["training_rollouts"] // run["updates"])
    card = ("---\nbase_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B\n"
            "library_name: transformers\npipeline_tag: text-generation\ntags:\n"
            "- rlve\n- data-scaling\n- " + ("opd-teacher" if item["role"] == "teacher" else run["kind"]) + "\n---\n\n"
            f"# RLVE OPD {item['role']}: {run['label']}\n\n"
            f"Final weights from the original RLVE data-scaling experiment. Method: **{method}**. "
            "All actors were initialized freshly from DeepSeek-R1-Distill-Qwen-1.5B.\n\n"
            f"Training environments: {len(run['environments'])}, fixed balanced difficulty 0/1. "
            "Generation uses 1024 initial tokens plus the configured 128-token answer-forcing continuation.\n\n"
            f"Training cap: {run['training_rollouts']:,} rollouts; final checkpoint update: {marker['checkpoint_step']} "
            f"(zero-based), corresponding to {actual:,} training rollouts. "
            + ("Teachers stop at 4,000 rollouts or ID validation pass@1 >= 0.98. " if item["role"] == "teacher" else "")
            + "Validation uses 32 prompts and 8 responses per prompt every 50 updates. "
            + ("Heroes also use the shared 32-prompt unseen-environment test set. " if item["role"] == "student" else "")
            + "Validation samples do not count toward training budgets.\n\n"
            f"Training run: [W&B]({wandb}). Group: `{item['group']}`.\n\n"
            f"This repository contains model weights and tokenizer/config files; optimizer and dataloader state are absent. "
            "The environment list and original run settings are recorded in `training_run.json`, "
            "with the exact configuration in `training_config.yaml`.\n")
    (staging / "README.md").write_text(card)
    atomic(staging / "training_run.json", run)
    atomic(staging / "archive_manifest.json", {
        "archive_date": "2026-10-25", "archive_id": item["id"],
        "source_group": item["group"], "relative_source": item["relative_source"],
        "role": item["role"], "wandb_url": wandb, "checkpoint": marker})
    atomic(staging / "weights_only_complete.json", marker)
    config = ARCHIVE / "configs" / Path(run["config"]).name
    if not config.is_file():
        raise FileNotFoundError(f"Missing frozen configuration: {config}")
    shutil.copy2(config, staging / "training_config.yaml")
    files = {p.name: p for p in (source / "hf-final").iterdir() if p.is_file()}
    if any(p.is_symlink() for p in (source / "hf-final").iterdir()):
        raise ValueError(f"Export contains symlinks: {source}")
    files.update({p.name: p for p in staging.iterdir() if p.is_file()})
    return files


def git_sha(path):
    h = hashlib.sha1()
    h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        while block := f.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def verify_remote(api, item, receipt, revision=None):
    info = api.model_info(item["repo_id"], revision=revision, files_metadata=True)
    siblings = {s.rfilename: s for s in info.siblings or []}
    for name, expected in receipt["files"].items():
        remote = siblings.get(name)
        if remote is None or remote.size != expected["size"]:
            raise ValueError(f"Remote missing/size mismatch: {item['repo_id']}/{name}")
        if remote.lfs is not None:
            if remote.lfs.sha256 != expected["sha256"]:
                raise ValueError(f"Remote SHA-256 mismatch: {item['repo_id']}/{name}")
        elif expected["git_sha1"] is None or remote.blob_id != expected["git_sha1"]:
            raise ValueError(f"Remote Git blob mismatch: {item['repo_id']}/{name}")
    return info.sha


def ensure_collections(api):
    path = ARCHIVE / "collections.json"
    if path.exists():
        saved = read(path)
        for role, entry in saved.items():
            collection = api.get_collection(entry["slug"])
            if collection.title != TITLES[role] or collection.private:
                raise ValueError("Collection metadata changed")
        return saved
    saved = {}
    for role, title in TITLES.items():
        c = api.create_collection(title, namespace=NAMESPACE, private=False, exists_ok=True,
                                  description=("128 RLVE GRPO teachers" if role == "teacher" else "8 RLVE MOPD and joint-RL heroes (N16/N32/N64/N128)")
                                  + " from the original data-scaling experiments, initialized from DeepSeek-R1-Distill-Qwen-1.5B.")
        saved[role] = {"slug": c.slug, "url": c.url, "title": c.title}
        print(f"Collection: {c.url}", flush=True)
    atomic(path, saved)
    return saved


def upload_one(item, collections):
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
    api = HfApi()
    path = receipt_path(item)
    receipt = read(path) if path.exists() else {}
    if receipt.get("status") in {"verified", "evicting", "evicted"}:
        verify_remote(api, item, receipt, receipt["revision"])
        return item["repo_id"]
    source = Path(item["source"])
    with lock_run(source):
        validate_export(source)
        files = model_files(item)
        print(f"Hashing {item['repo_id']}", flush=True)
        before = {name: fingerprint(local) for name, local in files.items()}
        operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=local) for name, local in sorted(files.items())]
        if any(fingerprint(local) != before[name] for name, local in files.items()):
            raise RuntimeError(f"Source changed while hashing: {source}")
        expected = {op.path_in_repo: {
            "size": op.upload_info.size, "sha256": op.upload_info.sha256.hex(),
            "git_sha1": git_sha(files[op.path_in_repo]) if op.upload_info.size < 20 * 1024**2 else None,
            "fingerprint": fingerprint(files[op.path_in_repo]), "path": str(files[op.path_in_repo])}
            for op in operations}
        receipt.update(status="hashed", source=item["source"], repo_id=item["repo_id"], files=expected,
                       collection=collections[item["role"]]["slug"])
        atomic(path, receipt)
        try:
            api.create_repo(item["repo_id"], repo_type="model", private=False, exist_ok=False)
            receipt["created_repo"] = True
            atomic(path, receipt)
        except HfHubHTTPError as error:
            if error.response.status_code != 409:
                raise
            if not receipt.get("created_repo"):
                remote_manifest = read(hf_hub_download(item["repo_id"], "archive_manifest.json"))
                if remote_manifest.get("archive_id") != item["id"]:
                    raise RuntimeError(f"Existing repository belongs to another source: {item['repo_id']}")
        try:
            revision = verify_remote(api, item, receipt)
        except (ValueError, FileNotFoundError):
            print(f"Uploading {item['repo_id']}", flush=True)
            commit = api.create_commit(item["repo_id"], operations=operations,
                                       commit_message="Archive validated RLVE scaling final weights and training provenance",
                                       num_threads=2)
            revision = verify_remote(api, item, receipt, commit.oid)
        note = f"{item['run']['label']}; {item['run']['kind']}; W&B {item['wandb_run_id']}"
        api.add_collection_item(receipt["collection"], item["repo_id"], "model", note=note, exists_ok=True)
        receipt.update(status="verified", revision=revision, verified_utc=now())
        atomic(path, receipt)
        return item["repo_id"]


def verify_collections(api, items, collections):
    for role, entry in collections.items():
        actual = {v.item_id for v in api.get_collection(entry["slug"]).items if v.item_type == "model"}
        expected = {v["repo_id"] for v in items if v["role"] == role}
        if actual != expected:
            raise ValueError(f"Collection mismatch {role}: missing={expected-actual}, extra={actual-expected}")


def reconcile_collections(api, items, collections):
    """Repair missing entries sequentially after concurrent uploads have ended."""
    repaired = []
    for role, entry in collections.items():
        actual = {v.item_id for v in api.get_collection(entry["slug"]).items if v.item_type == "model"}
        for item in items:
            if item["role"] == role and item["repo_id"] not in actual:
                path = receipt_path(item)
                receipt = read(path)
                if receipt["status"] not in {"verified", "evicting", "evicted"}:
                    raise RuntimeError(f"Cannot collect an unverified model: {item['repo_id']}")
                verify_remote(api, item, receipt, receipt["revision"])
                note = f"{item['run']['label']}; {item['run']['kind']}; W&B {item['wandb_run_id']}"
                api.add_collection_item(entry["slug"], item["repo_id"], "model", note=note, exists_ok=True)
                repaired.append(item["repo_id"])
    verify_collections(api, items, collections)
    atomic(ARCHIVE / "collection-reconciliation.json", {"utc": now(), "repaired": repaired})
    return repaired


def upload_all(manifest, workers):
    from huggingface_hub import HfApi
    collections = ensure_collections(HfApi())
    # Upload the eight heroes first, then the 128 teachers.
    items = sorted(manifest["models"], key=lambda item: (item["role"] == "teacher", item["run"]["label"]))
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(upload_one, item, collections): item for item in items}
        for count, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            try:
                result = future.result()
                print(f"Verified {count}/{len(items)}: {result}", flush=True)
            except Exception as e:
                failures.append({"repo_id": item["repo_id"], "error": str(e)})
                print(f"FAILED {item['repo_id']}: {type(e).__name__}: {e}", flush=True)
    if failures:
        atomic(ARCHIVE / "failures.json", failures)
        raise RuntimeError(f"{len(failures)} models failed; no eviction performed")
    reconcile_collections(HfApi(), manifest["models"], collections)
    atomic(ARCHIVE / "all-uploads-verified.json", {"utc": now(), "models": len(items), "collections": collections})


def audit_active_sources():
    """Check live Slurm work paths and each live async plan/task input."""
    raw = subprocess.check_output(["squeue", "-u", os.environ.get("USER", "dhei"), "-h", "-o", "%i|%j|%k"], text=True)
    roots = [str(p) for p in SOURCES]
    jobs = [line.split("|", 2) for line in raw.splitlines() if line.strip()]
    for job, _, _ in jobs:
        details = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True)
        if any(root in details for root in roots):
            raise RuntimeError(f"Live job {job} references an archive source")
    checked = []
    for experiment in (SCRATCH / "runs").glob("data-scaling-async-n*"):
        journal = experiment / "slurm_jobs.json"
        if not journal.exists():
            continue
        ids = {r["job"] for r in read(journal).values()}
        if not ids.intersection(j[0] for j in jobs):
            continue
        paths = [experiment / "plan.json"]
        state = read(experiment / "state.json")
        for device in state["devices"]:
            if device["active"]:
                p = experiment / "tasks" / device["active"]["id"] / "task.json"
                if p.exists():
                    paths.append(p)
        request = experiment / "student-service/request.json"
        if request.exists():
            paths.append(Path(read(request)["task"]))
        for path in paths:
            if any(root in path.read_text() for root in roots):
                raise RuntimeError(f"Active async input refers to archived source: {path}")
        checked.append(str(experiment))
    result = {"utc": now(), "live_jobs": jobs, "checked_async_experiments": checked,
              "source_paths_referenced_by_live_jobs": False}
    atomic(ARCHIVE / "active-source-audit.json", result)
    return result


def backup_provenance():
    """Preserve all small files, logs, and aliases outside scratch, excluding exports."""
    receipt = ARCHIVE / "provenance-backup.json"
    if receipt.exists():
        record = read(receipt)
        if digest(record["path"]) != record["sha256"]:
            raise RuntimeError("Compressed provenance changed")
        return
    target = ARCHIVE / "original-experiments-provenance.tar.gz"
    temporary = target.with_suffix(".incomplete")
    members = {}
    with tarfile.open(temporary, "w:gz", compresslevel=6, dereference=False) as archive:
        for root in SOURCES:
            for current, dirs, files in os.walk(root, followlinks=False):
                dirs[:] = [d for d in dirs if d != "hf-final"]
                for name in [*dirs, *files]:
                    path = Path(current) / name
                    if path.is_dir() and not path.is_symlink():
                        continue
                    if not path.is_symlink() and (path.suffix in {".safetensors", ".pt", ".bin"} or path.stat().st_size > 1024**3):
                        raise RuntimeError(f"Unexpected model outside declared export: {path}")
                    relative = path.relative_to(SCRATCH / "runs").as_posix()
                    members[relative] = {"symlink": os.readlink(path)} if path.is_symlink() else {"sha256": digest(path), "size": path.stat().st_size}
                    archive.add(path, arcname=relative, recursive=False)
    with tarfile.open(temporary, "r:gz") as archive:
        actual = set()
        for member in archive:
            actual.add(member.name)
            expected = members[member.name]
            if member.issym():
                if member.linkname != expected["symlink"]:
                    raise RuntimeError("Provenance alias mismatch")
            elif member.isfile() or member.islnk():
                with archive.extractfile(member) as f:
                    if hashlib.file_digest(f, "sha256").hexdigest() != expected["sha256"]:
                        raise RuntimeError(f"Provenance digest mismatch: {member.name}")
            else:
                raise RuntimeError(f"Unexpected provenance archive member: {member.name}")
        if actual != set(members):
            raise RuntimeError("Incomplete provenance archive")
    temporary.replace(target)
    atomic(receipt, {"utc": now(), "path": str(target), "sha256": digest(target), "members": members})
    print(f"Provenance backup verified: {target}", flush=True)


def verify_local_files(receipt, allow_removed=False):
    for name, expected in receipt["files"].items():
        path = Path(expected["path"])
        if allow_removed and not path.exists():
            continue
        # NFS st_dev is a node-local mount identifier. Its inode, file size,
        # and nanosecond mtime remain stable when resuming on another node.
        if not path.is_file() or path.is_symlink() or fingerprint(path)[1:] != expected["fingerprint"][1:]:
            raise RuntimeError(f"Local file changed since upload verification: {path}")


def evict_one(api, item):
    source = Path(item["source"])
    if (source.is_symlink() or source.resolve() != source or source.parent.parent not in SOURCES
            or source.name != item["run"]["label"]):
        raise ValueError(f"Deletion scope violation: {source}")
    path = receipt_path(item)
    receipt = read(path)
    if receipt["status"] == "evicted":
        if (source / "hf-final").exists():
            raise RuntimeError(f"An evicted export was recreated: {source}")
        return
    if receipt["status"] not in {"verified", "evicting"}:
        raise RuntimeError(f"No verified upload receipt: {source}")
    with lock_run(source):
        resuming = receipt["status"] == "evicting"
        verify_local_files(receipt, allow_removed=resuming)
        head = verify_remote(api, item, receipt)
        if head != receipt["revision"]:
            raise RuntimeError(f"Remote HEAD changed after verification: {item['repo_id']}")
        export = source / "hf-final"
        if export.is_symlink() or (not export.is_dir() and not resuming) or not checkpoint_metadata_only(source):
            raise RuntimeError(f"Unexpected eviction layout: {source}")
        record = {"repo_id": item["repo_id"], "revision": receipt["revision"],
                  "collection": receipt["collection"], "archive_receipt": str(path),
                  "restore": f"snapshot_download('{item['repo_id']}', revision='{receipt['revision']}')"}
        atomic(source / "uploaded_to_hf.json", record)
        receipt.update(status="evicting", eviction_started_utc=now())
        atomic(path, receipt)
        marker = source / "weights_only_complete.json"
        if marker.exists():
            marker.replace(source / "weights_only_complete.uploaded.json")
        if export.exists():
            shutil.rmtree(export)
        receipt.update(status="evicted", evicted_utc=now())
        atomic(path, receipt)


def evict_all(manifest):
    from huggingface_hub import HfApi
    api = HfApi()
    collections = read(ARCHIVE / "collections.json")
    verify_collections(api, manifest["models"], collections)
    if any(not receipt_path(item).exists() or read(receipt_path(item))["status"] not in {"verified", "evicting", "evicted"}
           for item in manifest["models"]):
        raise RuntimeError("Not all 136 models have verified upload receipts")
    audit_active_sources()
    backup_provenance()
    for count, item in enumerate(manifest["models"], 1):
        evict_one(api, item)
        print(f"Evicted {count}/136: {item['relative_source']}", flush=True)
    # Full logs were preserved and checked in the compressed provenance archive.
    for root in SOURCES:
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(current) / d).is_symlink()]
            for name in files:
                path = Path(current) / name
                if not path.is_symlink() and (name.endswith('.log') or '.log.' in name):
                    path.unlink()
    atomic(ARCHIVE / "completed.json", {"utc": now(), "models": 136,
           "collections": collections, "reclaimed_weight_bytes": sum(item["marker"]["weight_bytes"] for item in manifest["models"]),
           "remaining_local_data": "small metadata, symlink aliases, and pinned Hugging Face restore receipts",
           "free_scratch_gib": shutil.disk_usage(SCRATCH).free / 1024**3})


def main():
    global ROOT, ARCHIVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "upload", "evict", "run", "status"))
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    ROOT, ARCHIVE = args.repo_root.resolve(), args.archive.resolve()
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    with (ARCHIVE / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = inventory()
        if args.command in {"upload", "run"}:
            upload_all(manifest, args.workers)
        if args.command in {"evict", "run"}:
            evict_all(manifest)
        if args.command in {"inventory", "status"}:
            counts = Counter(read(receipt_path(item)).get("status") if receipt_path(item).exists() else "pending"
                             for item in manifest["models"])
            print(json.dumps({"models": len(manifest["models"]), "status": dict(counts)}), flush=True)


if __name__ == "__main__":
    main()
