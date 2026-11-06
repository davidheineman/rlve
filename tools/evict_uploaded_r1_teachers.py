"""Remove local copies of the already uploaded R1 teacher weights after SHA-256 verification."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi


GROUP = "opd-teachers-r1-nofilter16-20260929-231458"
COLLECTION = "davidheineman/rlve-opd-teachers-r1-distill-6abd28da5a830a9636eb3b84"
RUNS = Path("/juice5b/scr5b/dhei/runs")
ENVIRONMENTS = Path("scripts/slurm/sweeps/pairwise32.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evict", action="store_true", help="Remove verified local HF exports")
    args = parser.parse_args()

    root = RUNS / GROUP / "resumable"
    environments = ENVIRONMENTS.read_text().split()
    if len(environments) != 32 or len(set(environments)) != 32:
        raise RuntimeError("Expected 32 distinct original teacher environments")
    expected = {
        f"davidheineman/opd-teacher-R1Distill-{environment}-step149"
        for environment in environments
    }
    api = HfApi()
    found = {item.item_id for item in api.get_collection(COLLECTION).items if item.item_type == "model"}
    if found != expected:
        raise RuntimeError(f"Remote collection mismatch: missing={expected - found}, extra={found - expected}")

    reclaimed = 0
    for index, environment in enumerate(environments):
        run = root / f"{index:02d}-{environment}"
        export = run / "hf-final"
        backup = run / "uploaded_to_hf.json"
        marker = run / "weights_only_complete.json"
        archived_marker = run / "weights_only_complete.uploaded.json"
        repo = f"davidheineman/opd-teacher-R1Distill-{environment}-step149"
        if backup.exists() and archived_marker.exists():
            record = json.loads(backup.read_text())
            info = api.model_info(repo, files_metadata=True)
            sibling = next((item for item in info.siblings if item.rfilename == record["weight_file"]), None)
            if record["repo_id"] != repo or sibling is None or sibling.lfs is None or sibling.lfs.sha256 != record["weight_sha256"]:
                raise RuntimeError(f"Recorded remote backup no longer matches: {repo}")
            if args.evict and export.exists():
                if export.is_symlink() or export.resolve().parent != run.resolve():
                    raise RuntimeError(f"Unexpected local export layout: {run}")
                shutil.rmtree(export)
            print(f"{run.name}: remote backup verified" + ("; local copy absent" if not export.exists() else ""), flush=True)
            continue
        if not export.exists():
            if not backup.exists():
                raise RuntimeError(f"Missing local export without backup record: {run}")
            print(f"{run.name}: already evicted", flush=True)
            continue
        if export.is_symlink() or export.resolve().parent != run.resolve() or (run / "ckpt").exists():
            raise RuntimeError(f"Unexpected local export layout: {run}")
        weights = list(export.glob("model-*.safetensors"))
        if len(weights) != 1:
            raise RuntimeError(f"Expected one weight file: {run}")
        weight = weights[0]
        metadata = json.loads((marker if marker.exists() else archived_marker).read_text())
        if metadata.get("format") != "hf-safetensors" or metadata.get("weight_bytes") != weight.stat().st_size:
            raise RuntimeError(f"Local export marker mismatch: {run}")
        info = api.model_info(repo, files_metadata=True)
        sibling = next((item for item in info.siblings if item.rfilename == weight.name), None)
        with weight.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if sibling is None or sibling.lfs is None or sibling.lfs.sha256 != digest:
            raise RuntimeError(f"Remote SHA-256 mismatch: {repo}")
        if args.evict:
            record = {"collection": COLLECTION, "repo_id": repo, "revision": info.sha,
                      "weight_file": weight.name, "weight_sha256": digest,
                      "weight_bytes": weight.stat().st_size}
            temporary = run / "uploaded_to_hf.json.tmp"
            temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
            temporary.replace(backup)
            if marker.exists():
                marker.replace(archived_marker)
            reclaimed += sum(path.stat().st_size for path in export.rglob("*") if path.is_file())
            shutil.rmtree(export)
        print(f"{run.name}: verified {repo}" + ("; removed local weights" if args.evict else ""), flush=True)
    print(f"{'Reclaimed' if args.evict else 'Verified'} {reclaimed / 1024**3:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
