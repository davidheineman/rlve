"""Execute the approved Oct 26 storage review, with pinned HF restore receipts.

Only listed payload files are evicted. Logs, lineage, live experiments and the
original completed N16 async baseline remain local.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

ROOT = Path('/juice5b/u/dhei/tatsu/rlve')
SCRATCH = Path('/juice5b/scr5b/dhei')
OUT = ROOT / 'data/HF-STORAGE-ARCHIVE-20261026'
BASELINE = SCRATCH / 'runs/data-scaling-async-n16-20261023-171338'
TITLES = {
    'legacy_teachers': 'RLVE Legacy Teachers (Oct 26 2026)',
    'legacy_students': 'RLVE Legacy Students (Oct 26 2026)',
    'multiround_teachers': 'RLVE Multi-round Teachers (Oct 26 2026)',
    'multiround_students': 'RLVE Multi-round Students (Oct 26 2026)',
}
COLLECTION_LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def identity(path):
    value = Path(path).stat()
    # st_dev is a node-local NFS mount identifier, not a portable file identity.
    return [value.st_ino, value.st_size, value.st_mtime_ns]


def role(item):
    source = item['source']
    teacher = ('/teacher/' in source or '-teachers-' in source
               or 'p1r8-teachers' in source or '/fast-r1-' in source)
    prefix = 'multiround_' if item['category'] == 'Completed multi-round R2 models' else 'legacy_'
    return prefix + ('teachers' if teacher else 'students')


def check_scope(item):
    source, payload = Path(item['source']), Path(item['payload'])
    allowed = (SCRATCH / 'runs', SCRATCH / 'nano-omni-multiteacher/runs-inloop-300000000-2000000')
    if (source.is_symlink() or payload.is_symlink()
            or not any(source.resolve().is_relative_to(root) for root in allowed)
            or source.resolve().is_relative_to(BASELINE)
            or not payload.resolve().is_relative_to(source.resolve())
            or any(part.startswith(('data-scaling-async-', 'async-lora-')) for part in source.parts)):
        raise ValueError(f'Out-of-scope payload: {source} / {payload}')
    if item['repo_id'].split('/')[0] != 'davidheineman':
        raise ValueError('Unexpected HF namespace')


def payload_files(item):
    source, payload = Path(item['source']), Path(item['payload'])
    check_scope(item)
    prefix = f"checkpoint/iter_{item['checkpoint_step']:07d}/" if item['kind'] == 'megatron-torch-dist' else ''
    files = {}
    for directory, dirs, names in os.walk(payload, followlinks=False):
        dirs[:] = [name for name in dirs if name not in {'.cache', '__pycache__', 'wandb'}]
        if any((Path(directory) / name).is_symlink() for name in dirs):
            raise ValueError('Symlink directory in archive payload')
        for name in names:
            local = Path(directory) / name
            if local.parent == source and name in {'archive-storage.lock', 'storage_archive_restore.json'}:
                continue
            if local.is_symlink() or not local.is_file():
                raise ValueError(f'Nonregular payload file: {local}')
            files[prefix + local.relative_to(payload).as_posix()] = local
    if item['kind'] == 'megatron-torch-dist':
        for name in ('latest_checkpointed_iteration.txt', 'rlve_checkpoint_state.json'):
            local = source / 'ckpt' / name
            if local.is_file():
                files['checkpoint/' + name] = local
    if not any(path.stat().st_size > 100_000_000 for path in files.values()):
        raise ValueError(f'No substantial model payload: {payload}')
    return files


def prepare():
    path = OUT / 'inventory.json'
    if path.exists():
        value = read(path)
        changed = []
        for item in value['items']:
            normalized = re.sub(r'-+', '-', item['repo_id'])
            if normalized != item['repo_id']:
                changed.append({'archive_id': item['id'], 'previous': item['repo_id'], 'repo_id': normalized})
                item['repo_id'] = normalized
        if changed:
            atomic(OUT / 'repository-name-repairs.json', {'utc': now(), 'repairs': changed})
            atomic(path, value)
        return value
    plan = read(ROOT / 'data/STORAGE-REVIEW-20261026/archive-plan.json')
    old = {row['id']: row for row in read(ROOT / 'data/HF-SCRATCH-ARCHIVE-20261004/manifest.json')['runs']}
    items = []
    for original in plan['items']:
        item = dict(original)
        if item['id'] in old:
            for key in ('checkpoint_step', 'relative_source'):
                item[key] = old[item['id']].get(key)
        item.setdefault('checkpoint_step', item.get('marker', {}).get('checkpoint_step'))
        item.setdefault('relative_source', str(Path(item['source']).relative_to(SCRATCH)))
        item['role'] = role(item)
        item['repo_id'] = re.sub(r'-+', '-', item['repo_id'])
        files = payload_files(item)
        item['initial_files'] = {name: {'path': str(local), 'identity': identity(local)} for name, local in files.items()}
        item['allocated_bytes'] = sum(local.stat().st_blocks * 512 for local in files.values())
        items.append(item)
    if len(items) != 120 or len({item['payload'] for item in items}) != 120:
        raise ValueError('Approved inventory changed')
    value = {'created_utc': now(), 'approved_by_user': True, 'collections': TITLES, 'items': items,
             'approval': 'Archive and verify the reviewed payloads, remove verified local copies, clear reviewed rebuildable caches; use new collections and protect running experiments.'}
    atomic(path, value)
    return value


def live_inputs():
    # Expand task arrays: compressed IDs such as 123_[0-3%1] are not accepted
    # by scontrol show job and must not disable the live-input safety audit.
    queue_command = ['squeue', '--array', '-u', os.environ.get('USER', 'dhei'), '-h', '-o', '%i']
    raw = subprocess.check_output(queue_command, text=True)
    jobs = raw.split()
    details = []
    for job in jobs:
        result = subprocess.run(['scontrol', 'show', 'job', '-o', job], capture_output=True, text=True)
        if result.returncode:
            # A job may finish between the scheduler snapshot and the detail
            # query. Ignore only IDs confirmed absent from a fresh snapshot.
            if job not in subprocess.check_output(queue_command, text=True).split():
                continue
            raise RuntimeError(f'Unable to audit live job {job}: {result.stderr.strip()}')
        details.append(result.stdout)
    documents = []
    for experiment in (SCRATCH / 'runs').glob('*async*'):
        if not (experiment / 'plan.json').is_file():
            continue
        ids = set()
        for filename in ('job.json', 'submission.json', 'monitor.json'):
            path = experiment / filename
            if path.is_file():
                record = read(path)
                ids.update(str(record[k]) for k in ('job_id', 'job', 'coordinator_job', 'monitor_job') if record.get(k))
        journal = experiment / 'slurm_jobs.json'
        if journal.is_file():
            ids.update(str(record['job']) for record in read(journal).values())
        if not ids.intersection(jobs):
            continue
        paths = [experiment / 'plan.json']
        if (experiment / 'state.json').exists():
            state = read(experiment / 'state.json')
            paths.append(experiment / 'state.json')
            for device in state.get('devices', []):
                if device.get('active'):
                    path = experiment / 'tasks' / device['active']['id'] / 'task.json'
                    if path.is_file():
                        paths.append(path)
        request = experiment / 'student-service/request.json'
        if request.is_file():
            paths.append(Path(read(request)['task']))
        documents.extend((str(path), path.read_text()) for path in paths)
    return {'utc': now(), 'jobs': jobs, 'details': details, 'documents': documents}


def assert_unused(item, snapshot=None):
    snapshot = snapshot or live_inputs()
    source = item['source']
    if any(source in value for value in snapshot['details']):
        raise RuntimeError(f'Live Slurm job references {source}')
    if any(source in content or item['payload'] in content for path, content in snapshot['documents']):
        raise RuntimeError(f'Live experiment input references {source}')
    return snapshot


@contextmanager
def source_lock(item):
    check_scope(item)
    with (Path(item['source']) / 'archive-storage.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def ensure_collections(api):
    path = OUT / 'collections.json'
    saved = read(path) if path.exists() else {}
    for key, title in TITLES.items():
        if key not in saved:
            collection = api.create_collection(title, namespace='davidheineman', private=False, exists_ok=True,
                description='Verified RLVE checkpoints from the Oct 26 storage review, with source provenance and pinned restore revisions.')
            saved[key] = {'title': title, 'slug': collection.slug, 'url': collection.url}
            atomic(path, saved)
        else:
            collection = api.get_collection(saved[key]['slug'])
            if collection.title != title or collection.private:
                raise RuntimeError('Collection metadata changed')
    return saved


def git_sha(path):
    digest = hashlib.sha1()
    digest.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        while data := stream.read(8 * 1024**2):
            digest.update(data)
    return digest.hexdigest()


def metadata(item):
    directory = OUT / 'metadata' / item['id']
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(item['source'])
    wandb = (source / 'wandb_run_id.txt').read_text().strip() if (source / 'wandb_run_id.txt').is_file() else None
    card = ('---\ntags:\n- rlve\n- checkpoint-archive\n---\n\n'
            f"# RLVE archived checkpoint: {source.name}\n\n"
            f"Experiment group: `{item['relative_source']}`.\n\n"
            f"Archive category: {item['category']}. Format: `{item['kind']}`. "
            f"Checkpoint step: `{item.get('checkpoint_step')}`. W&B run ID: `{wandb}`.\n\n"
            'The original model configuration and tokenizer accompany HF exports. '
            'Distributed checkpoints retain their saved optimizer/RNG state when present; '
            'restore the complete `checkpoint/` directory before resuming. '
            'This archive preserves the exact reviewed checkpoint, including older stopped experiments. '
            'Training logs and local lineage metadata are retained at the source.\n')
    (directory / 'README.md').write_text(card)
    atomic(directory / 'archive_manifest.json', {
        'archive_id': item['id'], 'relative_source': item['relative_source'], 'kind': item['kind'],
        'category': item['category'], 'role': item['role'], 'checkpoint_step': item.get('checkpoint_step'),
        'wandb_run_id': wandb, 'archive_date': '2026-10-26'})
    return {path.name: path for path in directory.iterdir() if path.is_file()}


def verify_remote(api, item, receipt):
    info = api.model_info(item['repo_id'], revision=receipt.get('revision'), files_metadata=True)
    siblings = {value.rfilename: value for value in info.siblings or []}
    for name, expected in receipt['files'].items():
        remote = siblings.get(name)
        if remote is None or remote.size != expected['size']:
            raise ValueError(f'Remote file size mismatch: {item["repo_id"]}/{name}')
        if remote.lfs is not None:
            if remote.lfs.sha256 != expected['sha256']:
                raise ValueError(f'Remote SHA256 mismatch: {item["repo_id"]}/{name}')
        elif expected['git_sha1'] is None or remote.blob_id != expected['git_sha1']:
            raise ValueError(f'Remote Git blob mismatch: {item["repo_id"]}/{name}')
    return info.sha


def verify_local(receipt, allow_removed=False):
    for name, expected in receipt['files'].items():
        path = Path(expected['path'])
        if allow_removed and not path.exists():
            continue
        if path.is_symlink() or not path.is_file() or identity(path) != expected['identity']:
            raise RuntimeError(f'Local file changed since hashing: {path}')


def commit_operations(files, prior):
    """Reuse completed SHA256 scans only when portable file identity is intact."""
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.lfs import UploadInfo
    operations = []
    for name, local in sorted(files.items()):
        cached = prior.get('files', {}).get(name, {})
        if (cached.get('path') == str(local) and cached.get('identity') == identity(local)
                and cached.get('size', 0) > 1024**2):
            # The pinned hub release exposes these operation fields. Construct
            # its normal default state, then supply the already-scanned file.
            operation = CommitOperationAdd(path_in_repo=name, path_or_fileobj=b'')
            operation.path_or_fileobj = str(local)
            with local.open('rb') as stream:
                sample = stream.read(512)
            operation.upload_info = UploadInfo(size=cached['size'],
                sha256=bytes.fromhex(cached['sha256']), sample=sample)
        else:
            operation = CommitOperationAdd(path_in_repo=name, path_or_fileobj=local)
        operations.append(operation)
    return operations


def retained_payload_metadata(item, receipt):
    """Keep inventoried source metadata replaced by generated remote model cards."""
    retained = set()
    payload = Path(item['payload'])
    for name in ('README.md', 'archive_manifest.json'):
        original = item.get('initial_files', {}).get(name)
        uploaded = receipt['files'].get(name)
        if not original or not uploaded or original['path'] == uploaded['path']:
            continue
        path = Path(original['path'])
        if path != payload / name:
            raise RuntimeError(f'Unexpected source metadata path: {path}')
        if path.is_symlink() or not path.is_file() or identity(path) != original['identity']:
            raise RuntimeError(f'Original metadata changed since inventory: {path}')
        retained.add(path)
    return retained


def evict_payload(item, receipt):
    check_scope(item)
    verify_local(receipt, allow_removed=receipt['status'] == 'evicting')
    payload = Path(item['payload'])
    expected = {Path(value['path']) for value in receipt['files'].values()}
    # Generated cards can occupy the same remote filename as an original card.
    # Retain those exact inventoried originals locally, without relaxing the
    # rejection of new files or deleting any file absent from the remote receipt.
    expected.update(retained_payload_metadata(item, receipt))
    # Refuse deletion if any unrecorded file appeared under the approved payload.
    for directory, dirs, files in os.walk(payload, followlinks=False):
        if any((Path(directory) / name).is_symlink() for name in dirs):
            raise RuntimeError('Symlink appeared in payload')
        for name in files:
            path = Path(directory) / name
            if path.parent == Path(item['source']) and name in {'archive-storage.lock', 'storage_archive_restore.json'}:
                continue
            if path not in expected:
                raise RuntimeError(f'Unrecorded payload file: {path}')
    for value in receipt['files'].values():
        path = Path(value['path'])
        if not path.is_relative_to(payload):
            continue  # Preserve iteration pointers, transactional state and staged provenance.
        if item['kind'] == 'pytorch-checkpoint' and path.suffix not in {'.pt', '.bin', '.safetensors'}:
            continue  # Nano payload equals the run directory; keep its JSON results locally.
        path.unlink(missing_ok=True)
    if item['kind'] != 'pytorch-checkpoint':
        for directory, dirs, files in os.walk(payload, topdown=False):
            if not list(Path(directory).iterdir()):
                Path(directory).rmdir()


def upload_one(item, collections):
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    path = OUT / 'receipts' / (item['id'] + '.json')
    receipt = read(path) if path.exists() else {}
    if receipt.get('status') == 'evicted':
        verify_remote(api, item, receipt)
        return item['repo_id']
    with source_lock(item):
        assert_unused(item)
        if receipt.get('status') not in {'verified', 'evicting'}:
            files = payload_files(item)
            current = {name: {'path': str(local), 'identity': identity(local)} for name, local in files.items()}
            if current != item['initial_files']:
                raise RuntimeError(f'Payload changed since approved inventory: {item["source"]}')
            files.update(metadata(item))
            before = {name: identity(local) for name, local in files.items()}
            print(f'Hash/upload: {item["repo_id"]}', flush=True)
            operations = commit_operations(files, receipt)
            if any(identity(local) != before[name] for name, local in files.items()):
                raise RuntimeError('Source changed while hashing')
            receipt = {'status': 'hashed', 'source': item['source'], 'payload': item['payload'],
                'repo_id': item['repo_id'], 'role': item['role'], 'allocated_bytes': item['allocated_bytes'],
                'collection': collections[item['role']]['slug'],
                'files': {op.path_in_repo: {'path': str(files[op.path_in_repo]), 'identity': before[op.path_in_repo],
                    'size': op.upload_info.size, 'sha256': op.upload_info.sha256.hex(),
                    'git_sha1': git_sha(files[op.path_in_repo]) if op.upload_info.size < 20 * 1024**2 else None}
                    for op in operations}}
            atomic(path, receipt)
            api.create_repo(item['repo_id'], repo_type='model', private=False, exist_ok=True)
            commit = api.create_commit(item['repo_id'], operations=operations, num_threads=2,
                commit_message='Archive reviewed RLVE checkpoint and restore provenance')
            receipt['revision'] = commit.oid
            verify_remote(api, item, receipt)
            verify_local(receipt)
            receipt.update(status='verified', verified_utc=now())
            atomic(path, receipt)
        verify_remote(api, item, receipt)
        with COLLECTION_LOCK:
            api.add_collection_item(receipt['collection'], item['repo_id'], 'model', exists_ok=True)
            members = {value.item_id for value in api.get_collection(receipt['collection']).items}
            if item['repo_id'] not in members:
                raise RuntimeError('Collection membership missing')
        assert_unused(item)
        verify_local(receipt, allow_removed=receipt['status'] == 'evicting')
        marker = {'repo_id': item['repo_id'], 'revision': receipt['revision'], 'collection': receipt['collection'],
            'archive_receipt': str(path), 'payload': item['payload'], 'kind': item['kind'],
            'files': {name: {'size': value['size'], 'sha256': value['sha256']} for name, value in receipt['files'].items()},
            'verified_utc': receipt['verified_utc']}
        marker['retained_local_metadata'] = sorted(str(value) for value in retained_payload_metadata(item, receipt))
        atomic(Path(item['source']) / 'storage_archive_restore.json', marker)
        receipt.update(status='evicting', eviction_started_utc=now())
        atomic(path, receipt)
        evict_payload(item, receipt)
        receipt.update(status='evicted', evicted_utc=now())
        atomic(path, receipt)
        return item['repo_id']


def run(workers):
    from huggingface_hub import HfApi
    with (OUT / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory = prepare()
        snapshot = live_inputs()
        for item in inventory['items']:
            assert_unused(item, snapshot)
        atomic(OUT / 'initial-active-source-audit.json', snapshot)
        collections = ensure_collections(HfApi())
        # Largest completed RL checkpoints first; verified eviction makes space incrementally.
        items = sorted(inventory['items'], key=lambda item: -item['allocated_bytes'])
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(upload_one, item, collections): item for item in items}
            for count, future in enumerate(as_completed(futures), 1):
                item = futures[future]
                try:
                    future.result()
                    print(f'Archived and evicted {count}/{len(items)}: {item["repo_id"]}', flush=True)
                except Exception as error:
                    failure = {'repo_id': item['repo_id'], 'error': f'{type(error).__name__}: {error}'}
                    failures.append(failure)
                    print('FAILED ' + json.dumps(failure), flush=True)
                atomic(OUT / 'progress.json', {'utc': now(), 'finished': count, 'total': len(items),
                    'failures': failures, 'free_scratch_gib': shutil.disk_usage(SCRATCH).free / 1024**3})
        # Reconcile sequentially to avoid lost updates from parallel collection modifications.
        api = HfApi()
        for key, collection in collections.items():
            expected = {item['repo_id'] for item in items if item['role'] == key
                        and (OUT / 'receipts' / (item['id'] + '.json')).is_file()
                        and read(OUT / 'receipts' / (item['id'] + '.json'))['status'] == 'evicted'}
            actual = {value.item_id for value in api.get_collection(collection['slug']).items}
            for repo in expected - actual:
                api.add_collection_item(collection['slug'], repo, 'model', exists_ok=True)
            actual = {value.item_id for value in api.get_collection(collection['slug']).items}
            if not expected <= actual:
                raise RuntimeError('Collection reconciliation incomplete')
        if failures:
            atomic(OUT / 'failures.json', failures)
            raise RuntimeError(f'{len(failures)} payloads need retry; their local files were retained')
        atomic(OUT / 'completed.json', {'utc': now(), 'models': len(items), 'collections': collections,
            'archived_and_evicted_allocated_gib': sum(item['allocated_bytes'] for item in items) / 1024**3,
            'free_scratch_gib': shutil.disk_usage(SCRATCH).free / 1024**3})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == 'prepare':
        print(json.dumps({'models': len(prepare()['items']), 'output': str(OUT)}, indent=2))
    else:
        run(args.workers)


if __name__ == '__main__':
    main()
