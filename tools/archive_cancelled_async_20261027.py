"""Archive exactly the two user-selected cancelled dense async runs, then evict verified large files."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import time

from tools import archive_storage_review_20261026 as shared

ROOT = shared.ROOT
OUT = ROOT / 'data/HF-ASYNC-CANCELLED-ARCHIVE-20261027'
SOURCES = {
    'n32': {
        'name': 'data-scaling-async-n32-20261025-003350',
        'repo_id': 'davidheineman/rlve-async-mopd-n32-cancelled-20261027',
        'collection_title': 'RLVE Async MOPD N32 Archive (Oct 27 2026)',
    },
    'n16-replay30': {
        'name': 'data-scaling-async-n16-replay30-20261025-231220',
        'repo_id': 'davidheineman/rlve-async-mopd-n16-replay30-cancelled-20261027',
        'collection_title': 'RLVE Async MOPD N16 Replay 30% Archive (Oct 27 2026)',
    },
}
MIB = 1024**2
LIVE_LOCK = threading.Lock()
LIVE = {}


def allowed_source(source):
    source = Path(source)
    allowed = {shared.SCRATCH / 'runs' / item['name'] for item in SOURCES.values()}
    if source not in allowed or source.is_symlink() or source.resolve() != source:
        raise ValueError(f'Out-of-scope archive source: {source}')


def assert_unused(source, fresh=False):
    allowed_source(source)
    with LIVE_LOCK:
        if fresh or time.monotonic() - LIVE.get('at', -100) > 10:
            LIVE.update(at=time.monotonic(), snapshot=shared.live_inputs())
        shared.assert_unused({'source': str(source), 'payload': str(source)}, LIVE['snapshot'])


def check_record(source, record, missing_ok=False):
    relative = Path(record['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe inventory path')
    path = Path(source) / relative
    if any(parent.is_symlink() for parent in [path, *path.parents]):
        raise ValueError(f'Symlink in file ancestry: {path}')
    if missing_ok and not path.exists():
        return path
    if not path.is_file() or shared.identity(path) != record['identity']:
        raise RuntimeError(f'Source changed since inventory: {path}')
    return path


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def package_provenance(item, inventory):
    directory = OUT / item['key']
    target = directory / 'provenance.tar.gz'
    record_path = directory / 'provenance.json'
    if record_path.exists():
        record = shared.read(record_path)
        if sha256(target) != record['sha256']:
            raise RuntimeError('Staged provenance changed')
        return record
    directory.mkdir(parents=True, exist_ok=True)
    expected = {}
    with tarfile.open(target.with_suffix('.tmp'), 'w:gz', compresslevel=3) as archive:
        for record in inventory['files']:
            if record['identity'][1] >= MIB:
                continue
            path = check_record(inventory['root'], record)
            data = path.read_bytes()
            check_record(inventory['root'], record)
            info = tarfile.TarInfo(record['path'])
            info.size = len(data)
            info.mode = path.stat().st_mode & 0o777
            info.mtime = record['identity'][2] // 10**9
            archive.addfile(info, io.BytesIO(data))
            expected[record['path']] = {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                                        'storage': 'provenance'}
    target.with_suffix('.tmp').replace(target)
    value = {'sha256': sha256(target), 'files': expected}
    shared.atomic(record_path, value)
    return value


def expected_operations(files, prior):
    before = {name: shared.identity(path) for name, path in files.items()}
    operations = shared.commit_operations(files, prior)
    if any(shared.identity(path) != before[name] for name, path in files.items()):
        raise RuntimeError('Source changed while hashing')
    expected = {op.path_in_repo: {
        'path': str(files[op.path_in_repo]), 'identity': before[op.path_in_repo],
        'size': op.upload_info.size, 'sha256': op.upload_info.sha256.hex(),
        'git_sha1': shared.git_sha(files[op.path_in_repo]) if op.upload_info.size < 20 * MIB else None,
    } for op in operations}
    return operations, expected


def commit_files(item, files, receipt_path):
    from huggingface_hub import HfApi
    api = HfApi()
    prior = shared.read(receipt_path) if receipt_path.exists() else {}
    if prior.get('status') in {'verified', 'evicting', 'evicted'}:
        shared.verify_remote(api, item, prior)
        return prior
    operations, expected = expected_operations(files, prior)
    receipt = {'repo_id': item['repo_id'], 'status': 'hashed', 'files': expected}
    shared.atomic(receipt_path, receipt)
    commit = api.create_commit(item['repo_id'], operations=operations, num_threads=2,
                               commit_message='Archive cancelled async MOPD run with resumable state')
    receipt['revision'] = commit.oid
    shared.verify_remote(api, item, receipt)
    shared.verify_local(receipt)
    receipt.update(status='verified', verified_utc=shared.now())
    shared.atomic(receipt_path, receipt)
    return receipt


def evict_records(source, records, receipt):
    """Remove only immutable inventory entries whose remote bytes were verified."""
    allowed_source(source)
    if receipt.get('status') not in {'verified', 'evicting', 'evicted'} or not receipt.get('revision'):
        raise ValueError('No verified pinned remote checkpoint')
    # Validate the whole batch before removing any member.
    paths = []
    for record in records:
        expected = receipt['files'].get('run/' + record['path'])
        path = check_record(source, record, missing_ok=receipt['status'] in {'evicting', 'evicted'})
        if (not expected or expected['path'] != str(path) or expected['identity'] != record['identity']
                or expected['size'] != record['identity'][1]):
            raise ValueError('Inventory does not match verified remote receipt')
        paths.append(path)
    for path in paths:
        path.unlink(missing_ok=True)


def upload_chunk(item, inventory, number, records):
    path = OUT / item['key'] / 'receipts' / f'chunk-{number:04d}.json'
    prior = shared.read(path) if path.exists() else {}
    if prior.get('status') == 'evicted':
        from huggingface_hub import HfApi
        shared.verify_remote(HfApi(), item, prior)
        return prior
    source = Path(inventory['root'])
    assert_unused(source)
    files = {'run/' + record['path']: check_record(source, record,
             missing_ok=prior.get('status') == 'evicting') for record in records}
    receipt = commit_files(item, files, path)
    assert_unused(source, fresh=True)
    for record in records:
        check_record(source, record, missing_ok=receipt['status'] == 'evicting')
    receipt['status'] = 'evicting'
    shared.atomic(path, receipt)
    evict_records(source, records, receipt)
    receipt.update(status='evicted', evicted_utc=shared.now())
    shared.atomic(path, receipt)
    print(f'Verified and evicted {item["key"]} chunk {number}: '
          f'{sum(record["identity"][1] for record in records) / 1024**3:.2f} GiB', flush=True)
    return receipt


def prepare_item(key, config):
    from huggingface_hub import HfApi
    item = {**config, 'key': key}
    directory = OUT / key
    directory.mkdir(parents=True, exist_ok=True)
    inventory = shared.read(OUT / (item['name'] + '.inventory.json'))
    allowed_source(inventory['root'])
    if inventory['sensitive_filenames']:
        raise RuntimeError('Credential filenames found; refusing upload')
    if any(not Path(link['target']).is_relative_to(inventory['root']) for link in inventory['symlinks']):
        raise ValueError('Unexpected external source symlink')
    assert_unused(inventory['root'], fresh=True)
    stop = shared.read(Path(inventory['root']) / 'user-stop-20261027.json')
    if stop['status'] != 'cancelled_by_user':
        raise ValueError('Run was not explicitly cancelled by the user')
    api = HfApi()
    collection_path = directory / 'collection.json'
    if collection_path.exists():
        collection = shared.read(collection_path)
        actual = api.get_collection(collection['slug'])
        if actual.title != item['collection_title'] or actual.private:
            raise ValueError('Collection changed')
    else:
        actual = api.create_collection(item['collection_title'], namespace='davidheineman', private=False,
            exists_ok=True, description='Cancelled async MOPD student, retained teachers, and verified full-run recovery state.')
        collection = {'title': actual.title, 'slug': actual.slug, 'url': actual.url}
        shared.atomic(collection_path, collection)
    api.create_repo(item['repo_id'], repo_type='model', private=False, exist_ok=True)
    api.add_collection_item(collection['slug'], item['repo_id'], 'model', exists_ok=True)
    item['collection'] = collection
    provenance = package_provenance(item, inventory)
    inventory_path = directory / 'archive-inventory.json'
    if not inventory_path.exists():
        shared.atomic(inventory_path, inventory)
    card = directory / 'README.md'
    card.write_text('---\nbase_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B\n'
        'tags:\n- rlve\n- mopd\n- reinforcement-learning\n- resumable-checkpoint\n---\n\n'
        f'# Cancelled async MOPD: {key}\n\n'
        f'This run was stopped by its owner to release GPUs. Its latest committed student has '
        f'{inventory["student_snapshot"]["rollouts"]:,} rollouts; it did not finish its 16,000-rollout budget. '
        'The model files at the repository root are that student in Hugging Face format.\n\n'
        'The full retained run is archived under `run/` for large files and `provenance.tar.gz` '
        'for small files. It includes all retained teacher/student exports, optimizer checkpoints, '
        'replay data when present, scheduler state, data-source/RNG state when saved by training, '
        'frozen code, logs, and parent conversions. `archive_manifest.json` marks a fully verified archive.\n\n'
        'To restore, download `restore_archive.py` and run:\n\n```bash\n'
        f'python restore_archive.py --repo-id {item["repo_id"]} --revision COMMIT '
        f'--destination {inventory["root"]}\n```\n\n'
        'Use the pinned archive completion commit, install `huggingface_hub`, and restore to the original '
        'path to preserve absolute scheduler paths. This command verifies every restored file and launches no jobs. '
        'The base DeepSeek model and training container are external prerequisites listed in the run configuration. '
        'Unfinished teacher branches can replay from their saved parents; only committed training state is resumable.\n')
    restore = directory / 'restore_archive.py'
    if not restore.exists():
        shutil.copy2(ROOT / 'tools/restore_cancelled_async_archive.py', restore)
    files = {'README.md': card, 'provenance.tar.gz': directory / 'provenance.tar.gz',
             'archive-inventory.json': inventory_path, 'restore_archive.py': restore}
    hf = Path(inventory['student_snapshot']['hf'])
    records = {record['path']: record for record in inventory['files']}
    for path in hf.iterdir():
        if path.is_file():
            record = records[str(path.relative_to(inventory['root']))]
            files[path.name] = check_record(inventory['root'], record,
                missing_ok=(directory / 'primary.json').exists())
    primary = commit_files(item, files, directory / 'primary.json')
    shared.atomic(Path(inventory['root']) / 'storage_archive_restore.json', {
        'status': 'archiving', 'repo_id': item['repo_id'], 'collection': collection,
        'primary_revision': primary['revision'], 'receipts': str(directory / 'receipts'),
        'original_root': inventory['root'], 'automatic_resume_disabled': True})
    print(f'Archive collection ready: {collection["url"]}', flush=True)
    return item, inventory, provenance


def finalize(item, inventory, provenance):
    from huggingface_hub import HfApi
    directory = OUT / item['key']
    receipts = [shared.read(path) for path in sorted((directory / 'receipts').glob('chunk-*.json'))]
    records = dict(provenance['files'])
    for receipt in receipts:
        if receipt['status'] != 'evicted':
            raise ValueError('Chunk is not complete')
        for name, record in receipt['files'].items():
            records[name.removeprefix('run/')] = {'size': record['size'], 'sha256': record['sha256'], 'storage': 'run'}
    if set(records) != {record['path'] for record in inventory['files']}:
        raise ValueError('Final archive inventory mismatch')
    manifest = {'status': 'complete', 'utc': shared.now(), 'repo_id': item['repo_id'],
        'source': inventory['root'], 'student_snapshot': inventory['student_snapshot'],
        'files': records, 'symlinks': inventory['symlinks'], 'provenance_sha256': provenance['sha256'],
        'chunk_revisions': [receipt['revision'] for receipt in receipts],
        'collection': item['collection'], 'external_prerequisites': 'Original base HF model, base torch-dist conversion, and the container listed in archived launch scripts.'}
    path = directory / 'archive_manifest.json'
    shared.atomic(path, manifest)
    final = commit_files(item, {'archive_manifest.json': path}, directory / 'final.json')
    api = HfApi()
    expected = {}
    for receipt in [shared.read(directory / 'primary.json'), *receipts, final]:
        expected.update(receipt['files'])
    # A single pinned tree must contain every uploaded file, including chunks
    # committed concurrently to the same repository.
    shared.verify_remote(api, item, {'revision': final['revision'], 'files': expected})
    if item['repo_id'] not in {value.item_id for value in api.get_collection(item['collection']['slug']).items}:
        raise RuntimeError('Archive missing from collection')
    # Recheck every retained provenance file; large files are deliberately gone.
    for record in inventory['files']:
        if record['identity'][1] < MIB:
            check_record(inventory['root'], record)
    complete = {'utc': shared.now(), 'repo_id': item['repo_id'], 'revision': final['revision'],
        'collection': item['collection'], 'files_verified': len(records),
        'large_files_evicted': sum(record['storage'] == 'run' for record in records.values()),
        'logical_gib_evicted': sum(record['size'] for record in records.values() if record['storage'] == 'run') / 1024**3,
        'restore_command': f'python restore_archive.py --repo-id {item["repo_id"]} --revision {final["revision"]} --destination {inventory["root"]}',
        'local_metadata_retained': True, 'automatic_resume_disabled': True}
    shared.atomic(directory / 'completed.json', complete)
    shared.atomic(Path(inventory['root']) / 'storage_archive_restore.json', complete)
    print('ARCHIVE COMPLETE ' + json.dumps(complete), flush=True)
    return complete


def run(workers, chunk_size):
    OUT.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        writer = stack.enter_context((OUT / 'writer.lock').open('a'))
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for config in SOURCES.values():
            source = shared.SCRATCH / 'runs' / config['name']
            allowed_source(source)
            for name in ('run.lock', 'monitor.lock', 'archive-storage.lock'):
                lock = stack.enter_context((source / name).open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepared = [prepare_item(key, value) for key, value in SOURCES.items()]
        jobs = []
        for item, inventory, provenance in prepared:
            records = sorted((record for record in inventory['files'] if record['identity'][1] >= MIB),
                             key=lambda record: record['path'])
            chunks = [records[index:index + chunk_size] for index in range(0, len(records), chunk_size)]
            for number, chunk in enumerate(chunks):
                jobs.append((item, inventory, number, chunk))
        def priority(job):
            item, inventory, number, records = job
            snapshot = inventory['student_snapshot']
            source = Path(inventory['root'])
            paths = [source / record['path'] for record in records]
            if any(str(path) == snapshot.get('replay') or path.is_relative_to(snapshot['resume']) for path in paths):
                return (0, number, item['key'])
            if any(path.is_relative_to(snapshot['hf']) for path in paths):
                return (1, number, item['key'])
            # Rebuildable parent conversions follow the retained task models.
            return (3 if all(record['path'].startswith('parents/') for record in records) else 2,
                    number, item['key'])
        jobs.sort(key=priority)
        errors = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(upload_chunk, *job): job for job in jobs}
            for count, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()
                except Exception as error:
                    item, _, number, _ = futures[future]
                    failure = {'key': item['key'], 'chunk': number, 'error': f'{type(error).__name__}: {error}'}
                    errors.append(failure)
                    print('FAILED ' + json.dumps(failure), flush=True)
                shared.atomic(OUT / 'progress.json', {'utc': shared.now(), 'chunks_finished': count,
                    'chunks_total': len(jobs), 'errors': errors,
                    'free_scratch_gib': shutil.disk_usage(shared.SCRATCH).free / 1024**3})
        if errors:
            raise RuntimeError(f'{len(errors)} archive chunks failed; all unverified files retained')
        completed = [finalize(*item) for item in prepared]
        shared.atomic(OUT / 'completed.json', {'utc': shared.now(), 'runs': completed,
            'free_scratch_gib': shutil.disk_usage(shared.SCRATCH).free / 1024**3})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=8)
    args = parser.parse_args()
    run(args.workers, args.chunk_size)
