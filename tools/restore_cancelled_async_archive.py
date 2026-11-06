"""Restore a verified cancelled async run; this command never launches jobs."""

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError(f'Unsafe archive path: {relative}')
    path = Path(root).joinpath(*relative.parts)
    if any(parent.is_symlink() for parent in [path, *path.parents] if parent != Path(root).parent):
        raise ValueError(f'Symlink in destination path: {relative}')
    return path


def verify(path, expected):
    return (path.is_file() and not path.is_symlink()
            and path.stat().st_size == expected['size'] and sha256(path) == expected['sha256'])


def extract_provenance(package, manifest, destination):
    expected = {name: value for name, value in manifest['files'].items() if value['storage'] == 'provenance'}
    seen = set()
    with tarfile.open(package, 'r:gz') as archive:
        for member in archive:
            if member.name not in expected or not member.isfile() or member.name in seen:
                raise ValueError(f'Unexpected provenance member: {member.name}')
            record = expected[member.name]
            if member.size != record['size']:
                raise ValueError(f'Provenance size mismatch: {member.name}')
            path = safe_path(destination, member.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Locks are coordination objects, not training data. Keep the inode
            # whose flock the restore process currently holds.
            if member.name in {'run.lock', 'monitor.lock', 'archive-storage.lock'} and path.exists():
                if not verify(path, record):
                    raise ValueError(f'Lock contents changed: {member.name}')
                seen.add(member.name)
                continue
            temporary = path.with_name(path.name + '.archive-restore.tmp')
            if temporary.is_symlink():
                raise ValueError('Unsafe restore temporary file')
            with archive.extractfile(member) as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target)
            if not verify(temporary, record):
                temporary.unlink()
                raise ValueError(f'Provenance checksum mismatch: {member.name}')
            temporary.chmod(member.mode)
            os.replace(temporary, path)
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('Missing provenance files')


def restore_links(manifest, destination):
    original = Path(manifest['source'])
    destination = Path(destination)
    for record in manifest['symlinks']:
        target = Path(record['target'])
        if not target.is_absolute() or not target.is_relative_to(original):
            raise ValueError('Unexpected external archive symlink')
        relative = PurePosixPath(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe symlink name')
        # Validate ancestors separately: this leaf is intentionally a symlink.
        path = safe_path(destination, relative.parent) / relative.name
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = str(destination / target.relative_to(original))
        if path.is_symlink():
            if os.readlink(path) != expected:
                raise ValueError(f'Existing symlink target differs: {path}')
        elif path.exists():
            raise ValueError(f'Existing non-symlink at archive link: {path}')
        else:
            path.symlink_to(expected, target_is_directory=record['directory'])


def restore(repo_id, revision, destination):
    from huggingface_hub import hf_hub_download
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for name in ('run.lock', 'monitor.lock', 'archive-storage.lock'):
            lock = stack.enter_context((destination / name).open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        staging = destination / '.hf-archive-restore'
        staging.mkdir(exist_ok=True)
        def download(name):
            return Path(hf_hub_download(repo_id, name, revision=revision, local_dir=staging))
        manifest = json.loads(download('archive_manifest.json').read_text())
        if manifest['status'] != 'complete' or manifest['repo_id'] != repo_id:
            raise ValueError('Archive is incomplete or belongs to another repository')
        for name, record in manifest['files'].items():
            if record['storage'] != 'run':
                continue
            path = safe_path(destination, name)
            if verify(path, record):
                continue
            cached = download('run/' + name)
            if not verify(cached, record):
                raise ValueError(f'Download checksum mismatch: {name}')
            path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(cached, path)
        package = download('provenance.tar.gz')
        if sha256(package) != manifest['provenance_sha256']:
            raise ValueError('Provenance package checksum mismatch')
        extract_provenance(package, manifest, destination)
        restore_links(manifest, destination)
        for name, record in manifest['files'].items():
            if not verify(safe_path(destination, name), record):
                raise ValueError(f'Restored file failed final verification: {name}')
        if str(destination) != manifest['source']:
            print('Restored to a different location; update absolute paths in plan/state/task JSON before training.')
        print(f'Restored and verified {len(manifest["files"])} files at {destination}; no jobs launched.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    restore(args.repo_id, args.revision, args.destination)
