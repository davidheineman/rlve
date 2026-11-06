"""Clear only approved rebuildable caches, protecting live teacher parents."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tools.archive_storage_review_20261026 import OUT, ROOT, SCRATCH, atomic, read, now

BUILD_CACHES = (
    SCRATCH / 'cache/apptainer-miles',
    SCRATCH / 'nano-omni-multiteacher/uv_cache',
    SCRATCH / 'nano-omni-multiteacher/hf_cache',
)


def command(args):
    return subprocess.check_output(args, text=True, timeout=20)


def queue():
    output = command(['squeue', '-u', os.environ.get('USER', 'dhei'), '-h', '-o', '%i|%T|%j|%k'])
    return {parts[0]: {'state': parts[1], 'name': parts[2], 'comment': parts[3]}
            for line in output.splitlines() if len(parts := line.strip().split('|', 3)) == 4}


def allocated(path):
    total = 0
    for directory, dirs, files in os.walk(path, followlinks=False):
        if any((Path(directory) / name).is_symlink() for name in dirs):
            raise RuntimeError('Unexpected link in converted parent cache')
        for name in files:
            file = Path(directory) / name
            if file.is_symlink():
                raise RuntimeError('Unexpected linked parent file')
            total += file.stat().st_blocks * 512
    return total


def protected_parents(experiment, state, live):
    protected = {str(device['active']['parent']['rollouts']) for device in state['devices'] if device.get('active')}
    journal = read(experiment / 'slurm_jobs.json')
    token = hashlib.sha256(str(experiment.resolve()).encode()).hexdigest()[:16]
    tasks = {key for key, value in journal.items() if value['kind'] == 'teacher' and value['job'] in live}
    for value in live.values():
        prefix = f'rlve-async:{token}:'
        if value['comment'].startswith(prefix + 'teacher-'):
            tasks.add(value['comment'][len(prefix):])
    for key in tasks:
        path = experiment / 'tasks' / key / 'task.json'
        if not path.is_file():
            raise RuntimeError(f'Live teacher has no task metadata: {key}')
        protected.add(str(read(path)['parent']['rollouts']))
    return protected


@contextmanager
def quiesce(experiment):
    job = str(read(experiment / 'monitor.json')['coordinator_job'])
    live = queue()
    value = live.get(job)
    if value is None or not value['name'].startswith('scaling-async-') or value['name'].endswith('-monitor'):
        raise RuntimeError(f'Cannot identify the unique CPU coordinator: {job}')
    if any(key != job and record['name'] == value['name'] for key, record in live.items()):
        raise RuntimeError('Multiple coordinators are queued')
    details = command(['scontrol', 'show', 'job', '-o', job])
    if 'gres/gpu' in details or str(experiment) not in details:
        raise RuntimeError('Refusing to suspend a GPU job or unrelated coordinator')
    held = False
    release = None
    fail_safe = None
    try:
        if value['state'] == 'PENDING':
            command(['scontrol', 'hold', job])
            held = True
            release = ['scontrol', 'release', job]
            fail_safe = subprocess.Popen([sys.executable, '-c',
                'import subprocess,time;time.sleep(120);subprocess.run(' + repr(release) + ',check=False)'],
                start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if queue().get(job, {}).get('state') != 'PENDING':
                raise RuntimeError('Coordinator started while being held; retry cleanup')
        elif value['state'] == 'RUNNING':
            release = ['scancel', '--signal=CONT', '--full', job]
            # An independent host process guarantees CONT even if this cleaner dies.
            fail_safe = subprocess.Popen([sys.executable, '-c',
                'import subprocess,time;time.sleep(120);subprocess.run(' + repr(release) + ',check=False)'],
                start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            command(['scancel', '--signal=STOP', '--full', job])
            first = read(experiment / 'coordinator-lease.json')['timestamp']
            time.sleep(4)
            if read(experiment / 'coordinator-lease.json')['timestamp'] != first:
                raise RuntimeError('Coordinator heartbeat still advancing; refusing parent deletion')
        else:
            raise RuntimeError(f'Coordinator is {value["state"]}; retry later')
        if str(read(experiment / 'monitor.json')['coordinator_job']) != job:
            raise RuntimeError('Monitor changed coordinators during cleanup')
        yield {'job': job, 'state_before': value['state'], 'quiesced_utc': now()}
    finally:
        if release:
            subprocess.run(release, check=True, timeout=20, capture_output=True, text=True)
        if fail_safe:
            fail_safe.terminate()
            fail_safe.wait(timeout=5)


def prune_parents():
    receipts = OUT / 'parent-cache-cleanup.json'
    record = read(receipts) if receipts.exists() else {'operations': [], 'removed_bytes': 0}
    review = ROOT / 'data/STORAGE-REVIEW-20261026'
    for inventory in sorted(review.glob('*-parent-caches.json')):
        candidates = read(inventory)
        experiment = Path(candidates[0]['path']).parent.parent
        if experiment.name not in {'data-scaling-async-n32-20261025-003350', 'data-scaling-async-n16-replay30-20261025-231220'}:
            raise ValueError('Unapproved live experiment')
        with quiesce(experiment) as pause:
            state = read(experiment / 'state.json')
            live = queue()
            protected = protected_parents(experiment, state, live)
            operation = {**pause, 'experiment': str(experiment), 'protected_rollouts': sorted(protected), 'removed': [], 'skipped': []}
            atomic(OUT / 'parent-cache-in-progress.json', operation)
            for candidate in candidates:
                parent = Path(candidate['path'])
                if candidate['active_teacher_parent'] or parent.name in protected:
                    operation['skipped'].append(str(parent))
                    continue
                if parent.parent != experiment / 'parents' or not parent.name.isdigit() or parent.is_symlink():
                    raise ValueError('Unapproved parent cache path')
                ref = parent / 'ref'
                if not ref.exists():
                    continue
                if ref.is_symlink() or not (ref / 'latest_checkpointed_iteration.txt').is_file():
                    raise RuntimeError('Unexpected converted parent layout')
                with (parent / 'conversion.lock').open('a') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        operation['skipped'].append(str(parent))
                        continue
                    size = allocated(ref)
                    shutil.rmtree(ref)
                    operation['removed'].append({'path': str(ref), 'allocated_bytes': size})
                    record['removed_bytes'] += size
                    atomic(OUT / 'parent-cache-in-progress.json', operation)
            operation['resuming_utc'] = now()
            record['operations'].append(operation)
            atomic(receipts, record)
        record['operations'][-1]['coordinator_released_utc'] = now()
        atomic(receipts, record)
    return record


def clean_build_caches():
    cache_receipt = OUT / 'build-cache-cleanup.json'
    result = read(cache_receipt) if cache_receipt.exists() else {'removed': []}
    live = queue()
    descriptions = [command(['scontrol', 'show', 'job', '-o', job]) for job in live]
    for image in ['slime-v0.5.0rc0-cu126.sif', 'miles-dev-202610231907.sif']:
        file = SCRATCH / 'images' / image
        if not file.is_file() or file.is_symlink():
            raise RuntimeError('Built runtime image missing; retain download caches')
    for target in BUILD_CACHES:
        if not target.exists():
            continue
        if target.is_symlink() or target.resolve() != target:
            raise ValueError('Unsafe rebuildable cache root')
        if any(str(target) in text for text in descriptions):
            raise RuntimeError(f'Live Slurm job explicitly references cache: {target}')
        for line in command(['ps', '-u', os.environ.get('USER', 'dhei'), '-o', 'args=']).splitlines():
            if ('apptainer build ' in line or 'apptainer pull ' in line) and str(target) in line:
                raise RuntimeError('Container build is using approved download cache')
        # The completed review already measured these large package trees.
        # Avoid another long metadata traversal before removing a retired cache.
        measured = {}
        for line in (ROOT / 'data/STORAGE-REVIEW-20261026/du-other.tsv').read_text().splitlines():
            count, _, measured_path = line.partition('\t')
            if count.isdigit():
                measured[measured_path] = int(count)
        size = measured[str(target)] if any(target.iterdir()) else 0
        if size == 0:
            continue
        mode = target.stat().st_mode & 0o777
        shutil.rmtree(target)
        target.mkdir(mode=mode)
        result['removed'].append({'path': str(target), 'allocated_bytes': size, 'utc': now()})
        atomic(cache_receipt, result)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'cache-cleanup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = shutil.disk_usage(SCRATCH).free
        caches = clean_build_caches()
        parents = prune_parents()
        atomic(OUT / 'cache-cleanup-completed.json', {'utc': now(),
            'build_cache_removed_gib': sum(row['allocated_bytes'] for row in caches['removed']) / 1024**3,
            'parent_cache_removed_gib': parents['removed_bytes'] / 1024**3,
            'free_gib_before': before / 1024**3, 'free_gib_after': shutil.disk_usage(SCRATCH).free / 1024**3,
            'gpu_jobs_interrupted': False})
        print(json.dumps(read(OUT / 'cache-cleanup-completed.json'), indent=2))


if __name__ == '__main__':
    main()
