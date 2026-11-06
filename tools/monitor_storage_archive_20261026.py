"""Keep durable archive progress and confirm the existing N32 replay launch."""

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from tools.archive_storage_review_20261026 import OUT, ROOT, SCRATCH, atomic, read, now

EXPERIMENT = SCRATCH / 'runs/data-scaling-async-n32-replay30-20261026-192321'


def alive(pid):
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ')
        return b'HF-STORAGE-ARCHIVE-20261026/runtime.py run' in command
    except (FileNotFoundError, ProcessLookupError):
        return False


def main():
    with (OUT / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execution = read(OUT / 'execution.json')
        pid = execution['archive_pid']
        retries = execution.get('supervisor_retries', [])
        while True:
            receipts = [read(path) for path in (OUT / 'receipts').glob('*.json')]
            statuses = {}
            for receipt in receipts:
                statuses[receipt['status']] = statuses.get(receipt['status'], 0) + 1
            monitor = read(EXPERIMENT / 'monitor.json') if (EXPERIMENT / 'monitor.json').is_file() else {}
            state = read(EXPERIMENT / 'state.json') if (EXPERIMENT / 'state.json').is_file() else {}
            journal = read(EXPERIMENT / 'slurm_jobs.json') if (EXPERIMENT / 'slurm_jobs.json').is_file() else {}
            queue = subprocess.check_output(['squeue', '-u', os.environ.get('USER', 'dhei'), '-h', '-o', '%i|%T'], text=True)
            live = dict(line.split('|', 1) for line in queue.splitlines() if '|' in line)
            workers = {key: {**value, 'state': live.get(value['job'], 'ENDED')} for key, value in journal.items()}
            activated = bool(state and journal)
            training = any(value['state'] == 'RUNNING' for value in workers.values())
            result = {'utc': now(), 'archive_pid': pid, 'archive_alive': alive(pid),
                'archive_statuses': statuses, 'archive_complete': (OUT / 'completed.json').exists(),
                'free_scratch_gib': shutil.disk_usage(SCRATCH).free / 1024**3,
                'n32_replay_monitor': monitor.get('monitor_status'),
                'n32_replay_coordinator': monitor.get('coordinator_job'),
                'n32_replay_activated': activated, 'n32_replay_gpu_running': training,
                'n32_replay_student_rollouts': state.get('student_rollouts', 0),
                'n32_replay_workers': workers, 'retries': retries}
            if alive(pid):
                result['archive_io'] = Path(f'/proc/{pid}/io').read_text()
            atomic(OUT / 'live-status.json', result)
            if activated and training and not (OUT / 'n32-replay-launch-confirmed.json').exists():
                atomic(OUT / 'n32-replay-launch-confirmed.json', result)
            if result['archive_complete'] and activated and training:
                atomic(OUT / 'workflow-completed.json', result)
                return
            if not result['archive_complete'] and not alive(pid):
                if len(retries) >= 3:
                    result['attention_required'] = 'Archive retry allowance exhausted; unverified payloads retained.'
                    atomic(OUT / 'attention-required.json', result)
                    return
                log = OUT / f'archive-supervisor-retry-{len(retries) + 1}.log'
                environment = {**os.environ, 'HF_HOME': '/tmp/dhei/hf-storage-archive-20261026',
                    'HF_HUB_DISABLE_XET': '1', 'HF_HUB_DISABLE_PROGRESS_BARS': '1',
                    'HF_HUB_ENABLE_HF_TRANSFER': '1' if execution.get('parallel_multipart_upload') else '0',
                    'HF_HUB_DISABLE_TELEMETRY': '1', 'PYTHONUNBUFFERED': '1'}
                with log.open('a') as stream:
                    process = subprocess.Popen([str(OUT / 'venv/bin/python'), '-u', str(OUT / 'runtime.py'),
                        'run', '--workers', str(execution.get('workers', 3))], cwd=ROOT, env=environment, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                retries.append({'utc': now(), 'previous_pid': pid, 'pid': process.pid, 'log': str(log)})
                pid = process.pid
                execution.update(archive_pid=pid, supervisor_retries=retries)
                atomic(OUT / 'execution.json', execution)
            time.sleep(60)


if __name__ == '__main__':
    main()
