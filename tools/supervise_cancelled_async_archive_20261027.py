"""Keep the approved two-run archive alive without submitting training jobs."""

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from tools.archive_storage_review_20261026 import atomic, now


def main():
    root = Path('/juice5b/u/dhei/tatsu/rlve')
    out = root / 'data/HF-ASYNC-CANCELLED-ARCHIVE-20261027'
    with (out / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execution = json.loads((out / 'execution.json').read_text())
        env = {**os.environ, 'PYTHONPATH': str(out / 'source'),
               'HF_HOME': '/tmp/dhei/hf-cancelled-async-archive-20261027',
               'HF_HUB_DISABLE_XET': '1', 'HF_HUB_ENABLE_HF_TRANSFER': '0',
               'HF_HUB_DISABLE_PROGRESS_BARS': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
               'PYTHONUNBUFFERED': '1'}
        for attempt in range(1, 4):
            if (out / 'completed.json').exists():
                return
            log = out / f'archive-attempt-{attempt}.log'
            with log.open('a') as stream:
                process = subprocess.Popen(execution['command'], cwd=root, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            execution.update(archive_pid=process.pid, attempt=attempt, log=str(log))
            atomic(out / 'execution.json', execution)
            while process.poll() is None:
                progress = json.loads((out / 'progress.json').read_text()) if (out / 'progress.json').exists() else {}
                atomic(out / 'live-status.json', {'utc': now(), 'archive_pid': process.pid,
                    'archive_alive': True, 'attempt': attempt, 'progress': progress,
                    'free_scratch_gib': shutil.disk_usage('/juice5b/scr5b/dhei').free / 1024**3})
                time.sleep(10)
            if process.returncode == 0 and (out / 'completed.json').exists():
                atomic(out / 'live-status.json', {'utc': now(), 'archive_alive': False,
                    'archive_complete': True, 'attempt': attempt,
                    'free_scratch_gib': shutil.disk_usage('/juice5b/scr5b/dhei').free / 1024**3})
                return
            atomic(out / f'failed-attempt-{attempt}.json', {'utc': now(), 'returncode': process.returncode,
                                                          'log': str(log)})
            if attempt < 3:
                time.sleep(20)
        atomic(out / 'attention-required.json', {'utc': now(), 'reason': 'Three archive attempts failed; unverified local files remain.'})


if __name__ == '__main__':
    main()
