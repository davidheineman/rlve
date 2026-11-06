"""Persist follow-up progress and launch students only when their teachers exist."""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from modular_rl.mopd_export import build_banks
from modular_rl.mopd_suite import submit_ready
from modular_rl.prepare import write_json


def latest(path):
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    for index in range(len(lines) - 1, -1, -1):
        try:
            return json.loads(lines[index])
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return None


def inspect(root):
    jobs = []
    for kind in ('rl', 'students'):
        path = root / kind / 'jobs.json'
        if path.exists():
            jobs += [{**job, 'kind': kind} for job in json.loads(path.read_text())]
    states = {}
    if jobs:
        ids = ','.join(job['job_id'] for job in jobs)
        output = subprocess.check_output(['sacct', '-n', '-P', '-X', '-j', ids,
            '--format=JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,NodeList'], text=True)
        for line in output.splitlines():
            values = line.split('|')
            if values[0] in {j['job_id'] for j in jobs}:
                states[values[0]] = dict(zip(('job_id', 'state', 'exit_code', 'seconds', 'start', 'end', 'node'), values))
    rows = []
    for job in jobs:
        directory = Path(job['output']) / 'policy'
        progress = latest(directory / 'progress.jsonl')
        result = dict(name=job['name'], gpus=job['gpus'], **states.get(job['job_id'], {'state': 'UNKNOWN'}))
        result['progress'] = progress
        result['completed'] = json.loads((directory / 'completed.json').read_text()) if (directory / 'completed.json').exists() else None
        rows.append(result)
    record = {'utc': datetime.now(timezone.utc).isoformat(), 'runs': rows,
              'all_five_complete': len(rows) == 5 and all(row['state'] == 'COMPLETED'
                  and row['completed'] and row['completed']['completed'] for row in rows),
              'running_gpus': sum(row['gpus'] for row in rows if row['state'] == 'RUNNING')}
    write_json(root / 'monitor.json', record)
    return record


def monitor(root, seconds=30):
    export_process = None
    export_log = None
    while True:
        build_banks(root)
        submit_ready(root)
        status = inspect(root)
        print(json.dumps(status), flush=True)
        if status['all_five_complete']:
            return
        rl_done = [row for row in status['runs'] if row['name'] in ('mid-web-math', 'mid-web-code')
                   and row['state'] == 'COMPLETED' and row['completed']]
        missing_ready_exports = any(not (root / 'teachers/exports' / row['name'] / 'weights_only_complete.json').exists()
                                    for row in rl_done)
        if missing_ready_exports and export_process is None:
            export_log = (root / 'web-teacher-exports.log').open('a')
            repo = Path(__file__).resolve().parents[1]
            command = ('source scripts/slurm/common.sh\n'
                'rlve_apptainer_cpu env PYTHONPATH="$RLVE_REPO:/root/Megatron-LM" OMP_NUM_THREADS=2 '
                'python3 -u -m modular_rl.mopd_export --root "$1" --master "$2"')
            export_process = subprocess.Popen(['bash', '-c', command, 'export-web-teachers', str(root),
                '/juice5b/scr5b/dhei/modular-mopd-rl/v1/two-gpu-ready-20261029'],
                cwd=repo, stdout=export_log, stderr=subprocess.STDOUT)
            print('Exporting completed web math/code teachers:', export_process.pid, flush=True)
        if export_process is not None and export_process.poll() is not None:
            code = export_process.returncode
            export_log.close()
            export_process, export_log = None, None
            if code:
                raise RuntimeError('Web teacher export failed; inspect web-teacher-exports.log')
        errors = [r for r in status['runs'] if r['state'].split()[0] in
                  ('FAILED', 'CANCELLED', 'NODE_FAIL', 'TIMEOUT', 'OUT_OF_MEMORY')]
        if errors:
            print('NEEDS_ATTENTION', json.dumps(errors), flush=True)
            raise SystemExit(2)
        time.sleep(seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.once:
        print(json.dumps(inspect(args.root.resolve()), indent=2))
    else:
        monitor(args.root.resolve())
