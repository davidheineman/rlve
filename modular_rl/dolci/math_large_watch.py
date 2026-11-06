"""Curation -> audited capacity -> reviewed data -> preflight -> 1B SFT -> RL."""

import argparse
import fcntl
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .config import atomic_json
from .math_large_experiment import QUEUES, prepare_pilot_job, trainer
from .watch import states, TERMINAL_FAILURES
from .math_billion_watch import retry


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def launch(root, path):
    job = read(path)
    if not job.get('job_id'):
        job['job_id'] = subprocess.check_output(job['command'], text=True).strip().split(';')[0]
        job['status'] = 'submitted'
        atomic_json(path, job)
    return job


def prepare_merge_job(root):
    repo = Path(__file__).resolve().parents[2]
    script = root / 'merge.sbatch'
    script.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH={repo}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
cd {repo}
exec /nlp/scr/dhei/tinker-mopd/venv-rl/bin/python -u -m modular_rl.dolci.math_large_pack --root {root}
''')
    command = ['sbatch', '--parsable', '--account=miso', '--partition=' + QUEUES,
               '--nodes=1', '--ntasks=1', '--cpus-per-task=6', '--mem=96G',
               '--time=08:00:00', '--requeue', '--open-mode=append',
               '--job-name=math-large-merge', '--output=' + str(root / 'merge-%j.log'), str(script)]
    atomic_json(root / 'merge-job.json', {'name': 'math-large-merge', 'command': command, 'gpus': 0})


def tick(root):
    root = Path(root).resolve()
    status = {'utc': datetime.now(timezone.utc).isoformat(), 'stage': 'curation', 'jobs': [],
              'scratch_free_gib': os.statvfs(root).f_bavail * os.statvfs(root).f_frsize / 2**30}
    if (root / 'HALTED.json').exists():
        return {**status, 'stage': 'halted', 'finished': True}
    quality_policy = read(root / 'quality-policy.json', {})
    if quality_policy.get('mode') == 'available-checks' and not (root / 'curation-audit.json').exists():
        from .math_available_screen import preparation_tick
        result = {**status, **preparation_tick(root)}
        atomic_json(root / 'monitor.json', result)
        return result
    if (root / 'independent-screen-required.json').exists() and not (root / 'independent-screen-packed.json').exists():
        from .math_large_quality_watch import tick as quality_tick
        return {**status, **quality_tick(root)}
    jobs = read(root / 'curation-jobs.json')
    slurm = states(jobs)
    ready = True
    for job in jobs:
        state = slurm.get(job['job_id'], {'state': 'UNKNOWN'})
        name = f"{job['source']}-{job['partition']}"
        complete = read(root / 'candidates' / (name + '.complete.json'))
        status['jobs'].append({**state, 'name': job['name'], 'complete': complete,
                               'progress': read(root / 'candidates' / (name + '.progress.json'))})
        if state['state'] in TERMINAL_FAILURES:
            retry(root, root / 'curation-jobs.json', job, state)
        ready &= state['state'] == 'COMPLETED' and bool(complete)
    if not ready:
        atomic_json(root / 'monitor.json', status)
        return status
    status['stage'] = 'global-dedup-packing-capacity-audit'
    if not (root / 'merge-job.json').exists():
        prepare_merge_job(root)
    merge = launch(root, root / 'merge-job.json')
    ms = states([merge]).get(merge['job_id'], {'state': 'UNKNOWN'})
    status['merge'] = ms
    if ms['state'] in TERMINAL_FAILURES:
        merged = read(root / 'merge-complete.json')
        if merged:
            from .math_large import capacity
            try:
                capacity(merged['counts']['math']['tokens'], merged['counts']['general']['tokens'])
            except ValueError as exc:
                failure = {'requirements_not_met': str(exc), 'actual_unique_pool': merged,
                           'training_released': False, 'reason': 'Hard two-epoch budget guard; expand the data instead of cycling a tiny pool.'}
                atomic_json(root / 'capacity-shortfall.json', failure)
                return {**status, 'stage': 'capacity-shortfall', 'attention_required': True}
        retry(root, root / 'merge-job.json', merge, ms)
    audit = read(root / 'curation-audit.json')
    if ms['state'] != 'COMPLETED' or not audit:
        atomic_json(root / 'monitor.json', status)
        return status
    assert audit['passed'] and max(audit['source_epochs'].values()) <= 2
    status['curation'] = audit
    atomic_json(root / 'repetition-proof.json', {
        'unique_pool': audit['sources'], 'source_epochs': audit['source_epochs'],
        'maximum_block_exposures': audit['maximum_block_exposures'],
        'overall_effective_epochs': audit['overall_effective_epochs'],
        'hard_maximum': 2, 'all_questions_globally_deduplicated': True})
    review = read(root / 'REVIEWED.json')
    supplement_review = read(root / 'supplement-reviewed.json', {})
    if (not review or not review['passed'] or
            (root / 'supplement-plan.json').exists() and not supplement_review.get('passed')):
        status['stage'] = 'awaiting-agent-quality-review'
        atomic_json(root / 'monitor.json', status)
        return status
    from .math_large_storage import retire_merged_cache
    retire_merged_cache(root)
    status['stage'] = 'small-sft-and-rl-verification'
    if not (root / 'preflight-job.json').exists():
        prepare_pilot_job(root)
    preflight = launch(root, root / 'preflight-job.json')
    ps = states([preflight]).get(preflight['job_id'], {'state': 'UNKNOWN'})
    status['preflight'] = ps
    verification = read(root / 'preflight/validation.json')
    if verification and not verification['passed']:
        status.update(stage='preflight-quality-gate-failed', verification=verification, attention_required=True)
        atomic_json(root / 'monitor.json', status)
        return status
    if ps['state'] in TERMINAL_FAILURES:
        retry(root, root / 'preflight-job.json', preflight, ps)
    if ps['state'] != 'COMPLETED' or not verification:
        atomic_json(root / 'monitor.json', status)
        return status
    assert verification['passed']
    from .math_large_storage import compact_pilot
    compact_pilot(root)
    status['stage'] = 'full-1b-sft-and-paired-8k-rl'
    trainer(root)
    launch(root, root / 'sft-job.json')
    # Reuse the independently launched matched baseline control, which never
    # used the rejected repeated small pool.
    baseline = root / 'rl/baseline'
    if not baseline.exists():
        baseline.parent.mkdir(exist_ok=True)
        baseline.symlink_to('/nlp/scr/dhei/math-refresh-1b-20261101/rl/baseline')
    sft = read(root / 'sft-job.json')
    ss = states([sft]).get(sft['job_id'], {'state': 'UNKNOWN'})
    from .math_large_experiment import training_startup
    status['midtraining_startup'] = training_startup(root)
    if ss['state'] == 'COMPLETED' and (root / 'sft/math/results.json').exists():
        from .export import finalize_generation
        from .storage import retire
        from .math_billion import prepare_rl
        finalize_generation(root / 'sft/math')
        retire(root / 'sft/math')
        prepare_rl(root, 'math', run_name='math-large-1b-rl', group='math-large-1b-paired')
    from .math_billion_watch import tick as training_tick
    trained = training_tick(root)
    status['training'] = trained
    status['finished'] = trained['complete']
    atomic_json(root / 'monitor.json', status)
    return status


def run(root):
    root = Path(root).resolve()
    with (root / 'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        errors, last_plot = 0, None
        while True:
            try:
                value = tick(root)
                print(json.dumps(value), flush=True)
                if value['stage'] == 'full-1b-sft-and-paired-8k-rl':
                    files = sorted((root / 'rl/math').glob('runs/*/policy/rollouts/eval/*.jsonl'))
                    signature = [(p.name, p.stat().st_size) for p in files]
                    if signature != last_plot:
                        output = '/juice5b/u/dhei/tatsu/rlve/plot/mopd-modular-v2/math_large_1b.png'
                        subprocess.run(['/juice5b/u/dhei/tatsu/nano-omni/.venv-midtraining/bin/python',
                            '-m', 'modular_rl.dolci.math_billion_plot', '--root', str(root), '--output', output], check=True)
                        last_plot = signature
                errors = 0
                if value.get('finished'):
                    return
            except Exception:
                errors += 1
                atomic_json(root / 'controller-error.json', {'consecutive_errors': errors, 'traceback': traceback.format_exc()})
                traceback.print_exc()
                if errors >= 8:
                    raise
            time.sleep(45)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run(args.root)
