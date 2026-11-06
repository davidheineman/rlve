"""Durable independent solution screening before the large-math release gate."""

import hashlib
import json
import subprocess
from pathlib import Path

from .config import atomic_json
from .math_billion_watch import retry
from .math_large_experiment import QUEUES
from .watch import states, TERMINAL_FAILURES

MODEL = 'Qwen/Qwen3-32B'
REVISION = '9216db5781bf21249d130ec9da846c4624c16137'


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def submit(record):
    job = read(record)
    if not job.get('job_id'):
        job['job_id'] = subprocess.check_output(job['command'], text=True).strip().split(';')[0]
        atomic_json(record, job)
    return job


def cpu_job(root, name, commands):
    repo = Path(__file__).resolve().parents[2]
    script = root / (name + '.sbatch')
    script.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH={repo}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
cd {repo}
{commands}
''')
    command = ['sbatch', '--parsable', '--account=miso', '--partition=' + QUEUES,
               '--nodes=1', '--ntasks=1', '--cpus-per-task=6', '--mem=96G',
               '--time=08:00:00', '--requeue', '--open-mode=append',
               '--job-name=' + name, '--output=' + str(root / (name + '-%j.log')), str(script)]
    atomic_json(root / (name + '-job.json'), {'name': name, 'command': command, 'gpus': 0})


def screen_jobs(root, *, start=0, stop=6, append=False):
    repo = Path(__file__).resolve().parents[2]
    selected = read(root / 'selected-verifier.json', {'model': MODEL, 'revision': REVISION,
                                                     'calibration_file': 'judge-32b-calibration.json'})
    model, revision, calibration_file = [selected[k] for k in ('model', 'revision', 'calibration_file')]
    rows = read(root / 'solution-screen-jobs.json', []) if append else []
    for shard in range(start, stop):
        if any(row['shard'] == shard for row in rows):
            continue
        name = f'math-solution-screen-{shard}'
        script = root / (name + '.sbatch')
        script.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
judge_tmp=/tmp/dhei-math-large-verifier-{revision[:12]}
mkdir -p "$judge_tmp"
export JUDGE_MODEL_PATH="$judge_tmp/model"
export HF_HOME="$judge_tmp/hf"
export HUGGINGFACE_HUB_CACHE="$judge_tmp/hf/hub"
export HF_HUB_DISABLE_XET=1
/nlp/scr/dhei/tinker-mopd/venv-rl/bin/python -u - <<'DOWNLOAD'
import os
import json
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download
path = Path(os.environ['JUDGE_MODEL_PATH'])
def ready():
    index = path / 'model.safetensors.index.json'
    return (index.exists() and (path / 'config.json').exists() and
            all((path / name).exists() for name in set(json.loads(index.read_text())['weight_map'].values())))
if not ready() and '{model}' == 'Qwen/Qwen3-30B-A3B-Instruct-2507':
    # The source allocation already staged this exact pinned public release.
    # Reuse it over SSH if available; fall back to the primary release host.
    subprocess.run(['rsync', '-a', '--partial', '--size-only', '--exclude=.cache',
                    '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=5',
                    'jagupard39:/tmp/dhei-math-large-judge-moe-{revision[:12]}/model/', str(path) + '/'])
if not ready():
    snapshot_download('{model}',revision='{revision}',local_dir=str(path),allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'],max_workers=4)
assert ready(), 'Reviewer checkpoint did not finish staging'
DOWNLOAD
export RLVE_REPO=/juice5b/u/dhei/tatsu/rlve
export RLVE_SCRATCH=/juice5b/scr5b/dhei
export RLVE_HF_CACHE="$judge_tmp/hf"
export APPTAINERENV_PYTHONPATH={repo}
source "$RLVE_REPO/scripts/slurm/common.sh"
rlve_apptainer bash -c 'cd {repo}; exec python3 -u -m modular_rl.dolci.math_large_judge --root {root} --model '"$JUDGE_MODEL_PATH"' --shard {shard} --model-repo {model} --model-revision {revision} --calibration-file {calibration_file}'
''')
        command = ['sbatch', '--parsable', '--account=miso', '--partition=' + QUEUES,
                   '--nodes=1', '--ntasks=1', '--gres=gpu:4', '--constraint=40G|48G|80G|141G',
                   '--cpus-per-task=16', '--mem=128G', '--time=16:00:00', '--requeue',
                   '--open-mode=append', '--exclude=viscam10,jagupard33,jagupard34,pasteur6',
                   '--job-name=' + name, '--output=' + str(root / (name + '-%j.log')), str(script)]
        subprocess.run(['bash', '-n', str(script)], check=True)
        adopted = read(root / 'review-and-screen-job.json') if shard == 0 else None
        rows.append({**({'name': name, 'command': command, 'gpus': 4,
                        'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest()}
                        if adopted is None else adopted), 'shard': shard,
                     'model': model, 'revision': revision, 'calibration_file': calibration_file})
    atomic_json(root / 'solution-screen-jobs.json', rows)


def tick(root):
    root = Path(root).resolve()
    status = {'stage': 'quality-screen-reserve-source-curation', 'jobs': []}
    if (root / 'supplement-plan.json').exists() and not (root / 'supplement-ready.json').exists():
        from .math_large_supplement import tick as supplement_tick
        supplement = supplement_tick(root)
        if not supplement.get('ready'):
            atomic_json(root / 'monitor.json', supplement)
            return supplement
    jobs = read(root / 'reserve-curation-jobs.json')
    slurm = states(jobs)
    ready = True
    for job in jobs:
        state = slurm.get(job['job_id'], {'state': 'UNKNOWN'})
        name = f"{job['source']}-{job['partition']}"
        complete = read(root / 'candidates' / (name + '.complete.json'))
        status['jobs'].append({**state, 'source': job['source'],
                               'progress': read(root / 'candidates' / (name + '.progress.json'))})
        if state['state'] in TERMINAL_FAILURES:
            retry(root, root / 'reserve-curation-jobs.json', job, state)
        ready &= state['state'] == 'COMPLETED' and bool(complete)
    original = read(root / 'merge-job.json')
    original_state = states([original]).get(original['job_id'], {'state': 'UNKNOWN'})
    status['original_merge'] = original_state
    if original_state['state'] in TERMINAL_FAILURES and not (root / 'merge-complete.json').exists():
        retry(root, root / 'merge-job.json', original, original_state)
        atomic_json(root / 'monitor.json', status)
        return status
    if not ready or original_state['state'] not in {'COMPLETED', *TERMINAL_FAILURES}:
        atomic_json(root / 'monitor.json', status)
        return status
    if not (root / 'reserve-appended.json').exists() or not (root / 'judge-input/manifest.json').exists():
        status['stage'] = 'quality-screen-global-reserve-dedup'
        record = root / 'math-reserve-merge-job.json'
        if not record.exists():
            # These are preliminary, unscreened arrays. Keep their audits but
            # require final packing to rewrite them after independent screening.
            archive = root / 'before-independent-screen'
            archive.mkdir(exist_ok=True)
            for name in ('manifest.json', 'curation-audit.json', 'review-examples.json'):
                path = root / name
                if path.exists():
                    path.rename(archive / name)
            python = '/nlp/scr/dhei/tinker-mopd/venv-rl/bin/python -u'
            command = f'{python} -m modular_rl.dolci.math_large_quality_watch --root {root} --append-reserve'
            cpu_job(root, 'math-reserve-merge', command)
        merge = submit(record)
        ms = states([merge]).get(merge['job_id'], {'state': 'UNKNOWN'})
        status['reserve_merge'] = ms
        if ms['state'] in TERMINAL_FAILURES:
            retry(root, record, merge, ms)
        atomic_json(root / 'monitor.json', status)
        return status
    selected = read(root / 'selected-verifier.json')
    if (root / 'verifier-selection-required.json').exists() and not selected:
        status['stage'] = 'awaiting-verifier-policy-review'
        atomic_json(root / 'monitor.json', status)
        return status
    calibration_file = selected['calibration_file'] if selected else 'judge-32b-calibration.json'
    calibration = read(root / calibration_file)
    if not calibration:
        status['stage'] = 'awaiting-strong-verifier-calibration'
        atomic_json(root / 'monitor.json', status)
        return status
    if not calibration['passed']:
        status.update(stage='strong-verifier-calibration-failed', attention_required=True, calibration=calibration)
        atomic_json(root / 'monitor.json', status)
        return status
    status['stage'] = 'independent-math-solution-screen'
    record = root / 'solution-screen-jobs.json'
    if not record.exists():
        screen_jobs(root)
    jobs = read(record)
    for job in jobs:
        if not job.get('job_id'):
            job['job_id'] = subprocess.check_output(job['command'], text=True).strip().split(';')[0]
            atomic_json(record, jobs)
    slurm = states(jobs)
    ready = True
    for job in jobs:
        state = slurm.get(job['job_id'], {'state': 'UNKNOWN'})
        complete = read(root / 'judge-results' / f"{job['shard']:02d}.complete.json")
        status['jobs'].append({**state, 'progress': read(root / 'judge-results' / f"{job['shard']:02d}.progress.json")})
        if state['state'] in TERMINAL_FAILURES:
            retry(root, record, job, state)
        ready &= state['state'] == 'COMPLETED' and bool(complete)
    if not ready:
        atomic_json(root / 'monitor.json', status)
        return status
    status['stage'] = 'independent-screen-capacity-and-repacking'
    record = root / 'math-screen-pack-job.json'
    if not record.exists():
        cpu_job(root, 'math-screen-pack',
            f'/nlp/scr/dhei/tinker-mopd/venv-rl/bin/python -u -m modular_rl.dolci.math_large_quality_watch --root {root} --apply-and-pack')
    job = submit(record)
    state = states([job]).get(job['job_id'], {'state': 'UNKNOWN'})
    status['final_packing'] = state
    if state['state'] in TERMINAL_FAILURES:
        audit = read(root / 'independent-screen-audit.json')
        if audit:
            counts = audit['remaining_counts']
            try:
                from .math_large import capacity
                capacity(counts['math']['tokens'], counts['general']['tokens'])
            except ValueError as exc:
                status.update(stage='screened-data-capacity-shortfall', attention_required=True, reason=str(exc))
                atomic_json(root / 'capacity-shortfall.json', status)
                atomic_json(root / 'monitor.json', status)
                return status
        retry(root, record, job, state)
    if state['state'] == 'COMPLETED' and (root / 'curation-audit.json').exists():
        atomic_json(root / 'initial-merge-job.json', read(root / 'merge-job.json'))
        atomic_json(root / 'merge-job.json', job)
        atomic_json(root / 'independent-screen-packed.json', {'passed': True, 'job_id': job['job_id']})
    atomic_json(root / 'monitor.json', status)
    return status


if __name__ == '__main__':
    parser = __import__('argparse').ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--append-reserve', action='store_true')
    parser.add_argument('--apply-and-pack', action='store_true')
    args = parser.parse_args()
    if args.append_reserve:
        from .math_large_pack import merge
        from .math_large_judge import export_shards
        root = args.root
        if not (root / 'reserve-appended.json').exists():
            candidates = [root / 'candidates' / f"{v['source']}-{v['partition']}.sqlite"
                          for v in read(root / 'reserve-curation-jobs.json')]
            merge(root, candidates)
            atomic_json(root / 'reserve-appended.json', {'passed': True, 'sources': read(root / 'merge-complete.json')['sources']})
        export_shards(root)
    elif args.apply_and_pack:
        from .math_large_judge import apply_screen
        from .math_large_pack import pack
        apply_screen(args.root)
        pack(args.root)
    else:
        print(json.dumps(tick(args.root)))
