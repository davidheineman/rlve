"""Add novel screened problems without invalidating already running reviews."""

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from .config import atomic_json
from .math_billion_watch import retry
from .math_large_quality_watch import cpu_job, read, screen_jobs, submit
from .watch import states, TERMINAL_FAILURES


def append_and_export(root):
    root = Path(root)
    plan = read(root / 'supplement-plan.json')
    appended = root / 'supplement-appended.json'
    if not appended.exists():
        candidates = [root / 'candidates' / f"{row['source']}-{row['partition']}.sqlite"
                      for row in read(root / 'supplement-curation-jobs.json')]
        # A crash after the merge's durable commit/cache retirement must not
        # require missing derived caches or rescan millions of source rows.
        committed = read(root / 'merge-complete.json')
        if not all(not p.exists() for p in candidates) or plan['source'] not in committed['sources']:
            from .math_large_pack import merge
            merge(root, candidates, preserve_existing_solutions=True)
        atomic_json(appended, {'passed': True, 'sources': read(root / 'merge-complete.json')['sources'],
                              'existing_solution_payloads_preserved': True})
    destination = root / 'judge-input'
    manifest = read(destination / 'manifest.json')
    start, stop = plan['shard_start'], plan['shard_stop']
    if manifest['shards'] != stop:
        assert manifest['shards'] == start
        paths = [destination / f'{i:02d}.jsonl.partial' for i in range(start, stop)]
        counts = Counter()
        streams = [p.open('w') for p in paths]
        db = sqlite3.connect(f'file:{root}/merged.sqlite?mode=ro', uri=True)
        try:
            for key, payload in db.execute('SELECT key,payload FROM accepted WHERE source=? ORDER BY key', (plan['source'],)):
                value = json.loads(payload)
                value['solution_sha256'] = hashlib.sha256(value['response'].encode()).hexdigest()
                shard = int(key[:8], 16) % (stop - start)
                streams[shard].write(json.dumps(value, ensure_ascii=False) + '\n')
                counts[str(start + shard)] += 1
        finally:
            db.close()
            for stream in streams:
                stream.close()
        for path in paths:
            path.replace(path.with_suffix(''))
        manifest.update(shards=stop, questions={**manifest['questions'], **dict(counts)},
                        supplement_source=plan['source'], original_shards_unchanged=True)
        atomic_json(destination / 'manifest.json', manifest)
    atomic_json(root / 'supplement-exported.json', {'passed': True, 'shard_start': start, 'shard_stop': stop})


def tick(root):
    root = Path(root)
    status = {'stage': 'additional-unique-math-curation', 'jobs': [], 'ready': False}
    record = root / 'supplement-curation-jobs.json'
    jobs = read(record)
    slurm = states(jobs)
    ready = True
    for job in jobs:
        state = slurm.get(job['job_id'], {'state': 'UNKNOWN'})
        name = f"{job['source']}-{job['partition']}"
        status['jobs'].append({**state, 'progress': read(root / 'candidates' / (name + '.progress.json'))})
        if state['state'] in TERMINAL_FAILURES:
            retry(root, record, job, state)
        ready &= state['state'] == 'COMPLETED' and (root / 'candidates' / (name + '.complete.json')).exists()
    if not ready:
        return status
    status['stage'] = 'additional-unique-math-global-dedup'
    record = root / 'math-supplement-merge-job.json'
    if not record.exists():
        cpu_job(root, 'math-supplement-merge',
                f'/nlp/scr/dhei/tinker-mopd/venv-rl/bin/python -u -m modular_rl.dolci.math_large_supplement --root {root}')
    job = submit(record)
    state = states([job]).get(job['job_id'], {'state': 'UNKNOWN'})
    status['merge'] = state
    if state['state'] in TERMINAL_FAILURES:
        retry(root, record, job, state)
    if state['state'] == 'COMPLETED' and read(root / 'supplement-exported.json'):
        plan = read(root / 'supplement-plan.json')
        screen_jobs(root, start=plan['shard_start'], stop=plan['shard_stop'], append=True)
        atomic_json(root / 'supplement-ready.json', {'passed': True, 'merge_job_id': job['job_id']})
        status['ready'] = True
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    append_and_export(parser.parse_args().root)
