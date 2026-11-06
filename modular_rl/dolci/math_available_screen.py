"""Apply committed available verdicts without claiming a full model screen."""

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from .config import atomic_json
from . import math_large as core


def preparation_tick(root):
    from .math_large_quality_watch import cpu_job, read, submit
    from .watch import states, TERMINAL_FAILURES
    from .math_billion_watch import retry
    root = Path(root)
    status = {'stage': 'available-checks-final-data-preparation'}
    policy = read(root / 'quality-policy.json')
    if not policy.get('use_committed_merge_snapshot'):
        record = root / 'math-supplement-merge-job.json'
        job = read(record)
        state = states([job]).get(job['job_id'], {'state': 'UNKNOWN'})
        status['supplement_merge'] = state
        if state['state'] in TERMINAL_FAILURES:
            retry(root, record, job, state)
        if state['state'] != 'COMPLETED' or not (root / 'supplement-exported.json').exists():
            return status
    record = root / 'math-available-pack-job.json'
    if not record.exists():
        cpu_job(root, 'math-available-pack',
                f'/nlp/scr/dhei/tinker-mopd/venv-rl/bin/python -u -m modular_rl.dolci.math_available_screen --root {root}')
    job = submit(record)
    atomic_json(root / 'merge-job.json', job)
    state = states([job]).get(job['job_id'], {'state': 'UNKNOWN'})
    status['packing'] = state
    if state['state'] in TERMINAL_FAILURES:
        retry(root, record, job, state)
    return status


def apply_available(root):
    root = Path(root)
    policy = json.loads((root / 'quality-policy.json').read_text())
    assert policy['mode'] == 'available-checks' and policy['user_authorized']
    audit_path = root / 'available-screen-audit.json'
    if audit_path.exists():
        return json.loads(audit_path.read_text())
    db = sqlite3.connect(root / 'merged.sqlite')
    db.execute('PRAGMA cache_size=-1048576')
    db.execute('PRAGMA temp_store=MEMORY')
    plan_path = root / 'available-screen-plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        verdicts, evidence = {}, []
        for shard in range(6):
            path = root / 'judge-results' / f'{shard:02d}.jsonl'
            progress = path.with_suffix('.progress.json')
            if not path.exists() or not progress.exists():
                continue
            committed = json.loads(progress.read_text())['reviewed']
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for index in range(committed):
                    line = next(stream)
                    digest.update(line)
                    value = json.loads(line)
                    assert value['key'] not in verdicts
                    verdicts[value['key']] = value
            evidence.append({'shard': shard, 'committed_rows': committed,
                             'committed_prefix_sha256': digest.hexdigest()})
        manual = json.loads((root / 'manual-rejections.json').read_text())
        manual_keys = {v.get('key') or core.key(v['problem']) for v in manual if v.get('key') or v.get('problem')}
        manual_ids = {v['id'] for v in manual}
        rejected, examined = [], Counter()
        for key, payload, tokens in db.execute("SELECT key,payload,tokens FROM accepted WHERE bucket!='general'"):
            value = json.loads(payload)
            verdict = verdicts.pop(key, None)
            if verdict:
                assert hashlib.sha256(value['response'].encode()).hexdigest() == verdict['solution_sha256']
                examined['reviewed_questions'] += 1
                examined['reviewed_tokens'] += tokens
            else:
                examined['unreviewed_questions'] += 1
                examined['unreviewed_tokens'] += tokens
            if key in manual_keys or value['id'] in manual_ids or verdict and not verdict['accepted']:
                rejected.append(key)
                examined['removed_questions'] += 1
                examined['removed_tokens'] += tokens
        plan = {'rejected_keys': rejected, 'counts_before_deletion': dict(examined),
                'partial_screen_evidence': evidence, 'verdicts_for_previously_excluded_questions': len(verdicts)}
        atomic_json(plan_path, plan)
    keys = [(key,) for key in plan['rejected_keys']]
    print('APPLYING_AVAILABLE_REJECTIONS', len(keys), flush=True)
    for start in range(0, len(keys), 5000):
        db.executemany('DELETE FROM accepted WHERE key=?', keys[start:start + 5000])
        db.commit()
        print('REJECTION_BATCH_COMMITTED', min(start + 5000, len(keys)), '/', len(keys), flush=True)
    counts = {b: {'questions': n, 'tokens': t} for b, n, t in db.execute(
        'SELECT bucket,count(*),sum(tokens) FROM accepted GROUP BY bucket')}
    counts['math'] = {k: sum(counts.get(b, {}).get(k, 0) for b in ('math-labeled', 'math-unlabeled'))
                      for k in ('questions', 'tokens')}
    sources = {s: {'questions': n, 'tokens': t} for s, n, t in db.execute(
        'SELECT source,count(*),sum(tokens) FROM accepted GROUP BY source')}
    db.close()
    merged = json.loads((root / 'merge-complete.json').read_text())
    merged.update(counts=counts, sources=sources)
    if policy.get('use_committed_merge_snapshot'):
        merged.update(committed_merge_snapshot=True, dedup_statistics_cover_completed_merge_only=True)
    atomic_json(root / 'merge-complete.json', merged)
    report = {'passed': True, 'mode': 'available-checks', 'all_solutions_model_reviewed': False,
              'user_authorized_partial_screen': True, 'remaining_counts': counts,
              'review_coverage': plan['counts_before_deletion'],
              'partial_screen_evidence': plan['partial_screen_evidence'],
              'known_model_rejections_and_manual_vetoes_applied': True,
              'use_committed_merge_snapshot': bool(policy.get('use_committed_merge_snapshot'))}
    atomic_json(audit_path, report)
    core.capacity(counts['math']['tokens'], counts['general']['tokens'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(apply_available(args.root)), flush=True)
    from .math_large_pack import pack
    pack(args.root)
