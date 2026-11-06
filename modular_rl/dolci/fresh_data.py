"""Streaming, resumable construction of a unique-question 1B-token code mix.

Prefer verified Python traces, retain one conversation per normalized user
question across all three pinned datasets, and never cycle packed blocks.
Long conversations are filtered rather than truncated. Packing is the existing
continuous-document format, now with 8192-token context for reasoning traces.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import sqlite3
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests

from .config import ACTUAL_TOKENS, SEED, atomic_json, fingerprint
from .prepare import initialize, encode, normalized

DATASETS = {
    'python': ('allenai/Dolci-Think-SFT-Python', '36fcd3f93e8e702f793e1d77be44e7849d727a95', 45),
    'think': ('allenai/Dolci-Think-SFT-7B', '83f618e2fe0aed29ee3440a3658a0326c175d09d', 156),
    'instruct': ('allenai/Dolci-Instruct-SFT', 'bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221', 15),
}
SEQ_LEN = 8192


def question_key(messages):
    users = [normalized(m.get('content') or '') for m in messages if m['role'] == 'user']
    # Sharing an initial question counts as repetition even with different follow-up turns.
    return hashlib.sha256(users[0].encode()).hexdigest() if users and users[0] else None


def encode_candidate(item):
    key, bucket, provenance, messages = item
    ids, mask, _, _ = encode(messages)
    if len(ids) > SEQ_LEN or len(ids) < 2:
        return key, bucket, provenance, None, None
    return key, bucket, provenance, ids, mask


def download(repo, revision, filename, target):
    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}'
    for attempt in range(6):
        try:
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with target.open('wb') as stream:
                    for chunk in response.iter_content(4 * 1024**2): stream.write(chunk)
            pq.ParquetFile(target)  # Reject incomplete downloads before using them.
            return
        except Exception:
            target.unlink(missing_ok=True)
            if attempt == 5: raise
            time.sleep(min(5 * 2**attempt, 60))


def prepare(root, parent, workers=12):
    root, parent = Path(root), Path(parent)
    if (root / 'manifest.json').exists(): return
    (root / 'units').mkdir(exist_ok=True)
    (root / 'downloads').mkdir(exist_ok=True)
    db = sqlite3.connect(root / 'dedup.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS questions (key TEXT PRIMARY KEY, unit TEXT NOT NULL)')
    excluded = set()
    # Exclude every MBPP problem, not just the 64 validation questions.
    refs = json.loads((root / 'mbpp-reference-all.json').read_text())
    excluded.update(normalized(r['text']) for r in refs)
    for task in ('math', 'code', 'if', 'chat'):
        for line in (root / 'selection' / task / 'val.jsonl').read_text().splitlines():
            for m in json.loads(line)['metadata']['messages']:
                if m['role'] == 'user':
                    excluded.add(normalized(m['content']))
                    # The SFT source may omit the RL-only answer/API suffix.
                    core = m['content'].split('\nRequired API', 1)[0]
                    core = core.split('\n\nSolve the problem and put your final answer', 1)[0]
                    excluded.add(normalized(core))
    excluded_keys = {hashlib.sha256(s.encode()).hexdigest() for s in excluded}
    totals = Counter()
    completed = {}
    for p in sorted((root / 'units').glob('*/complete.json')):
        value = json.loads(p.read_text()); completed[p.parent.name] = value
        totals.update(value['tokens'])
    desired_code = int(ACTUAL_TOKENS * .8) + 4 * SEQ_LEN
    desired_general = ACTUAL_TOKENS - int(ACTUAL_TOKENS * .8) + 4 * SEQ_LEN
    general_caps = {'instruct': desired_general // 2, 'think': desired_general - desired_general // 2}
    source_tokens = Counter()
    for value in completed.values(): source_tokens.update(value.get('source_tokens', {}))
    audit = Counter()
    # Arrow has background threads by the first map call; forking that process
    # can deadlock tokenizer imports. Use independent spawned workers.
    pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                               initializer=initialize, initargs=(str(root / 'model'), []))
    try:
        # Prefer the explicitly verified Python data before overlapping aggregate data.
        for dataset in ('python', 'instruct', 'think', 'instruct-topup'):
            kind = 'instruct' if dataset == 'instruct-topup' else dataset
            repo, revision, shards = DATASETS[kind]
            for index in range(shards):
                unit = f'{dataset}-{index:05d}'
                if unit in completed: continue
                if dataset == 'think' and totals['code'] >= desired_code and totals['general'] >= desired_general: break
                if dataset == 'instruct-topup' and sum(totals.values()) >= ACTUAL_TOKENS + 8 * SEQ_LEN: break
                directory = root / 'units' / unit
                # Roll back an interrupted unit. Its complete marker is the commit point.
                db.execute('DELETE FROM questions WHERE unit=?', (unit,)); db.commit()
                if directory.exists(): shutil.rmtree(directory)
                directory.mkdir()
                remote = kind != 'instruct'
                filename = f'data/train-{index:05d}-of-{shards:05d}.parquet'
                path = root / 'downloads/current.parquet' if remote else parent / 'raw' / filename
                if remote: download(repo, revision, filename, path)
                counts, unit_tokens, unit_sources = Counter(), Counter(), Counter()
                provenance = (directory / 'examples.jsonl').open('w')
                files = {b: ((directory / (b + '.bin')).open('wb'), (directory / (b + '.mask')).open('wb')) for b in ('code', 'general')}
                try:
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                        candidates = []
                        for row in batch.to_pylist():
                            counts['rows_scanned'] += 1
                            if kind == 'python' and row['correct'] is not True:
                                counts['incorrect'] += 1; continue
                            messages = row['messages']
                            source = str(row.get('dataset_source', row.get('source_dataset', repo)))
                            if kind == 'python': is_code = True
                            elif kind == 'instruct': is_code = row['domain'] == 'Coding'
                            else:
                                lower = source.casefold()
                                # The aggregate includes verified Python, Nemotron-code,
                                # and mixed OpenThoughts traces. Detect coding in the latter.
                                is_code = any(w in lower for w in ('python', 'coding', 'code', 'nemotron')) or any(
                                    '```python' in (m.get('content') or '').casefold() for m in messages if m['role'] == 'assistant')
                                if 'python' in lower and 'correct' not in lower:
                                    counts['unverified_python_source'] += 1; continue
                            bucket = 'code' if is_code else 'general'
                            if bucket == 'code' and totals['code'] + unit_tokens['code'] >= desired_code: continue
                            if bucket == 'general':
                                limit = general_caps.get(kind, 0)
                                if dataset == 'instruct-topup': limit = ACTUAL_TOKENS + 8 * SEQ_LEN - totals['code']
                                if source_tokens[kind + ':general'] + unit_sources[kind + ':general'] >= limit: continue
                            key = question_key(messages)
                            if not key: counts['empty_question'] += 1; continue
                            if key in excluded_keys or any(normalized(m.get('content') or '').split('\nrequired api')[0] in excluded
                                                           for m in messages if m['role'] == 'user'):
                                counts['rl_exclusion'] += 1; continue
                            try: db.execute('INSERT INTO questions VALUES (?, ?)', (key, unit))
                            except sqlite3.IntegrityError: counts['duplicate_question'] += 1; continue
                            info = {'dataset': kind, 'source': source, 'id': str(row.get('id', row.get('sample_id', '')))}
                            candidates.append((key, bucket, info, messages))
                        for key, bucket, info, ids, mask in pool.map(encode_candidate, candidates, chunksize=8):
                            if ids is None:
                                counts['over_context'] += 1
                                # A later shorter correct trace can still be selected.
                                db.execute('DELETE FROM questions WHERE key=?', (key,)); continue
                            np.asarray(ids, dtype='<u4').tofile(files[bucket][0]); mask.tofile(files[bucket][1])
                            unit_tokens[bucket] += len(ids); unit_sources[kind + ':' + bucket] += len(ids)
                            counts['accepted'] += 1; counts['assistant_tokens'] += int(mask.sum())
                            provenance.write(json.dumps({'question_sha256': key, 'tokens': len(ids), 'bucket': bucket, **info}) + '\n')
                    for pair in files.values():
                        for stream in pair: stream.flush(); os.fsync(stream.fileno())
                    provenance.flush(); os.fsync(provenance.fileno()); db.commit()
                    value = {'dataset': kind, 'revision': revision, 'tokens': dict(unit_tokens), 'counts': dict(counts), 'source_tokens': dict(unit_sources)}
                    atomic_json(directory / 'complete.json', value)
                    completed[unit] = value; totals.update(unit_tokens); source_tokens.update(unit_sources); audit.update(counts)
                    atomic_json(root / 'data-progress.json', {'unit': unit, 'tokens': dict(totals), 'counts': dict(audit), 'source_tokens': dict(source_tokens), 'utc': time.time()})
                    print('UNIT_COMPLETE', unit, dict(totals), dict(counts), flush=True)
                finally:
                    provenance.close()
                    for pair in files.values():
                        for stream in pair: stream.close()
                    if remote: path.unlink(missing_ok=True)
    finally:
        pool.shutdown(); db.close()
    pack(root, completed)


def pack(root, completed):
    sources = {}
    (root / 'sources').mkdir(exist_ok=True)
    width = SEQ_LEN + 1
    # Each token belongs to one stored block; no duplicated boundary tokens.
    for bucket in ('code', 'general'):
        target = root / 'sources' / bucket
        total = 0; digests = {}; supervised = 0
        for suffix, dtype in (('.bin', '<u4'), ('.mask', 'u1')):
            path = target.with_suffix(suffix)
            with path.open('wb') as out:
                for unit in sorted(completed):
                    source = root / 'units' / unit / (bucket + suffix)
                    with source.open('rb') as stream: shutil.copyfileobj(stream, out, 8 * 1024**2)
            n = path.stat().st_size // np.dtype(dtype).itemsize
            n = n // width * width
            with path.open('r+b') as out: out.truncate(n * np.dtype(dtype).itemsize)
            h = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(8 * 1024**2), b''): h.update(chunk)
            digests[suffix] = h.hexdigest()
            total = n
        masks = np.memmap(target.with_suffix('.mask'), dtype='u1', mode='r').reshape(-1, width)
        counts = masks[:, 1:].sum(axis=1, dtype=np.uint32)
        np.save(target.with_suffix('.counts.npy'), counts)
        supervised = int(counts.sum())
        sources[bucket] = {'packed_tokens': total, 'blocks': total // width, 'tokens': total,
                           'assistant_tokens': supervised, 'tokens_sha256': digests['.bin'], 'mask_sha256': digests['.mask']}
    blocks = ACTUAL_TOKENS // SEQ_LEN
    code_quota = min(int(blocks * .8), sources['code']['blocks'])
    general_quota = blocks - code_quota
    if general_quota > sources['general']['blocks']:
        raise ValueError('Unique pool below budget; refusing to repeat examples')
    weights = {'code': code_quota / blocks, 'general': general_quota / blocks}
    manifest = {'repo': list(DATASETS.values()), 'revision': {k: v[1] for k, v in DATASETS.items()},
                'seq_len': SEQ_LEN, 'seed': SEED, 'objective': 'assistant',
                'pool_prediction_tokens': ACTUAL_TOKENS, 'natural_definition': 'unique-question fresh code mix',
                'sources': sources, 'quotas': {'code': {'code': code_quota, 'general': general_quota}},
                'recipe': {'mixes': {'code': weights}}, 'require_unique_blocks': True,
                'unique_question_policy': 'one conversation per casefolded whitespace-normalized first user question across all sources',
                'python_filter': 'correct=true; aggregate Python constituent must be explicitly correct-python',
                'no_truncation': True, 'max_conversation_tokens': SEQ_LEN,
                'packing': 'continuous complete conversations; a conversation may straddle adjacent blocks',
                'units': completed}
    manifest['signature'] = fingerprint(manifest)
    atomic_json(root / 'manifest.json', manifest)
    # Keep hashes, per-example provenance and counts; remove only redundant token copies.
    for p in (root / 'units').glob('*/*'):
        if p.suffix in ('.bin', '.mask'): p.unlink()
    print('READY', ACTUAL_TOKENS, weights, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    prepare(args.root, args.parent, args.workers)
