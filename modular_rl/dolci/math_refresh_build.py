"""Resumable preparation of a reference-checked one-epoch 80/20 math mix."""

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .config import SEED, atomic_json, fingerprint
from . import math_refresh as core

PARENT = Path('/nlp/scr/dhei/dolci-mix-20261029')
RUNTIME = PARENT / 'rl/base/runtime'


def open_database(root):
    db = sqlite3.connect(Path(root) / 'corpus.sqlite', timeout=120)
    db.execute('PRAGMA cache_size=-524288')
    db.execute('PRAGMA temp_store=FILE')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS refs (key TEXT, answer TEXT, source TEXT, rank INTEGER,
                                        PRIMARY KEY(key,answer));
        CREATE TABLE IF NOT EXISTS accepted (key TEXT PRIMARY KEY, source TEXT, tokens INTEGER,
                                            payload TEXT, unit TEXT);
        CREATE INDEX IF NOT EXISTS accepted_source ON accepted(source,key);
        CREATE TABLE IF NOT EXISTS units (name TEXT PRIMARY KEY, stats TEXT);
    ''')
    columns = {row[1] for row in db.execute('PRAGMA table_info(accepted)')}
    for name in ('ids', 'mask'):
        if name not in columns: db.execute('ALTER TABLE accepted ADD COLUMN ' + name + ' BLOB')
    db.commit()
    return db


def unit_done(db, name):
    return db.execute('SELECT stats FROM units WHERE name=?', (name,)).fetchone() is not None


def commit_unit(root, db, name, stats):
    db.execute('INSERT OR REPLACE INTO units VALUES (?,?)', (name, json.dumps(stats)))
    db.commit()
    print('UNIT_COMPLETE', name, dict(stats), flush=True)
    atomic_json(Path(root) / 'progress.json', {'stage': name, 'counts': dict(stats),
                'accepted': dict(db.execute('SELECT source,count(*) FROM accepted GROUP BY source')),
                'math_tokens': db.execute("SELECT coalesce(sum(tokens),0) FROM accepted WHERE source!='general'").fetchone()[0]})


def raw_file(root, inventory, name, item):
    path = Path(root) / 'raw' / name / Path(item['path']).name
    if not path.exists() or path.stat().st_size != item['bytes']:
        core.download(inventory[name]['repo'], inventory[name]['revision'], item['path'], path)
    return path


def exclusions(root):
    root = Path(root)
    if (root / 'exclusions.json').exists():
        return json.loads((root / 'exclusions.json').read_text())
    from huggingface_hub import HfApi
    from modular_rl.v2.data import boxed_answer
    records = []
    with (PARENT / 'rl-data/math/val.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line); md = row['metadata']
            records.append({'key': core.question_key(md['messages'][0]['content']),
                            'problem': core.canonical_question(md['messages'][0]['content']),
                            'origin': 'custom-math-holdout', 'id': md['id']})
    info = HfApi().dataset_info(core.MATH_REPO, revision=core.MATH_REVISION, files_metadata=True)
    for item in info.siblings:
        if '/test-' not in item.rfilename or not item.rfilename.endswith('.parquet'): continue
        path = root / 'holdout-source' / item.rfilename.replace('/', '-')
        if not path.exists(): core.download(core.MATH_REPO, core.MATH_REVISION, item.rfilename, path)
        for row in pq.read_table(path).to_pylist():
            records.append({'key': core.question_key(row['problem']), 'problem': core.canonical_question(row['problem']),
                            'origin': 'official-math-test'})
    for task in ('code', 'if', 'chat'):
        with (PARENT / 'rl/base/selection' / task / 'val.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                for message in row['user_prompt']:
                    if message['role'] != 'user': continue
                    problem = message['content'].split('\nRequired API', 1)[0]
                    records.append({'key': core.question_key(problem), 'problem': core.canonical_question(problem),
                                    'origin': task + '-rl-holdout'})
    manual = Path(__file__).with_name('math_refresh_exclusions.json')
    if manual.exists():
        for row in json.loads(manual.read_text()):
            records.append({'key': core.question_key(row['problem']), 'problem': row['problem'],
                            'origin': 'manual-invalid-question', 'id': row.get('id'),
                            'reason': row['rejection_reason']})
    counts = Counter(row['origin'] for row in records)
    if counts['custom-math-holdout'] != 750 or counts['official-math-test'] != 5000:
        raise ValueError('Expected the complete custom holdout and official MATH test')
    atomic_json(root / 'exclusions.json', records)
    atomic_json(root / 'exclusion-summary.json', {'counts': dict(counts), 'unique_questions': len({r['key'] for r in records}),
                                               'math_revision': core.MATH_REVISION})
    return records


def build_references(root, inventory, db):
    """Index external expected answers before inspecting Olmo teacher solutions."""
    if not unit_done(db, 'refs-human-math-train'):
        with (PARENT / 'rl-data/math/train.jsonl').open() as stream:
            rows = [json.loads(line) for line in stream]
        db.executemany('INSERT OR IGNORE INTO refs VALUES (?,?,?,?)',
                       [(core.question_key(r['metadata']['messages'][0]['content']),
                         r['metadata']['target']['answer'], 'human-math-train', 0) for r in rows])
        commit_unit(root, db, 'refs-human-math-train', {'rows': len(rows)})
    for source in ('nemotron', 'openmath'):
        for item in inventory[source]['files']:
            name = 'refs-' + source + '-' + Path(item['path']).stem
            if unit_done(db, name): continue
            path = raw_file(root, inventory, source, item)
            columns = ['problem', 'expected_answer'] + (['problem_source'] if source == 'openmath' else [])
            stats = Counter()
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=columns):
                refs = []
                for row in batch.to_pylist():
                    stats['rows_scanned'] += 1
                    if source == 'openmath' and row['problem_source'] not in ('math', 'augmented_math'): continue
                    if not row['expected_answer'] or len(row['expected_answer']) > 1000: continue
                    refs.append((core.question_key(row['problem']), row['expected_answer'], source,
                                 1 if source == 'nemotron' else 2))
                db.executemany('INSERT OR IGNORE INTO refs VALUES (?,?,?,?)', refs)
                stats['reference_rows'] += len(refs)
            commit_unit(root, db, name, stats)


@lru_cache(maxsize=131072)
def references(db, key):
    # The augmented OpenMath expected answers alone are not a quality gate:
    # manual review found inconsistent questions with mutually agreeing wrong
    # answers. Require a human MATH or curated Nemotron reference instead.
    rows = db.execute("SELECT answer,source,rank FROM refs WHERE key=? AND source IN ('human-math-train','nemotron') ORDER BY rank LIMIT 8", (key,)).fetchall()
    if not rows: return [], []
    # The human training reference, when available, has priority over synthetic labels.
    rows = [r for r in rows if r[2] == rows[0][2]]
    return [r[0] for r in rows], [r[1] for r in rows]


def prune_legacy_quality(root, db):
    if unit_done(db, 'trusted-reference-quality-v1'): return
    stats = Counter()
    with (Path(root) / 'quality-removals.jsonl').open('a') as out:
        for key, payload in db.execute('SELECT key,payload FROM accepted').fetchall():
            value = json.loads(payload)
            reason = (None if set(value['reference_sources']) & {'human-math-train', 'nemotron'}
                      else 'synthetic_only_reference')
            reason = reason or core.quality_reason(value['problem'], value['response'])
            if reason is None and not core.numeric_consistency(value['response']):
                reason = 'incorrect_numeric_equation'
            if reason:
                db.execute('DELETE FROM accepted WHERE key=?', (key,))
                out.write(json.dumps({'reason': reason, 'example': value}) + '\n'); stats[reason] += 1
            else: stats['retained'] += 1
    commit_unit(root, db, 'trusted-reference-quality-v1', stats)


def verify_group(candidates):
    counts = Counter()
    for candidate in sorted(candidates, key=lambda c: len(c['response'])):
        value, reason = core.verify_candidate(candidate)
        if value is not None: return value, dict(counts)
        counts[reason] += 1
    return None, dict(counts)


def prepare_math(root, inventory, db, heldout, workers):
    accepted = {row[0] for row in db.execute('SELECT key FROM accepted')}
    trusted = {row[0] for row in db.execute("SELECT DISTINCT key FROM refs WHERE source IN ('human-math-train','nemotron')")}
    pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                               initializer=core.initialize_worker,
                               initargs=(str(PARENT / 'model'), str(RUNTIME)))
    try:
        # Short externally checked Olmo solutions are selected first where a
        # reference is available; their long hidden traces are never truncated.
        for source in ('olmo', 'nemotron', 'openmath'):
            for item in inventory[source]['files']:
                if source == 'olmo' and not any('math' in c.casefold() for c in item['categories']): continue
                if source == 'nemotron' and 'cot' not in item['categories']: continue
                name = 'math-' + source + '-' + Path(item['path']).stem
                if unit_done(db, name): continue
                path = raw_file(root, inventory, source, item)
                # Reprocess an interrupted unit, retaining every completed unit.
                accepted.difference_update(r[0] for r in db.execute('SELECT key FROM accepted WHERE unit=?', (name,)))
                db.execute('DELETE FROM accepted WHERE unit=?', (name,)); db.commit()
                stats = Counter()
                columns = None
                if source == 'nemotron':
                    # The separate hidden-reasoning column dominates this
                    # release's bytes; the final-solution experiment does not
                    # need to decode or tokenize it.
                    columns = ['problem', 'expected_answer', 'uuid', 'source', 'subset', 'license',
                               'url', 'user_url', 'username', 'tools',
                               'messages.list.element.role', 'messages.list.element.content',
                               'messages.list.element.tool_calls']
                for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=columns):
                    groups = defaultdict(list)
                    for row in batch.to_pylist():
                        stats['rows_scanned'] += 1
                        if source == 'openmath' and row['problem_source'] not in ('math', 'augmented_math'):
                            stats['not_math_subset'] += 1; continue
                        if source == 'nemotron' and row['subset'].casefold() != 'cot':
                            stats['tool_assisted_subset'] += 1; continue
                        if source == 'nemotron' and row['source'] != 'AoPS':
                            stats['outside_selected_competition_source'] += 1; continue
                        if source == 'olmo' and 'math' not in row['dataset_source'].casefold():
                            stats['not_math_subset'] += 1; continue
                        problem = row.get('problem') or next((m['content'] for m in row['messages'] if m['role'] == 'user'), '')
                        key = core.question_key(problem)
                        if key in heldout: stats['heldout_question'] += 1; continue
                        if key in accepted: stats['duplicate_question'] += 1; continue
                        if key not in trusted: stats['synthetic_only_reference'] += 1; continue
                        answers, origins = references(db, key)
                        if not answers: stats['no_independent_reference'] += 1; continue
                        value, reason = core.candidate(source, row)
                        if value is None: stats[reason] += 1; continue
                        key = value['key']
                        if len(value['response']) > 14000: stats['clearly_long_response'] += 1; continue
                        value.update(reference_answers=answers, reference_sources=origins)
                        # Keep several alternatives so an answer-only first trace
                        # does not hide a later complete, short worked solution.
                        if len(groups[key]) < 4: groups[key].append(value)
                        else: stats['extra_alternative'] += 1
                    for value, failures in pool.map(verify_group, groups.values(), chunksize=8):
                        stats.update(failures)
                        if value is None: continue
                        ids, mask = value.pop('encoded_ids'), value.pop('encoded_mask')
                        db.execute('INSERT INTO accepted VALUES (?,?,?,?,?,?,?)',
                                   (value['key'], source, value['tokens'], json.dumps(value), name, ids, mask))
                        accepted.add(value['key'])
                        stats['accepted_questions'] += 1
                        stats['tokens'] += value['tokens']; stats['response_tokens'] += value['response_tokens']
                commit_unit(root, db, name, stats)
                path.unlink()  # References and accepted records are committed; the raw shard is reproducible.
    finally:
        pool.shutdown()


def simhash(text):
    words = re.findall(r'\\[A-Za-z]+|[A-Za-z]+|\d+(?:\.\d+)?|[^\w\s]', text)
    shingles = set(' '.join(words[i:i + 3]) for i in range(max(1, len(words) - 2)))
    hashes = np.array([int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'little')
                       for s in shingles], dtype='<u8')
    bits = np.unpackbits(hashes.view('u1').reshape(-1, 8), axis=1, bitorder='little')
    positive = bits.sum(axis=0) * 2 >= len(hashes)
    value = sum(int(bit) << i for i, bit in enumerate(positive))
    return value, words


def near_equal(left, right):
    """Conservative fuzzy dedup: retain changed constants and logical qualifiers."""
    if re.findall(r'-?\d+(?:\.\d+)?', left) != re.findall(r'-?\d+(?:\.\d+)?', right): return False
    qualifiers = r'\b(?:least|greatest|maximum|minimum|not|except|negative|positive|integer|real|complex|distinct|ordered|unordered|exactly)\b'
    if Counter(re.findall(qualifiers, left.casefold())) != Counter(re.findall(qualifiers, right.casefold())): return False
    # A high shingle overlap alone can merge long questions whose requested
    # operation or variable differs by one word. Permit presentation changes
    # only: preserve the ordered mathematical and substantive language tokens.
    def meaning_tokens(text):
        words = re.findall(r'\\[A-Za-z]+|\w+|[^\w\s]', text.rstrip().rstrip('.'))
        return tuple(w if len(w) == 1 or w.startswith('\\') else w.casefold()
                     for w in words if w.casefold() not in ('the', 'please', 'kindly'))
    if meaning_tokens(left) != meaning_tokens(right): return False
    def shingles(text):
        words = re.findall(r'\w+|[^\w\s]', text)
        return set(tuple(words[i:i + 3]) for i in range(max(1, len(words) - 2)))
    a, b = shingles(left), shingles(right)
    return bool(a | b) and len(a & b) / len(a | b) >= .97


def register_near(db, key, problem, is_heldout=False):
    h, _ = simhash(problem); bands = [(h >> (16 * i)) & 65535 for i in range(4)]
    if not is_heldout:
        for previous, text, oldhash, hold in db.execute(
                'SELECT key,problem,hash,heldout FROM near_seen WHERE b0=? OR b1=? OR b2=? OR b3=?', bands):
            if (h ^ int(oldhash, 16)).bit_count() <= 3 and near_equal(problem, text):
                return previous, hold
    db.execute('INSERT OR IGNORE INTO near_seen VALUES (?,?,?,?,?,?,?,?)',
               (key, problem, hex(h), *bands, int(is_heldout)))
    return None


def dedupe_near(root, db, excluded):
    if unit_done(db, 'near-dedup'): return
    # A retry can safely rebuild the index. Rows deleted as near duplicates
    # remain deleted; records of each removal are kept separately.
    db.executescript('''DROP TABLE IF EXISTS near_seen;
        CREATE TABLE near_seen (key TEXT PRIMARY KEY, problem TEXT, hash TEXT,
                                b0 INTEGER,b1 INTEGER,b2 INTEGER,b3 INTEGER,heldout INTEGER);
        CREATE INDEX near0 ON near_seen(b0); CREATE INDEX near1 ON near_seen(b1);
        CREATE INDEX near2 ON near_seen(b2); CREATE INDEX near3 ON near_seen(b3);''')
    for row in excluded: register_near(db, row['key'], row['problem'], True)
    stats = Counter()
    removed = Path(root) / 'near-duplicate-removals.jsonl'
    with removed.open('a') as out:
        for source in ('olmo', 'nemotron', 'openmath'):
            cursor = db.execute('SELECT key,payload FROM accepted WHERE source=? ORDER BY key', (source,))
            while rows := cursor.fetchmany(512):
                for key, payload in rows:
                    value = json.loads(payload); match = register_near(db, key, value['problem'])
                    if match:
                        db.execute('DELETE FROM accepted WHERE key=?', (key,))
                        stats['near_holdout' if match[1] else 'near_duplicate'] += 1
                        matched_problem = db.execute('SELECT problem FROM near_seen WHERE key=?', (match[0],)).fetchone()[0]
                        out.write(json.dumps({'key': key, 'matches': match[0], 'heldout': bool(match[1]),
                                              'source': source, 'id': value.get('id'),
                                              'problem': value['problem'], 'matched_problem': matched_problem}) + '\n')
                    else: stats['retained'] += 1
    commit_unit(root, db, 'near-dedup', stats)


def general_worker(value):
    from . import prepare
    messages = value['messages']
    assistant = messages[-1]['content'] or ''
    n = len(core.TOKENIZER(assistant, add_special_tokens=False)['input_ids'])
    if n > core.RESPONSE_LIMIT: return None, 'general_response_over_1536'
    ids, mask, clean, _ = prepare.encode(messages)
    if len(ids) > core.CONTEXT_LIMIT: return None, 'general_conversation_over_2048'
    value.update(messages=clean, tokens=len(ids), response_tokens=n, assistant_tokens=int(mask.sum()))
    value['encoded_ids'] = np.asarray(ids, dtype='<u4').tobytes()
    value['encoded_mask'] = mask.tobytes()
    return value, None


def prepare_general(root, db, heldout, workers):
    if unit_done(db, 'general-complete'): return
    # Reserve a deterministic internal validation set before computing the train ratio.
    val_keys = {r[0] for r in db.execute("SELECT key FROM accepted WHERE source!='general' ORDER BY key LIMIT 512")}
    math_tokens = db.execute("SELECT coalesce(sum(tokens),0) FROM accepted WHERE source!='general'").fetchone()[0]
    val_tokens = sum(db.execute('SELECT tokens FROM accepted WHERE key=?', (k,)).fetchone()[0] for k in val_keys)
    target = int((math_tokens - val_tokens) / 4)
    manifest = json.loads((PARENT / 'manifest.json').read_text())
    permitted = {v['source_dataset']: n for n, v in manifest['sources'].items()
                 if v['domain'] not in ('Math', 'Tool Use', 'ToolUse')}
    masses = {n: manifest['recipe']['mixes']['baseline'][source_id] for n, source_id in permitted.items()}
    denominator = sum(masses.values())
    targets = {n: int(target * mass / denominator) for n, mass in masses.items()}
    used = Counter()
    for (payload,) in db.execute("SELECT payload FROM accepted WHERE source='general'"):
        value = json.loads(payload); used[value['origin']] += value['tokens']
    pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                               initializer=core.initialize_worker, initargs=(str(PARENT / 'model'), str(RUNTIME)))
    try:
        for path in sorted((PARENT / 'raw/data').glob('*.parquet')):
            name = 'general-' + path.stem
            if unit_done(db, name): continue
            stats = Counter()
            # Keep completed shards; interrupted units are replayed by skipping
            # their already accepted keys, without repeating any training row.
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
                candidates, pending = [], set()
                for row in batch.to_pylist():
                    origin = row['source_dataset']
                    if origin not in targets or used[origin] >= targets[origin]: continue
                    messages = row['messages']
                    if len(messages) != 2 or [m['role'] for m in messages] != ['user', 'assistant']: continue
                    if any(m.get('functions') or m.get('function_calls') for m in messages): continue
                    problem = core.canonical_question(messages[0]['content'] or ''); key = core.question_key(problem)
                    if not problem or key in heldout or key in pending: continue
                    if db.execute('SELECT 1 FROM accepted WHERE key=?', (key,)).fetchone():
                        stats['general_duplicate'] += 1; continue
                    pending.add(key)
                    candidates.append({'key': key, 'problem': problem, 'source': 'general', 'origin': origin,
                                       'id': row['id'], 'messages': messages, 'license': 'odc-by'})
                for value, reason in pool.map(general_worker, candidates, chunksize=8):
                    if value is None: stats[reason] += 1; continue
                    if used[value['origin']] >= targets[value['origin']]: continue
                    if register_near(db, value['key'], value['problem']):
                        stats['general_near_duplicate_or_holdout'] += 1; continue
                    ids, mask = value.pop('encoded_ids'), value.pop('encoded_mask')
                    db.execute('INSERT INTO accepted VALUES (?,?,?,?,?,?,?)',
                               (value['key'], 'general', value['tokens'], json.dumps(value), name, ids, mask))
                    used[value['origin']] += value['tokens']; stats['accepted_general'] += 1
            commit_unit(root, db, name, stats)
    finally: pool.shutdown()
    if sum(used.values()) < target * .98:
        raise ValueError(f'Only {sum(used.values())} unique general tokens for {target} required; do not repeat to fill')
    atomic_json(Path(root) / 'general-selection.json', {'target_tokens': target, 'source_targets': targets,
                'actual_source_tokens': dict(used), 'definition': 'Short single-turn tool-free non-Math Dolci Instruct, approximately baseline token-proportional'})
    atomic_json(Path(root) / 'sft-validation-keys.json', sorted(val_keys))
    commit_unit(root, db, 'general-complete', {'tokens': sum(used.values()), 'target': target})


def prepare(root, workers=16):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    if (root / 'prepared.json').exists():
        from .math_epoch import pack
        pack(root); return
    inventory = json.loads((root / 'inventory.json').read_text())
    if set(inventory) != set(core.DATASETS):
        raise ValueError('Finish all three source inventories before preparing the corpus')
    excluded = exclusions(root); heldout = {v['key'] for v in excluded}
    db = open_database(root)
    try:
        build_references(root, inventory, db)
        prune_legacy_quality(root, db)
        prepare_math(root, inventory, db, heldout, workers)
        dedupe_near(root, db, excluded)
        counts = dict(db.execute('SELECT source,count(*) FROM accepted GROUP BY source'))
        if any(counts.get(source, 0) == 0 for source in core.DATASETS):
            raise ValueError('At least one requested math source has no independently checked short solutions: ' + repr(counts))
        prepare_general(root, db, heldout, workers)
        atomic_json(root / 'prepared.json', {'passed': True, 'counts': dict(db.execute('SELECT source,count(*) FROM accepted GROUP BY source')),
                    'response_limit': core.RESPONSE_LIMIT, 'context_limit': core.CONTEXT_LIMIT,
                    'math_epochs': 1, 'math_share_target': .8, 'independent_math_references_required': True,
                    'sources': {n: {'repo': v['repo'], 'revision': v['revision']} for n, v in inventory.items()}})
    finally: db.close()
    from .math_epoch import pack
    pack(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args(); prepare(args.root, args.workers)
