"""Parallel, resumable source scans for the large short-math experiment."""

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import sqlite3
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from . import math_large as core
from .config import atomic_json
from .math_refresh import download


def database(path):
    db = sqlite3.connect(path, timeout=120)
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=NORMAL')
    db.execute('PRAGMA cache_size=-131072')
    db.executescript('''CREATE TABLE IF NOT EXISTS accepted
        (key TEXT PRIMARY KEY, presentation TEXT UNIQUE, source TEXT, origin TEXT,
         tokens INTEGER, response_tokens INTEGER, quality INTEGER, payload TEXT, ids BLOB, mask BLOB);
        CREATE TABLE IF NOT EXISTS units (name TEXT PRIMARY KEY, stats TEXT);''')
    return db


def human_references():
    values = {}
    with (core.PARENT / 'rl-data/math/train.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)['metadata']
            values[core.key(row['messages'][0]['content'])] = row['target']['answer']
    return values


def general(row):
    if row['domain'] in ('Math', 'Tool Use', 'ToolUse'):
        return None, 'not_general'
    messages = row['messages']
    if [m['role'] for m in messages] != ['user', 'assistant']:
        return None, 'not_single_turn'
    if any(m.get('functions') or m.get('function_calls') or m.get('tool_calls') for m in messages):
        return None, 'tool_dependent'
    problem, response = messages[0]['content'] or '', messages[1]['content'] or ''
    if not problem or not response:
        return None, 'empty'
    return {'key': core.key(problem), 'presentation_key': core.presentation_key(problem),
            'problem': problem, 'response': response, 'messages': messages,
            'source': 'general', 'origin': row['source_dataset'], 'domain': row['domain'],
            'id': row['id'], 'license': 'odc-by'}, None


def verify_general(value):
    if core.EXCLUDED.contains(value['problem']):
        return None, 'holdout_or_embedded_holdout'
    n = len(core.TOKENIZER(value['response'], add_special_tokens=False)['input_ids'])
    if n > core.RESPONSE_LIMIT:
        return None, 'response_over_1792'
    ids, mask, messages, _ = core.ENCODE(value['messages'])
    if len(ids) > core.CONTEXT_LIMIT:
        return None, 'conversation_over_2048'
    value.update(messages=messages, tokens=len(ids), response_tokens=n,
                 assistant_tokens=int(mask.sum()), quality=10000, verification='general_sft')
    value['ids'] = np.asarray(ids, dtype='<u4').tobytes()
    value['mask'] = mask.tobytes()
    return value, None


def process_group(item):
    kind, values = item
    return verify_general(values[0]) if kind == 'general' else core.group(values)


def save(db, value):
    ids, mask = value.pop('ids'), value.pop('mask')
    db.execute('INSERT OR IGNORE INTO accepted VALUES (?,?,?,?,?,?,?,?,?,?)', (
        value['key'], value['presentation_key'], value['source'], value['origin'],
        value['tokens'], value['response_tokens'], value['quality'], json.dumps(value), ids, mask))


def run(root, source, partition=0, partitions=1, workers=12):
    root = Path(root).resolve()
    name = f'{source}-{partition}'
    directory = root / 'candidates'
    directory.mkdir(exist_ok=True)
    complete_path = directory / (name + '.complete.json')
    if complete_path.exists():
        return
    db = database(directory / (name + '.sqlite'))
    humans = human_references()
    inventory = json.loads((root / 'inventory.json').read_text())
    done = {v[0] for v in db.execute('SELECT name FROM units')}
    accepted = {v[0] for v in db.execute('SELECT key FROM accepted')}
    if source in ('dolci', 'general'):
        files = [{'path': str(p)} for p in sorted((core.PARENT / 'raw/data').glob('*.parquet'))]
    else:
        files = inventory[source]['files']
    file_partitioned = source in ('reasoning', 'prism') and partitions > 1
    if file_partitioned:
        files = [item for i, item in enumerate(files) if i % partitions == partition]
    source_stats = Counter()
    general_tokens = Counter()
    for origin, total in db.execute('SELECT origin,sum(tokens) FROM accepted GROUP BY origin'):
        general_tokens[origin] = total
    baseline = json.loads((core.PARENT / 'manifest.json').read_text())
    allowed = {v['source_dataset']: baseline['recipe']['mixes']['baseline'][k]
               for k, v in baseline['sources'].items() if v['domain'] not in ('Math', 'Tool Use', 'ToolUse')}
    denominator = sum(allowed.values())
    general_targets = {k: int(260_000_000 * v / denominator) for k, v in allowed.items()}
    with tempfile.TemporaryDirectory(prefix='math-large-' + name + '-', dir='/tmp') as temporary:
        temporary = Path(temporary)
        reference = None
        if source == 'olmo':
            shutil.copy2(core.EARLIER / 'corpus.sqlite', temporary / 'references.sqlite')
            reference = sqlite3.connect(temporary / 'references.sqlite')

        @lru_cache(maxsize=131072)
        def references(question):
            if reference is None:
                return []
            records = reference.execute('SELECT answer FROM refs WHERE key=? ORDER BY rank LIMIT 4',
                                        (core.old.question_key(question),)).fetchall()
            return [v[0] for v in records]

        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=core.initialize, initargs=(str(root),)) as pool:
            for item in files:
                unit = Path(item['path']).stem
                if unit in done:
                    continue
                if source == 'olmo' and not any('math' in c.casefold() for c in item['categories']):
                    continue
                if source == 'nemotron' and 'cot' not in item['categories']:
                    continue
                local = source not in ('dolci', 'general')
                path = temporary / (unit + '.parquet') if local else Path(item['path'])
                if local:
                    info = inventory[source]
                    download(info['repo'], info['revision'], item['path'], path)
                columns = None
                if source == 'nemotron':
                    columns = ['problem', 'expected_answer', 'uuid', 'source', 'subset', 'license',
                               'url', 'user_url', 'username', 'tools', 'messages.list.element.role',
                               'messages.list.element.content', 'messages.list.element.tool_calls']
                if source == 'olmo':
                    columns = ['id', 'dataset_source', 'messages']
                stats = Counter()
                for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=columns):
                    groups = defaultdict(list)
                    for row in batch.to_pylist():
                        stats['raw_rows'] += 1
                        if source in ('openmath', 'reasoning', 'numina', 'prism'):
                            if source == 'openmath' and row['problem_source'] not in ('math', 'augmented_math', 'gsm8k', 'augmented_gsm8k'):
                                stats['not_math'] += 1
                                continue
                            raw_key = core.key(row['problem'])
                            if not file_partitioned and int(raw_key[:8], 16) % partitions != partition:
                                continue
                            if raw_key in accepted:
                                stats['duplicate_question'] += 1
                                continue
                        if source == 'general':
                            origin = row['source_dataset']
                            if origin not in general_targets or general_tokens[origin] >= general_targets[origin]:
                                continue
                            value, reason = general(row)
                        else:
                            value, reason = core.standardize(source, row)
                        if value is None:
                            stats[reason] += 1
                            continue
                        k = value['key']
                        if not file_partitioned and int(k[:8], 16) % partitions != partition:
                            continue
                        if k in accepted:
                            stats['duplicate_question'] += 1
                            continue
                        if k in humans:
                            value['references'] = [humans[k]]
                            value['reference_kind'] = 'human_reference'
                        elif source == 'olmo':
                            value['references'] = references(value['problem'])
                            if value['references']:
                                value['reference_kind'] = 'provider_reference'
                        if len(groups[k]) < 4:
                            groups[k].append(value)
                    items = [('general' if source == 'general' else 'math', values) for values in groups.values()]
                    for value, failures in pool.map(process_group, items, chunksize=8):
                        if source == 'general':
                            if value is None:
                                stats[failures] += 1
                                continue
                        else:
                            stats.update(failures)
                        if value is None:
                            continue
                        if source == 'general' and general_tokens[value['origin']] >= general_targets[value['origin']]:
                            continue
                        count = db.total_changes
                        save(db, value)
                        if db.total_changes != count:
                            accepted.add(value['key'])
                            stats['accepted'] += 1
                            stats['tokens'] += value['tokens']
                            general_tokens[value['origin']] += value['tokens']
                    db.commit()
                db.execute('INSERT OR REPLACE INTO units VALUES (?,?)', (unit, json.dumps(stats)))
                db.commit()
                source_stats.update(stats)
                totals = db.execute('SELECT count(*),coalesce(sum(tokens),0) FROM accepted').fetchone()
                atomic_json(directory / (name + '.progress.json'), {'unit': unit,
                    'unique_questions': totals[0], 'unique_tokens': totals[1], 'stats': dict(stats),
                    'partition': partition, 'partitions': partitions})
                print('UNIT', name, unit, totals, dict(stats), flush=True)
                if local:
                    path.unlink()
    totals = db.execute('SELECT count(*),coalesce(sum(tokens),0) FROM accepted').fetchone()
    atomic_json(complete_path, {'source': source, 'partition': partition, 'partitions': partitions,
                               'unique_questions': totals[0], 'unique_tokens': totals[1],
                               'database': str(directory / (name + '.sqlite'))})
    db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', choices=['openmath', 'reasoning', 'numina', 'prism', 'olmo', 'nemotron', 'general', 'dolci'], required=True)
    parser.add_argument('--partition', type=int, default=0)
    parser.add_argument('--partitions', type=int, default=1)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    run(args.root, args.source, args.partition, args.partitions, args.workers)
