"""Global question deduplication, capacity gates, and whole-conversation packing."""

import argparse
import hashlib
import json
import math
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import math_large as core
from .config import ACTUAL_TOKENS, SEED, atomic_json, fingerprint
from .math_epoch import assignments
from .math_refresh_build import simhash


def close_question(a, b):
    """Near-copy detection preserves constants, variables, and the requested task."""
    if re.findall(r'\d+(?:\.\d+)?', a) != re.findall(r'\d+(?:\.\d+)?', b):
        return False
    operations = r'\b(?:sum|product|difference|quotient|area|perimeter|volume|probability|mean|median|'
    operations += r'minimum|maximum|least|greatest|positive|negative|distinct|ordered|unordered|not|except)\b'
    if Counter(re.findall(operations, a.lower())) != Counter(re.findall(operations, b.lower())):
        return False
    symbols = r'\\[A-Za-z]+|(?<![A-Za-z])[A-Za-z](?![A-Za-z])|[+*/^=<>-]'
    if re.findall(symbols, a) != re.findall(symbols, b):
        return False
    left, right = core.shingles(a), core.shingles(b)
    return len(left) >= 8 and len(left & right) / max(1, len(left | right)) >= .9


def global_rejection(value):
    return (core.unreliable(value['response']) or core.outside_scope(value['problem'])
            or core.invented_integer_roots(value['problem'], value['response'])
            or bool(re.search(r'<image>|!\[.*?\]\(', value['problem'] + '\n' + value['response'], re.I))
            or bool(re.search(r'(?i)\bdifferential equations?\b|\bintegrat(?:e|ion)\b|'
                              r'\bantiderivative\b|\bjacobian\b|\bmilp\b|\blinear programming\b|'
                              r'\brecommendation algorithm\b|\bformulate.{0,40}optimization problem\b|'
                              r'\bfind a norm\b|\bvector space\b|\bcrossing number\b|cr\(Q_', value['problem']))
            or bool(re.search(r'\\(?:i?i?int|oint)(?![A-Za-z])', value['response'])))


def sampling_quotas(labeled_blocks, unlabeled_blocks, general_blocks, total_tokens=ACTUAL_TOKENS,
                    average_lengths=None):
    total = total_tokens // 2048

    def allocate(math_quota):
        general = total - math_quota
        if general > 2 * general_blocks or math_quota > 2 * (labeled_blocks + unlabeled_blocks):
            raise ValueError('Insufficient unique packed data for the two-pass hard limit')
        # Prefer labeled solutions within 1.5 passes; otherwise minimize the
        # necessary exposure instead of immediately pushing one pool to two.
        limit = 1.5
        if math.floor(limit * labeled_blocks) + math.floor(limit * unlabeled_blocks) < math_quota:
            low, high = 1.5, 2.
            for _ in range(60):
                middle = (low + high) / 2
                if math.floor(middle * labeled_blocks) + math.floor(middle * unlabeled_blocks) >= math_quota:
                    high = middle
                else:
                    low = middle
            limit = high
        for limit in (limit, 2.):
            lower = max(0, math_quota - math.floor(limit * unlabeled_blocks))
            upper = min(math_quota, math.floor(limit * labeled_blocks))
            if lower <= upper:
                labeled = min(max(round(math_quota * .8), lower), upper)
                return {'math-labeled': labeled, 'math-unlabeled': math_quota - labeled,
                        'general': general}, limit
        raise ValueError('Insufficient unique packed data for the two-pass hard limit')

    if average_lengths is None:
        return allocate(round(total * .8))
    if any(not 0 < v <= 2048 for v in average_lengths.values()):
        raise ValueError('Invalid mean nonpadding block lengths')

    def share(math_quota):
        quotas, _ = allocate(math_quota)
        tokens = {k: quotas[k] * average_lengths[k] for k in quotas}
        return (tokens['math-labeled'] + tokens['math-unlabeled']) / sum(tokens.values())

    # The requested 80/20 mix concerns actual input text, not padding slots.
    # Whole-conversation packing can have substantially different fill rates
    # for long unlabeled answers and shorter labeled/general answers.
    minimum, maximum = max(0, total - 2 * general_blocks), min(total, 2 * (labeled_blocks + unlabeled_blocks))
    if minimum > maximum:
        raise ValueError('Insufficient unique packed data for the two-pass hard limit')
    low, high = minimum, maximum
    while low < high:
        middle = (low + high) // 2
        if share(middle) < .8:
            low = middle + 1
        else:
            high = middle
    chosen = min({low, max(minimum, low - 1)}, key=lambda m: abs(share(m) - .8))
    if abs(share(chosen) - .8) > .002:
        raise ValueError('The 80/20 nonpadding-token mix cannot fit within two passes')
    return allocate(chosen)


def merge(root, additional=None, *, preserve_existing_solutions=False):
    root = Path(root)
    path = root / 'merged.sqlite'
    if (root / 'merge-complete.json').exists() and not additional:
        return
    # Rebuild a partially merged cache; original source candidates remain intact
    # until the merge is committed and its counts are recorded.
    if not additional:
        path.unlink(missing_ok=True)
    elif not path.exists():
        raise ValueError('Append requires the retained committed global merge cache')
    db = sqlite3.connect(path)
    db.execute('PRAGMA cache_size=-1048576')
    db.execute('PRAGMA synchronous=OFF')  # Reproducible intermediate cache only.
    db.execute('PRAGMA temp_store=MEMORY')
    db.executescript('''CREATE TABLE IF NOT EXISTS accepted
        (key TEXT PRIMARY KEY, presentation TEXT UNIQUE, bucket TEXT, source TEXT,
         tokens INTEGER, quality INTEGER, problem TEXT, payload TEXT, ids BLOB, mask BLOB,
         hash TEXT,b0 INTEGER,b1 INTEGER,b2 INTEGER,b3 INTEGER);
        ''')
    stats = Counter()
    manual_path = root / 'manual-rejections.json'
    rejected_ids = {v['id'] for v in json.loads(manual_path.read_text())} if manual_path.exists() else set()
    known_bad = json.loads(Path(__file__).with_name('math_refresh_exclusions.json').read_text())
    reviewed = json.loads(manual_path.read_text()) if manual_path.exists() else []
    rejected_questions = {core.key(v['problem']) for v in [*known_bad, *reviewed] if v.get('problem')}
    nearby = [defaultdict(list) for _ in range(4)]
    seen_questions = {}
    if additional:
        discard = []
        for k, h, problem, payload, *bands in db.execute('SELECT key,hash,problem,payload,b0,b1,b2,b3 FROM accepted'):
            value = json.loads(payload)
            if value['source'] != 'general' and (value['id'] in rejected_ids or k in rejected_questions or global_rejection(value)):
                discard.append((k,))
                rejected_questions.add(k)
                stats['rechecked_existing_quality_rejection'] += 1
                continue
            seen_questions[k] = int(h, 16), problem
            for i, band in enumerate(bands):
                nearby[i][band].append(k)
        db.executemany('DELETE FROM accepted WHERE key=?', discard)
        db.commit()
        old = json.loads((root / 'merge-complete.json').read_text())
        stats.update(old['dedup'])
    removals = (root / 'cross-source-dedup.jsonl').open('a' if additional else 'w')
    files = ([Path(p) for p in additional] if additional else
             [root / 'candidates' / (n + '.sqlite') for n in (
                 'openmath-0', 'openmath-1', 'nemotron-0', 'olmo-0', 'dolci-0', 'general-0')])
    for source_path in files:
        if not source_path.with_suffix('.complete.json').exists():
            raise ValueError('Finish the source scan before global dedup: ' + str(source_path))
        source = sqlite3.connect(f'file:{source_path}?mode=ro', uri=True)
        atomic_json(root / 'source-unit-audits' / (source_path.stem + '.json'),
                    {name: json.loads(stats) for name, stats in source.execute('SELECT name,stats FROM units')})
        for row in source.execute('SELECT key,presentation,source,tokens,quality,payload,ids,mask FROM accepted'):
            key, presentation, origin, tokens, quality, payload, ids, mask = row
            value = json.loads(payload)
            if origin != 'general' and global_rejection(value):
                stats['global_reasoning_quality_rejection'] += 1
                continue
            if value.get('id') in rejected_ids or key in rejected_questions:
                stats['documented_manual_rejection'] += 1
                continue
            problem = value['problem']
            bucket = ('general' if origin == 'general' else
                      'math-labeled' if value.get('verification') in ('human_reference', 'provider_reference')
                      else 'math-unlabeled')
            previous = db.execute('SELECT key,quality,payload FROM accepted WHERE key=? OR presentation=?',
                                  (key, presentation)).fetchone()
            if previous:
                stats['exact_or_presentation_duplicates'] += 1
                if preserve_existing_solutions:
                    continue
                old_value = json.loads(previous[2])
                new_rank = hashlib.sha256(('solution-choice:' + origin + ':' + str(value['id'])).encode()).hexdigest()
                old_rank = hashlib.sha256(('solution-choice:' + old_value['source'] + ':' + str(old_value['id'])).encode()).hexdigest()
                if quality < previous[1] or quality == previous[1] and new_rank >= old_rank:
                    continue
                db.execute('DELETE FROM accepted WHERE key=?', (previous[0],))
                seen_questions.pop(previous[0], None)
            h, _ = simhash(problem)
            bands = [(h >> (16 * i)) & 65535 for i in range(4)]
            duplicate = None
            candidates = {k for i, band in enumerate(bands) for k in nearby[i].get(band, [])}
            for prior_key in candidates:
                prior = seen_questions.get(prior_key)
                if prior is None:
                    continue
                prior_hash, prior_problem = prior
                if (h ^ prior_hash).bit_count() <= 5 and close_question(problem, prior_problem):
                    duplicate = prior_key
                    break
            if duplicate:
                stats['near_duplicates'] += 1
                removals.write(json.dumps({'removed_key': key, 'retained_key': duplicate,
                                            'source': origin, 'problem': problem}) + '\n')
                continue
            # The messages repeat problem/response verbatim in source caches.
            # Reconstruct them during packing to keep the committed cache small.
            value.pop('messages', None)
            db.execute('INSERT INTO accepted VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (key, presentation, bucket, origin, tokens, quality, problem,
                        json.dumps(value), ids, mask, hex(h), *bands))
            seen_questions[key] = h, problem
            for i, band in enumerate(bands):
                nearby[i][band].append(key)
            stats['insertions'] += 1
            if stats['insertions'] % 5000 == 0:
                db.commit()
                print('MERGED', source_path.stem, dict(stats), flush=True)
        source.close()
        db.commit()
    removals.close()
    counts = {bucket: {'questions': n, 'tokens': t} for bucket, n, t in
              db.execute('SELECT bucket,count(*),sum(tokens) FROM accepted GROUP BY bucket')}
    counts['math'] = {k: sum(counts.get(n, {}).get(k, 0) for n in ('math-labeled', 'math-unlabeled'))
                      for k in ('questions', 'tokens')}
    sources = {source: {'questions': n, 'tokens': t} for source, n, t in
               db.execute('SELECT source,count(*),sum(tokens) FROM accepted GROUP BY source')}
    db.close()
    atomic_json(root / 'merge-complete.json', {'counts': counts, 'sources': sources, 'dedup': dict(stats)})
    # All retained examples, masks, provenance and IDs now live in the merged
    # database; source caches can be regenerated from pinned source releases.
    retired = []
    for candidate in files:
        retired.append({'path': str(candidate), 'bytes': candidate.stat().st_size})
    prior_receipt = root / 'source-caches-retired.json'
    prior_files = json.loads(prior_receipt.read_text())['files'] if additional and prior_receipt.exists() else []
    atomic_json(prior_receipt, {'reason': 'Verified committed global merge; preserve source completion counts and final accepted records', 'files': prior_files + retired})
    for candidate in files:
        candidate.unlink()


def pack(root):
    root = Path(root)
    if (root / 'curation-audit.json').exists():
        return
    merged = json.loads((root / 'merge-complete.json').read_text())
    capacity = core.capacity(merged['counts']['math']['tokens'], merged['counts']['general']['tokens'])
    db = sqlite3.connect(root / 'merged.sqlite')
    db.execute('PRAGMA cache_size=-1048576')
    all_docs = {}
    for row in db.execute('SELECT rowid,key,bucket,source,tokens,payload,ids,mask FROM accepted'):
        all_docs[row[0]] = row[1:]
    output = root / 'sources'
    output.mkdir(exist_ok=True)
    sources, reports, review = {}, {}, []
    providers = sorted(merged['sources'])
    excluded = core.Exclusions(json.loads((root / 'exclusions.json').read_text()) +
                               json.loads((root / 'additional-exclusions.json').read_text()))
    for bucket in ('math-labeled', 'math-unlabeled', 'general'):
        docs = [(i, row[3]) for i, row in all_docs.items() if row[1] == bucket]
        bins = assignments(docs)
        targets, real_lengths, source_tokens = [], [], Counter()
        response_lengths = []
        provider_rows, provider_question_rows, origin_tokens = [], [], Counter()
        token_hash, mask_hash = hashlib.sha256(), hashlib.sha256()
        expected_targets = 0
        seen = set()
        examples = (root / (bucket + '.jsonl')).open('w')
        with (output / (bucket + '.bin')).open('wb') as tokens, (output / (bucket + '.mask')).open('wb') as masks:
            for block in bins:
                ids = np.full(2049, 151643, dtype='<u4')
                mask = np.zeros(2049, dtype='u1')
                position = 0
                block_providers, block_provider_questions = Counter(), Counter()
                for i in block:
                    key, _, source, length, payload, encoded, encoded_mask = all_docs[i]
                    assert key not in seen
                    seen.add(key)
                    value = json.loads(payload)
                    assert not excluded.contains(value['problem'])
                    assert value['response_tokens'] <= 1792 and length <= 2048
                    sequence = np.frombuffer(encoded, dtype='<u4')
                    keep = np.frombuffer(encoded_mask, dtype='u1')
                    assert len(sequence) == len(keep) == length and not keep[0]
                    assert int(keep.sum()) == value['assistant_tokens']
                    expected_targets += value['assistant_tokens']
                    response_lengths.append(value['response_tokens'])
                    ids[position:position + length], mask[position:position + length] = sequence, keep
                    position += length
                    source_tokens[source] += length
                    origin_tokens[source + '/' + str(value['origin'])] += length
                    block_providers[source] += length
                    block_provider_questions[source] += 1
                    messages = [{'role': 'user', 'content': value['problem'] + (value.get('prompt_suffix', core.old.PROMPT_SUFFIX) if bucket != 'general' else '')},
                                {'role': 'assistant', 'content': value['response']}]
                    metadata = {k: v for k, v in value.items() if k not in ('messages', 'problem', 'response')}
                    metadata.update(packed_bucket=bucket, packed_block=len(targets))
                    examples.write(json.dumps({'messages': messages, 'metadata': metadata}, ensure_ascii=False) + '\n')
                    if int(hashlib.sha256(('review-independent-v1:' + key).encode()).hexdigest()[:8], 16) < 2**32 / 3000:
                        review.append({'messages': messages, 'metadata': metadata})
                assert position <= 2048
                ids.tofile(tokens)
                mask.tofile(masks)
                token_hash.update(ids.tobytes())
                mask_hash.update(mask.tobytes())
                targets.append(int(mask[1:].sum()))
                real_lengths.append(position)
                provider_rows.append([block_providers[n] for n in providers])
                provider_question_rows.append([block_provider_questions[n] for n in providers])
        examples.close()
        assert len(seen) == len(docs) == merged['counts'][bucket]['questions']
        assert sum(targets) == expected_targets
        assert (output / (bucket + '.bin')).stat().st_size == len(bins) * 2049 * 4
        assert (output / (bucket + '.mask')).stat().st_size == len(bins) * 2049
        np.save(output / (bucket + '.counts.npy'), np.asarray(targets, dtype=np.uint32))
        np.save(output / (bucket + '.lengths.npy'), np.asarray(real_lengths, dtype=np.uint32))
        np.save(output / (bucket + '.providers.npy'), np.asarray(provider_rows, dtype=np.uint32))
        np.save(output / (bucket + '.provider-questions.npy'), np.asarray(provider_question_rows, dtype=np.uint16))
        sources[bucket] = {'packed_tokens': len(bins) * 2049, 'blocks': len(bins),
                           'tokens': sum(real_lengths), 'questions': len(docs),
                           'sha256': {'.bin': token_hash.hexdigest(), '.mask': mask_hash.hexdigest()}}
        reports[bucket] = {'unique_questions': len(docs), 'unique_input_tokens': sum(real_lengths),
                           'supervised_targets': sum(targets), 'blocks': len(bins),
                           'source_input_tokens': dict(source_tokens),
                           'origin_input_tokens': dict(origin_tokens),
                           'response_tokens': {str(q): float(np.percentile(response_lengths, q))
                                               for q in (50, 90, 99, 100)},
                           'padding_fraction': 1 - sum(real_lengths) / (len(bins) * 2048)}
        print('PACKED', bucket, reports[bucket], flush=True)
    db.close()
    atomic_json(root / 'review-examples.json', review)
    # Prefer labeled math, first within the user's preferred 1.5-pass limit.
    # Use the authorized two-pass ceiling only if that preferred allocation
    # is impossible with the available unique data.
    quotas, allocation_limit = sampling_quotas(sources['math-labeled']['blocks'],
        sources['math-unlabeled']['blocks'], sources['general']['blocks'],
        average_lengths={k: v['tokens'] / v['blocks'] for k, v in sources.items()})
    epochs = {k: quotas[k] / sources[k]['blocks'] for k in quotas}
    if max(epochs.values()) > 2:
        raise ValueError('Packed source would exceed the two-epoch hard cap')
    manifest = {'repo': 'math-large-20261101', 'revision': {
        **{k: {'repo': v['repo'], 'revision': v['revision']} for k, v in json.loads((root / 'inventory.json').read_text()).items()},
        'dolci': {'repo': 'allenai/Dolci-Instruct-SFT', 'revision': 'bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221'}},
        'seq_len': 2048, 'seed': SEED, 'objective': 'assistant', 'sources': sources,
        'quotas': {'math': quotas}, 'pool_prediction_tokens': ACTUAL_TOKENS,
        'recipe': {'mixes': {'math': {k: v / sum(quotas.values()) for k, v in quotas.items()}}},
        'natural_definition': '80% short math / 20% general; globally deduplicated, maximum two passes',
        'whole_conversations': True, 'no_truncation': True, 'response_limit': 1792,
        'require_unique_blocks': max(epochs.values()) <= 1, 'maximum_epochs': 2,
        'effective_source_epochs': epochs, 'preferred_maximum_epochs': 1.5,
        'provenance_provider_columns': providers,
        'allocation_epoch_limit': allocation_limit,
        'allocation_policy': 'Prefer reference-labeled math within 1.5 passes; otherwise minimize necessary math-pool exposure, hard maximum two.',
        'mix_weight_basis': 'nonpadding_input_tokens',
        'training_tokens': ACTUAL_TOKENS, 'train_steps': ACTUAL_TOKENS // 262144}
    manifest['signature'] = fingerprint(manifest)
    atomic_json(root / 'manifest.json', manifest)
    for name in ('model', 'eval', 'eval_inloop_64'):
        link = root / name
        if not link.exists():
            link.symlink_to(core.PARENT / name)
    shutil.copy2(core.PARENT / 'model_revision.json', root / 'model_revision.json')
    from .data import MixedBlocks
    stream = MixedBlocks(root, 'math')
    sampled_tokens, maximum_exposure, unique_consumed = {}, {}, {}
    provider_sampled = np.zeros(len(providers), dtype=np.uint64)
    provider_unique_used = np.zeros(len(providers), dtype=np.uint64)
    provider_questions = np.zeros(len(providers), dtype=np.uint64)
    for label, name in enumerate(stream.names):
        indices = stream.indices[stream.labels == label]
        frequencies = np.bincount(indices, minlength=sources[name]['blocks'])
        assert frequencies.max() <= 2
        maximum_exposure[name] = int(frequencies.max())
        lengths = np.load(output / (name + '.lengths.npy'))
        sampled_tokens[name] = int(lengths[indices].sum())
        unique_consumed[name] = int(lengths[frequencies > 0].sum())
        contributions = np.load(output / (name + '.providers.npy'))
        provider_sampled += contributions[indices].sum(0, dtype=np.uint64)
        provider_unique_used += contributions[frequencies > 0].sum(0, dtype=np.uint64)
        q = np.load(output / (name + '.provider-questions.npy'))
        provider_questions += q[indices].sum(0, dtype=np.uint64)
    share = sum(v for k, v in sampled_tokens.items() if k != 'general') / sum(sampled_tokens.values())
    assert .798 <= share <= .802
    assert stream.loss_tokens(0, len(stream.labels)) > 0
    audit = {'passed': True, 'capacity_before_packing': capacity, 'sources': reports,
             'source_epochs': epochs, 'maximum_block_exposures': maximum_exposure,
             'math_fraction_actual_nonpadding': share, 'sampled_nonpadding_tokens': sampled_tokens,
             'input_token_slots': ACTUAL_TOKENS, 'question_deduplication': merged['dedup'],
             'all_conversations_whole': True, 'all_assistant_targets_preserved': True,
             'all_sources_present': set(merged['sources']) >= {'openmath', 'olmo', 'nemotron', 'dolci', 'general'},
             'response_token_cap': 1792, 'rl_response_token_cap': 2048,
             'unique_nonpadding_tokens_consumed': unique_consumed,
             'overall_effective_epochs': sum(sampled_tokens.values()) / sum(unique_consumed.values()),
             'effective_epochs_over_used_examples': {k: sampled_tokens[k] / unique_consumed[k] for k in sampled_tokens},
             'labeled_fraction_of_math_tokens': sampled_tokens['math-labeled'] / sum(v for k, v in sampled_tokens.items() if k != 'general'),
             'provider_training': {n: {'sampled_input_tokens': int(provider_sampled[i]),
                 'unique_input_tokens_used': int(provider_unique_used[i]),
                 'sampled_questions': int(provider_questions[i]),
                 'effective_epochs': float(provider_sampled[i] / max(1, provider_unique_used[i]))}
                 for i, n in enumerate(providers)},
             'meets_preferred_epoch_limit': sum(sampled_tokens.values()) / sum(unique_consumed.values()) <= 1.5,
             'all_buckets_meet_preferred_epoch_limit': max(epochs.values()) <= 1.5,
             'indices_sha256': hashlib.sha256(stream.indices.tobytes()).hexdigest()}
    assert audit['all_sources_present']
    atomic_json(root / 'curation-audit.json', audit)
    print('AUDIT_PASSED', json.dumps(audit), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--append', type=Path, action='append')
    parser.add_argument('--merge-only', action='store_true')
    args = parser.parse_args()
    merge(args.root, args.append)
    if not args.merge_only:
        pack(args.root)
