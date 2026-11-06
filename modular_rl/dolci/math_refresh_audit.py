"""Validate prepared one-epoch math artifacts and publish a reviewable summary."""

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .config import atomic_json
from .math_refresh import question_key, quality_reason
from .math_refresh_build import PARENT


def finalize_review(root):
    """Apply explicit, documented manual rejections before publishing/training."""
    root = Path(root)
    path = root / 'manual-invalid-questions.json'
    if not path.exists() or (root / 'manual-review-applied.json').exists(): return
    invalid = json.loads(path.read_text())
    if not invalid: return
    if (root / 'pilot/sft/math/config.json').exists():
        raise ValueError('The corpus is already in use; publish a new version for manual edits')
    db = sqlite3.connect(root / 'corpus.sqlite')
    removed = []
    manual = {question_key(v['problem']): v for v in invalid}
    with (root / 'final-quality-removals.jsonl').open('a') as log:
        for key, payload in db.execute("SELECT key,payload FROM accepted WHERE source!='general'").fetchall():
            value = json.loads(payload)
            reason = (manual[key]['rejection_reason'] if key in manual
                      else quality_reason(value['problem'], value['response']))
            if reason is None and value['source'] == 'nemotron' and value['origin'] != 'AoPS':
                reason = 'outside_selected_competition_source'
            if not reason: continue
            db.execute('DELETE FROM accepted WHERE key=?', (key,))
            removed.append({'key': key, 'source': value['source'], 'tokens': value['tokens'], 'reason': reason,
                            'manual': key in manual})
            log.write(json.dumps({'reason': reason, 'example': value}) + '\n')
    # Preserve 80/20 after final quality exclusions by reducing general data,
    # never by adding another pass over math. Reserve 512 internal math examples
    # from the final pool; the external RL validation panel stays unchanged.
    val_math = {r[0] for r in db.execute("SELECT key FROM accepted WHERE source!='general' ORDER BY key LIMIT 512")}
    atomic_json(root / 'sft-validation-keys.json', sorted(val_math))
    math_tokens = sum(n for k, n in db.execute("SELECT key,tokens FROM accepted WHERE source!='general'") if k not in val_math)
    val_general = set()  # Existing frozen continuation suite evaluates general text.
    general = []
    for key, payload in db.execute("SELECT key,payload FROM accepted WHERE source='general' ORDER BY key"):
        if key not in val_general: general.append((key, json.loads(payload)))
    old_masses = Counter()
    for key, value in general: old_masses[value['origin']] += value['tokens']
    target = math_tokens // 4
    actual = sum(old_masses.values()); trimmed = 0
    if actual > target:
        quotas = {s: n * target / actual for s, n in old_masses.items()}; used = Counter()
        for key, value in general:
            origin = value['origin']
            if used[origin] < quotas[origin]: used[origin] += value['tokens']
            else:
                db.execute('DELETE FROM accepted WHERE key=?', (key,)); trimmed += 1
    db.commit()
    prepared_path = root / 'prepared.json'; prepared = json.loads(prepared_path.read_text())
    prepared['counts'] = dict(db.execute('SELECT source,count(*) FROM accepted GROUP BY source'))
    prepared['manual_invalid_questions'] = len(invalid)
    prepared['nemotron_solution_subset'] = 'AoPS COT only: competition mathematics aligned with MATH'
    prepared['final_quality_policy'] = 'trusted-references-competition-math-v1'
    atomic_json(prepared_path, prepared)
    db.close()
    selection_path = root / 'general-selection.json'
    if selection_path.exists():
        before_selection = root / 'pre-final-review-general-selection.json'
        if not before_selection.exists(): before_selection.write_bytes(selection_path.read_bytes())
        final_masses = Counter()
        with sqlite3.connect(root / 'corpus.sqlite') as connection:
            for key, payload in connection.execute("SELECT key,payload FROM accepted WHERE source='general'"):
                value = json.loads(payload)
                if key not in val_general: final_masses[value['origin']] += value['tokens']
        atomic_json(selection_path, {'target_training_tokens': target, 'actual_source_training_tokens': dict(final_masses),
                    'definition': 'Short single-turn tool-free non-Math Dolci Instruct, approximately baseline token-proportional',
                    'final_review_rebalanced': True, 'questions_repeated': False})
    excluded_path = root / 'exclusions.json'; excluded = json.loads(excluded_path.read_text())
    old = {v['key'] for v in excluded}
    for row in invalid:
        key = question_key(row['problem'])
        if key not in old:
            excluded.append({'key': key, 'problem': row['problem'], 'origin': 'manual-invalid-question',
                             'id': row.get('id'), 'reason': row['rejection_reason']}); old.add(key)
    atomic_json(excluded_path, excluded)
    summary_path = root / 'exclusion-summary.json'; summary = json.loads(summary_path.read_text())
    summary.update(counts=dict(Counter(v['origin'] for v in excluded)), unique_questions=len(old))
    atomic_json(summary_path, summary)
    manifest = root / 'manifest.json'; before = root / 'pre-manual-review-manifest.json'
    if manifest.exists():
        if not before.exists(): before.write_bytes(manifest.read_bytes())
        manifest.unlink()
    from .math_epoch import pack
    pack(root)
    atomic_json(root / 'manual-review-applied.json', {'rejected_questions': len(invalid), 'removed': removed,
                'general_questions_trimmed_to_keep_80_20': trimmed,
                'before_training': True, 'repacked_after_edits': True,
                'evidence': str(path), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()})


def audit(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    excluded = {r['key'] for r in json.loads((root / 'exclusions.json').read_text())}
    keys, source_stats, examples = {}, {}, defaultdict(list)
    math_keys, origins, checks, references = {}, Counter(), Counter(), Counter()
    for split in ('train', 'val'):
        seen, source_lengths, counts, token_counts = set(), defaultdict(list), Counter(), Counter()
        total_assistant = 0
        math_keys[split] = set()
        with (root / (split + '.jsonl')).open() as stream:
            for line in stream:
                row = json.loads(line); metadata = row['metadata']; key = metadata['key']; source = metadata['source']
                if key in seen or key in excluded: raise ValueError('Duplicate or held-out training question')
                seen.add(key); counts[source] += 1; token_counts[source] += metadata['tokens']
                source_lengths[source].append(metadata['response_tokens'])
                total_assistant += metadata['assistant_tokens']
                if metadata['response_tokens'] > 1536 or metadata['tokens'] > 2048:
                    raise ValueError('A retained conversation exceeds the declared response/context budget')
                if source != 'general' and not (metadata.get('matched_reference') and metadata.get('reference_sources')):
                    raise ValueError('Math solution lacks an external expected answer')
                if source != 'general' and not set(metadata['reference_sources']) & {'human-math-train', 'nemotron'}:
                    raise ValueError('Synthetic-only answer agreement is insufficient')
                if source != 'general':
                    math_keys[split].add(key)
                    origins[source + ':' + str(metadata['origin'])] += 1
                    checks[metadata['verification_method']] += 1
                    references.update(set(metadata['reference_sources']))
                # Independent salted ordering avoids both longest-first packing
                # bias and correlation with the hash-selected SFT holdout.
                if split == 'train':
                    rank = hashlib.sha256(('independent-train-review-' + key).encode()).hexdigest()
                    bucket = examples[source]
                    if len(bucket) < 12 or rank < bucket[-1][0]:
                        bucket.append((rank, row)); bucket.sort(key=lambda v: v[0]); del bucket[12:]
        info = manifest['splits'][split]
        if len(seen) != info['documents'] or total_assistant != info['supervised_tokens']:
            raise ValueError('Packing changed document count or supervised-token count')
        mappings = set()
        with (root / (split + '-blocks.jsonl')).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['real_tokens'] + row['padding_tokens'] != 2048: raise ValueError('Invalid packed context size')
                for key in row['questions']:
                    if key in mappings: raise ValueError('A conversation is packed twice')
                    mappings.add(key)
        if mappings != seen: raise ValueError('Packed blocks do not exactly partition the examples')
        for suffix, width in (('.bin', 4), ('.mask', 1)):
            path = root / 'sources' / (split + suffix)
            if path.stat().st_size != info['blocks'] * 2049 * width: raise ValueError('Incomplete packed file')
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b''): h.update(chunk)
            if h.hexdigest() != info['sha256'][suffix]: raise ValueError('Packed content hash changed')
        keys[split] = seen
        source_stats[split] = {n: {'questions': counts[n], 'input_tokens': token_counts[n],
                                  'response_p50': float(np.percentile(v, 50)), 'response_p95': float(np.percentile(v, 95)),
                                  'response_max': max(v)} for n, v in source_lengths.items()}
    if keys['train'] & keys['val']: raise ValueError('Internal SFT validation overlaps training')
    db = sqlite3.connect(root / 'corpus.sqlite')
    filters = {row[0]: json.loads(row[1]) for row in db.execute('SELECT name,stats FROM units')}
    db.close()
    coverage = defaultdict(Counter)
    with (PARENT / 'rl-data/math/train.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line); md = row['metadata']
            key = question_key(md['messages'][0]['content'])
            for label in ('all', 'subject:' + md['subject'], 'level:' + md['level']):
                coverage[label]['rl_training_questions'] += 1
                coverage[label]['present_in_sft_train'] += int(key in math_keys['train'])
                coverage[label]['reserved_for_sft_validation'] += int(key in math_keys['val'])
    report = {'passed': True, 'unique_train_questions': len(keys['train']), 'validation_questions': len(keys['val']),
              'exact_holdout_matches': 0, 'train_validation_overlap': 0, 'whole_conversations': True,
              'math_epochs': 1, 'cycling': False, 'response_limit': 1536, 'rl_response_limit': 2048,
              'sources': source_stats, 'training_tokens_with_padding': manifest['training_tokens'],
              'nonpadding_training_tokens': manifest['splits']['train']['nonpadding_input_tokens'],
              'math_input_share': manifest['splits']['train']['math_input_share'],
              'packing_padding_fraction': manifest['splits']['train']['padding_fraction'],
              'math_origins': dict(origins), 'verification_methods': dict(checks),
              'reference_sources': dict(references),
              'math_training_coverage': {k: dict(v) for k, v in sorted(coverage.items())},
              'reference_checks': 'Each math final answer agrees with a human MATH or curated Nemotron expected answer. Synthetic-only OpenMath labels are excluded; this does not prove every reasoning step.',
              'deduplication': 'Canonical exact matching plus conservative SimHash/Jaccard presentation matching, preserving substantive words, variables, operators, numbers and qualifiers.',
              'filters': filters}
    atomic_json(root / 'validation.json', report)
    atomic_json(root / 'review-examples.json', {s: [row for _, row in values] for s, values in examples.items()})
    lines = ['# Refreshed math SFT corpus', '',
             'Prepared for Qwen/Qwen3-0.6B-Base, assistant-token loss, one epoch, and a 2K RL response budget.', '',
             f"Training: {report['unique_train_questions']:,} unique questions; {report['nonpadding_training_tokens']:,} nonpadding input tokens.",
             f"Math input-token share: {report['math_input_share']:.2%}; packing padding: {report['packing_padding_fraction']:.2%}.", '',
             '| Source | Training questions | Input tokens | Response p50 | Response p95 | Response max |',
             '|---|---:|---:|---:|---:|---:|']
    for n, v in source_stats['train'].items():
        lines.append(f"| {n} | {v['questions']:,} | {v['input_tokens']:,} | {v['response_p50']:.0f} | {v['response_p95']:.0f} | {v['response_max']} |")
    direct = report['math_training_coverage']['all']
    lines += ['', f"Direct coverage of the existing MATH RL training pool: {direct['present_in_sft_train']:,} / {direct['rl_training_questions']:,} questions.",
              'This training overlap is allowed; both the custom RL validation questions and official MATH test are excluded.']
    lines += ['', 'Every retained conversation fits wholly within 2,048 tokens. No solution is cut by token index.',
              'Math responses are complete final worked solutions, with a maximum of 1,536 Qwen tokens.',
              'Nemotron hidden reasoning and tool-assisted trajectories are excluded. Olmo hidden thinking is removed only at its explicit closing delimiter.',
              'Nemotron solutions use its AoPS competition-math subset; the Math StackExchange/MathOverflow solutions are excluded to avoid undergraduate/research topic drift.',
              'Olmo examples require an independently matched reference from the answer index; their own final answer is not treated as its own reference.',
              'Synthetic-only OpenMath expected answers were rejected after manual review found inconsistent questions despite agreeing labels. OpenMath/Olmo solutions require human MATH or curated Nemotron references.',
              'Calculus and advanced topics outside the MATH scope, explicit uncertainty about missing/inconsistent conditions, and detectable false numeric equations are filtered.',
              'All 750 custom MATH holdouts and 5,000 official MATH test questions are excluded, along with the other RL validation prompts.',
              'Canonical and conservative fuzzy deduplication is applied across sources; one solution is retained per accepted question.',
              'The 20% general portion uses short, single-turn, tool-free Dolci-Instruct examples with approximately empirical source weighting.', '',
              'The token budget follows this one-pass corpus; padding in the final global batch is masked rather than filled by repeating real examples.',
              'This is an audited data/training recipe. Better final MATH performance must be measured in the full SFT/RL experiment.', '',
              'See `validation.json`, `review-examples.json`, `inventory.json`, and `manifest.json` for exact counts, provenance and hashes.']
    (root / 'README.md').write_text('\n'.join(lines) + '\n')
    print('DATA_AUDIT_PASSED', json.dumps({k: v for k, v in report.items() if k not in ('filters', 'sources')}), flush=True)
    return report
