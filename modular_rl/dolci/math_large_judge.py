"""Independent short-solution review; first calibrate against manually checked cases."""

import argparse
import hashlib
import inspect
import json
import os
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

from .config import atomic_json
from . import math_large as core

MODEL = 'Qwen/Qwen3-8B'
REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
SYSTEM = '''You are a strict mathematics data reviewer. Independently check the problem and proposed solution.
Reject if the problem is inconsistent or underspecified for its requested answer, if the solution imposes unstated constraints, changes the question, invents an answer after deriving a contradiction, or contains a substantive mathematical error. A matching external answer label is not evidence of correctness. Accept valid concise solutions; harmless verbosity is acceptable. If unsure, reject.
Begin with exactly ACCEPT or REJECT, followed by a brief mathematical reason (at most 60 words).'''
BASE_SYSTEM = SYSTEM
REVIEW_TOKENS = 128
REASONED = False


def configure_reviewer(model_repo):
    global SYSTEM, REVIEW_TOKENS, REASONED
    REASONED = model_repo == 'Qwen/Qwen3-30B-A3B-Instruct-2507'
    REVIEW_TOKENS = 384 if REASONED else 128
    SYSTEM = (BASE_SYSTEM[:BASE_SYSTEM.index('Begin with exactly')] +
        'Check the mathematics before deciding. In at most 180 words, give a concrete verification or counterexample, '
        'checking unstated assumptions, omitted counting factors, signs, and domains. Do not merely paraphrase the solution. '
        'End with a separate final line VERDICT: ACCEPT or VERDICT: REJECT.' if REASONED else BASE_SYSTEM)


def accepted(text):
    if not REASONED:
        return text.lstrip().startswith('ACCEPT')
    verdicts = re.findall(r'(?im)^\s*VERDICT:\s*(ACCEPT|REJECT)\b', text)
    return bool(verdicts and verdicts[-1] == 'ACCEPT')


def generate_reviews(engine, tokenizer, values):
    started = time.monotonic()
    prompts = [prompt(tokenizer, v) for v in values]
    responses = engine.generate(prompts, {'temperature': 0., 'max_new_tokens': REVIEW_TOKENS})
    assert len(responses) == len(values)
    if REASONED:
        missing = [i for i, v in enumerate(responses)
                   if not re.search(r'(?im)^\s*VERDICT:\s*(?:ACCEPT|REJECT)\b', v['text'])]
        if missing:
            suffix = '\n\nVERDICT:'
            final = engine.generate([prompts[i] + responses[i]['text'] + suffix for i in missing],
                                    {'temperature': 0., 'max_new_tokens': 8})
            assert len(final) == len(missing)
            for i, value in zip(missing, final):
                responses[i]['text'] += suffix + value['text']
                responses[i]['forced_verdict'] = True
    seconds = time.monotonic() - started
    print(json.dumps({'review_batch_cases': len(values), 'review_seconds': seconds,
                      'cases_per_second': len(values) / max(seconds, .001)}), flush=True)
    return responses


def prefilter(value):
    return (core.outside_scope(value['problem']) or core.unreliable(value['response'])
            or core.invented_integer_roots(value['problem'], value['response']))


def prefilter_signature():
    return hashlib.sha256(''.join(inspect.getsource(f) for f in
        (core.outside_scope, core.unreliable, core.invented_integer_roots)).encode()).hexdigest()


def calibration_report(rows, examples, model_repo, model_revision):
    """Score the deployed filter plus reviewer, retaining standalone results."""
    configure_reviewer(model_repo)
    values = {v['id']: v for v in examples}
    raw = [{**row, 'accepted': row.get('raw_model_accepted', row['accepted'])} for row in rows]
    def scores(items):
        positive = [v for v in items if v['expected_accept']]
        negative = [v for v in items if not v['expected_accept']]
        return {'bad_case_recall': sum(not v['accepted'] for v in negative) / len(negative),
                'good_case_retention': sum(v['accepted'] for v in positive) / len(positive)}
    combined = [{**v, 'raw_model_accepted': v['accepted'],
                 'prefilter_rejected': prefilter(values[v['id']]),
                 'accepted': v['accepted'] and not prefilter(values[v['id']])} for v in raw]
    score = scores(combined)
    return {'passed': score['bad_case_recall'] >= .875 and score['good_case_retention'] >= .625,
            'model': model_repo, 'revision': model_revision, 'score': score,
            'standalone_model_score': scores(raw), 'cases': combined,
            'prefilter_signature': prefilter_signature(),
            'review_prompt_sha256': hashlib.sha256(SYSTEM.encode()).hexdigest(),
            'review_tokens': REVIEW_TOKENS, 'reason_before_verdict': REASONED,
            'force_complete_verdict': REASONED,
            'scope': 'End-to-end static curriculum/reasoning filter plus model reviewer. No ID-based manual vetoes enter this calibration. Small calibration is not a mathematical certificate; require a fresh retained-example review.'}


def prompt(tokenizer, value):
    return tokenizer.apply_chat_template([
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': 'PROBLEM:\n' + value['problem'] + '\n\nPROPOSED SOLUTION:\n' + value['response']}
    ], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def calibration(root, model, *, model_repo=MODEL, model_revision=REVISION, output='judge-calibration.json'):
    import sglang as sgl
    from transformers import AutoTokenizer

    root = Path(root)
    configure_reviewer(model_repo)
    existing = root / output
    if existing.exists():
        saved = json.loads(existing.read_text())
        if (saved['model'] == model_repo and saved['revision'] == model_revision
                and saved.get('review_prompt_sha256') == hashlib.sha256(SYSTEM.encode()).hexdigest()
                and saved.get('force_complete_verdict') == REASONED
                and saved.get('prefilter_signature') == prefilter_signature()):
            print('REUSING_PINNED_CALIBRATION', existing, flush=True)
            return
    examples = json.loads((root / 'premerge-random-review.json').read_text())
    bad = json.loads((root / 'premerge-random-review-findings.json').read_text())['findings']
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    engine = sgl.Engine(model_path=str(model), tp_size=4, dtype='bfloat16',
                        mem_fraction_static=.7, context_length=4096,
                        max_running_requests=32, cuda_graph_max_bs=32,
                        attention_backend='flashinfer', random_seed=20261101)
    try:
        responses = generate_reviews(engine, tokenizer, examples)
        rows = [{'id': value['id'], 'expected_accept': value['id'] not in bad,
                 'accepted': accepted(output['text']),
                 'review': output['text'], 'source': value['source']}
                for value, output in zip(examples, responses)]
        report = calibration_report(rows, examples, model_repo, model_revision)
        atomic_json(root / output, report)
        print(json.dumps({'passed': report['passed'], **report['score']}), flush=True)
    finally:
        engine.shutdown()


def export_shards(root, shards=6):
    root = Path(root)
    assert (root / 'reserve-appended.json').exists()
    destination = root / 'judge-input'
    if (destination / 'manifest.json').exists():
        return
    destination.mkdir(exist_ok=True)
    streams = [(destination / f'{i:02d}.jsonl').open('w') for i in range(shards)]
    counts = Counter()
    db = sqlite3.connect(f'file:{root}/merged.sqlite?mode=ro', uri=True)
    for key, payload in db.execute("SELECT key,payload FROM accepted WHERE bucket!='general' ORDER BY key"):
        value = json.loads(payload)
        value['solution_sha256'] = hashlib.sha256(value['response'].encode()).hexdigest()
        shard = int(key[:8], 16) % shards
        streams[shard].write(json.dumps(value, ensure_ascii=False) + '\n')
        counts[shard] += 1
    for stream in streams:
        stream.close()
    db.close()
    atomic_json(destination / 'manifest.json', {'shards': shards, 'questions': dict(counts),
                'solution_hash_bound': True})


def screen(root, model, shard, *, calibration_file, model_repo, model_revision):
    import sglang as sgl
    from transformers import AutoTokenizer

    root = Path(root)
    configure_reviewer(model_repo)
    calibration = json.loads((root / calibration_file).read_text())
    assert calibration['passed'] and calibration['model'] == model_repo and calibration['revision'] == model_revision
    assert calibration['prefilter_signature'] == prefilter_signature()
    if REASONED:
        assert calibration['review_prompt_sha256'] == hashlib.sha256(SYSTEM.encode()).hexdigest()
        assert calibration['force_complete_verdict']
    incoming = root / 'judge-input' / f'{shard:02d}.jsonl'
    records = [json.loads(line) for line in incoming.open()]
    destination = root / 'judge-results'
    destination.mkdir(exist_ok=True)
    output = destination / f'{shard:02d}.jsonl'
    marker = destination / f'{shard:02d}.complete.json'
    if marker.exists():
        return
    prior = []
    if output.exists():
        # Preserve complete batches after preemption; discard only an incomplete
        # trailing JSON line, which can never constitute a committed verdict.
        with output.open('r+b') as f:
            while line := f.readline():
                try:
                    prior.append(json.loads(line))
                except json.JSONDecodeError:
                    f.truncate(f.tell() - len(line))
                    break
    for original, verdict in zip(records, prior):
        assert original['key'] == verdict['key'] and original['solution_sha256'] == verdict['solution_sha256']
    assert len(prior) <= len(records)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    engine = sgl.Engine(model_path=str(model), tp_size=4, dtype='bfloat16',
                        mem_fraction_static=.85, context_length=4096,
                        max_running_requests=512, cuda_graph_max_bs=512,
                        attention_backend='flashinfer', random_seed=20261101)
    try:
        with output.open('a') as stream:
            for start in range(len(prior), len(records), 512):
                batch = records[start:start + 512]
                responses = generate_reviews(engine, tokenizer, batch)
                assert len(responses) == len(batch)
                for value, response in zip(batch, responses):
                    row = {k: value[k] for k in ('key', 'id', 'source', 'tokens', 'solution_sha256')}
                    row.update(accepted=accepted(response['text']) and not prefilter(value),
                               review=response['text'])
                    stream.write(json.dumps(row) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                atomic_json(destination / f'{shard:02d}.progress.json',
                            {'reviewed': start + len(batch), 'total': len(records)})
        atomic_json(marker, {'questions': len(records), 'model': model_repo,
                    'revision': model_revision, 'input_sha256': hashlib.sha256(incoming.read_bytes()).hexdigest(),
                    'review_prompt_sha256': hashlib.sha256(SYSTEM.encode()).hexdigest(),
                    'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
    finally:
        engine.shutdown()


def apply_screen(root):
    root = Path(root)
    audit_path = root / 'independent-screen-audit.json'
    if audit_path.exists():
        previous = json.loads(audit_path.read_text())['remaining_counts']
        core.capacity(previous['math']['tokens'], previous['general']['tokens'])
        return
    manifest = json.loads((root / 'judge-input/manifest.json').read_text())
    verdicts = {}
    for shard in range(manifest['shards']):
        path = root / 'judge-results' / f'{shard:02d}.jsonl'
        proof = json.loads(path.with_suffix('.complete.json').read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == proof['output_sha256']
        assert hashlib.sha256((root / 'judge-input' / path.name).read_bytes()).hexdigest() == proof['input_sha256']
        rows = [json.loads(line) for line in path.open()]
        assert len(rows) == proof['questions']
        for row in rows:
            assert row['key'] not in verdicts
            verdicts[row['key']] = row
    manual_rows = json.loads((root / 'manual-rejections.json').read_text())
    manual = {v['id'] for v in manual_rows}
    manual_keys = {v.get('key') or core.key(v['problem']) for v in manual_rows if v.get('key') or v.get('problem')}
    from .math_large_pack import global_rejection
    db = sqlite3.connect(root / 'merged.sqlite')
    current = {key: hashlib.sha256(json.loads(payload)['response'].encode()).hexdigest()
               for key, payload in db.execute("SELECT key,payload FROM accepted WHERE bucket!='general'")}
    removed, retained, review = Counter(), Counter(), []
    reject = (root / 'independent-screen-rejections.jsonl').open('w')
    keys = []
    for shard in range(manifest['shards']):
        for line in (root / 'judge-input' / f'{shard:02d}.jsonl').open():
            value = json.loads(line)
            key = value['key']
            verdict = verdicts.pop(key)
            digest = hashlib.sha256(value['response'].encode()).hexdigest()
            assert digest == value['solution_sha256'] == verdict['solution_sha256']
            existing = current.pop(key, None)
            assert existing is None or existing == digest
            rejected = (not verdict['accepted'] or value['id'] in manual or key in manual_keys
                        or global_rejection(value))
            # A retry after a committed delete may revisit already removed
            # rejected rows. Accepted rows must always still exist unchanged.
            assert existing is not None or rejected
            if rejected:
                keys.append((key,))
                removed[value['source']] += value['tokens']
                reject.write(json.dumps({'example': value, 'verdict': verdict}) + '\n')
            else:
                retained[value['source']] += value['tokens']
                if int(hashlib.sha256(('post-screen-review:' + key).encode()).hexdigest()[:8], 16) < 2**32 / 3000:
                    review.append({'example': value, 'verdict': verdict})
    assert not verdicts and not current
    reject.close()
    # The shared scratch volume is tight. A single deletion transaction can
    # create a rollback journal almost as large as the corpus. Each rejection
    # is already bound to its response hash; committed batches can safely be
    # revisited after preemption (see the existing-is-None check above).
    for start in range(0, len(keys), 5000):
        db.executemany('DELETE FROM accepted WHERE key=?', keys[start:start + 5000])
        db.commit()
    counts = {b: {'questions': n, 'tokens': t} for b, n, t in db.execute(
        'SELECT bucket,count(*),sum(tokens) FROM accepted GROUP BY bucket')}
    counts['math'] = {k: sum(counts.get(b, {}).get(k, 0) for b in ('math-labeled', 'math-unlabeled'))
                      for k in ('questions', 'tokens')}
    sources = {s: {'questions': n, 'tokens': t} for s, n, t in db.execute(
        'SELECT source,count(*),sum(tokens) FROM accepted GROUP BY source')}
    db.close()
    old = json.loads((root / 'merge-complete.json').read_text())
    old.update(counts=counts, sources=sources)
    atomic_json(root / 'merge-complete.json', old)
    atomic_json(root / 'post-screen-review.json', review)
    atomic_json(root / 'independent-screen-audit.json', {'all_solutions_reviewed': True,
                'removed_math_tokens': dict(removed), 'retained_math_tokens': dict(retained),
                'remaining_counts': counts, 'model_weight_revision': proof['revision']})
    # The capacity check uses retained data only; rejected examples cannot inflate it.
    core.capacity(counts['math']['tokens'], counts['general']['tokens'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--model-repo', default=MODEL)
    parser.add_argument('--model-revision', default=REVISION)
    parser.add_argument('--output', default='judge-calibration.json')
    parser.add_argument('--calibration-file', default='judge-32b-calibration.json')
    parser.add_argument('--shard', type=int)
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.export:
        export_shards(args.root)
    elif args.apply:
        apply_screen(args.root)
    elif args.shard is not None:
        assert args.model is not None
        screen(args.root, args.model, args.shard, calibration_file=args.calibration_file,
               model_repo=args.model_repo, model_revision=args.model_revision)
    else:
        assert args.model is not None
        calibration(args.root, args.model, model_repo=args.model_repo,
                    model_revision=args.model_revision, output=args.output)
