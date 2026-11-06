"""Scan every pinned Dolci row, tokenize conversations, and build five virtual mixes."""

import argparse
import bisect
import csv
import hashlib
import json
import multiprocessing
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .config import (ACTUAL_TOKENS, DATASET, MODEL, MODEL_REVISION, REVISION,
                     SEED, SEQ_LEN, apportion, atomic_json, family, fingerprint, mixtures)

TOKENIZER = None
TEMPLATE = None
EXCLUDED = None


def normalized(text):
    return ' '.join(text.split()).casefold()


def messages_for_qwen(messages):
    """Retain Dolci's XML function-call convention, including non-text fields."""
    result = []
    for message in messages:
        role = message['role']
        role = 'tool' if role in ('environment', 'function', 'ipython') else role
        if role not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError(f'Unsupported conversation role {role}')
        content = message['content'] or ''
        if message.get('functions'):
            content += '\n<functions>\n' + message['functions'] + '\n</functions>'
        if message.get('function_calls'):
            content += '\n<function_calls>\n' + message['function_calls'] + '\n</function_calls>'
        result.append({'role': role, 'content': content})
    return result


def initialize(model, excluded):
    global TOKENIZER, TEMPLATE, EXCLUDED
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(model, local_files_only=True)
    TEMPLATE = TOKENIZER.chat_template
    start = '{%- elif message.role == "assistant" %}'
    end = '{%- elif message.role == "tool" %}'
    assert TEMPLATE.count(start) == TEMPLATE.count(end) == 1
    TEMPLATE = TEMPLATE.replace(start, start + '{%- generation -%}').replace(end, '{%- endgeneration -%}' + end)
    EXCLUDED = set(excluded)


def encode(messages):
    from transformers.utils.chat_template_utils import render_jinja_template
    clean = messages_for_qwen(messages)
    rendered, spans = render_jinja_template(conversations=[clean], chat_template=TEMPLATE,
                                            return_assistant_tokens_mask=True,
                                            add_generation_prompt=False, tools=None)
    encoding = TOKENIZER.backend_tokenizer.encode(rendered[0], add_special_tokens=False)
    mask = np.zeros(len(encoding.ids), dtype='u1')
    starts = [offset[0] for offset in encoding.offsets]
    for begin, end in spans[0]:
        begin += len('<|im_start|>assistant\n')
        left, right = bisect.bisect_left(starts, begin), bisect.bisect_left(starts, end)
        mask[left:right] = 1
    if not mask.any():
        raise ValueError('Conversation has no supervised assistant tokens')
    return encoding.ids, mask, clean, rendered[0]


def tokenize_shard(arguments):
    path, destination, pairs, row_groups = arguments
    unit = path.stem if row_groups is None else path.stem + f'-rg{row_groups[0]:04d}-{row_groups[-1] + 1:04d}'
    directory = Path(destination) / unit
    result_path = directory / 'counts.json'
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        intact = all((directory / (name + suffix)).exists()
                     and (directory / (name + suffix)).stat().st_size == counts.get('tokens', 0) * width
                     for name, counts in cached['counts'].items() if counts.get('tokens', 0)
                     for suffix, width in (('.bin', 4), ('.mask', 1)))
        if intact:
            return cached
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    files, masks, counts = {}, {}, defaultdict(Counter)
    rng_key = path.name if row_groups is None else unit
    rng = np.random.default_rng(SEED + int(fingerprint(rng_key)[:8], 16))
    examples = {}
    try:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, row_groups=row_groups):
            rows = batch.to_pylist()
            for index in rng.permutation(len(rows)):
                row = rows[index]
                name = pairs[(row['source_dataset'], row['domain'])]
                counts[name]['original_rows'] += 1
                if any(normalized(m['content'] or '') in EXCLUDED for m in row['messages'] if m['role'] == 'user'):
                    counts[name]['excluded_rl_validation_prompt'] += 1
                    continue
                ids, mask, clean, rendered = encode(row['messages'])
                if name not in files:
                    files[name] = (directory / f'{name}.bin').open('wb')
                    masks[name] = (directory / f'{name}.mask').open('wb')
                np.asarray(ids, dtype='<u4').tofile(files[name])
                mask.tofile(masks[name])
                counts[name].update(rows=1, tokens=len(ids), assistant_tokens=int(mask.sum()))
                if name not in examples:
                    examples[name] = {'id': row['id'], 'messages': clean, 'rendered': rendered[:12000],
                                      'tokens': len(ids), 'assistant_tokens': int(mask.sum())}
    finally:
        for f in list(files.values()) + list(masks.values()):
            f.close()
    result = {'shard': unit, 'counts': dict(counts)}
    atomic_json(directory / 'examples.json', examples)
    atomic_json(result_path, result)
    return result


def prepare(root, model, previous, workers=12, objective='assistant', natural='complement'):
    root, model, previous = Path(root).resolve(), Path(model).resolve(), Path(previous).resolve()
    if (root / 'manifest.json').exists():
        previous_manifest = json.loads((root / 'manifest.json').read_text())
        if previous_manifest['objective'] != objective or previous_manifest['natural_definition'] != natural:
            raise ValueError('Prepared objective or natural-domain mapping differs')
        print('ALREADY_PREPARED', root / 'manifest.json', flush=True)
        return
    raw = root / 'raw'
    files = sorted((raw / 'data').glob('*.parquet'))
    if len(files) != 15:
        raise ValueError('All fifteen pinned parquet shards are required')
    from huggingface_hub import HfApi
    expected = HfApi().dataset_info(DATASET, revision=REVISION, files_metadata=True)
    sizes = {f.rfilename: f.size for f in expected.siblings}
    if any(p.stat().st_size != sizes['data/' + p.name] for p in files):
        raise ValueError('Incomplete pinned dataset download')
    initial = Counter()
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(columns=['source_dataset', 'domain']):
            initial.update(zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()))
    if sum(initial.values()) != 2152112:
        raise ValueError('Unexpected Dolci row count')
    pairs = {p: f's{i:02}' for i, p in enumerate(sorted(initial))}
    atomic_json(root / 'initial-row-distribution.json', [
        {'source_dataset': p[0], 'domain': p[1], 'source_id': n, 'rows': initial[p],
         'row_share': initial[p] / sum(initial.values())} for p, n in pairs.items()])
    selection = previous / 'selection'
    excluded = set()
    panel_hashes = {}
    for task in ('math', 'code', 'chat', 'if'):
        path = selection / task / 'val.jsonl'
        panel_hashes[task] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            for message in json.loads(line)['metadata']['messages']:
                if message['role'] == 'user':
                    excluded.add(normalized(message['content']))
    (root / 'model').symlink_to(model, target_is_directory=True) if not (root / 'model').exists() else None
    atomic_json(root / 'model_revision.json', {'repo': MODEL, 'revision': MODEL_REVISION})
    for directory in ('eval', 'eval_inloop_64'):
        target = root / directory
        if not target.exists():
            target.symlink_to(Path('/nlp/scr/dhei/nano-omni-midtraining') / directory, target_is_directory=True)
    initialize(model, excluded)
    units = []
    for path in files:
        if (root / 'parts' / path.stem / 'counts.json').exists():
            units.append((path, str(root / 'parts'), pairs, None))
        else:
            count = pq.ParquetFile(path).num_row_groups
            units += [(path, str(root / 'parts'), pairs, list(range(start, min(start + 8, count))))
                      for start in range(0, count, 8)]
    # Fork only before tokenization starts: workers share an immutable tokenizer.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
        futures = [pool.submit(tokenize_shard, unit) for unit in units]
        for future in as_completed(futures):
            info = future.result()
            print('TOKENIZED', info['shard'], sum(c['tokens'] for c in info['counts'].values()), flush=True)
    totals = defaultdict(Counter)
    for path in sorted((root / 'parts').glob('*/counts.json')):
        for name, counts in json.loads(path.read_text())['counts'].items():
            totals[name].update(counts)
    if sum(c['original_rows'] for c in totals.values()) != 2152112:
        raise ValueError('Preparation parts duplicate or omit source rows')
    sources = {}
    source_directory = root / 'sources'
    source_directory.mkdir(exist_ok=True)
    for pair, name in pairs.items():
        count = totals[name]
        packed_tokens = count['tokens'] // (SEQ_LEN + 1) * (SEQ_LEN + 1)
        hashes = {}
        if not packed_tokens:
            raise ValueError(f'Source has no full blocks: {pair}')
        for suffix, width in (('.bin', 4), ('.mask', 1)):
            target = source_directory / (name + suffix)
            temporary = target.with_suffix(target.suffix + '.partial')
            remaining = packed_tokens * width
            digest = hashlib.sha256()
            with temporary.open('wb') as output:
                complete_parts = [p.parent / (name + suffix) for p in sorted((root / 'parts').glob('*/counts.json'))]
                for part in (p for p in complete_parts if p.exists()):
                    with part.open('rb') as source:
                        while remaining:
                            chunk = source.read(min(remaining, 8 * 1024 * 1024))
                            if not chunk:
                                break
                            output.write(chunk)
                            digest.update(chunk)
                            remaining -= len(chunk)
                    if not remaining:
                        break
            assert not remaining
            temporary.replace(target)
            hashes['tokens_sha256' if suffix == '.bin' else 'mask_sha256'] = digest.hexdigest()
        sources[name] = {**count, 'source_dataset': pair[0], 'domain': pair[1],
                         'packed_tokens': packed_tokens, 'blocks': packed_tokens // (SEQ_LEN + 1), **hashes}
    recipe = {'sources': sources, 'mixes': mixtures(sources, natural)}
    manifest = {'repo': DATASET, 'revision': REVISION, 'seq_len': SEQ_LEN, 'seed': SEED,
                'objective': objective, 'natural_definition': natural,
                'pool_prediction_tokens': ACTUAL_TOKENS, 'recipe': recipe, 'sources': sources,
                'quotas': {m: apportion(ACTUAL_TOKENS // SEQ_LEN, weights) for m, weights in recipe['mixes'].items()},
                'baseline_weighting': 'empirical rendered Qwen input-token shares; row shares also recorded',
                'repetition': 'each source reshuffled without replacement within each repeated cycle',
                'rl_validation_exact_user_message_exclusion': {'panel_sha256': panel_hashes,
                                                               'excluded_rows': sum(c['excluded_rl_validation_prompt'] for c in totals.values())}}
    manifest['signature'] = fingerprint(manifest)
    atomic_json(root / 'manifest.json', manifest)
    rows = []
    total_rows, total_tokens = sum(c['rows'] for c in sources.values()), sum(c['tokens'] for c in sources.values())
    for name, source in sources.items():
        row = {'source_dataset': source['source_dataset'], 'domain': source['domain'], 'family': family(source['domain'], natural),
               'rows': source['rows'], 'tokens': source['tokens'], 'row_share': source['rows'] / total_rows,
               'token_share': source['tokens'] / total_tokens}
        for mix, weights in recipe['mixes'].items():
            row[mix + '_token_share'] = weights[name]
            row[mix + '_upsample'] = weights[name] / row['token_share']
            row[mix + '_actual_train_tokens'] = manifest['quotas'][mix][name] * SEQ_LEN
        rows.append(row)
    with (root / 'mixture-ratios.csv').open('w') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Parts are reproducible preparation intermediates, not model checkpoints.
    for path in (root / 'parts').glob('*/*'):
        if path.suffix in ('.bin', '.mask'):
            path.unlink()
    print('PREPARED', root / 'manifest.json', 'tokens', total_tokens, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('/nlp/scr/dhei/models/Qwen3-0.6B-Base'))
    parser.add_argument('--previous', type=Path, default=Path('/nlp/scr/dhei/modular-mopd-rl/v1/two-gpu-ready-20261029'))
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--objective', choices=['assistant', 'all'], default='assistant')
    parser.add_argument('--natural', choices=['complement', 'knowledge'], default='complement')
    args = parser.parse_args()
    prepare(**vars(args))
