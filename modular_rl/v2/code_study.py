"""Prepare independent MBPP training, monitoring and confirmation for Base RL."""
import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from modular_rl.prepare import write_json

from .base_models import BASE_MODELS


def split_examples(rows, monitor=32, confirmation=256, seed=20261029):
    # Older frozen MBPP pools predate the environment label required by native
    # evaluation rollout tables. Normalize metadata without touching the source.
    rows = copy.deepcopy(rows)
    for row in rows:
        if row['metadata'].get('task', 'code') != 'code':
            raise ValueError('Expected only MBPP code examples')
        row['metadata']['environment'] = 'code'
    if len({r['metadata']['key'] for r in rows}) != len(rows):
        raise ValueError('Duplicate MBPP keys')
    counts = Counter(r['metadata']['split_group'] for r in rows)
    eligible = sorted((r for r in rows if counts[r['metadata']['split_group']] == 1),
                      key=lambda r: r['metadata']['key'])
    rng = random.Random(seed)
    rng.shuffle(eligible)
    if monitor < 1 or confirmation < 1 or len(eligible) < monitor + confirmation:
        raise ValueError('Not enough independent held-out problems')
    held = eligible[:monitor + confirmation]
    held_keys = {r['metadata']['key'] for r in held}
    train = sorted((r for r in rows if r['metadata']['key'] not in held_keys),
                   key=lambda r: r['metadata']['key'])
    if not train:
        raise ValueError('Training split is empty')
    rng.shuffle(train)
    return {'train': train, 'val': held[:monitor], 'confirmation': held[monitor:]}


def cells(include_prefix=False):
    recipes = [('control', 2e-6, 0, 'binary', 512),
               ('lr1e-6', 1e-6, 0, 'binary', 512),
               ('kl0p01', 2e-6, .01, 'binary', 512),
               ('kl0p03', 2e-6, .03, 'binary', 512),
               ('partial-kl0p01', 2e-6, .01, 'partial', 512),
               ('answer1024-kl0p01', 2e-6, .01, 'binary', 1024)]
    result = [{'model': model, 'cap': cap, 'variant': name, 'lr': lr,
             'reference_kl': kl, 'responses': 8000, 'passes': 1, 'task': 'code',
             'answer_reserve': reserve, 'reward_variant': reward}
            for model in BASE_MODELS for cap in (2048, 4096)
            for name, lr, kl, reward, reserve in recipes]
    if include_prefix:
        result.extend({'model': model, 'cap': 2048, 'variant': 'prefix-kl0p01', 'lr': 2e-6,
                       'reference_kl': .01, 'responses': 8000, 'passes': 1, 'task': 'code',
                       'answer_reserve': 512, 'reward_variant': 'binary', 'code_format': 'prefix'}
                      for model in BASE_MODELS)
    return result


def prepare(base, review, include_prefix=False):
    from transformers import AutoTokenizer

    base, review = base.resolve(), review.resolve()
    if (review / 'data-protocol.json').exists():
        raise ValueError('Use a fresh code-study review directory')
    rows = [json.loads(s) for name in ('train', 'val')
            for s in (base / f'data/code/{name}.jsonl').read_text().split('\n') if s.strip()]
    parts = split_examples(rows)
    proof = []
    for model in BASE_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(base.parents[1] / 'models' / model, local_files_only=True)
        lengths = [len(tokenizer(tokenizer.apply_chat_template(r['metadata']['messages'], tokenize=False,
                       add_generation_prompt=True, enable_thinking=True), add_special_tokens=False)['input_ids'])
                   for r in rows]
        if max(lengths) > 2048 or tokenizer.eos_token_id != 151643:
            raise ValueError('MBPP prompt/Base-tokenizer contract changed')
        proof.append({'model': model, 'max_prompt_tokens': max(lengths), 'examples_checked': len(rows)})
    output = review / 'selection/code'
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / (name + '.jsonl') for name in ('train', 'val')}
    paths['confirmation'] = review / 'confirmation/code/val.jsonl'
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in parts[name]))
    rounds = []
    for index in range(2):
        path = review / f'confirmation/round{index + 1}/code/val.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        selected_round = parts['confirmation'][index * 128:(index + 1) * 128]
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected_round))
        rounds.append({'round': index + 1, 'path': str(path), 'prompts': 128,
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {'created_utc': datetime.now(UTC).isoformat(), 'seed': 20261029,
                'dataset': 'Muennighoff/mbpp', 'revision': 'd81b8291e5998f5726ab7f35a0a557e761532aac',
                'counts': {k: len(v) for k, v in parts.items()},
                'keys': {k: [r['metadata']['key'] for r in v] for k, v in parts.items()},
                'sha256': {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()},
                'paths': {k: str(p) for k, p in paths.items()},
                'confirmation_rounds': rounds,
                'enable_thinking': True, 'validation_prompts': 32,
                'checkpoint_family': 'pretrained-base', 'confirmation_outcomes_generated': False,
                'split_group_disjoint': True, 'duplicate_problem_groups_kept_in_train_only': 2,
                'tokenizer_contract': proof}
    write_json(review / 'data-protocol.json', manifest)
    selected = {**manifest, 'keys': {k: manifest['keys'][k] for k in ('train', 'val')},
                'sha256': {k: manifest['sha256'][k] for k in ('train', 'val')}}
    write_json(output / 'selection.json', selected)
    grid = cells(include_prefix=include_prefix)
    write_json(review / 'code-cells.json', grid)
    protocol = {'declared_utc': manifest['created_utc'], 'objective': 'Confirm Base MBPP RL before IFEval',
                'models': ['Qwen/' + m for m in BASE_MODELS], 'conditions': grid,
                'data_protocol': str(review / 'data-protocol.json'),
                'train_prompts': len(parts['train']), 'monitor_prompts': 32, 'confirmation_prompts': 128,
                'confirmation_rounds': rounds, 'primary_family_size': 4,
                'samples_per_prompt': 8, 'eval_interval': 50, 'early_stop_pass1': .98,
                'candidate_selection': 'Per model size, choose an eligible completed FINAL8k policy with highest '
                    'mean of last5 post-training monitor pass1 evaluations. Ties: fewer responses, lower LR, '
                    'lower KL, variant name, cap. Freeze BOTH choices (or explicit no-eligible-candidate entries) '
                    'before any reserved outcomes. Caps/reserves are chosen using monitoring only.',
                'quality_guard': {'mean_repeated_6word_fraction_at_most': .3,
                                  'response_fraction_repetition_above_half_at_most': .25},
                'selection_timing': 'When each size has an eligible completed FINAL8k policy (or all '
                    'its conditions complete with none eligible), choose among completed policies and '
                    'disclose all pending conditions. Freeze BOTH choices before any fresh outcomes.',
                'confirmation': 'Two predeclared disjoint128-problem rounds, each with up to2 size comparisons. '
                    'Use20000-draw paired problem bootstrap with family size4 (98.75% intervals,95% Bonferroni '
                    'confidence across all4 possible slots). Freeze both policies before each round. Round2 '
                    'remains unused unless round1 yields no code policy passing accuracy/quality/integrity. '
                    'Match baseline/candidate cap, reserve, generation format, template, sampling and GPU. '
                    'Strict all-tests accuracy including partial-training-reward variants.',
                'success': 'At least one requested Base size has positive adjusted accuracy gain, passes '
                    'response-quality and manual code/grader integrity review. Report all conditions/both '
                    'sizes. Then proceed to IFEval with Dolci-RL-Zero-IF-7B.',
                'slurm_dependencies': False, 'reserved_confirmation_outcomes_used': False,
                'project': 'modular-mopd-rl-initial'}
    write_json(review / 'code-confirmation-protocol.json', protocol)
    print(json.dumps({'review': str(review), 'counts': manifest['counts'], 'conditions': len(grid),
                      'tokenizers': proof}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('/nlp/scr/dhei/modular-mopd-rl/v2'))
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--include-prefix', action='store_true',
                        help='Prospectively include the two full8k Python-prefix conditions (26 total)')
    args = parser.parse_args()
    prepare(args.base, args.review, args.include_prefix)
