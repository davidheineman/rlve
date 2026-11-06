"""Short-solution curation with provider labels and explicit exposure limits."""

import hashlib
import json
import logging
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import math_refresh as old

PARENT = Path('/nlp/scr/dhei/dolci-mix-20261029')
EARLIER = Path('/nlp/scr/dhei/math-refresh-20261101')
RESPONSE_LIMIT = 1792
CONTEXT_LIMIT = 2048
TOKENIZER = None
ENCODE = None
EXCLUDED = None


@lru_cache(maxsize=131072)
def canonical(text):
    # NFKC alone maps superscripts to ordinary digits, e.g. x² -> x2.
    superscripts = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾', '0123456789+-()')
    subscripts = str.maketrans('₀₁₂₃₄₅₆₇₈₉₊₋₍₎', '0123456789+-()')
    text = re.sub('[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾]+', lambda m: '^{' + m[0].translate(superscripts) + '}', text)
    text = re.sub('[₀₁₂₃₄₅₆₇₈₉₊₋₍₎]+', lambda m: '_{' + m[0].translate(subscripts) + '}', text)
    return old.canonical_question(text)


@lru_cache(maxsize=262144)
def key(problem):
    return hashlib.sha256(canonical(problem).encode()).hexdigest()


def meaning(problem):
    tokens = re.findall(r'\\[A-Za-z]+|\w+|[^\w\s]', canonical(problem).rstrip().rstrip('.'))
    return tuple(t if len(t) == 1 or t.startswith('\\') else t.casefold()
                 for t in tokens if t.casefold() not in ('the', 'please', 'kindly'))


def presentation_key(problem):
    return hashlib.sha256(json.dumps(meaning(problem), ensure_ascii=False).encode()).hexdigest()


def shingles(problem):
    words = meaning(problem)
    return {' '.join(words[i:i + 5]) for i in range(max(0, len(words) - 4))}


class Exclusions:
    """Exact/presentation matching plus embedded near-verbatim holdout checks."""
    def __init__(self, records):
        self.exact = {key(r['problem']) for r in records}
        self.presentation = {presentation_key(r['problem']) for r in records}
        self.grams, self.index = [], {}
        for row in records:
            grams = shingles(row['problem'])
            self.grams.append(grams)
            if len(grams) >= 8:
                for gram in grams:
                    self.index.setdefault(gram, []).append(len(self.grams) - 1)

    def contains(self, problem):
        if key(problem) in self.exact or presentation_key(problem) in self.presentation:
            return True
        counts = Counter(i for gram in shingles(problem) for i in self.index.get(gram, []))
        return any(n >= 8 and n / len(self.grams[i]) >= .85 for i, n in counts.items())


def initialize(root):
    global TOKENIZER, ENCODE, EXCLUDED
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    old.initialize_worker(str(PARENT / 'model'), str(PARENT / 'rl/base/runtime'))
    TOKENIZER, ENCODE = old.TOKENIZER, old.ENCODE
    records = json.loads((Path(root) / 'exclusions.json').read_text())
    records += json.loads((Path(root) / 'additional-exclusions.json').read_text())
    EXCLUDED = Exclusions(records)
    logging.getLogger('math_verify').setLevel(logging.CRITICAL)


def outside_scope(problem):
    # Target the MATH curriculum without rejecting words such as "integral"
    # when they mean an integer, or valid calculus-free optimization working.
    return bool(re.search(r'(?i)\\(?:i?i?int|oint|lim)(?![A-Za-z])|'
        r'\b(?:evaluate|compute|find|solve).{0,50}\b(?:integral(?!\s+(?:values?|solutions?|points?|coefficients?|roots?|coordinates?)\b)|derivative|differential equation)\b|'
        r'\b(?:hilbert|banach|lebesgue|martingale|homology|cohomology|manifold|'
        r'galois group|splitting field|fourier transform|laplace transform|'
        r'category theory|measure theory|complex analysis|differentiat(?:e|ion)|'
        r'rate of change|definite integral|indefinite integral|dijkstra|bellman.ford|'
        r'ford.fulkerson|push.relabel|maximum flow|data flow|polynomial time)\b', problem))


def invented_integer_roots(problem, response):
    """Reject a common synthetic failure: imposing unstated integer-root bounds."""
    return bool(re.search(r'(?i)integer (?:coefficients and )?roots', response)
                and not re.search(r'(?i)integer|integral|whole.number|\\mathbb\s*\{?Z', problem))


def unreliable(response):
    return bool(re.search(r'(?is)(?:problem|question|statement|given conditions).{0,55}'
        r'(?:contradict|inconsisten|ill[ -]?posed|typo|insufficient|incorrect)|'
        r'(?:contradict|inconsisten).{0,55}(?:problem|question|given)|'
        r'(?:missing|insufficient|not enough) information|'
        r'problem (?:most likely|likely) intends|assuming.{0,80}intended|'
        r'(?:possible|likely|potential) (?:typo|misprint)|'
        r'(?:likely|probable|presumed) (?:final )?answer|'
        r'(?:equation|condition|statement).{0,45}should (?:instead )?be|'
        r'despite (?:the )?(?:earlier |previous )?contradictions|'
        r'cannot (?:be determined|determine a unique)', response))


def standardize(source, row):
    if source == 'prism':
        problem, response = row['problem'], row['solution']
        identifier = row['id']
        expected, origin, license_name = None, 'prismatic_synthesis', 'cc-by-4.0'
    elif source == 'numina':
        if (row['problem_is_valid'] != 'Yes' or row['solution_is_valid'] != 'Yes'
                or row['question_type'] != 'math-word-problem' or row['problem_type'] == 'Calculus'
                or not row.get('answer') or row['answer'] in ('proof', 'notfound')):
            return None, 'not_validated_answered_school_math'
        problem, response = row['problem'], row['solution']
        identifier = hashlib.sha256((problem + response).encode()).hexdigest()
        problem = re.sub(r'(?i)^\s*(?:#+\s*)?problem\s+\d+\s*[.:]?\s+', '', problem)
        expected, origin, license_name = row['answer'], row['source'], 'apache-2.0'
    elif source in ('openmath', 'reasoning'):
        if source == 'reasoning' and (row['problem_source'] not in
                ('aops_c4_high_school_math', 'aops_c6_high_school_olympiads', 'MATH_training_set')
                or row['problem_type'] != 'has_answer_extracted'):
            return None, 'not_reference_labeled_school_math'
        if source == 'openmath' and row['problem_source'] not in ('math', 'augmented_math', 'gsm8k', 'augmented_gsm8k'):
            return None, 'not_math'
        problem, response = row['problem'], row['generated_solution']
        expected, origin = row['expected_answer'], row['problem_source']
        identifier = hashlib.sha256((problem + response).encode()).hexdigest()
        license_name = 'cc-by-4.0'
    else:
        if source == 'olmo' and 'math' not in row['dataset_source'].casefold():
            return None, 'not_math'
        if source == 'nemotron' and row['subset'].casefold() != 'cot':
            return None, 'tool_assisted'
        if source == 'dolci' and row['domain'] != 'Math':
            return None, 'not_math'
        messages = row['messages']
        if row.get('tools') or any(m['role'] == 'tool' or m.get('tool_calls') or
                                  m.get('function_calls') or m.get('functions') for m in messages):
            return None, 'tool_dependent'
        if [m['role'] for m in messages] != ['user', 'assistant']:
            return None, 'not_single_turn'
        problem = row.get('problem') or messages[0]['content']
        response = messages[1].get('content') or ''
        expected = row.get('expected_answer')
        origin = row.get('source', row.get('dataset_source', row.get('source_dataset')))
        identifier = row.get('uuid', row.get('id'))
        license_name = row.get('license', 'odc-by')
    if not problem or not response:
        return None, 'empty'
    response = old.short_solution(response)
    if not response:
        return None, 'incomplete_thinking'
    problem = canonical(problem)
    if outside_scope(problem):
        return None, 'outside_scope'
    if re.search(r'(?i)^(?:prove|show|demonstrate)\b', problem) and not re.search(
            r'(?i)\b(?:find|compute|calculate|evaluate|what|how many)\b', problem):
        return None, 'proof_only'
    if re.search(r'<image>|!\[.*?\]\(', problem + '\n' + response, re.I):
        return None, 'image'
    if re.search(r'(?i)(?:shown in|from|see)\s+(?:the\s+)?(?:diagram|figure|picture)', problem) and '[asy]' not in problem:
        return None, 'missing_diagram'
    if unreliable(response):
        return None, 'inconsistent_working'
    boxes = old.boxed_answers(response)
    if (not boxes or not boxes[-1]) and source not in ('dolci', 'olmo', 'numina'):
        return None, 'no_final_box'
    if len(boxes) != len(re.findall(r'\\(?:boxed|fbox)\s*\{', response)):
        return None, 'incomplete_box'
    if len(response) > 18000:
        return None, 'clearly_long'
    return {'problem': problem, 'response': response, 'expected': expected,
            'prediction': boxes[-1] if boxes else None,
            'answer_format': 'boxed' if boxes else 'natural',
            'source': source, 'origin': origin,
            'id': identifier, 'license': license_name, 'key': key(problem),
            'presentation_key': presentation_key(problem),
            **{n: row.get(n) for n in ('url', 'user_url', 'username') if row.get(n)}}, None


def verify(value):
    if EXCLUDED.contains(value['problem']):
        return None, 'holdout_or_embedded_holdout'
    response = value['response']
    n = len(TOKENIZER(response, add_special_tokens=False)['input_ids'])
    if n < 24:
        return None, 'answer_only'
    if n > RESPONSE_LIMIT:
        return None, 'response_over_1792'
    if len(TOKENIZER(value['problem'], add_special_tokens=False)['input_ids']) > 1024:
        return None, 'prompt_over_1024'
    labels = value.get('references', [])
    if value.get('expected') and value.get('reference_kind') != 'human_reference':
        labels = [value['expected'], *labels]
    if labels:
        matched = None
        try:
            for label in labels:
                if value['prediction'] is not None:
                    agrees = old.equivalent(value['prediction'], label)
                else:
                    from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse, verify
                    actual = parse(response, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
                                   parsing_timeout=1)
                    gold = old.parsed_answer(label)
                    agrees = bool(actual and gold and not isinstance(actual[0], str)
                                  and not isinstance(gold[0], str) and verify(gold, actual, timeout_seconds=1))
                if agrees:
                    matched = label
                    break
        except Exception:
            return None, 'verification_timeout'
        if matched is None:
            return None, 'provider_reference_mismatch'
        value['matched_reference'] = matched
        value['verification'] = value.get('reference_kind') or 'provider_reference'
    else:
        if value['source'] not in ('olmo', 'dolci', 'prism'):
            return None, 'missing_provider_reference'
        value['verification'] = 'provider_filtered_release_no_external_label'
    # Most short solutions contain only a handful of simple numerical equalities.
    # This check is supplementary; it does not certify all symbolic reasoning.
    if not old.numeric_consistency(response):
        return None, 'incorrect_numeric_equation'
    value['prompt_suffix'] = old.PROMPT_SUFFIX if value['answer_format'] == 'boxed' else ''
    messages = [{'role': 'user', 'content': value['problem'] + value['prompt_suffix']},
                {'role': 'assistant', 'content': response}]
    ids, mask, clean, _ = ENCODE(messages)
    if len(ids) > CONTEXT_LIMIT:
        return None, 'conversation_over_2048'
    value.update(messages=clean, tokens=len(ids), response_tokens=n, assistant_tokens=int(mask.sum()))
    value['ids'] = np.asarray(ids, dtype='<u4').tobytes()
    value['mask'] = mask.tobytes()
    # Prefer independently labeled examples, then complete concise working;
    # never select a longer answer just to manufacture a larger token budget.
    confidence = {'human_reference': 3, 'provider_reference': 2}.get(value['verification'], 1)
    value['quality'] = confidence * 10000 + min(n, 96) - max(0, n - 1400) // 8
    value.pop('references', None)
    return value, None


def group(values):
    failures = Counter()
    # Stable source IDs avoid systematically choosing the shortest trajectory.
    for value in sorted(values, key=lambda x: str(x['id'])):
        result, reason = verify(value)
        if result is not None:
            return result, dict(failures)
        failures[reason] += 1
    return None, dict(failures)


def capacity(math_tokens, general_tokens, budget=999817216, limit=2.):
    required = {'math': .8 * budget, 'general': .2 * budget}
    available = {'math': math_tokens, 'general': general_tokens}
    epochs = {k: required[k] / available[k] if available[k] else float('inf') for k in required}
    if any(v > limit for v in epochs.values()):
        raise ValueError(f'Insufficient unique data for the {limit:g}-epoch hard cap: {epochs}')
    return {'epochs': epochs, 'hard_limit': limit, 'preferred_limit': 1.5,
            'meets_preferred': max(epochs.values()) <= 1.5}
