"""Prepare short, reference-checked, unique-question MATH SFT conversations.

Only data preparation and audit commands live here. Published experiments are
immutable; no running trainer/source snapshots are modified by this module.
"""

import argparse
import bisect
import hashlib
import html
import json
import logging
import math
import multiprocessing
import os
import re
import shutil
import sqlite3
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from .config import SEED, atomic_json, fingerprint

DATASETS = {
    'olmo': ('allenai/Dolci-Think-SFT-7B', '83f618e2fe0aed29ee3440a3658a0326c175d09d'),
    'openmath': ('nvidia/OpenMathInstruct-2', '469216e3f46f4dacf476b382e192485ea51a143e'),
    'nemotron': ('nvidia/Nemotron-SFT-Math-v4', '84d42ad0cb960f07f951b9baa9ed2b46a5a18c66'),
}
MATH_REPO = 'EleutherAI/hendrycks_math'
MATH_REVISION = '21a5633873b6a120296cce3e2df9d5550074f4a3'
RESPONSE_LIMIT = 1536
CONTEXT_LIMIT = 2048
PROMPT_SUFFIX = '\n\nSolve the problem and put your final answer in \\boxed{}.'
TOKENIZER = None
ENCODE = None


def canonical_question(text):
    """Normalize presentation while retaining variable case, constants and signs."""
    text = unicodedata.normalize('NFKC', html.unescape(text)).strip()
    for suffix in (PROMPT_SUFFIX, '\nPlease reason step by step, and put your final answer within \\boxed{}.'):
        if text.endswith(suffix): text = text[:-len(suffix)]
    text = re.sub(r'\\(?:left|right)\b', '', text)
    text = text.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    text = re.sub(r'\\[()[\]]', ' ', text).replace('$', ' ')
    text = re.sub(r'\\[,!;:]', ' ', text)
    return ' '.join(text.split())


def question_key(text):
    return hashlib.sha256(canonical_question(text).encode()).hexdigest()


def boxed_answers(text):
    answers = []
    for match in re.finditer(r'\\(?:boxed|fbox)\s*\{', text):
        start, depth = match.end(), 1
        for index in range(start, len(text)):
            depth += (text[index] == '{') - (text[index] == '}')
            if depth == 0:
                answers.append(text[start:index].strip()); break
    return answers


def presentation_latex(text):
    text = re.sub(r'\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle|bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr|left|right)(?![A-Za-z])', '', text)
    text = re.sub(r'\\[!,:;>]', ' ', text)
    return ' '.join(text.split())


def answer_key(text):
    text = presentation_latex(text)
    text = text.strip()
    boxes = boxed_answers(text)
    if boxes: text = boxes[-1]
    text = re.sub(r'^\\\(|\\\)$|^\\\[|\\\]$', '', text).strip('$ ')
    text = text.replace('\\left', '').replace('\\right', '').replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    return re.sub(r'\s+', '', text)


def short_solution(text):
    """Keep the complete final worked solution; never cut a trace by token index."""
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[-1]
    elif '<think>' in text:
        return None
    return text.strip()


def quality_reason(problem, response):
    """Reject explicit uncertainty about the problem and out-of-scope math."""
    advanced = (r'(?i)\\(?:int|iint|iiint|oint|lim)(?![A-Za-z])|\b(?:integrals?|integrat(?:e|ion)|'
                r'antiderivative|derivatives?|differential equations?|laplace|fourier|'
                r'topology|manifolds?|hilbert|banach|lebesgue|martingales?|'
                r'group theory|ring theory|complex analysis|minimal polynomial|galois|'
                r'automorphism|eigenvalues?|eigenvectors?|diagonalization|splitting field|'
                r'nonsingular|singular values?|orthonormal|matrix rank)\b|\\operatorname\{rank\}')
    if re.search(advanced, problem): return 'outside_math_benchmark_scope'
    uncertainty = (r'(?is)contradict|inconsisten|ill[ -]?posed|under[ -]?(?:specified|determined)|'
                   r'no such (?:real |positive |integer )?(?:function|triangle|number|configuration)|'
                   r'not (?:well[ -]defined|uniquely determined)|cannot be determined|'
                   r'(?:intended|missing).{0,50}(?:configuration|sides|diagram|condition|constraint)|'
                   r'based on.{0,60}(?:clean answer|intended)|'
                   r'assum(?:e|ing).{0,80}(?:isosceles|equilateral)|'
                   r'(?:after|through) (?:a )?(?:detailed analysis|careful consideration)|'
                   r'problem might expect|does not resolve|resolving inconsistencies')
    if re.search(uncertainty, response): return 'unreliable_or_incomplete_working'
    reinterpretation = (r'(?is)(?:problem|question).{0,100}(?:most likely|likely intends|'
                        r'should (?:say|be|read)|is interpreted as)|'
                        r'(?:most likely|presumably) (?:intends|intended|means|meant)')
    if re.search(reinterpretation, response): return 'reinterprets_problem_statement'
    if len(response.split()) < 110 and re.search(r'(?i)\boptimi[sz]ation (?:leads to|gives|yields|shows|produces|results in)\b', response):
        return 'optimization_result_without_derivation'
    return None


def candidate(source, row):
    """Standardize source schemas and reject tool-dependent/proof/image tasks."""
    if source == 'openmath':
        if row['problem_source'] not in ('math', 'augmented_math'): return None, 'not_math_subset'
        problem, response, expected = row['problem'], row['generated_solution'], row['expected_answer']
        identity = hashlib.sha256((problem + response).encode()).hexdigest()
        origin = row['problem_source']; license_name = 'cc-by-4.0'
    else:
        messages = row['messages']
        if source == 'olmo' and 'math' not in row['dataset_source'].casefold(): return None, 'not_math_subset'
        if source == 'nemotron' and row['subset'].casefold() != 'cot': return None, 'tool_assisted_subset'
        if source == 'nemotron' and row['source'] != 'AoPS': return None, 'outside_selected_competition_source'
        if row.get('tools') or any(m['role'] == 'tool' or m.get('tool_calls') for m in messages):
            return None, 'tool_dependent'
        users = [m['content'] for m in messages if m['role'] == 'user']
        assistants = [m for m in messages if m['role'] == 'assistant']
        if len(users) != 1 or len(assistants) != 1: return None, 'not_single_turn'
        problem = row.get('problem') or users[0]
        response = assistants[0].get('content') or ''
        expected = row.get('expected_answer')
        identity = row.get('uuid', row.get('id'))
        origin = row.get('source', row.get('dataset_source'))
        license_name = row.get('license', 'odc-by')
        # reasoning_content is deliberately omitted: the whole final solution
        # must stand on its own and fit the RL response budget.
    response = short_solution(response)
    if not response: return None, 'no_complete_final_solution'
    problem = canonical_question(problem)
    if not problem: return None, 'empty_problem'
    if re.search(r'(?i)\b(prove|proof|show that|demonstrate that)\b', problem): return None, 'proof_task'
    if re.search(r'(?i)\[asy\]|<image>|!\[.*?\]\(|(?:in|from|shown in)\s+(?:the\s+)?(?:diagram|figure|picture)', problem):
        return None, 'diagram_or_image'
    bad = r'(?is)(?:problem|question|statement).{0,45}(?:typo|inconsistent|incorrect|mistake)|assuming.{0,100}common configuration|(?:insufficient|not enough) information'
    if re.search(bad, response): return None, 'ambiguous_or_inconsistent'
    reason = quality_reason(problem, response)
    if reason: return None, reason
    boxes = boxed_answers(response)
    if not boxes or not boxes[-1]: return None, 'no_final_box'
    if len(boxes) != len(re.findall(r'\\(?:boxed|fbox)\s*\{', response)):
        return None, 'unfinished_box'
    return {'problem': problem, 'response': response, 'expected': expected,
            'prediction': boxes[-1], 'key': question_key(problem),
            'source': source, 'origin': origin, 'id': identity, 'license': license_name,
            'url': row.get('url'), 'user_url': row.get('user_url'), 'username': row.get('username')}, None


def initialize_worker(model, runtime):
    global TOKENIZER, ENCODE
    import sys
    if runtime: sys.path.insert(0, runtime)
    from . import prepare
    prepare.initialize(model, [])
    TOKENIZER, ENCODE = prepare.TOKENIZER, prepare.encode
    logging.getLogger('math_verify').setLevel(logging.CRITICAL)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@lru_cache(maxsize=16384)
def parsed_answer(text):
    from math_verify import LatexExtractionConfig, parse
    clean = presentation_latex(text)
    boxes = boxed_answers(clean)
    if boxes: clean = boxes[-1]
    clean = re.sub(r'^\\\(|\\\)$|^\\\[|\\\]$', '', clean).strip('$ ')
    parsed = parse('$' + clean + '$', extraction_config=[LatexExtractionConfig()], parsing_timeout=1)
    # A reference may express a numeric answer as an identity A=B. Only use
    # its scalar value after independently verifying the constant identity.
    if parsed and getattr(parsed[0], 'is_Equality', False) and not parsed[0].free_symbols:
        eq = parsed[0]
        if eq.lhs.is_number is True and eq.rhs.is_number is True:
            from math_verify import verify
            if verify([eq.lhs], [eq.rhs], timeout_seconds=1): parsed[0] = eq.rhs
    return parsed


def equivalent(predicted, expected):
    if not expected: return False
    if answer_key(predicted) == answer_key(expected): return True
    from math_verify import verify
    gold, actual = parsed_answer(expected), parsed_answer(predicted)
    if not gold or not actual: return False
    # Do not call the parser's unparsed-string fallback an independent check.
    if isinstance(gold[0], str) or isinstance(actual[0], str): return False
    return bool(verify(gold, actual, timeout_seconds=1))


def numeric_consistency(response):
    """Check short, fully numeric exact equations; skip prose and variables."""
    spans = re.findall(r'\\\((.*?)\\\)|\\\[(.*?)\\\]|(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', response, re.S)
    checked = 0
    for groups in spans:
        text = next((v for v in groups if v), '')
        if '=' not in text or len(text) > 180: continue
        rest = re.sub(r'\\(?:frac|dfrac|tfrac|sqrt|cdot|times|pi|binom|left|right)(?![A-Za-z])', '', text)
        if re.search(r'[^\d\s+\-*/^{}()\[\].,=!]', rest): continue
        sides = [s.strip() for s in text.split('=')]
        if not all(re.search(r'\d', s) for s in sides): continue
        for a, b in zip(sides, sides[1:]):
            if checked >= 12: return True
            checked += 1
            try:
                gold, actual = parsed_answer(a), parsed_answer(b)
                if gold and actual and not isinstance(gold[0], str) and not isinstance(actual[0], str):
                    from math_verify import verify
                    if not verify(gold, actual, timeout_seconds=1): return False
            except Exception:
                continue
    return True


def verify_candidate(value):
    result = dict(value)
    problem, response = value['problem'], value['response']
    response_tokens = len(TOKENIZER(response, add_special_tokens=False)['input_ids'])
    if response_tokens < 32: return None, 'too_little_working'
    if response_tokens > RESPONSE_LIMIT: return None, 'response_over_1536'
    # Formula-only final answers are short but do not teach a worked solution.
    if not (response.count('=') >= 2 or re.search(r'(?i)\b(since|because)\b', response)
            or response_tokens >= 96 and re.search(r'(?i)\b(therefore|thus|hence|then)\b', response)):
        return None, 'no_worked_solution'
    if len(TOKENIZER(problem, add_special_tokens=False)['input_ids']) > 768:
        return None, 'problem_too_long'
    try:
        if value['reference_answers'] and any(not equivalent(value['reference_answers'][0], other)
                                               for other in value['reference_answers'][1:]):
            return None, 'conflicting_reference_answers'
        matches = [answer for answer in value['reference_answers'] if equivalent(value['prediction'], answer)]
    except Exception:
        return None, 'verification_error_or_timeout'
    if not matches: return None, 'reference_answer_mismatch'
    if not numeric_consistency(response): return None, 'incorrect_numeric_equation'
    messages = [{'role': 'user', 'content': problem + PROMPT_SUFFIX}, {'role': 'assistant', 'content': response}]
    ids, mask, _, _ = ENCODE(messages)
    if len(ids) > CONTEXT_LIMIT: return None, 'conversation_over_2048'
    result.update(messages=messages, response_tokens=response_tokens, tokens=len(ids),
                  assistant_tokens=int(mask.sum()), matched_reference=matches[0],
                  verification_method='exact_reference' if answer_key(matches[0]) == answer_key(value['prediction']) else 'symbolic_reference')
    import numpy as np
    result['encoded_ids'] = np.asarray(ids, dtype='<u4').tobytes()
    result['encoded_mask'] = mask.tobytes()
    result.pop('reference_answers')
    return result, None


def inventory(root):
    from huggingface_hub import HfApi
    import fsspec
    import pyarrow.parquet as pq
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, (repo, revision) in DATASETS.items():
        info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
        files = [s for s in info.siblings if re.search(r'/train-\d+-of-\d+\.parquet$', s.rfilename)]
        def describe(item):
            cache = root / 'inventory-units' / name / (Path(item.rfilename).stem + '.json')
            if cache.exists(): return json.loads(cache.read_text())
            url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{item.rfilename}'
            with fsspec.open(url, block_size=262144, cache_type='bytes') as stream:
                parquet = pq.ParquetFile(stream)
                column = {'olmo': 'dataset_source', 'openmath': 'problem_source', 'nemotron': 'subset'}[name]
                if name == 'openmath':
                    # This shuffled corpus has hundreds of row groups per file.
                    # Only the schema/row count is needed for this source; the
                    # builder counts exact categories from local parquet rows.
                    counts = {c: None for c in ('math', 'augmented_math', 'gsm8k', 'augmented_gsm8k')}
                else:
                    table = parquet.read(columns=[column])
                    counts = {v['values']: v['counts'] for v in table.column(0).value_counts().to_pylist()}
                result = {'path': item.rfilename, 'bytes': item.size, 'rows': parquet.metadata.num_rows,
                          'categories': counts}
                atomic_json(cache, result)
                print('INVENTORY_FILE', name, item.rfilename, flush=True)
                return result
        with ThreadPoolExecutor(8) as workers: results = list(workers.map(describe, files))
        sources[name] = {'repo': repo, 'revision': revision, 'files': results}
        atomic_json(root / 'inventory.json', sources)
        print('INVENTORY', name, len(results), sum(v['rows'] for v in results), flush=True)
    return sources


def download(repo, revision, filename, path):
    import requests
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    for attempt in range(6):
        try:
            with requests.get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}',
                              stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with temporary.open('wb') as out:
                    for chunk in response.iter_content(4 * 1024**2): out.write(chunk)
            import pyarrow.parquet as pq
            pq.ParquetFile(temporary)
            temporary.replace(path); return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 5: raise
            time.sleep(min(5 * 2**attempt, 45))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inventory', action='store_true')
    args = parser.parse_args()
    if args.inventory: inventory(args.root)
