"""Descriptive response checks for saved monitoring and confirmation panels."""
import ast
import json
import re
from collections import Counter
from statistics import mean

from modular_rl.graders import extract_code

from .inspect_rollouts import read_events


def panel_quality(path, prompts, samples=8):
    rows = read_events(path)
    groups = Counter(json.loads(r['metadata_json'])['key'] for r in rows)
    if len(rows) != prompts * samples or len(groups) != prompts or set(groups.values()) != {samples}:
        raise ValueError('Quality panel must retain every declared problem and sample')
    repeated, tokens, parsed, api_present = [], [], [], []
    for row in rows:
        words = re.findall(r'\S+', row['response'])
        grams = [tuple(words[i:i + 6]) for i in range(max(0, len(words) - 5))]
        repeated.append(1 - len(set(grams)) / len(grams) if grams else 0.)
        meta = json.loads(row['metadata_json'])
        state = meta.get('_modular_v2_forcing', {})
        tokens.append(row['response_tokens'] - state.get('prefix_tokens', 0))
        if meta['task'] == 'code':
            # Match the exact response segment passed to the code reward.
            response = row['response'][state['answer_char_start']:] if state else row['response']
            if state.get('mode') == 'python-prefix':
                from .code_generation import submission_text

                response = submission_text(response)
            try:
                tree = ast.parse(extract_code(response))
                names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
            except (SyntaxError, ValueError, RecursionError):
                names = set()
            parsed.append(bool(names))
            expected = {name for msg in meta['messages'] for name in re.findall(r'\bdef\s+(\w+)\s*\(', msg['content'])}
            api_present.append(bool(expected) and expected <= names)
    result = {'responses': len(rows), 'problems': prompts, 'samples_per_prompt': samples,
              'mean_sampled_tokens': mean(tokens),
              'mean_repeated_6word_gram_fraction': mean(repeated),
              'response_fraction_repetition_above_half': mean(x > .5 for x in repeated)}
    if parsed:
        result.update(parseable_implementation_fraction=mean(parsed), required_api_present_fraction=mean(api_present))
    return result


def passes_guard(quality, guard):
    return (quality['mean_repeated_6word_gram_fraction'] <= guard['mean_repeated_6word_fraction_at_most']
            and quality['response_fraction_repetition_above_half'] <= guard['response_fraction_repetition_above_half_at_most'])
