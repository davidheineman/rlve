"""Manually reviewed MBPP validation contracts, version 2026-11-01.

The 64 problem IDs, test setup and assertions stay fixed. Corrections describe
public interfaces, return formats and conventions; no reference solutions or
test answers are inserted into model prompts. Historic results are immutable.
"""

import copy
import hashlib
import json
import shutil
from pathlib import Path

from modular_rl.prepare import digest
from .config import atomic_json

VERSION = 'mbpp-contracts-20261101'
# All 64 reference programs and their tests were reviewed. Unlisted prompts
# already communicate the tested contract sufficiently and remain unchanged.
CONTRACTS = {
    631: 'Replace each space character with an underscore and return the resulting string. Existing underscores remain unchanged; do not perform the reverse substitution.',
    294: 'Return the largest integer element, ignoring non-integer elements of the heterogeneous list. An integer element is present.',
    343: 'Return a tuple (number_of_letters, number_of_digits), in that order. Use Python isalpha for letters and isdigit for digits; ignore other characters.',
    601: 'Include a class Pair with constructor Pair(a, b) and public attributes a and b in your solution. arr contains n such objects. Return the longest chain length, retaining input order, where each next pair has a strictly greater first component than the previous pair\'s second component.',
    742: 'side is the edge length of a regular tetrahedron. Return its total surface area as a float, without rounding.',
    261: 'Return a tuple of elementwise integer floor divisions of the two tuples, using Python // semantics.',
    422: 'Return the arithmetic mean of the cubes of integers 1 through n inclusive, rounded to six decimal places.',
    276: 'r is the radius and h is the height. Use the fixed constant pi = 3.1415. Return the floating-point volume, evaluating pi times r times r times h from left to right, without rounding.',
    313: 'Return the first nonnegative number (including zero) in list order, as a scalar. Return None if there is none. Do not print or return a list of all positive numbers.',
    314: 'Return the maximum sum of selected cells. Cells sharing an edge or a corner cannot both be selected; thus at most one cell can be selected in a column, and selected columns cannot be adjacent.',
    305: 'Each list entry is a string. Return the first matching pair of words as a tuple (word1, word2), where the string starts with two words beginning with uppercase P, separated by one non-word character. Return None if no entry matches.',
    960: 'Return the number of tilings of a 2-by-(n-1) rectangle with 2-by-1 dominoes, for n >= 1. For this API the special case n = 0 returns 0.',
    215: 'Return a flat decoded list. A list item [count, value] represents count copies of value; an item that is not a list represents itself once.',
    699: 'Return the minimum number of swaps of two positions within the first binary string needed to obtain the second, or the exact string "Not Possible" if this cannot be done.',
    830: 'Return a rounded-up numeric value at digits decimal places, toward positive infinity (ceiling), rather than ordinary nearest rounding.',
    680: 'Return True when the list is nondecreasing, allowing equal neighboring values; otherwise return False.',
    890: 'arr1 has one extra element relative to sorted arr2; n is len(arr2). Return the zero-based index of the extra element in arr1.',
    878: 'K is a collection of allowed values, not a count. Return True exactly when every element of test_tuple belongs to K.',
    304: 'Each of the first rotations entries in ranges gives inclusive zero-based endpoints of a subarray to rotate right by one position. Apply these rotations in list order and return the value at the specified final index.',
    87: 'Return one dictionary containing the keys from all three arguments. On a duplicate key, the earlier argument takes precedence.',
    84: 'Use the Newman-Conway sequence indexed from 1, whose first two terms are both 1. Return the nth term as an integer.',
    323: 'Return the rearranged list, starting with a negative item and then alternating negative and nonnegative items while both remain. Preserve relative order within each sign group, then append remaining items in their original order. Zero is nonnegative.',
    457: 'Return the lexicographically smallest sublist using Python list comparison, not a length or a collection of sublists.',
    398: 'Return one integer totaling all decimal digits found in the string representation of each item in nums. Ignore signs, letters, punctuation and brackets; an item can itself be a list.',
    543: 'Return the number of decimal digits in the positive sum num1 + num2, as an integer. Do not print it.',
    784: 'Find the first even and first odd element in input order and return their product. If either category is missing, use -1 for its missing value.',
    387: 'N is a hexadecimal string. Return the exact string "Even" or "Odd" for the represented integer.',
    649: 'm and n are zero-based indices. Return the sum from m through n inclusive.',
    836: 'Return the length, not the sum, of the nonempty contiguous subarray with maximum sum among the first size items. For ties choose the earliest ending subarray; for that ending index retain the earliest start giving that sum.',
    528: 'Return a tuple (minimum_sublist_length, lexicographically_smallest_sublist). Compute these two components independently using Python list comparison for the second.',
    92: 'n is a digit string, not an integer. Return True exactly when its length exceeds two and every character equals the character two positions earlier.',
    686: 'Return a string containing the standard Python dictionary representation of the element counts (str of a dict), not a dictionary object. Insert keys in order of first appearance; preserve the usual spaces after colons and commas.',
    124: 'a and b are addends of the complex number a + b; b may already be imaginary, such as 1j. Return its phase in radians in the range [-pi, pi]. Do not interpret b as a second real coordinate.',
    223: 'Return True if x occurs strictly more than n/2 times in the first n elements of the sorted array, otherwise False.',
    263: 'Return a merged dictionary; values from d2 take precedence on duplicate keys.',
    966: 'Return a list with empty tuples removed, preserving every other item and its order, including strings and tuples containing an empty string.',
    648: 'lst has even length. Return the list with adjacent pairs swapped: indices 0 and 1, 2 and 3, and so on.',
    270: 'Consider the first n items. Sum only even-valued elements whose zero-based index is even.',
    751: 'i is the zero-based root index of the subtree to inspect in the array representation of a binary tree. Return whether every parent in that subtree is <= its children; i=0 checks the full heap.',
    643: 'Match a lowercase z strictly inside a word (with a word character on both sides). Return exactly "Found a match!" for a match, otherwise "Not matched!".',
    522: 'Return the length of the longest subsequence that is strictly increasing then strictly decreasing. Either side may have length one, so a monotone subsequence is allowed.',
}


def prepare(original, destination):
    original, destination = Path(original), Path(destination)
    if destination.exists():
        raise ValueError('Use a new versioned validation directory')
    shutil.copytree(original, destination)
    path = destination / 'code/val.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    changes = []
    audited = []
    for row in rows:
        meta = row['metadata']
        identifier = int(meta['id'])
        audited.append(identifier)
        if identifier not in CONTRACTS:
            continue
        before = copy.deepcopy(row)
        messages = copy.deepcopy(row['user_prompt'])
        # Replace inaccurate introductory prose, retaining the exact required API.
        prefix, api = messages[-1]['content'].split('\nRequired API', 1)
        messages[-1]['content'] = CONTRACTS[identifier] + '\nRequired API' + api
        row['user_prompt'] = messages
        meta.update(messages=messages, key=digest(messages), validation_contract_version=VERSION)
        assert meta['target'] == before['metadata']['target']
        changes.append({'id': identifier, 'old_key': before['metadata']['key'], 'new_key': meta['key'],
                        'original_description': prefix, 'corrected_description': CONTRACTS[identifier]})
    assert len(rows) == len(set(audited)) == 64
    assert set(CONTRACTS) <= set(audited)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    info = json.loads((destination / 'code/selection.json').read_text())
    info['keys']['val'] = [r['metadata']['key'] for r in rows]
    info['sha256']['val'] = hashlib.sha256(path.read_bytes()).hexdigest()
    info['validation_contract_version'] = VERSION
    atomic_json(destination / 'code/selection.json', info)
    atomic_json(destination / 'code/contract-repairs.json', {'version': VERSION, 'audited_ids': audited,
                'modified_prompts': len(changes), 'changes': changes, 'tests_unchanged': True,
                'training_data_unchanged': True, 'original_selection': str(original)})
    return changes
