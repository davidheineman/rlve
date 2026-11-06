"""Explicitly install the terminal-marker fix into a new frozen RL snapshot."""

import hashlib
from pathlib import Path


def install(source, implementation):
    path = Path(source) / 'modular_rl/graders.py'
    original = path.read_text()
    canonical = Path(implementation).read_text()
    start = 'def extract_code(response):\n'
    end = '    fenced = re.findall('
    old_prefix = start + '    response = response.rsplit("</think>", 1)[-1].strip()\n'
    if canonical.count(start) != 1 or canonical.count(end) != 1:
        raise ValueError('Inspect the current code extractor before installing')
    fixed_prefix = start + canonical.split(start, 1)[1].split(end, 1)[0]
    if 'response.removesuffix(marker)' not in fixed_prefix:
        raise ValueError('The canonical code extractor lacks terminal normalization')
    if original.count(old_prefix + end) == 1:
        updated = original.replace(old_prefix + end, fixed_prefix + end)
    elif original.count(fixed_prefix + end) == 1:
        updated = original
    else:
        raise ValueError('Inspect the frozen code extractor before patching')
    path.write_text(updated)
    return {'version': 1, 'scope': 'Code reward input only; raw responses and policy loss preserved',
            'strip_terminal_markers': ['<|im_end|>', '<|endoftext|>'],
            'source_file': 'modular_rl/graders.py',
            'before_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'after_sha256': hashlib.sha256(updated.encode()).hexdigest()}
