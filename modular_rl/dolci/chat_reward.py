"""Remove transport turn endings before text is scored by Skywork."""

import hashlib
import re
import shutil
from pathlib import Path

ENDINGS = ('<|endoftext|>', '<|im_end|>')
_ENDING = re.compile(r'(?:(?:<\|endoftext\|>|<\|im_end\|>)\s*)+$')


def normalize_conversation(messages):
    result = []
    for message in messages:
        value = dict(message)
        if value['role'] == 'assistant' and isinstance(value.get('content'), str):
            text = value['content']
            if _ENDING.search(text):
                value['content'] = _ENDING.sub('', text).rstrip()
        result.append(value)
    return result


def install(source, implementation):
    """Make the one explicit RM-input change to an otherwise frozen control."""
    source, implementation = Path(source), Path(implementation)
    path = source / 'modular_rl/skywork.py'
    original = path.read_text()
    marker = '            for messages in conversations:\n'
    if original.count(marker) != 1 or 'normalize_conversation' in original:
        raise ValueError('Inspect the pinned Skywork scoring loop before patching')
    updated = original.replace(marker, marker +
        '                from modular_rl.dolci.chat_reward import normalize_conversation\n'
        '                messages = normalize_conversation(messages)\n')
    package = source / 'modular_rl/dolci'
    package.mkdir(parents=True, exist_ok=True)
    (package / '__init__.py').touch(exist_ok=True)
    shutil.copy2(implementation, package / 'chat_reward.py')
    path.write_text(updated)
    return {'version': 1, 'scope': 'Skywork input only; rollout text and policy loss unchanged',
            'strip_terminal_assistant_markers': list(ENDINGS), 'reference_and_generation_same_processing': True,
            'source_file': 'modular_rl/skywork.py',
            'before_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'after_sha256': hashlib.sha256(updated.encode()).hexdigest()}
