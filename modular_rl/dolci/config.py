"""Pinned, token-weighted Dolci conversation-mixture experiment."""

import hashlib
import json
import math
from pathlib import Path

DATASET = 'allenai/Dolci-Instruct-SFT'
REVISION = 'bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221'
MODEL = 'Qwen/Qwen3-0.6B-Base'
MODEL_REVISION = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
SEED = 20261027
SEQ_LEN = 2048
TOKENS_PER_STEP = 262144
STEPS = 1000000000 // TOKENS_PER_STEP
ACTUAL_TOKENS = STEPS * TOKENS_PER_STEP
QUEUES = 'miso-lo,sphinx-lo,jag-lo,sc-loprio'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def family(domain, natural='complement'):
    if domain in ('Math', 'Coding', 'Chat'):
        return {'Math': 'math', 'Coding': 'code', 'Chat': 'chat'}[domain]
    if natural == 'complement' or domain in ('Science', 'Other', 'Reasoning'):
        return 'web'
    return 'other'


def mixtures(sources, natural='complement'):
    total = sum(s['tokens'] for s in sources.values())
    baseline = {name: s['tokens'] / total for name, s in sources.items()}
    mixes = {'baseline': baseline}
    for target in ('web', 'chat', 'math', 'code'):
        mass = sum(baseline[n] for n, s in sources.items() if family(s['domain'], natural) == target)
        if not 0 < mass < 1:
            raise ValueError(f'No positive target and remainder mass for {target}')
        mixes[target] = {n: w * (.8 / mass if family(sources[n]['domain'], natural) == target
                                else .2 / (1 - mass)) for n, w in baseline.items()}
        assert math.isclose(sum(mixes[target].values()), 1.)
    return mixes


def apportion(total, weights):
    raw = {k: total * w / sum(weights.values()) for k, w in weights.items()}
    counts = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - counts[k]), k))[:total - sum(counts.values())]:
        counts[k] += 1
    return counts
