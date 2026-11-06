"""Launch independent, read-only raw Base evaluations on the fixed Dolci panel."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from modular_rl.prepare import write_json

from .submit_endpoint import submit


def launch(base, review, models, caps):
    base, review = base.resolve(), review.resolve()
    if review.exists():
        raise ValueError('Use a fresh IF review directory; inspect partial submissions before retrying')
    review.mkdir(parents=True)
    selection = json.loads((base / 'selection/if/selection.json').read_text())
    plan = {'utc': datetime.now(UTC).isoformat(), 'dataset': 'allenai/Dolci-RL-Zero-IF-7B',
            'models': models, 'caps': caps, 'validation_prompts': selection['validation_prompts'],
            'samples_per_prompt': 8, 'answer_forcing': False, 'optimizer_updates': 0,
            'panel_sha256': selection['sha256']['val'], 'experiments': []}
    write_json(review / 'if-eval-experiments.json', plan)
    timestamp = datetime.now(UTC).strftime('%Y%m%d-%H%M%S')
    for model in models:
        for cap in caps:
            experiment = base / 'runs' / f'modular-v2-dolci-if-{model}-gen{cap}-{timestamp}'
            checkpoint = base.parents[1] / 'models' / (model + '_torch_dist')
            submit(Path.cwd(), base, experiment,
                   [{'task': 'if', 'name': 'base', 'checkpoint': str(checkpoint)}],
                   model=model, max_sampled_tokens=cap, if_prompts=selection['validation_prompts'])
            plan['experiments'].append(str(experiment))
            write_json(review / 'if-eval-experiments.json', plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=['Qwen3-0.6B-Base', 'Qwen3-1.7B-Base'],
                        default=['Qwen3-0.6B-Base', 'Qwen3-1.7B-Base'])
    parser.add_argument('--caps', type=int, nargs='+', choices=[2048, 4096], default=[2048, 4096])
    args = parser.parse_args()
    launch(args.base, args.review, args.models, args.caps)
