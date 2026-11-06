"""Unsmooth Math RL comparison with the original and matched SFT baselines."""

import argparse
import json
from pathlib import Path

from .math_billion import PARENT
from .report import events, scores


def plot(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    root, output = Path(root), Path(output)
    candidates = [('RL on baseline SFT (original settings)', PARENT / 'rl/baseline',
                   PARENT / 'rl/baseline/runs/dolci-baseline-math/policy', '0.65'),
                  ('RL on baseline SFT (matched settings)', root / 'rl/baseline', None, 'tab:orange'),
                  ('RL on refreshed math SFT (1B)', root / 'rl/math', None, 'tab:green')]
    series = []
    for label, suite, policy, color in candidates:
        if not (suite / 'jobs.json').exists():
            continue
        if policy is None:
            job, = json.loads((suite / 'jobs.json').read_text())
            policy = Path(job['output']) / 'policy'
        keys = set(json.loads((suite / 'selection/math/selection.json').read_text())['keys']['val'])
        files = sorted((policy / 'rollouts/eval').glob('*.jsonl'))
        points = []
        for path in files:
            try:
                value = scores(events(path), keys)
            except (ValueError, json.JSONDecodeError):
                if path == files[-1] and not (policy / 'completed.json').exists():
                    continue
                raise
            step = int(path.stem)
            points.append({'train_rollouts': 0 if step == 0 else (step + 1) * 8, **value})
        if points:
            series.append({'label': label, 'color': color, 'points': points})
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for row in series:
        ax.plot([p['train_rollouts'] for p in row['points']], [p['pass@1'] for p in row['points']],
                color=row['color'], label=row['label'], linewidth=1.5)
    ax.set(title='Math', xlabel='train rollouts', ylabel='pass@1', xlim=(0, 8000), ylim=(0, 1))
    ax.set_xticks([0, 2000, 4000, 6000, 8000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda n, _: '0' if n == 0 else f'{n / 1000:g}K'))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps({'smoothing': False, 'series': series,
        'comparison': 'Orange and green share the updated protocol. Gray is the previously published baseline with its original protocol.'}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.root, args.output)
