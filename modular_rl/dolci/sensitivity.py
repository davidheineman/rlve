"""Compare fixed-panel results with literal related-task matches omitted."""

import argparse
import csv
import json
from pathlib import Path

from .config import atomic_json
from .report import events, scores


def sensitivity(root):
    root = Path(root)
    primary = json.loads((root / 'final-report.json').read_text())
    assert primary['passed'] and len(primary['rl']) == 14
    overlap = json.loads((root / 'validation-related-variants.json').read_text())
    excluded = {task: set(keys) for task, keys in overlap['sensitivity_excluded_keys'].items()}
    result = {'passed': True, 'primary_validation_unchanged': True,
              'scope': 'Omit literal related-question matches; this is not semantic decontamination',
              'excluded_keys': overlap['sensitivity_excluded_keys'], 'runs': []}
    table = []
    for run in primary['rl']:
        task = run['task']
        path = root / 'rl/base/selection' / task / 'selection.json'
        keys = set(json.loads(path.read_text())['keys']['val'])
        omit = excluded.get(task, set())
        assert omit <= keys and len(keys) == 64
        retained = keys - omit
        panels = []
        for path in sorted((Path(run['run_directory']) / 'rollouts/eval').glob('*.jsonl')):
            rows = events(path)
            filtered = [r for r in rows if json.loads(r['metadata_json'])['key'] in retained]
            panels.append({'rollout_id': int(path.stem), 'full': scores(rows, keys),
                           'related_tasks_omitted': scores(filtered, retained)})
        assert len(panels) == run['evaluation_panels']
        assert panels[-1]['full']['pass@1'] == run['final']['pass@1']
        row = {'name': run['name'], 'task': task, 'validation_prompts': len(retained),
               'excluded_prompts': len(omit), 'initial': panels[0], 'final': panels[-1], 'panels': panels}
        result['runs'].append(row)
        table.append({'name': run['name'], 'task': task, 'retained_prompts': len(retained),
                      'full_initial_pass1': panels[0]['full']['pass@1'],
                      'full_final_pass1': panels[-1]['full']['pass@1'],
                      'subset_initial_pass1': panels[0]['related_tasks_omitted']['pass@1'],
                      'subset_final_pass1': panels[-1]['related_tasks_omitted']['pass@1']})
    with (root / 'related-task-sensitivity.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    atomic_json(root / 'related-task-sensitivity.json', result)
    print(json.dumps(table, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    sensitivity(parser.parse_args().root)
