import json

import pytest

from .repair_summaries import final_monitor_summary


def completed_panel(tmp_path):
    (tmp_path / 'completed.json').write_text(json.dumps({'completed': True, 'updates': 48, 'train_rollouts': 384}))
    directory = tmp_path / 'rollouts/eval'
    directory.mkdir(parents=True)
    rows = [{'metadata_json': json.dumps({'task': 'code', 'key': str(i)}), 'correct': sample == 0}
            for i in range(32) for sample in range(8)]
    path = directory / '0047.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    progress = tmp_path / 'progress.jsonl'
    progress.write_text(json.dumps({'phase': 'eval', 'completed_updates': 48, 'accuracy': .125}) + '\n')
    return path, progress, rows


def test_final_summary_uses_strict_raw_grades_and_the_final_training_budget(tmp_path):
    completed_panel(tmp_path)
    result = final_monitor_summary(tmp_path)
    assert result['david/eval/val_envs/pass@1'] == .125
    assert result['david/eval/val_envs/pass@8'] == 1.
    assert result['david/rollouts'] == 384 and result['david/step'] == 47


def test_missing_response_cannot_be_published_as_a_complete_summary(tmp_path):
    path, _, rows = completed_panel(tmp_path)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows[:-1]))
    with pytest.raises(ValueError, match='Incomplete'):
        final_monitor_summary(tmp_path)


def test_progress_cannot_override_disagreeing_exported_grades(tmp_path):
    _, progress, _ = completed_panel(tmp_path)
    progress.write_text(json.dumps({'phase': 'eval', 'completed_updates': 48, 'accuracy': .9}) + '\n')
    with pytest.raises(ValueError, match='disagree'):
        final_monitor_summary(tmp_path)


@pytest.mark.parametrize('task', ['math', 'code', 'if', 'chat'])
def test_expanded_suite_summary_requires_all64_complete_groups(tmp_path, task):
    path, _, _rows = completed_panel(tmp_path)
    expanded = []
    for group in range(64):
        for sample in range(8):
            expanded.append({'metadata_json': json.dumps({'task': task, 'key': str(group)}),
                             'correct': sample == 0})
    path.write_text(''.join(json.dumps(row) + '\n' for row in expanded))
    result = final_monitor_summary(tmp_path, task=task, validation_prompts=64)
    assert result['david/eval/val_envs/pass@1'] == .125
    assert result['david/eval/val_envs/pass@8'] == 1
    path.write_text(''.join(json.dumps(row) + '\n' for row in expanded[:-1]))
    with pytest.raises(ValueError, match='Incomplete'):
        final_monitor_summary(tmp_path, task=task, validation_prompts=64)
