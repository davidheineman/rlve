import pytest

from .followup import if_mixtures, safe_command


def test_precise_if_mixes_hit_targets_and_preserve_source_ratios():
    manifest = {'sources': {'a': {'domain': 'Precise IF'}, 'b': {'domain': 'Precise IF'},
                            'c': {'domain': 'Chat'}, 'd': {'domain': 'Coding'}},
                'recipe': {'mixes': {'baseline': {'a': .01, 'b': .03, 'c': .8, 'd': .16}}}}
    mixes = if_mixtures(manifest)
    for name, target in [('if20', .2), ('if50', .5)]:
        weights = mixes[name]
        assert sum(weights.values()) == pytest.approx(1)
        assert weights['a'] + weights['b'] == pytest.approx(target)
        assert weights['a'] / weights['b'] == pytest.approx(1 / 3)
        assert weights['c'] / weights['d'] == pytest.approx(5)


def test_followup_submissions_clear_parent_service_and_exclude_bad_cuda_node():
    command = ['sbatch', '--exclude=viscam10', '--export=ALL,WANDB_SERVICE=stale', 'run.sbatch']
    fixed = safe_command(command)
    assert '--exclude=viscam10,pasteur6' in fixed
    assert fixed.count('--export=ALL,WANDB_SERVICE=') == 1
    assert safe_command(fixed) == fixed
