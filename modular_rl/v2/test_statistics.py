import pytest

from .ablation_report import paired_interval, training_metric_groups


def test_four_primary_comparisons_report_9875_percent_intervals():
    result = paired_interval([-1, 0, .125, 1] * 8, family_size=4)
    assert result["comparison_confidence_level"] == .9875
    assert result["familywise_confidence_level"] == .95
    assert result["paired_problem_adjusted_interval"][0] <= result["paired_problem_95_interval"][0]
    assert result["paired_problem_adjusted_interval"][1] >= result["paired_problem_95_interval"][1]
    assert result["problems"] == 32


def test_identical_paired_gains_have_a_point_interval():
    result = paired_interval([.25] * 32, family_size=4)
    assert result["difference"] == .25
    assert result["paired_problem_95_interval"] == [.25, .25]
    assert result["paired_problem_adjusted_interval"] == [.25, .25]


@pytest.mark.parametrize("differences,family", [([], 4), ([float("nan")], 4), ([0], 0)])
def test_invalid_paired_problem_inputs_are_rejected(differences, family):
    with pytest.raises(ValueError):
        paired_interval(differences, family_size=family)


def test_reused_objective_metrics_do_not_collide_with_skipped_prompt_indices():
    log = "Resumable run first\n" + "\n".join(
        f"step {n}: {{'train/grad_norm': {n + 1}}}" for n in range(4))
    log += "\nstep 1: {'train/grad_norm': 0, 'train/optimizer_update_skipped': 1}\n"
    log += "step 8: {'train/grad_norm': 99}\n"
    log += "Resumable run retry\nstep 8: {'train/grad_norm': 5}\nstep 9: {'train/grad_norm': 6}\n"
    groups = training_metric_groups(log, passes=4, updates=3)
    assert [row['train/grad_norm'] for row in groups[0]] == [1, 2, 3, 4]
    assert groups[1] == [{'train/grad_norm': 0, 'train/optimizer_update_skipped': 1}]
    assert [row['train/grad_norm'] for row in groups[2]] == [5, 6]


def test_concatenated_worker_logs_keep_each_optimizer_record():
    log = ("step 944: {'train/grad_norm': .83, 'train/step': 944}"
           "david 472: {'david/step': 472}\n"
           "step 945: {'train/grad_norm': .7}step 946: {'train/grad_norm': .6}\n")
    groups = training_metric_groups(log, passes=2, updates=474)
    assert [row['train/grad_norm'] for row in groups[472]] == [.83, .7]
    assert groups[473] == [{'train/grad_norm': .6}]


def test_metric_dictionary_prefix_handles_quoted_and_nested_braces():
    log = "step 0: {'note': 'brace }', 'nested': {'value': 3}} trailing\n"
    assert training_metric_groups(log, passes=1, updates=1)[0] == [
        {'note': 'brace }', 'nested': {'value': 3}}]
