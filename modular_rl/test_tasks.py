import json
from fractions import Fraction
from types import SimpleNamespace

import pytest

from modular_rl.graders import extract_code, grade_code, grade_math, numeric_answer
from modular_rl.prepare import assign_splits, normalize


@pytest.mark.parametrize(
    "text,expected",
    [
        (r"We compute 21. So \boxed{1,234}", Fraction(1234)),
        (r"\boxed{\frac{1}{2}}", Fraction(1, 2)),
        ("<answer>0.50</answer>", Fraction(1, 2)),
        ("reasoning has 9 and 3\n42", Fraction(42)),
        ("Still reasoning about 42 apples", None),
        ("<think>42", None),
        ("</think>\\boxed{3.5%}", Fraction(7, 2)),
    ],
)
def test_final_answer_extraction(text, expected):
    assert numeric_answer(text) == expected


def test_math_controls():
    assert grade_math(r"\boxed{0.5}", "Calculate.\n#### 0.50")["accuracy"] == 1
    assert grade_math(r"\boxed{7}", "#### 8")["accuracy"] == 0
    assert grade_math("The intermediate result is 8.", "#### 8")["accuracy"] == 0
    with pytest.raises(ValueError):
        grade_math("7", "invalid reference")


def test_code_controls_setup_reasoning_failure_timeout_and_stdout():
    target = {
        "setup": "from math import sqrt",
        "tests": ["assert root(9)==3", "assert root(4)==2"],
    }
    good = "<think>some thoughts</think>\n```python\ndef root(x):\n    print('extra output')\n    return sqrt(x)\n```"
    assert grade_code(good, target)["accuracy"] == 1
    assert grade_code("def root(x): return 0", target)["accuracy"] == 0
    assert grade_code("def root(:", target)["accuracy"] == 0
    assert grade_code("while True: pass", target, timeout=0.5)["error"] == "timeout"
    assert (
        grade_code("import subprocess; subprocess.Popen(['true'])", target)["accuracy"]
        == 0
    )
    assert extract_code(good).startswith("def root")


def test_code_grader_uses_implementation_instead_of_example_or_output_blocks():
    target = {"tests": ["assert add(1,2)==3", "assert add(-2,2)==0"]}
    response = (
        "Solution:\n```python\ndef add(a,b): return a+b\n```\n"
        "Example:\n```python\nprint(add(1,2))\n```\nOutput:\n```\nResult: 3\n```"
    )
    assert grade_code(response, target)["accuracy"] == 1
    revised = response + "\nCorrected solution:\n```python\ndef add(a,b): return 0\n```"
    assert grade_code(revised, target)["accuracy"] == 0


def test_code_diagnostics_distinguish_missing_api_from_wrong_result():
    target = {"tests": ["assert add(1,2)==3", "assert add(0,0)==0"]}
    wrong = grade_code("def add(a,b): return 0", target, diagnostics=True)
    missing = grade_code("def Add(a,b): return a+b", target, diagnostics=True)
    assert wrong["passes"] == [False, True]
    assert wrong["test_results"][0]["error"] == "AssertionError"
    assert missing["test_results"][0]["error"] == "NameError"
    assert missing["test_results"][0]["test"] == target["tests"][0]
    assert wrong["accuracy"] == missing["accuracy"] == 0
    assert "test_results" not in grade_code("def add(a,b): return 0", target)


@pytest.mark.parametrize("ending", ["<|im_end|>", "<|endoftext|>", "<|im_end|>\n<|endoftext|>  "])
def test_code_terminal_markers_do_not_change_bare_or_fenced_solution_reward(ending):
    target = {"tests": ["assert add(1,2)==3", "assert add(-2,2)==0"]}
    code = "def add(a,b): return a+b"
    for answer in (code, "```python\n" + code + "\n```"):
        response = "<think>reasoning</think>\n" + answer + ending
        assert extract_code(response) == code
        assert grade_code(response, target)["accuracy"] == 1
    assert grade_code("def add(a,b): return 0" + ending, target)["accuracy"] == 0
    assert grade_code("def add(:" + ending, target)["accuracy"] == 0


def test_code_terminal_cleanup_preserves_literal_marker_inside_python():
    response = 'def token(): return "<|im_end|>"<|endoftext|>'
    target = {"tests": ['assert token() == "<|im_end|>"']}
    assert extract_code(response) == 'def token(): return "<|im_end|>"'
    assert grade_code(response, target)["accuracy"] == 1


def test_gsm_templates_never_cross_split_and_selection_reproduces():
    rows = []
    for template in range(10):
        for instance in range(5):
            rows.append(
                normalize(
                    "math",
                    {
                        "id": template,
                        "original_id": template + 100,
                        "instance": instance,
                        "question": f"question {template} {instance}",
                        "answer": "#### 3",
                    },
                )
            )
    splits = assign_splits(rows, seed=42, val_fraction=0.2)
    assert len(splits["train"]) == 40 and len(splits["val"]) == 10
    groups = [
        {x["metadata"]["split_group"] for x in splits[s]} for s in ("train", "val")
    ]
    assert not groups[0] & groups[1]
    assert splits == assign_splits(rows, seed=42, val_fraction=0.2)


def test_code_prompt_exposes_api_without_solution_or_tests():
    row = {
        "task_id": 1,
        "text": "Add numbers.",
        "code": "def add(a,b): return a+b",
        "test_list": ["assert add(1,2)==3"],
        "test_setup_code": "",
        "challenge_test_list": [],
    }
    example = normalize("code", row)
    prompt = json.dumps(example["user_prompt"])
    assert "def add(a, b): ..." in prompt
    assert "return a+b" not in prompt and "assert add" not in prompt
    assert grade_code(row["code"], example["metadata"]["target"])["accuracy"] == 1


def test_preference_prompt_never_contains_reference_completion():
    messages = [{"role": "user", "content": "Question"}]
    example = normalize(
        "chat",
        {
            "id": "a",
            "chosen": messages + [{"role": "assistant", "content": "chosen"}],
            "rejected": messages + [{"role": "assistant", "content": "rejected"}],
        },
    )
    assert example["user_prompt"] == messages
    assert example["metadata"]["target"]["rejected_reference"] == "rejected"
    with pytest.raises(ValueError, match="inconsistent"):
        normalize(
            "chat",
            {
                "id": "a",
                "chosen": messages + [{"role": "assistant", "content": "a"}],
                "rejected": [
                    {"role": "user", "content": "different"},
                    {"role": "assistant", "content": "b"},
                ],
            },
        )


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 2])
def test_invalid_split_fraction(fraction):
    with pytest.raises(ValueError):
        assign_splits([], val_fraction=fraction)


def test_static_task_metadata_supports_native_eval_tables_and_rollout_export(tmp_path):
    from slime.utils.eval_rollouts import select_eval_rollouts
    from slime.utils.rollout_export import export_training_rollouts

    example = normalize(
        "math",
        {
            "id": 1,
            "original_id": 1,
            "instance": 0,
            "question": "Add 1 and 2.",
            "answer": "#### 3",
        },
    )
    samples = [
        SimpleNamespace(
            index=i,
            prompt="Add 1 and 2.",
            response=r"\boxed{3}",
            response_length=5,
            reward={"reward": 1.0, "accuracy": 1.0},
            metadata=example["metadata"],
            label=None,
            status=SimpleNamespace(value="completed"),
        )
        for i in range(8)
    ]
    rows = select_eval_rollouts(samples, 8, "accuracy")
    assert len(rows) == 8 and all(row["environment"] == "math" for row in rows)
    destination = export_training_rollouts(
        str(tmp_path / "{rollout_id}.jsonl"), 0, samples, "pilot", "group"
    )
    exported = [json.loads(line) for line in destination.read_text().splitlines()]
    assert len(exported) == 8 and all(row["accuracy"] == 1 for row in exported)
