"""Generate fixed training questions, disjoint from the student's randomized evaluation inputs.

Run in the pinned Miles container, which provides the RLVE generator dependencies.
"""

import argparse
import json
import random
from pathlib import Path

from async_lora.questions import question_id
from lora_teachers.plan import ROOT
from make_rlve_eval_set import _records_for_environment


def prepare(size=16, difficulty=1, seed=1234, *, allow_teacher_eval_overlap_for=()):
    protocol = json.loads((ROOT / f"data/DATA-SCALING-ASYNC-V1/n{size}.json").read_text())
    allowed = set(allow_teacher_eval_overlap_for)
    if allowed - set(protocol["environments"]):
        raise ValueError("Teacher-eval overlap exceptions must belong to the selected environment panel")
    questions, teacher_overlap = [], {}
    for index, environment in enumerate(protocol["environments"]):
        original = json.loads((ROOT / protocol["teacher_eval"][environment]).read_text())
        original_prompts = {row["user_prompt"] for row in original}
        held_out = set() if environment in allowed else set(original_prompts)
        for path in (protocol["id_eval"], protocol["test_eval"]):
            held_out.update(row["user_prompt"] for row in json.loads((ROOT / path).read_text()))
        row = _records_for_environment(
            environment,
            difficulty,
            1,
            90_000_000 + index * 100_000 + seed % 10000,
            seed + index,
            seen_prompts=held_out,
        )[0]
        metadata = json.loads(row["metadata"])
        metadata["difficulty"] = difficulty
        row["metadata"] = metadata
        metadata["question_id"] = question_id(row)
        if row["user_prompt"] in original_prompts:
            teacher_overlap[environment] = {
                "question_id": metadata["question_id"],
                "unique_teacher_eval_prompts": len(original_prompts),
                "student_validation_and_test_overlap": False,
            }
        questions.append(row)
        print(f"Frozen {environment}: {metadata['question_id'][:12]}", flush=True)
    result = {
        "version": 1,
        "size": size,
        "environments": protocol["environments"],
        "difficulty": difficulty,
        "seed": seed,
        "questions": questions,
    }
    if teacher_overlap:
        result["teacher_eval_overlap"] = teacher_overlap
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=(16, 32), default=16)
    parser.add_argument("--difficulty", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-teacher-eval-overlap-for",
        action="append",
        default=[],
        metavar="ENVIRONMENT",
        help="Allow a named tiny-domain environment to overlap its teacher eval; student val/test stay disjoint",
    )
    args = parser.parse_args()
    destination = (
        args.output
        or ROOT / f"data/DATA-SCALING-ASYNC-LORA-V1/single_questions_n{args.size}_d{args.difficulty}_s{args.seed}.json"
    )
    if destination.exists():
        raise FileExistsError(f"Fixed question sets are immutable: {destination}")
    random.seed(args.seed)
    result = prepare(
        args.size, args.difficulty, args.seed, allow_teacher_eval_overlap_for=args.allow_teacher_eval_overlap_for
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
