"""Frozen PCFG v3 string-tree training pools and paired ID evaluations."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .experiment import RLVE_ROOT, SOURCE_ROOT, function_code, prompt, sha256
from .next_experiments import eval_inputs, sample_input, verified_rows, walk, write_json, write_jsonl
from envs.core import REGISTRY


DEST = RLVE_ROOT / "data/PROGRAM-TREES-PCFG-V3"
MODEL = "davidheineman/RL-Base-R1-Distill-Q1.5B"
DEPTHS = (2, 3, 4, 5)
BRANCH_PROBABILITY = 1 / 3
TRAIN_PROMPTS = 2000
EVAL_PROMPTS_PER_DEPTH = 15
EVAL_VALUES = eval_inputs()


def take_even_chars(s: str) -> str:
    """Keep characters at even positions."""
    return s[::2]


def swap_case(s: str) -> str:
    """Switch each letter's case."""
    return s.swapcase()


def reverse_pairs(s: str) -> str:
    """Reverse each consecutive pair of characters."""
    return "".join(s[i:i + 2][::-1] for i in range(0, len(s), 2))


# All names are distinct. The three local functions fill out the available
# string-to-string primitives without changing rl-data or ongoing experiments.
N = (
    "compress_repeats", "remove_vowels", "sort_chars", "alternate_case",
    "mirror_str", "add_prefix", "insert_separator", "take_even_chars",
)
M = (
    "recursive_reverse", "shift_chars", "while_rotate", "add_suffix",
    "duplicate_every_char", "interlace_str", "swap_case", "reverse_pairs",
)
FIXED = {
    "add_prefix": "#", "insert_separator": "-", "shift_chars": 1,
    "while_rotate": 2, "add_suffix": "!", "interlace_str": "|",
}
LOCAL = {
    "take_even_chars": take_even_chars,
    "swap_case": swap_case,
    "reverse_pairs": reverse_pairs,
}


def primitive(name: str):
    fn = LOCAL[name] if name in LOCAL else REGISTRY[name].fn
    return fn, FIXED.get(name)


def make_tree(rng: random.Random, depth: int, alpha: float) -> dict:
    pool = M if rng.random() < alpha else N
    name = rng.choice(pool)
    child_count = 0 if depth == 1 else (2 if rng.random() < BRANCH_PROBABILITY else 1)
    return {
        "fn": name,
        "fixed": FIXED.get(name),
        "children": [make_tree(rng, depth - 1, alpha) for _ in range(child_count)],
        "masked": False,
    }


def expression(tree: dict) -> str:
    children = tree["children"]
    argument = "x" if not children else " + ".join(expression(child) for child in children)
    if len(children) == 2:
        argument = f"({argument})"
    return f"{tree['fn']}({argument})"


def evaluate(tree: dict, value: str) -> str:
    children = tree["children"]
    argument = value if not children else "".join(evaluate(child, value) for child in children)
    fn, fixed = primitive(tree["fn"])
    return fn(argument) if fixed is None else fn(argument, fixed)


def source_code(tree: dict) -> str:
    definitions = [function_code(*primitive(name)) for name in sorted({n["fn"] for n in walk(tree)})]
    definitions.append(f"def main(x):\n    return {expression(tree)}")
    return "\n\n".join(definitions) + "\n"


class Sampler:
    def __init__(self, *, seed: int, alpha: float, run_id: str):
        self.rng = random.Random(seed)
        self.alpha = alpha
        self.run_id = run_id

    def draw(self, *, depth: int | None = None, value: str | None = None) -> dict:
        depth = self.rng.choice(DEPTHS) if depth is None else depth
        if depth not in DEPTHS:
            raise ValueError(f"Unsupported depth: {depth}")
        tree = make_tree(self.rng, depth, self.alpha)
        if value is None:
            value = sample_input(self.rng)
            while value in EVAL_VALUES:
                value = sample_input(self.rng)
        return {
            "user_prompt": prompt(source_code(tree), value),
            "label": evaluate(tree, value),
            "metadata": {
                "run_id": self.run_id, "environment": f"depth_{depth}",
                "input": value, "depth": depth,
                "breadth": max(len(n["children"]) for n in walk(tree)),
                "active_functions": sum(1 for _ in walk(tree)), "tree": tree,
            },
        }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def shape(tree: dict) -> tuple:
    return tuple(shape(child) for child in tree["children"])


def prepare() -> dict:
    assert len(N) == len(M) == 8 and set(N).isdisjoint(M)
    eval_sets = {}
    for name, alpha in (("N_id", 0.0), ("M_id", 1.0)):
        sampler = Sampler(seed=12003, alpha=alpha, run_id=f"eval-{name}-v3")
        rows = [
            sampler.draw(depth=depth, value=EVAL_VALUES[(depth - 2) * EVAL_PROMPTS_PER_DEPTH + i])
            for depth in DEPTHS for i in range(EVAL_PROMPTS_PER_DEPTH)
        ]
        path = DEST / "eval" / f"{name}.jsonl"
        write_jsonl(path, verified_rows(rows))
        eval_sets[name] = {
            "path": str(path.relative_to(RLVE_ROOT)), "prompts": len(rows),
            "sha256": sha256(path),
        }

    runs = []
    for index in range(8):
        run_id = f"pcfg-m{index}of7-v3"
        spec = {
            "run_id": run_id, "kind": "pcfg_v3", "seed": 7000 + index,
            "alpha": index / 7, "N": list(N), "M": list(M),
            "depths": list(DEPTHS), "branch_probability": BRANCH_PROBABILITY,
            "budget": 16000, "evals": ["N_id", "M_id"],
            "stop_evals": ["N_id", "M_id"],
        }
        spec_path = DEST / "specs" / f"{run_id}.json"
        write_json(spec_path, spec)
        sampler = Sampler(seed=spec["seed"], alpha=spec["alpha"], run_id=run_id)
        rows = [sampler.draw() for _ in range(TRAIN_PROMPTS)]
        train_path = DEST / "train" / f"{run_id}.jsonl"
        write_jsonl(train_path, rows)
        runs.append({
            **spec,
            "spec_path": str(spec_path.relative_to(RLVE_ROOT)),
            "spec_sha256": sha256(spec_path),
            "train_path": str(train_path.relative_to(RLVE_ROOT)),
            "train_sha256": sha256(train_path),
            "train_prompts": len(rows),
            "max_label_chars": max(len(row["label"]) for row in rows),
            "max_prompt_chars": max(len(row["user_prompt"]) for row in rows),
        })

    manifest = {
        "model": MODEL, "model_revision": "a01bb85838a2157c12519a36138cbce4855118df",
        "source": "rl-data/envs/strings.py", "source_sha256": sha256(SOURCE_ROOT / "envs/strings.py"),
        "generator_sha256": sha256(Path(__file__)),
        "max_initial_tokens": 1024, "answer_forcing_extra_tokens": 128,
        "rollouts_per_prompt": 8, "eval_samples_per_prompt": 8,
        "eval_interval_updates": 50, "early_stop_accuracy": 0.98,
        "eval_prompts_per_depth": EVAL_PROMPTS_PER_DEPTH,
        "eval_sets": eval_sets, "runs": runs,
    }
    write_json(DEST / "manifest.json", manifest)
    return verify()


def verify() -> dict:
    manifest = json.loads((DEST / "manifest.json").read_text())
    assert manifest["model"] == MODEL
    assert manifest["source_sha256"] == sha256(SOURCE_ROOT / "envs/strings.py")
    assert manifest["generator_sha256"] == sha256(Path(__file__))
    assert len(manifest["runs"]) == 8
    assert {run["alpha"] for run in manifest["runs"]} == {i / 7 for i in range(8)}
    eval_rows = {}
    for name, pool in (("N_id", N), ("M_id", M)):
        entry = manifest["eval_sets"][name]
        path = RLVE_ROOT / entry["path"]
        assert sha256(path) == entry["sha256"]
        rows = read_jsonl(path)
        assert len(rows) == entry["prompts"] == 60
        assert Counter(row["metadata"]["depth"] for row in rows) == {depth: 15 for depth in DEPTHS}
        assert all({n["fn"] for n in walk(row["metadata"]["tree"])} <= set(pool) for row in rows)
        verified_rows(rows)
        eval_rows[name] = rows
    for n_row, m_row in zip(eval_rows["N_id"], eval_rows["M_id"]):
        assert n_row["metadata"]["input"] == m_row["metadata"]["input"]
        assert n_row["metadata"]["depth"] == m_row["metadata"]["depth"]
        assert n_row["metadata"]["breadth"] == m_row["metadata"]["breadth"]
        assert shape(n_row["metadata"]["tree"]) == shape(m_row["metadata"]["tree"])
    for run in manifest["runs"]:
        spec_path, train_path = (RLVE_ROOT / run[key] for key in ("spec_path", "train_path"))
        assert sha256(spec_path) == run["spec_sha256"]
        assert sha256(train_path) == run["train_sha256"]
        spec = json.loads(spec_path.read_text())
        assert spec["N"] == list(N) and spec["M"] == list(M)
        assert spec["depths"] == list(DEPTHS)
        assert spec["branch_probability"] == BRANCH_PROBABILITY
        rows = read_jsonl(train_path)
        assert len(rows) == run["train_prompts"] == TRAIN_PROMPTS
        assert all(row["metadata"]["input"] not in EVAL_VALUES for row in rows)
        assert all(row["metadata"]["depth"] in DEPTHS for row in rows)
        assert all(row["metadata"]["breadth"] <= 2 for row in rows)
        assert max(len(row["label"]) for row in rows) == run["max_label_chars"]
        verified_rows(rows[:10] + rows[-10:])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    args = parser.parse_args()
    manifest = prepare() if args.action == "prepare" else verify()
    print(f"Verified {len(manifest['runs'])} v3 PCFG runs and two 60-prompt eval sets")


if __name__ == "__main__":
    main()
