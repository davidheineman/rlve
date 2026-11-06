"""Build the 24 fixed string program-tree datasets and their manifest.

Run from the RLVE repository: python3 -m program_trees.experiment
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import random
import sys
import textwrap
from pathlib import Path


RLVE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = RLVE_ROOT.parent / "rl-data"
sys.path.insert(0, str(SOURCE_ROOT))
from envs import strings  # noqa: E402
from envs.core import REGISTRY  # noqa: E402


SETS = {
    "A": [
        ("compress_repeats", None),
        ("remove_vowels", None),
        ("mirror_str", None),
        ("sort_chars", None),
        ("shift_chars", 1),
        ("alternate_case", None),
    ],
    "B": [
        ("duplicate_every_char", None),
        ("recursive_reverse", None),
        ("add_suffix", "!"),
        ("add_prefix", "#"),
        ("insert_separator", "-"),
        ("while_rotate", 2),
    ],
}
FAMILIES = ("depth", "breadth")
TRAIN_COUNT = 500
EVAL_COUNT = 32
ROLLOUTS_PER_PROMPT = 8
EVAL_SAMPLES_PER_PROMPT = 8
EVAL_INTERVAL = 50
MAX_RESPONSE_TOKENS = 1024
SEED = 20261019


def functions(set_name: str):
    """Return frozen primitive functions and fixed optional arguments."""
    result = []
    for name, fixed in SETS[set_name]:
        primitive = REGISTRY[name]
        if primitive.domain != strings.DOMAIN:
            raise ValueError(f"{name} is outside the string domain")
        result.append((primitive.fn, fixed))
    return result


def function_code(fn, fixed) -> str:
    """Copy a primitive definition into runnable prompt code."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    original = definition.name
    definition.decorator_list = []
    if fixed is not None:
        if len(definition.args.args) != 2 or definition.args.defaults:
            raise ValueError(f"Expected a two-argument primitive: {original}")
        definition.args.defaults = [ast.Constant(fixed)]

    tree = ast.fix_missing_locations(tree)
    return ast.unparse(tree.body[0])


def main_expression(set_name: str, family: str, n: int) -> str:
    names = [fn.__name__ for fn, _ in functions(set_name)[:n]]
    if family == "depth":
        expression = "x"
        for name in names:
            expression = f"{name}({expression})"
        return expression
    if family == "breadth":
        if n == 1:
            return f"{names[0]}(x)"
        return "''.join([" + ", ".join(f"{name}(x)" for name in names) + "])"
    raise ValueError(family)


def source_code(set_name: str, family: str, n: int) -> str:
    definitions = [function_code(fn, fixed) for fn, fixed in functions(set_name)[:n]]
    return "\n\n".join(definitions + [f"def main(x):\n    return {main_expression(set_name, family, n)}"]) + "\n"


def answer(set_name: str, family: str, n: int, value: str) -> str:
    fns = functions(set_name)[:n]

    def call(fn, fixed, x):
        return fn(x) if fixed is None else fn(x, fixed)

    if family == "depth":
        for fn, fixed in fns:
            value = call(fn, fixed, value)
        return value
    return "".join(call(fn, fixed, value) for fn, fixed in fns)


def prompt(code: str, value: str) -> str:
    call = f"main({json.dumps(value)})"
    return (
        "Use the following code:\n\n```python\n"
        + code
        + f"\n>>> {call}\n>>>\n```\n\n"
        + f"Can you predict the output of `{call}` without writing any code? "
        + "Please reason and put your final answer in the following JSON format: "
        + '{"output": <your output>}, where <your output> is the final output string.'
    )


def inputs(count: int = TRAIN_COUNT + EVAL_COUNT) -> list[str]:
    """One disjoint train/eval input sequence, shared by all 24 runs."""
    rng = random.Random(SEED)
    alphabet = "abcdeilmnoprstuvxyz"
    seen: set[str] = set()
    values = []
    while len(values) < count:
        length = rng.randint(4, 9)
        chars = []
        for _ in range(length):
            ch = chars[-1] if chars and rng.random() < 0.22 else rng.choice(alphabet)
            if rng.random() < 0.13:
                ch = ch.upper()
            chars.append(ch)
        value = "".join(chars)
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values


def run_id(set_name: str, family: str, n: int) -> str:
    return f"{set_name}-{family}{n}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frozen(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"Frozen dataset differs: {path}")
    path.write_text(content)


def generate(output_dir: Path) -> dict:
    values = inputs()
    run_rows = []
    for set_name in SETS:
        for family in FAMILIES:
            for n in range(1, 7):
                identifier = run_id(set_name, family, n)
                code = source_code(set_name, family, n)
                # Catch source-generation mistakes before freezing labels.
                namespace = {}
                exec(compile(code, f"<{identifier}>", "exec"), namespace)
                rows = []
                for value in values:
                    result = answer(set_name, family, n, value)
                    if namespace["main"](value) != result:
                        raise AssertionError(f"Generated source differs: {identifier}, {value}")
                    rows.append({
                        "user_prompt": prompt(code, value),
                        "label": result,
                        "metadata": {"run_id": identifier, "input": value,
                                     "set": set_name, "family": family, "size": n},
                    })
                paths = {}
                for split, selected in (("train", rows[:TRAIN_COUNT]), ("eval", rows[TRAIN_COUNT:])):
                    path = output_dir / identifier / f"{split}.jsonl"
                    write_frozen(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
                    paths[split] = {"path": str(path.relative_to(RLVE_ROOT)), "sha256": sha256(path),
                                    "prompts": len(selected)}
                run_rows.append({
                    "id": identifier, "set": set_name, "family": family,
                    "size": n, "depth": n if family == "depth" else 1,
                    "breadth": n if family == "breadth" else 1,
                    "functions": [fn.__name__ for fn, _ in functions(set_name)[:n]],
                    "train": paths["train"], "eval": paths["eval"],
                })
    manifest = {
        "description": "String program-tree breadth and depth sweep",
        "seed": SEED, "source": str((SOURCE_ROOT / "envs/strings.py").relative_to(RLVE_ROOT.parent)),
        "source_sha256": sha256(SOURCE_ROOT / "envs/strings.py"),
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "train_prompts_per_run": TRAIN_COUNT, "eval_prompts_per_run": EVAL_COUNT,
        "eval_kind": "id", "eval_samples_per_prompt": EVAL_SAMPLES_PER_PROMPT,
        "eval_interval": EVAL_INTERVAL,
        "rollouts_per_prompt": ROLLOUTS_PER_PROMPT, "updates": TRAIN_COUNT,
        "train_rollouts_per_run": TRAIN_COUNT * ROLLOUTS_PER_PROMPT,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "sets": {label: [{"function": name, "fixed_second_argument": fixed}
                         for name, fixed in sequence] for label, sequence in SETS.items()},
        "runs": run_rows,
    }
    write_frozen(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RLVE_ROOT / "data/PROGRAM-TREES")
    args = parser.parse_args()
    manifest = generate(args.output_dir.resolve())
    print(f"Prepared {len(manifest['runs'])} runs, {TRAIN_COUNT} train and {EVAL_COUNT} eval prompts each")


if __name__ == "__main__":
    main()
