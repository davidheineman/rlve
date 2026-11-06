"""Online samplers and frozen evaluations for the next string-tree studies."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .experiment import RLVE_ROOT, SETS, SOURCE_ROOT, function_code, prompt, sha256
from envs.core import REGISTRY


DATA_ROOT = RLVE_ROOT / "data/PROGRAM-TREES-NEXT"
PRIMITIVES = {name: fixed for sequence in SETS.values() for name, fixed in sequence}
SINGLE_CHAIN = SETS["A"][:5]
ALL_FUNCTIONS = SETS["A"] + SETS["B"]
_single_rng = random.Random(20261022)
SINGLE_CHAINS = [SINGLE_CHAIN, SETS["B"][:5]]
while len(SINGLE_CHAINS) < 4:
    candidate = _single_rng.sample(ALL_FUNCTIONS, 5)
    if candidate not in SINGLE_CHAINS:
        SINGLE_CHAINS.append(candidate)
MEGA_FUNCTIONS = [
    ("compress_repeats", None), ("remove_vowels", None), ("sort_chars", None),
    ("recursive_reverse", None), ("alternate_case", None),
    ("shift_chars", 1), ("while_rotate", 2),
]
PCFG_N = [
    ("compress_repeats", None), ("remove_vowels", None),
    ("sort_chars", None), ("alternate_case", None),
]
PCFG_M = [
    ("recursive_reverse", None), ("shift_chars", 1),
    ("while_rotate", 2), ("add_suffix", "!"),
]


def multi_chains(count: int = 32) -> list[list[list]]:
    """Distinct five-function trees; include all four single-env controls."""
    rng = random.Random(20261020)
    chains = [[list(op) for op in chain] for chain in SINGLE_CHAINS]
    seen = {tuple(tuple(op) for op in chain) for chain in chains}
    while len(chains) < count:
        chain = [list(op) for op in rng.sample(ALL_FUNCTIONS, 5)]
        key = tuple(tuple(op) for op in chain)
        if key not in seen:
            seen.add(key)
            chains.append(chain)
    return chains


def sample_input(rng: random.Random) -> str:
    alphabet = "abcdeilmnoprstuvxyz"
    chars = []
    for _ in range(rng.randint(4, 9)):
        char = chars[-1] if chars and rng.random() < 0.22 else rng.choice(alphabet)
        chars.append(char.upper() if rng.random() < 0.13 else char)
    return "".join(chars)


def eval_inputs(count: int = 128) -> list[str]:
    rng = random.Random(20261021)
    values = []
    seen = set()
    while len(values) < count:
        value = sample_input(rng)
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


EVAL_INPUT_SET = set(eval_inputs())


def node(op, children=(), masked=False):
    name, fixed = op
    if name not in PRIMITIVES or PRIMITIVES[name] != fixed:
        raise ValueError(f"Unknown string primitive: {op}")
    return {"fn": name, "fixed": fixed, "children": list(children), "masked": masked}


def chain_tree(chain, depth):
    if not 1 <= depth <= len(chain):
        raise ValueError(f"Depth {depth} exceeds chain length {len(chain)}")
    branch = node(chain[0])
    for op in chain[1:depth]:
        branch = node(op, [branch])
    return branch


def mega_tree(rng, active_count):
    active = set(rng.sample(range(7), active_count))

    def build(index):
        children = [] if index >= 3 else [build(index * 2 + 1), build(index * 2 + 2)]
        return node(MEGA_FUNCTIONS[index], children, masked=index not in active)

    return build(0)


def mega_path_tree(rng, active_count):
    active = set(rng.sample(range(7), active_count))
    branch = node(MEGA_FUNCTIONS[6], masked=6 not in active)
    for index in range(5, -1, -1):
        branch = node(MEGA_FUNCTIONS[index], [branch], masked=index not in active)
    return branch


def pcfg_tree(rng, depth, alpha, branch_probability=0.35):
    pool = PCFG_M if rng.random() < alpha else PCFG_N
    op = rng.choice(pool)
    if depth == 1:
        return node(op)
    child_count = 2 if rng.random() < branch_probability else 1
    return node(op, [pcfg_tree(rng, depth - 1, alpha, branch_probability) for _ in range(child_count)])


def continued_tree(rng, depth):
    op = rng.choice(ALL_FUNCTIONS)
    if depth == 1:
        return node(op)
    child_count = rng.choice((1, 2))
    return node(op, [node(rng.choice(ALL_FUNCTIONS)) for _ in range(child_count)])


def expression(tree):
    children = tree["children"]
    if not children:
        argument = "x"
    elif len(children) == 1:
        argument = expression(children[0])
    elif len(children) == 2:
        argument = f"({expression(children[0])} + {expression(children[1])})"
    else:
        raise ValueError("String trees support at most two children per node")
    name = f"masked_{tree['fn']}" if tree["masked"] else tree["fn"]
    return f"{name}({argument})"


def evaluate(tree, value):
    children = tree["children"]
    if not children:
        argument = value
    elif len(children) == 1:
        argument = evaluate(children[0], value)
    else:
        argument = "".join(evaluate(child, value) for child in children)
    if tree["masked"]:
        return argument
    fn = REGISTRY[tree["fn"]].fn
    return fn(argument) if tree["fixed"] is None else fn(argument, tree["fixed"])


def source_code(tree):
    used = {}

    def visit(current):
        used[(current["fn"], current["masked"])] = current["fixed"]
        for child in current["children"]:
            visit(child)

    visit(tree)
    definitions = []
    for (name, masked), fixed in sorted(used.items()):
        if masked:
            definitions.append(f"def masked_{name}(s):\n    \"\"\"A masked tree node; pass through its input.\"\"\"\n    return s")
        else:
            definitions.append(function_code(REGISTRY[name].fn, fixed))
    definitions.append(f"def main(x):\n    return {expression(tree)}")
    return "\n\n".join(definitions) + "\n"


def tree_depth(tree):
    return 1 + max((tree_depth(child) for child in tree["children"]), default=0)


def tree_breadth(tree):
    return max([len(tree["children"])] + [tree_breadth(child) for child in tree["children"]])


class ProgramTreeSampler:
    def __init__(self, spec):
        self.spec = spec
        self.rng = random.Random(spec["seed"])
        self.draws = 0

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def state_dict(self):
        return {"rng": self.rng.getstate(), "draws": self.draws}

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        self.draws = state["draws"]

    def draw(self, *, depth=None, active_count=None, value=None):
        spec = self.spec
        kind = spec["kind"]
        rng = self.rng
        if kind in ("diversity_single", "diversity_multi"):
            depth = depth or rng.randint(1, 5)
            chain = spec["chain"] if kind == "diversity_single" else rng.choice(spec["chains"])
            tree = chain_tree(chain, depth)
            environment = f"depth_{depth}"
        elif kind == "mega_primitive":
            tree = node(spec["primitive"])
            environment = spec["primitive"][0]
        elif kind == "mega_tree":
            active_count = active_count or rng.randint(1, 7)
            tree = mega_tree(rng, active_count)
            environment = f"active_{active_count}"
        elif kind == "mega_path":
            active_count = active_count or rng.randint(1, 7)
            tree = mega_path_tree(rng, active_count)
            environment = f"active_{active_count}"
        elif kind == "pcfg":
            depth = depth or rng.randint(1, 3)
            tree = pcfg_tree(rng, depth, spec["alpha"], spec["branch_probability"])
            environment = f"depth_{depth}"
        elif kind == "continued":
            depth = depth or rng.randint(1, 2)
            tree = continued_tree(rng, depth)
            environment = f"depth_{depth}_breadth_{tree_breadth(tree)}"
        else:
            raise ValueError(kind)
        if value is None:
            value = sample_input(rng)
            while value in EVAL_INPUT_SET:
                value = sample_input(rng)
        code = source_code(tree)
        answer = evaluate(tree, value)
        self.draws += 1
        return {
            "user_prompt": prompt(code, value),
            "label": answer,
            "metadata": {
                "run_id": spec["run_id"], "environment": environment,
                "input": value, "depth": tree_depth(tree), "breadth": tree_breadth(tree),
                "active_functions": sum(not n["masked"] for n in walk(tree)),
                "tree": tree,
            },
        }


def walk(tree):
    yield tree
    for child in tree["children"]:
        yield from walk(child)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def verified_rows(rows):
    for row in rows:
        namespace = {}
        text = row["user_prompt"]
        code = text.split("```python\n", 1)[1].split("\n```", 1)[0].split("\n>>> main(", 1)[0]
        exec(compile(code, "<program-tree-eval>", "exec"), namespace)
        assert namespace["main"](row["metadata"]["input"]) == row["label"]
    return rows


def specs():
    items = []
    for index in range(4):
        items.append({"run_id": f"div-single-s{index + 1}", "family": "diversity", "kind": "diversity_single",
                      "seed": 3000 + index, "chain": [list(op) for op in SINGLE_CHAINS[index]], "budget": 8000,
                      "evals": [f"single_s{index + 1}_id", "multi_id"],
                      "stop_evals": [f"single_s{index + 1}_id"]})
    chains = multi_chains()
    for index in range(2):
        items.append({"run_id": f"div-multi-s{index + 1}", "family": "diversity", "kind": "diversity_multi",
                      "seed": 4000 + index, "chains": chains, "budget": 8000,
                      "evals": ["single_pooled_id", "multi_id"], "stop_evals": ["multi_id"]})
    for index, primitive in enumerate(MEGA_FUNCTIONS):
        items.append({"run_id": f"mega-primitive-{primitive[0]}", "family": "mega", "kind": "mega_primitive",
                      "seed": 5000 + index, "primitive": list(primitive), "budget": 8000,
                      "evals": ["binary_id", "path_id", "primitives_id"],
                      "stop_evals": [f"primitives_id/{primitive[0]}"]})
    items.append({"run_id": "mega-tree", "family": "mega", "kind": "mega_tree", "seed": 6000,
                  "budget": 80000, "evals": ["binary_id", "path_id", "primitives_id"],
                  "stop_evals": ["binary_id"]})
    items.append({"run_id": "mega-path", "family": "mega", "kind": "mega_path", "seed": 6001,
                  "budget": 80000, "evals": ["binary_id", "path_id", "primitives_id"],
                  "stop_evals": ["path_id"]})
    for index in range(8):
        items.append({"run_id": f"pcfg-m{index}of7", "family": "pcfg", "kind": "pcfg",
                      "seed": 7000 + index, "alpha": index / 7, "branch_probability": 0.35, "budget": 8000,
                      "evals": ["N_id", "M_id"], "stop_evals": ["N_id", "M_id"]})
    items.append({"run_id": "continued-b2d2", "family": "continued", "kind": "continued",
                  "seed": 8000, "budget": 20000, "evals": ["val_envs"], "stop_evals": ["val_envs"]})
    return items


def build_eval_sets(spec_by_id):
    values = eval_inputs()
    datasets = {}
    multi = ProgramTreeSampler({**spec_by_id["div-multi-s1"], "seed": 10002})
    for index in range(4):
        name = f"single_s{index + 1}_id"
        sampler = ProgramTreeSampler({**spec_by_id[f"div-single-s{index + 1}"], "seed": 10001 + index})
        rows = [sampler.draw(depth=depth, value=values[(depth - 1) * 10 + i])
                for depth in range(1, 6) for i in range(10)]
        datasets[name] = verified_rows(rows)
    pooled = [ProgramTreeSampler({**spec_by_id[f"div-single-s{index + 1}"], "seed": 10101 + index})
              for index in range(4)]
    datasets["single_pooled_id"] = verified_rows([
        pooled[(i + depth - 1) % 4].draw(depth=depth, value=values[(depth - 1) * 10 + i])
        for depth in range(1, 6) for i in range(10)
    ])
    for name, sampler in (("multi_id", multi),):
        rows = [sampler.draw(depth=depth, value=values[(depth - 1) * 10 + i])
                for depth in range(1, 6) for i in range(10)]
        datasets[name] = verified_rows(rows)
    for name, run_id, seed in (("binary_id", "mega-tree", 10003),
                               ("path_id", "mega-path", 10004)):
        mega = ProgramTreeSampler({**spec_by_id[run_id], "seed": seed})
        datasets[name] = verified_rows([
            mega.draw(active_count=active, value=values[(active - 1) * 8 + i])
            for active in range(1, 8) for i in range(8)
        ])
    rows = []
    for index, primitive in enumerate(MEGA_FUNCTIONS):
        sampler = ProgramTreeSampler({"run_id": f"eval-{primitive[0]}", "kind": "mega_primitive",
                                      "primitive": list(primitive), "seed": 11000 + index})
        rows.extend(sampler.draw(value=values[56 + index * 4 + i]) for i in range(4))
    datasets["primitives_id"] = verified_rows(rows)
    for name, alpha in (("N_id", 0.0), ("M_id", 1.0)):
        sampler = ProgramTreeSampler({"run_id": f"eval-{name}", "kind": "pcfg", "seed": 12000,
                                      "alpha": alpha, "branch_probability": 0.35})
        datasets[name] = verified_rows([
            sampler.draw(depth=depth, value=values[(depth - 1) * 10 + i])
            for depth in range(1, 4) for i in range(10)
        ])
    sampler = ProgramTreeSampler({**spec_by_id["continued-b2d2"], "seed": 13000})
    datasets["val_envs"] = verified_rows([
        sampler.draw(depth=depth, value=values[84 + depth_index * 16 + i])
        for depth_index, depth in enumerate((1, 2)) for i in range(16)
    ])
    return datasets


def prepare(root=DATA_ROOT):
    root = Path(root)
    runs = specs()
    by_id = {item["run_id"]: item for item in runs}
    eval_sets = build_eval_sets(by_id)
    for item in runs:
        write_json(root / "specs" / f"{item['run_id']}.json", item)
    for name, rows in eval_sets.items():
        write_jsonl(root / "eval" / f"{name}.jsonl", rows)
    manifest = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "seed": 20261020,
        "source": "rl-data/envs/strings.py", "source_sha256": sha256(SOURCE_ROOT / "envs/strings.py"),
        "generator_sha256": sha256(Path(__file__)), "max_initial_tokens": 1024,
        "answer_forcing_extra_tokens": 128, "rollouts_per_prompt": 8, "eval_samples_per_prompt": 8,
        "eval_interval_updates": 50, "early_stop_accuracy": 0.98,
        "eval_sets": {name: {"path": str((root / "eval" / f"{name}.jsonl").relative_to(RLVE_ROOT)),
                              "prompts": len(rows), "sha256": sha256(root / "eval" / f"{name}.jsonl")}
                      for name, rows in eval_sets.items()},
        "runs": [{**item, "spec_path": str((root / "specs" / f"{item['run_id']}.json").relative_to(RLVE_ROOT)),
                  "spec_sha256": sha256(root / "specs" / f"{item['run_id']}.json")}
                 for item in runs],
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def verify(root=DATA_ROOT):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["source_sha256"] == sha256(SOURCE_ROOT / "envs/strings.py")
    assert manifest["generator_sha256"] == sha256(Path(__file__))
    assert len(manifest["runs"]) == 24
    assert Counter(run["family"] for run in manifest["runs"]) == {"diversity": 6, "mega": 9, "pcfg": 8, "continued": 1}
    assert len({tuple(chain) for chain in SINGLE_CHAINS}) == 4
    assert all([list(op) for op in chain] in multi_chains() for chain in SINGLE_CHAINS)
    assert {name for name, _ in PCFG_N}.isdisjoint({name for name, _ in PCFG_M})
    for name, item in manifest["eval_sets"].items():
        path = RLVE_ROOT / item["path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert item["sha256"] == sha256(path) and len(rows) == item["prompts"]
        verified_rows(rows)
        if name.startswith("single_") or name == "multi_id":
            assert Counter(row["metadata"]["depth"] for row in rows) == {depth: 10 for depth in range(1, 6)}
        elif name in ("N_id", "M_id"):
            assert Counter(row["metadata"]["depth"] for row in rows) == {depth: 10 for depth in range(1, 4)}
            allowed = {op[0] for op in (PCFG_N if name == "N_id" else PCFG_M)}
            assert all({part["fn"] for part in walk(row["metadata"]["tree"])} <= allowed for row in rows)
        elif name in ("binary_id", "path_id"):
            assert Counter(row["metadata"]["active_functions"] for row in rows) == {
                count: 8 for count in range(1, 8)
            }
            assert all(len(list(walk(row["metadata"]["tree"]))) == 7 for row in rows)
            assert all(row["metadata"]["depth"] == (3 if name == "binary_id" else 7) for row in rows)
        elif name == "primitives_id":
            assert Counter(row["metadata"]["environment"] for row in rows) == {
                primitive[0]: 4 for primitive in MEGA_FUNCTIONS
            }
    for run in manifest["runs"]:
        path = RLVE_ROOT / run["spec_path"]
        assert run["spec_sha256"] == sha256(path)
        sampler = ProgramTreeSampler.from_file(path)
        first = [sampler.draw() for _ in range(12)]
        verified_rows(first)
        state = sampler.state_dict()
        expected = [sampler.draw() for _ in range(4)]
        sampler.load_state_dict(state)
        assert [sampler.draw() for _ in range(4)] == expected
        assert run["budget"] % 8 == 0
    assert len(manifest["eval_sets"]["single_pooled_id"]["path"]) > 0
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    manifest = prepare(args.root) if args.action == "prepare" else verify(args.root)
    print(f"{args.action}: {len(manifest['runs'])} runs, {len(manifest['eval_sets'])} eval sets")


if __name__ == "__main__":
    main()
